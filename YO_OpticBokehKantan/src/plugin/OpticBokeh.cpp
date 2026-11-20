// Optic Bokeh - After Effects SmartFX wrapper around lenscore (camera lens blur).
// One effect; the blur engine is chosen per instance by the Quality popup (Fast / Standard / Best).
// v2.3: the same source also builds "Optic Bokeh Kantan" (the simple edition, a separate effect) with
// /DOB_KANTAN (plugin\build_kantan.bat): only its params (kantan_ui.inc), ReadSettings (Kantan controls
// -> kantan::ToSettings) and the popup handling differ; render, pickers and everything else are shared.
//
// Coordinate conventions used throughout:
//   "layer coords" = downsampled layer pixel coordinates, as used by SmartFX request/result
//   rects. A checked-out smart world's pixel (0,0) sits at layer coords (origin_x, origin_y).
// Pixel layout: AE worlds are ARGB (alpha first); lenscore wants RGBA float, premultiplied.

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectSuites.h"
#include "AE_GeneralPlug.h"  // AEGP stream / utility suites (expression check in QUERY_DYNAMIC_FLAGS)
#include "adobesdk/DrawbotSuite.h"  // v2.3 thumbnail pickers (custom ECW UI)

#include "OpticBokehConfig.h"
#include "lenscore.h"
#include "lens_presets.h"
#include "hq/lenshq_seam.h"  // Quality = Best (core\hq)
#include "ob_strings.h"         // v2.3 English / Japanese UI strings
#include "ob_thumbs.h"          // v2.3 engine-rendered preset thumbnails
#include "ob_settings.h"        // v2.3: Settings + engine adapter (shared with Kantan)
#include "ob_mapx.h"           // v2.3: blur-map features (Map Blur, Focus Range, Front / Back, Focus Point)
#include "ob_gpu.h"            // v2.4: the GPU blur stage (gpu\obgpu)
#include "kantan.h"             // v2.3 Kantan: the link curves
#include "../platform/ob_platform.h"  // mac port: getenv / fopen (Windows: the v2.4.2 CRT calls)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

extern "C" DllExport PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra);

const A_long kOutFlags = PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_DEEP_COLOR_AWARE |
                         PF_OutFlag_SEND_UPDATE_PARAMS_UI | PF_OutFlag_CUSTOM_UI;  // CUSTOM_UI: v2.3 pickers
const A_long kOutFlags2 = PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
                          PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                          PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
static_assert(OB_OUTFLAGS == kOutFlags, "PiPL out_flags mismatch");
static_assert(OB_OUTFLAGS2 == kOutFlags2, "PiPL out_flags2 mismatch");
#define OB_STAGE PF_Stage_RELEASE
static_assert(
    OB_PIPL_VERSION ==
        PF_VERSION(OB_MAJOR_VERSION, OB_MINOR_VERSION, OB_BUG_VERSION, OB_STAGE, OB_BUILD_VERSION),
    "PiPL version mismatch");

namespace {
using namespace obfx;  // ob_settings.h

// AE effect worlds hold STRAIGHT (unpremultiplied) colour at 8, 16 and 32 bpc, and AE reads the
// output world as straight too. Verified by host test in AE 2026 (26.5.3): with `true` here,
// soft-alpha layers got dark/grey fringes, and the rendered result equalled
// engine(straight input) * alpha exactly (aetest/compare_alpha.py, mask case).
// The converters premultiply on input and unpremultiply on output.
const bool kHostPixelsPremultiplied = false;

// kDisplayGamma, the popup enums, Settings and the engine adapter: ob_settings.h (v2.3, shared with Kantan)

// ------------------------------------------------------------------ params ---

// Param indices (order in the Effect Controls). Saved projects match params by disk ID, so the
// order may change between versions; disk IDs below may not.
// ---- Optic Bokeh Kantan (simple edition, v2.3): its own params and disk IDs (a separate effect with
// its own match name, so the numbering starts at 1). The index names shared with the full effect
// (OB_PRESET, OB_QUALITY, ...) let the common code (pickers, render, grain flags) serve both editions.
enum
{
    OB_INPUT = 0,
    OB_PRESET,          // "Lens": the 16 lens presets, no "Custom"
    OB_LENS_PICKER,
    OB_QUALITY,
    OB_VIEW,
    OB_RADIUS,
    OB_MAP_GROUP_START,
    OB_MAP_LAYER,
    OB_MAP_CHANNEL,
    OB_MAP_INVERT,
    OB_FOCUS,
    OB_FOCUS_POINT_ON,
    OB_FOCUS_POINT,
    OB_FOCUS_RANGE,
    OB_MAP_SCALE,       // "Depth Strength" (%)
    OB_MAP_GAMMA,       // "Depth Curve"
    OB_FRONT_BLUR,
    OB_BACK_BLUR,
    OB_MAP_BLUR,
    OB_MAP_GROUP_END,
    OB_APERTURE,
    OB_CHARACTER,
    OB_CHROMA,          // "Chromatic Aberration"
    OB_HIGHLIGHTS,
    OB_FILM_GROUP_START,
    OB_HAL_AMOUNT,
    OB_LA_STYLE,
    OB_STYLE_PICKER,
    OB_LA_AMOUNT,
    OB_GRAIN_STOCK,
    OB_STOCK_PICKER,
    OB_GRAIN_AMOUNT,
    OB_FILM_GROUP_END,
    OB_EDGE_MODE,
    OB_USE_GPU,         // v2.4, at the bottom
    OB_KANTAN_VERSION,  // hidden: the version of the link curves (kantan.h)
    OB_NUM_PARAMS
};
// Disk IDs: stable forever, append only.
enum
{
    KDISK_LENS = 1,
    KDISK_LENS_PICKER,
    KDISK_QUALITY,
    KDISK_VIEW,
    KDISK_RADIUS,
    KDISK_MAP_GROUP_START,
    KDISK_MAP_LAYER,
    KDISK_FOCUS,
    KDISK_MAP_INVERT,
    KDISK_MAP_GROUP_END,
    KDISK_APERTURE,
    KDISK_CHARACTER,
    KDISK_HIGHLIGHTS,
    KDISK_FILM_GROUP_START,
    KDISK_HALATION,
    KDISK_LA_STYLE,
    KDISK_STYLE_PICKER,
    KDISK_LA_AMOUNT,
    KDISK_GRAIN_STOCK,
    KDISK_STOCK_PICKER,
    KDISK_GRAIN_AMOUNT,
    KDISK_FILM_GROUP_END,
    KDISK_EDGE_MODE,
    KDISK_VERSION,      // 24
    KDISK_MAP_CHANNEL,  // 25
    KDISK_MAP_SCALE,    // 26 Depth Strength
    KDISK_MAP_GAMMA,    // 27 Depth Curve
    KDISK_FOCUS_RANGE,  // 28
    KDISK_FRONT_BLUR,   // 29
    KDISK_BACK_BLUR,    // 30
    KDISK_MAP_BLUR,     // 31
    KDISK_FOCUS_POINT_ON, // 32
    KDISK_FOCUS_POINT,  // 33
    KDISK_CHROMA,       // 34 Chromatic Aberration
    KDISK_USE_GPU       // 35 (v2.4)
};
static_assert(KDISK_VERSION == 24 && KDISK_CHROMA == 34 && KDISK_USE_GPU == 35, "Kantan disk ID layout v2.4");


// SmartFX checkout ids.
enum
{
    CHECKOUT_INPUT = 0,
    CHECKOUT_MAP_PROBE = 1,
    CHECKOUT_MAP = 2,
    CHECKOUT_INPUT_PROBE = 3,
    CHECKOUT_HL_PROBE = 4,
    CHECKOUT_HL = 5,
    CHECKOUT_MAP_PT_PROBE = 6,  // v2.3 Focus at Point: the map around the point
    CHECKOUT_MAP_PT = 7
};


// A layer parameter sampled onto the effect layer (stretch to fit), e.g. blur map, highlight layer.
struct LayerSampler
{
    bool has = false;
    double kx = 1, ky = 1;  // sampled-layer px per effect-layer px
    A_long w = 0, h = 0;    // sampled-layer bounds (downsampled)
    int channel = CH_LUMINANCE;
};

// Everything SMART_RENDER needs, computed in SMART_PRE_RENDER (heap, owned by AE).
struct RenderPlan
{
    Settings s;
    lenscore::Params ep;
    lenscore::HqParams hq;
    PF_LRect layerBounds; // effect layer bounds in layer coords (downsampled): the lens frame
    PF_LRect content;     // layerBounds U the input's max_result_rect: where input pixels can be
    bool marginExact = true;  // false: the window budget truncated the wide post-effect apron
    LayerSampler map, hl;
    obmap::Xform mx;      // v2.3 map features (inactive = the classic map path)
    int mapExt = 0;       // Map Blur: extra map pixels needed on every side
    LayerSampler mapPt;   // Focus at Point: the map around the point (its own checkout)
    A_long margin = 0;    // engine input margin (px)
    bool emptyResult = false;  // v2.2: the request lies outside max_result_rect, nothing was checked out
};

void DeletePlan(void* p)
{
    delete reinterpret_cast<RenderPlan*>(p);
}

inline double Ratio(const PF_RationalScale& r)
{
    return (r.den != 0 && r.num != 0) ? (double)r.num / (double)r.den : 1.0;
}

inline bool RectEmpty(const PF_LRect& r)
{
    return r.right <= r.left || r.bottom <= r.top;
}

inline PF_LRect Intersect(const PF_LRect& a, const PF_LRect& b)
{
    PF_LRect r;
    r.left = std::max(a.left, b.left);
    r.top = std::max(a.top, b.top);
    r.right = std::min(a.right, b.right);
    r.bottom = std::min(a.bottom, b.bottom);
    if (RectEmpty(r))
    {
        r.left = r.top = r.right = r.bottom = 0;
    }
    return r;
}

inline PF_LRect Union(const PF_LRect& a, const PF_LRect& b)
{
    if (RectEmpty(a))
        return b;
    if (RectEmpty(b))
        return a;
    PF_LRect r;
    r.left = std::min(a.left, b.left);
    r.top = std::min(a.top, b.top);
    r.right = std::max(a.right, b.right);
    r.bottom = std::max(a.bottom, b.bottom);
    return r;
}

inline PF_LRect Grow(PF_LRect r, A_long m)
{
    r.left -= m;
    r.top -= m;
    r.right += m;
    r.bottom += m;
    return r;
}

inline A_long ScaledExtent(A_long fullRes, double ds)
{
    return (A_long)std::ceil(fullRes * ds - 1e-6);
}

// Kantan: the Kantan controls -> kantan::ToSettings (the full effect's UI values, kantan.cpp); from
// there on everything is the full effect's code.
PF_Err ReadSettings(PF_InData* in_data, Settings& s)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef p[OB_NUM_PARAMS];
    bool out[OB_NUM_PARAMS] = {};
    const int idx[] = {OB_PRESET,      OB_QUALITY,      OB_VIEW,       OB_RADIUS,      OB_FOCUS,    OB_MAP_INVERT,
                       OB_APERTURE,    OB_CHARACTER,    OB_HIGHLIGHTS, OB_HAL_AMOUNT,  OB_LA_STYLE, OB_LA_AMOUNT,
                       OB_GRAIN_STOCK, OB_GRAIN_AMOUNT, OB_EDGE_MODE,  OB_KANTAN_VERSION, OB_MAP_CHANNEL, OB_MAP_SCALE,
                       OB_MAP_GAMMA,   OB_FOCUS_RANGE,  OB_FRONT_BLUR, OB_BACK_BLUR,   OB_MAP_BLUR, OB_FOCUS_POINT_ON,
                       OB_FOCUS_POINT, OB_CHROMA,       OB_USE_GPU};
    for (int i : idx)
    {
        AEFX_CLR_STRUCT(p[i]);
        ERR(PF_CHECKOUT_PARAM(in_data, i, in_data->current_time, in_data->time_step, in_data->time_scale, &p[i]));
        if (!err)
            out[i] = true;
    }
    if (!err)
    {
        kantan::Values v;
        v.lens = p[OB_PRESET].u.pd.value;
        v.quality = p[OB_QUALITY].u.pd.value;
        v.view = p[OB_VIEW].u.pd.value;
        v.radius = p[OB_RADIUS].u.fs_d.value;
        v.focusPct = p[OB_FOCUS].u.fs_d.value;
        v.invertMap = p[OB_MAP_INVERT].u.bd.value != 0;
        v.aperturePct = p[OB_APERTURE].u.fs_d.value;
        v.characterPct = p[OB_CHARACTER].u.fs_d.value;
        v.highlightsPct = p[OB_HIGHLIGHTS].u.fs_d.value;
        v.halationPct = p[OB_HAL_AMOUNT].u.fs_d.value;
        v.laStyle = p[OB_LA_STYLE].u.pd.value;
        v.laPct = p[OB_LA_AMOUNT].u.fs_d.value;
        v.grainStock = p[OB_GRAIN_STOCK].u.pd.value;
        v.grainPct = p[OB_GRAIN_AMOUNT].u.fs_d.value;
        v.edgeMode = p[OB_EDGE_MODE].u.pd.value;
        v.version = p[OB_KANTAN_VERSION].u.sd.value;
        v.mapChannel = p[OB_MAP_CHANNEL].u.pd.value;
        v.depthStrengthPct = p[OB_MAP_SCALE].u.fs_d.value;
        v.depthCurve = p[OB_MAP_GAMMA].u.fs_d.value;
        v.focusRangePct = p[OB_FOCUS_RANGE].u.fs_d.value;
        v.frontPct = p[OB_FRONT_BLUR].u.fs_d.value;
        v.backPct = p[OB_BACK_BLUR].u.fs_d.value;
        v.mapBlurPx = p[OB_MAP_BLUR].u.fs_d.value;
        v.focusPointOn = p[OB_FOCUS_POINT_ON].u.bd.value != 0;
        v.focusPointX = (double)p[OB_FOCUS_POINT].u.td.x_value / 65536.0;
        v.focusPointY = (double)p[OB_FOCUS_POINT].u.td.y_value / 65536.0;
        v.chromaPct = p[OB_CHROMA].u.fs_d.value;
        v.useGpu = p[OB_USE_GPU].u.pd.value;
        kantan::ToSettings(v, s);
    }
    for (int i : idx)
    {
        if (out[i])
            ERR2(PF_CHECKIN_PARAM(in_data, &p[i]));
    }
    return err;
}

// "Blur Amount" view: the engine's own per-pixel amount.
inline float EngineBlurAmount(const lenscore::Params& p, const float* mapValue)
{
    return mapValue ? lenscore::blurAmount(*mapValue, p) : 1.f;
}

// Kantan: the popups write nothing (the links read the preset tables at render time), so nothing is
// supervised and there is no "Custom" state. A picker click only sets its popup.
inline PF_ParamFlags SupFlag(A_long)
{
    return 0;
}
bool KantanPopupOnly(int, PF_ParamDef*[])
{
    return true;
}

struct AbortCtx
{
    PF_InData* in_data;
    PF_Err err;
};

bool AbortCallback(void* user)
{
    // lenscore calls this only on the thread that called render(), i.e. AE's render thread.
    AbortCtx* c = reinterpret_cast<AbortCtx*>(user);
    if (c->err)
        return true;
    c->err = PF_ABORT(c->in_data);
    return c->err != PF_Err_NONE;
}

// v2.2 (A10): an error the user sees gets a message (PF_OutFlag_DISPLAY_ERROR_MESSAGE + return_msg).
// Out of memory stays PF_Err_OUT_OF_MEMORY (AE purges its cache and reports it itself); cancel stays
// silent. Nothing returns PF_Err_INTERNAL_STRUCT_DAMAGED (AE would suggest saving and quitting).
PF_Err ReportError(PF_OutData* out_data, PF_Err err, const char* what)
{
    if (out_data && err != PF_Err_NONE && err != PF_Err_OUT_OF_MEMORY && err != PF_Interrupt_CANCEL)
    {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d: %s", OB_NAME, OB_MAJOR_VERSION,
                      OB_MINOR_VERSION, obs::T(what));
    }
    return err;
}

PF_Err ToPfErr(lenscore::Status st, const AbortCtx& ctx, PF_OutData* out_data)
{
    switch (st)
    {
    case lenscore::Status::Ok:
        return PF_Err_NONE;
    case lenscore::Status::Aborted:
        return ctx.err ? ctx.err : PF_Interrupt_CANCEL;
    case lenscore::Status::OutOfMemory:
        return PF_Err_OUT_OF_MEMORY;
    case lenscore::Status::BadArgs:
        return ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "the render engine rejected the frame (invalid size or settings).");
    default:
        return ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render engine error. The frame was not rendered.");
    }
}

// ---- process-wide render accounting (v2.2, review A3 / A4) ----
// These are the plug-in's ONLY mutable globals. They hold no per-instance or per-frame data, only
// counters shared by concurrent renders (multi-frame rendering runs SMART_RENDER on several threads
// at once, each frame otherwise spawning one engine worker per CPU):
//  - g_activeRenders: renders inside the engine right now; each splits the CPU, threads =
//    max(1, hardware threads / active), fixed for the frame when it starts.
//  - the Best gate: at most kMaxConcurrentBest (1) Best render runs at a time (each can hold GBs at UHD,
//    outside AE's memory manager); the others wait, polling PF_ABORT every 20 ms, so a cancelled
//    frame never hangs in the queue. Fast / Standard never wait.
std::atomic<int> g_activeRenders{0};
struct ActiveRenderGuard
{
    ActiveRenderGuard() { g_activeRenders.fetch_add(1, std::memory_order_relaxed); }
    ~ActiveRenderGuard() { g_activeRenders.fetch_sub(1, std::memory_order_relaxed); }
    ActiveRenderGuard(const ActiveRenderGuard&) = delete;
    ActiveRenderGuard& operator=(const ActiveRenderGuard&) = delete;
};
int RenderThreads()
{
    const unsigned hw = std::thread::hardware_concurrency();
    const int n = hw > 0 ? (int)hw : 4;
    const int active = std::max(1, g_activeRenders.load(std::memory_order_relaxed));
    return std::max(1, n / active);
}

bool g_serialConversions = false;  // test / timing hook: the v2.2 serial conversion loops

// v2.3 (wrapper change B): the pixel conversions (WorldToFloat, map sampling, FloatToWorld) run on
// worker threads, rows split in contiguous bands. Pure memory work, per pixel exactly what the serial
// loop computes; no AE call happens on a worker (abort is polled by the engine on the render thread).
// The thread count follows the same MFR split as the engine (this render counted as active).
int ConversionThreads()
{
    const unsigned hw = std::thread::hardware_concurrency();
    const int n = hw > 0 ? (int)hw : 4;
    const int active = std::max(0, g_activeRenders.load(std::memory_order_relaxed)) + 1;
    return g_serialConversions ? 1 : std::max(1, n / active);
}
template <typename F>
void ParallelRows(A_long begin, A_long end, double pixelsPerRow, F f)
{
    const A_long rows = end - begin;
    int n = ConversionThreads();
    if (rows * pixelsPerRow < 65536.0)  // small rects: threads cost more than they save
        n = 1;
    n = (int)std::min<A_long>(n, std::max<A_long>(1, rows));
    if (n <= 1)
    {
        f(begin, end);
        return;
    }
    std::vector<std::thread> pool;
    try
    {
        pool.reserve((size_t)n - 1);
        for (int i = 1; i < n; ++i)
            pool.emplace_back([&, i] { f(begin + rows * i / n, begin + rows * (i + 1) / n); });
    }
    catch (...)
    {
        // no more threads: the remaining bands run here (the result is the same either way)
        for (int i = (int)pool.size() + 1; i < n; ++i)
            f(begin + rows * i / n, begin + rows * (i + 1) / n);
    }
    f(begin, begin + rows / n);
    for (auto& th : pool)
        th.join();
}

// test hook (probes / mockhost only): true = the v2.2 uncropped engine path for every render version
bool g_disableCrop = false;
bool g_disableLut8 = false;  // test hook: the general per-pixel path for 8 bpc too

// One Best render at a time (v2.2 final): lenshq already uses every core, so two at once gave no
// throughput (1080p R160 premium85: 61.7 s alone vs 120.6 / 125.2 s for two in parallel) but more
// memory (peak 1.68 GB vs 2.31 GB; at UHD each render's work buffers grow ~4x). Other frames
// (Fast / Standard) never wait.
const int kMaxConcurrentBest = 1;
std::mutex g_bestM;
std::condition_variable g_bestCv;
int g_bestActive = 0;
class BestSlot
{
public:
    PF_Err Acquire(PF_InData* in_data)
    {
        std::unique_lock<std::mutex> lk(g_bestM);
        while (g_bestActive >= kMaxConcurrentBest)
        {
            if (g_bestCv.wait_for(lk, std::chrono::milliseconds(20), [] { return g_bestActive < kMaxConcurrentBest; }))
                break;
            lk.unlock();
            const PF_Err e = PF_ABORT(in_data);
            lk.lock();
            if (e)
                return e;
        }
        ++g_bestActive;
        held_ = true;
        return PF_Err_NONE;
    }
    ~BestSlot()
    {
        if (!held_)
            return;
        {
            std::lock_guard<std::mutex> lk(g_bestM);
            --g_bestActive;
        }
        g_bestCv.notify_one();
    }
    BestSlot() = default;
    BestSlot(const BestSlot&) = delete;
    BestSlot& operator=(const BestSlot&) = delete;

private:
    bool held_ = false;
};

// ---- THE BLUR-STAGE SEAM (selected by the Quality popup) ----
// Fast     : lenscore::blurAlpha with Params.fastBlur (set in MakeEngineParams).
// Standard : lenscore::blurAlpha, exact (v1.2 multi-resolution rule).
// Best     : the quality engine lenshq (core/hq, through lenshq_seam.h): continuous aperture model,
//            exact pixel-area integration, spectral fringes, in-disc texture (HqParams). Its margin
//            is lenshq::inputMarginHq (SmartPreRender). Slow by design; abort is polled on the
//            calling thread (AbortCallback -> PF_ABORT), so Esc / a parameter change cancels it.
const bool kBestUsesHq = true;
lenscore::Status BlurStage(const lenscore::Prepared& prep, const lenscore::Params& p, const lenscore::HqParams& hq,
                           int quality, float* dst, int stride, float* eLOut, lenscore::AbortFn ab, void* user)
{
    if (quality == QUALITY_BEST && kBestUsesHq)
    {
        const lenscore::Status st = lenshq::blurPrepared(prep, p, hq, dst, stride, eLOut, ab, user);
        // v2.2: out of memory -> release lenshq's shared lens-table cache before AE purges its own
        // cache and retries the frame (cached tables are shared_ptr: renders using them keep theirs)
        if (st == lenscore::Status::OutOfMemory)
            lenshq::purgeTableCache();
        return st;
    }
    return lenscore::blurAlpha(prep, p, dst, stride, eLOut, ab, user);
}

// Engine margins for the selected quality (Best uses the lenshq blur margin).
int EngineInputMargin(const lenscore::Params& p, const lenscore::HqParams& hq, int quality)
{
    return quality == QUALITY_BEST && kBestUsesHq ? lenshq::inputMarginHq(p, hq) : lenscore::inputMargin(p);
}
int EngineBlurMargin(const lenscore::Params& p, const lenscore::HqParams& hq, int quality)
{
    return quality == QUALITY_BEST && kBestUsesHq ? lenshq::blurMarginHq(p, hq) : lenscore::blurMargin(p);
}
// Output expansion beyond the content: the full blur spread + the post-effect spread capped at `cap`
// (a very wide veiling glow / halation halo, sigma up to 25 % of the diagonal, is not extended more
// than the cap beyond the layer content - its far tail is a few % of an already faint glow).
int EngineSpread(const lenscore::Params& p, const lenscore::HqParams& hq, int quality, A_long cap)
{
    return EngineBlurMargin(p, hq, quality) + std::min((int)cap, lenscore::postMargin(p));
}

// v2.1f memory guard. Post-effect cap: 6 % of the frame diagonal (>= 32 px).
A_long PostSpreadCap(const RenderPlan& plan)
{
    const double w = plan.layerBounds.right - plan.layerBounds.left, h = plan.layerBounds.bottom - plan.layerBounds.top;
    return (A_long)std::max(32.0, 0.06 * std::sqrt(w * w + h * h));
}

inline double RectArea(const PF_LRect& r)
{
    return RectEmpty(r) ? 0.0 : (double)(r.right - r.left) * (double)(r.bottom - r.top);
}

// Engine margin for an output rect. Exact (tile-exact) margin unless the padded processing window
// would exceed the budget (max(24 Mpx, 3 x the output area)); then the margin is reduced, but never
// below prepare + blur + capped post margin: everything except the far apron of very wide glows /
// halation scene means stays exact; those wide fields see a truncated neighbourhood at the window
// edge (error <= the Gaussian tail beyond the kept apron, i.e. a few % of an already faint glow,
// only near tile / window borders). Documented in 覚書.md; plan.marginExact tells which case ran.
A_long WindowMargin(RenderPlan& plan, const PF_LRect& result, A_long cap)
{
    const lenscore::Params& p = plan.ep;
    const int q = plan.s.quality;
    const A_long full = (A_long)EngineInputMargin(p, plan.hq, q);
    const A_long minM = (A_long)std::min<long long>(
        full, (long long)lenscore::prepareMargin(p) + EngineBlurMargin(p, plan.hq, q) + std::min((int)cap, lenscore::postMargin(p)) + 2);
    auto windowArea = [&](A_long m) {
        const PF_LRect lim = plan.s.edge == EDGE_CLIP     ? plan.content
                             : plan.s.edge == EDGE_REPEAT ? Grow(plan.content, m)
                                                          : Grow(plan.content, 2 * m + 2);
        return RectArea(Intersect(Grow(result, m), lim));
    };
    const double budget = std::max(24.0e6, 3.0 * RectArea(result));
    plan.marginExact = true;
    if (windowArea(full) <= budget)
        return full;
    plan.marginExact = false;
    A_long lo = minM, hi = full;  // largest m in [minM, full] whose window fits (area grows with m)
    while (hi - lo > 1)
    {
        const A_long mid = lo + (hi - lo) / 2;
        if (windowArea(mid) <= budget)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// Runs the three engine stages for the requested view.
// v2.3 (wrapper change A): `out` = the output rect inside the w x h engine buffer. For render version 2
// at Fast / Standard in the Result view, the blur and post stages run on a crop of the prepared buffer:
// the output rect grown by blurMargin + postMargin (everything those stages read for the output
// pixels; the prepare stage needed prepareMargin more, which can be most of the window at large radii
// or with the frame-wide highlight / halation fields). The frame rect and the output rows are shifted
// into the crop. Results inside the output rect equal the uncropped run up to double round-off of the
// row prefix sums (they start at the crop's left edge, exactly as they start at a tile's window edge
// today); host_probe measures it. Render version 1 (v1.2 / v2.1 projects) and Best keep the uncropped
// path, bit-exact with v2.2.
static PF_Err RunEngineBody(
    PF_InData* in_data, PF_OutData* out_data, const RenderPlan& plan, const lenscore::Params& p, const float* src, int w,
    int h, const float* map, const float* hlLayer, float* dst, const PF_LRect& out)
{
    AbortCtx ctx = {in_data, PF_Err_NONE};
    // progress per stage (prepare / blur / post): AE shows it and polls for cancel; the engines also
    // poll PF_ABORT through AbortCallback (lenscore between row bands, lenshq every ~20 ms)
    auto progress = [&](int done) -> PF_Err { return in_data->inter.progress ? PF_PROGRESS(in_data, done, 3) : PF_Err_NONE; };
    try
    {
        PF_Err perr = progress(0);
        if (perr)
            return perr;
        bool gpuFailed = false;  // v2.4: the GPU frame path failed / was routed away: CPU for the rest of this frame
        double cpuWork = 0;      // > 0: report this CPU frame's time to the GPU router (obgpu ABI 1.2)
        auto cpuT0 = std::chrono::steady_clock::now();  // (v2.4.2: restarted when the GPU declines the frame, below)
        auto reportCpu = [&]() {
            if (cpuWork > 0)
                obgpuw::ReportCpu(cpuWork, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuT0).count());
        };
#ifndef OB_OLD_ENGINE
        // v2.4 (obgpu ABI 1.1): prepare + blur on the GPU in one job (one upload of the window, one readback of
        // the blurred crop), post on the CPU exactly as in the crop branch below. Automatic highlight source,
        // render version 2, Fast / Standard, Result view; anything else (or any GPU failure) falls through to
        // the CPU prepare, which may still use the GPU blur (phase 1) and finally the CPU blur.
        if (plan.s.renderVersion >= 2 && plan.s.quality != QUALITY_BEST && plan.s.view == VIEW_RESULT && !RectEmpty(out) &&
            !hlLayer &&
            obgpuw::Eligible(p, plan.s.renderVersion, false, plan.s.useGpu, plan.layerBounds.right - plan.layerBounds.left,
                             plan.layerBounds.bottom - plan.layerBounds.top))
        {
            const int g = lenscore::blurMargin(p) + lenscore::postMargin(p);
            const int cx0 = std::max(0, (int)out.left - g), cy0 = std::max(0, (int)out.top - g);
            const int cx1 = std::min(w, (int)out.right + g), cy1 = std::min(h, (int)out.bottom + g);
            const int cw = cx1 - cx0, ch = cy1 - cy0;
            lenscore::Params pq = p;
            pq.frameLeft -= (float)cx0;
            pq.frameTop -= (float)cy0;
            pq.outRowBegin = (int)out.top - cy0;
            pq.outRowEnd = (int)out.bottom - cy0;
            std::vector<float> dc((size_t)cw * ch * 4), eLc;
            lenscore::Prepared pc;
            lenscore::Params prest = pq;
            const int fr = obgpuw::Frame(src, w, h, map, p, cx0, cy0, cw, ch, pq, dc.data(), cw * 4, eLc, pc, prest, cpuWork, AbortCallback,
                                         &ctx);
            if (fr < 0)
                return ToPfErr(lenscore::Status::Aborted, ctx, out_data);
            gpuFailed = fr == 2 || fr == 3;  // the GPU did not take it / routed to the CPU: CPU for the whole frame
            if (!gpuFailed)
                cpuWork = 0;  // (nothing to report)
            else
                cpuT0 = std::chrono::steady_clock::now();  // v2.4.2: report the CPU render only, not the GPU queue wait before it
            if (fr == 1)
            {
                if ((perr = progress(2)) != PF_Err_NONE)
                    return perr;
                const lenscore::Status sp = lenscore::postProcess(pc, prest, dc.data(), cw * 4,
                                                                  lenscore::postNeedsEL(prest, pc) ? eLc.data() : nullptr, AbortCallback, &ctx);
                if (sp != lenscore::Status::Ok)
                    return ToPfErr(sp, ctx, out_data);
                for (A_long y = out.top; y < out.bottom; ++y)
                    std::memcpy(dst + ((size_t)y * w + out.left) * 4, dc.data() + ((size_t)(y - cy0) * cw + (out.left - cx0)) * 4,
                                sizeof(float) * 4 * (size_t)(out.right - out.left));
                progress(3);
                return PF_Err_NONE;
            }
        }
#endif
        lenscore::Prepared prep;
        lenscore::PrepareOptions opt;
        opt.keepExtra = plan.s.view == VIEW_HL_SOURCE || plan.s.view == VIEW_HL_ONLY;
        opt.highlightLayer = hlLayer;
        opt.highlightLayerStride = w;
        lenscore::Status st =
            lenscore::prepareHighlights(src, w, h, w * 4, map, w, p, opt, prep, AbortCallback, &ctx);
        if (st != lenscore::Status::Ok)
            return ToPfErr(st, ctx, out_data);
        if (plan.s.view == VIEW_HL_SOURCE)
        {
            // added highlight energy on black (units: disc brightness of a 4x4 px source, x4)
            for (size_t i = 0; i < (size_t)w * h; ++i)
            {
                dst[i * 4] = prep.extra[i * 3];
                dst[i * 4 + 1] = prep.extra[i * 3 + 1];
                dst[i * 4 + 2] = prep.extra[i * 3 + 2];
                dst[i * 4 + 3] = 1.f;
            }
            return PF_Err_NONE;
        }
        if (plan.s.view == VIEW_HL_ONLY)
        {
            const lenscore::Prepared q = lenscore::highlightsOnly(prep);
            lenscore::Params pq = p;
            pq.highlightRolloff = 0;
            return ToPfErr(BlurStage(q, pq, plan.hq, plan.s.quality, dst, w * 4, nullptr, AbortCallback, &ctx), ctx, out_data);
        }
#ifndef OB_OLD_ENGINE  // (timing builds against the v2.2.1 engine copy, which has no cropPrepared)
        const bool canCrop = !g_disableCrop && plan.s.renderVersion >= 2 && plan.s.quality != QUALITY_BEST &&
                             plan.s.view == VIEW_RESULT && !RectEmpty(out);
        if (canCrop)
        {
            const int g = lenscore::blurMargin(p) + lenscore::postMargin(p);
            const int cx0 = std::max(0, (int)out.left - g), cy0 = std::max(0, (int)out.top - g);
            const int cx1 = std::min(w, (int)out.right + g), cy1 = std::min(h, (int)out.bottom + g);
            if ((long long)(cx1 - cx0) * (cy1 - cy0) < (long long)w * h)
            {
                const int cw = cx1 - cx0, ch = cy1 - cy0;
                lenscore::Prepared pc = lenscore::cropPrepared(prep, cx0, cy0, cw, ch);
                prep = lenscore::Prepared();  // the window-sized buffers are not needed any more
                lenscore::Params pq = p;
                pq.frameLeft -= (float)cx0;
                pq.frameTop -= (float)cy0;
                pq.outRowBegin = (int)out.top - cy0;
                pq.outRowEnd = (int)out.bottom - cy0;
                std::vector<float> dc((size_t)cw * ch * 4);
                std::vector<float> eLc(lenscore::postNeedsEL(pq, pc) ? (size_t)cw * ch : 0);
                if ((perr = progress(1)) != PF_Err_NONE)
                    return perr;
                // v2.4: the GPU blur when eligible; any failure -> the CPU blur below (same semantics)
                int gpu = 0;
                // (the size rule uses the LAYER, not this crop: every tile of a frame takes the same path)
                if (!gpuFailed && obgpuw::Eligible(pq, plan.s.renderVersion, plan.s.quality == QUALITY_BEST, plan.s.useGpu,
                                     plan.layerBounds.right - plan.layerBounds.left, plan.layerBounds.bottom - plan.layerBounds.top))
                    gpu = obgpuw::Blur(pc, pq, !eLc.empty(), dc.data(), cw * 4, eLc.empty() ? nullptr : eLc.data(), AbortCallback, &ctx);
                if (gpu < 0)
                    return ToPfErr(lenscore::Status::Aborted, ctx, out_data);
                if (gpu == 0)
                {
                    st = BlurStage(pc, pq, plan.hq, plan.s.quality, dc.data(), cw * 4, eLc.empty() ? nullptr : eLc.data(), AbortCallback,
                                   &ctx);
                    if (st != lenscore::Status::Ok)
                        return ToPfErr(st, ctx, out_data);
                }
                if ((perr = progress(2)) != PF_Err_NONE)
                    return perr;
                st = lenscore::postProcess(pc, pq, dc.data(), cw * 4, eLc.empty() ? nullptr : eLc.data(), AbortCallback, &ctx);
                if (st != lenscore::Status::Ok)
                    return ToPfErr(st, ctx, out_data);
                reportCpu();
                for (A_long y = out.top; y < out.bottom; ++y)
                    std::memcpy(dst + ((size_t)y * w + out.left) * 4, dc.data() + ((size_t)(y - cy0) * cw + (out.left - cx0)) * 4,
                                sizeof(float) * 4 * (size_t)(out.right - out.left));
                progress(3);
                return PF_Err_NONE;
            }
        }
#endif
        std::vector<float> eL(lenscore::postNeedsEL(p, prep) ? (size_t)w * h : 0);
        if ((perr = progress(1)) != PF_Err_NONE)
            return perr;
        // v2.4: the GPU blur here too (render version >= 2 Fast / Standard Result views that need no crop,
        // e.g. Clip to Layer on a whole-layer request); render version 1 never qualifies (Eligible)
        int gpuU = 0;
        if (plan.s.view == VIEW_RESULT && !gpuFailed &&
            obgpuw::Eligible(p, plan.s.renderVersion, plan.s.quality == QUALITY_BEST, plan.s.useGpu,
                             plan.layerBounds.right - plan.layerBounds.left, plan.layerBounds.bottom - plan.layerBounds.top))
            gpuU = obgpuw::Blur(prep, p, !eL.empty(), dst, w * 4, eL.empty() ? nullptr : eL.data(), AbortCallback, &ctx);
        if (gpuU < 0)
            return ToPfErr(lenscore::Status::Aborted, ctx, out_data);
        st = gpuU == 1 ? lenscore::Status::Ok
                       : BlurStage(prep, p, plan.hq, plan.s.quality, dst, w * 4, eL.empty() ? nullptr : eL.data(), AbortCallback, &ctx);
        if (st != lenscore::Status::Ok)
            return ToPfErr(st, ctx, out_data);
        if ((perr = progress(2)) != PF_Err_NONE)
            return perr;
        st = lenscore::postProcess(prep, p, dst, w * 4, eL.empty() ? nullptr : eL.data(), AbortCallback, &ctx);
        if (st == lenscore::Status::Ok)
        {
            reportCpu();
            progress(3);
        }
        return ToPfErr(st, ctx, out_data);
    }
    catch (const std::bad_alloc&)
    {
        return PF_Err_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render engine error. The frame was not rendered.");
    }
}

// v2.4.2: RunEngineBody plus the per-frame diagnostic line (OPTICBOKEH_GPU_LOG=<file>, aetest): window size, the
// path the frame took (GPU / GPU-blur / CPU), why not the GPU, GPU queue wait and run ms, render ms. With the
// variable unset (the default) this is one relaxed atomic load: nothing is recorded, timed or written.
PF_Err RunEngine(
    PF_InData* in_data, PF_OutData* out_data, const RenderPlan& plan, const lenscore::Params& p, const float* src, int w,
    int h, const float* map, const float* hlLayer, float* dst, const PF_LRect& out)
{
    if (!obgpuw::FrameLogOn())
        return RunEngineBody(in_data, out_data, plan, p, src, w, h, map, hlLayer, dst, out);
    obgpuw::FrameLogBegin();
    const auto t0 = std::chrono::steady_clock::now();
    const PF_Err err = RunEngineBody(in_data, out_data, plan, p, src, w, h, map, hlLayer, dst, out);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const char* why = nullptr;
    if (plan.s.renderVersion < 2)
        why = "render-version-1";
    else if (plan.s.quality == QUALITY_BEST)
        why = "quality-best";
    else if (plan.s.view != VIEW_RESULT)
        why = "not-result-view";
    else if (plan.s.useGpu == obgpuw::USE_OFF)
        why = "use-gpu-off";
    obgpuw::FrameLogEnd(w, h, 0, 0, ms, (int)err, why);
    return err;
}

// ------------------------------------------------------- pixel conversion ---

template <typename Pix> struct Traits;
template <> struct Traits<PF_Pixel8>
{
    static inline float In(A_u_char v) { return v * (1.f / 255.f); }
    static inline A_u_char Out(float v)
    {
        v = v * 255.f + 0.5f;
        return (A_u_char)(!(v > 0.f) ? 0 : (v >= 255.f ? 255 : (int)v));  // NaN -> 0
    }
};
template <> struct Traits<PF_Pixel16>
{
    static inline float In(A_u_short v) { return v * (1.f / PF_MAX_CHAN16); }
    static inline A_u_short Out(float v)
    {
        v = v * PF_MAX_CHAN16 + 0.5f;
        return (A_u_short)(!(v > 0.f) ? 0 : (v >= (float)PF_MAX_CHAN16 ? PF_MAX_CHAN16 : (int)v));  // NaN -> 0
    }
};
template <> struct Traits<PF_PixelFloat>
{
    static inline float In(PF_FpShort v) { return v; }
    static inline PF_FpShort Out(float v) { return v; }
};

// non-finite -> 0 (+Inf -> a large finite value), magnitudes clamped (same rule as the engine)
inline float SanitizeF(float v)
{
    const float kMax = 1.0e6f;
    return std::isfinite(v) ? std::min(kMax, std::max(-kMax, v)) : (v > 0 ? kMax : 0.f);
}

inline float SignedPow(float v, float e)
{
    return v >= 0.f ? std::pow(v, e) : -std::pow(-v, e);
}

template <typename Pix>
inline const Pix* RowPtr(const PF_EffectWorld* w, A_long y)
{
    return reinterpret_cast<const Pix*>(reinterpret_cast<const char*>(w->data) + (size_t)y * w->rowbytes);
}
template <typename Pix>
inline Pix* RowPtrW(PF_EffectWorld* w, A_long y)
{
    return reinterpret_cast<Pix*>(reinterpret_cast<char*>(w->data) + (size_t)y * w->rowbytes);
}

// v2.3: 8 bpc, alpha 255, Work in Linear Light: WorldToFloat computes SignedPow(v / 255 * 1, 2.2) * 1
// per channel; this table holds exactly those values (same expression, same float operations).
template <typename Pix>
const float* Lut8()
{
    return nullptr;
}
template <>
const float* Lut8<PF_Pixel8>()
{
    static const std::vector<float> lut = [] {
        std::vector<float> l(256);
        typedef Traits<PF_Pixel8> T;
        for (int v = 0; v < 256; ++v)
        {
            const float a = std::min(1.f, std::max(0.f, SanitizeF(T::In(255))));
            float c = SanitizeF(T::In((A_u_char)v));
            if (!kHostPixelsPremultiplied)
                c *= a;
            const float inv = 1.f / a;
            l[v] = SignedPow(c * inv, kDisplayGamma) * a;
        }
        return l;
    }();
    return lut.data();
}
template <typename Pix>
inline bool Opaque8(const Pix*)
{
    return false;
}
template <>
inline bool Opaque8<PF_Pixel8>(const PF_Pixel8* s)
{
    return s->alpha == 255;
}
template <typename C>
inline int Chan8(C v)
{
    return (int)v;
}

// AE ARGB -> premultiplied (optionally linearized) float RGBA for the layer-coord rect `r`.
// Pixels of `r` outside the input world are transparent black, or - with repeatEdge - copies of
// the nearest pixel of `clampTo` (the layer bounds, which must lie inside the input world where
// sampled). Dense output, stride 4 * width.
template <typename Pix>
void WorldToFloat(const PF_EffectWorld* in, const PF_LRect& r, bool linear, bool repeatEdge, const PF_LRect& clampTo, float* dst)
{
    typedef Traits<Pix> T;
    const A_long w = r.right - r.left;
    const A_long ix0 = in->origin_x, iy0 = in->origin_y, ix1 = in->origin_x + in->width, iy1 = in->origin_y + in->height;
    const A_long cx0 = std::max(ix0, clampTo.left), cy0 = std::max(iy0, clampTo.top);
    const A_long cx1 = std::min(ix1, clampTo.right) - 1, cy1 = std::min(iy1, clampTo.bottom) - 1;
    const float* lut = linear && !g_disableLut8 ? Lut8<Pix>() : nullptr;  // 8 bpc opaque pixels: the same pow, tabulated
    ParallelRows(r.top, r.bottom, (double)w, [&](A_long yb, A_long ye) {
    for (A_long y = yb; y < ye; ++y)
    {
        float* d = dst + (size_t)(y - r.top) * w * 4;
        for (A_long x = r.left; x < r.right; ++x, d += 4)
        {
            A_long sx = x, sy = y;
            if (repeatEdge && cx1 >= cx0 && cy1 >= cy0)
            {
                sx = std::min(cx1, std::max(cx0, sx));
                sy = std::min(cy1, std::max(cy0, sy));
            }
            if (sx < ix0 || sx >= ix1 || sy < iy0 || sy >= iy1)
            {
                d[0] = d[1] = d[2] = d[3] = 0.f;
                continue;
            }
            const Pix* s = RowPtr<Pix>(in, sy - iy0) + (sx - ix0);
            if (lut && Opaque8(s))
            {
                d[0] = lut[Chan8(s->red)];
                d[1] = lut[Chan8(s->green)];
                d[2] = lut[Chan8(s->blue)];
                d[3] = std::min(1.f, std::max(0.f, SanitizeF(T::In(s->alpha))));  // = the general path's alpha
                continue;
            }
            // 32 bpc can carry NaN / Inf from upstream effects: sanitise (the engine does it too)
            float a = SanitizeF(T::In(s->alpha));
            float rr = SanitizeF(T::In(s->red)), gg = SanitizeF(T::In(s->green)), bb = SanitizeF(T::In(s->blue));
            a = std::min(1.f, std::max(0.f, a));
            if (!kHostPixelsPremultiplied)
            {
                rr *= a;
                gg *= a;
                bb *= a;
            }
            if (linear)
            {
                if (a > 0.f)
                {
                    const float inv = 1.f / a;
                    rr = SignedPow(rr * inv, kDisplayGamma) * a;
                    gg = SignedPow(gg * inv, kDisplayGamma) * a;
                    bb = SignedPow(bb * inv, kDisplayGamma) * a;
                }
                else
                {
                    rr = gg = bb = 0.f;
                }
            }
            d[0] = rr;
            d[1] = gg;
            d[2] = bb;
            d[3] = a;
        }
    }
    });
}

// Float RGBA (dense, covering layer rect `src_r`) -> output world. Output pixels outside src_r (or
// outside `clip` if given) become transparent.
template <typename Pix>
void FloatToWorld(const float* src, const PF_LRect& src_r, bool linear, PF_EffectWorld* out, const PF_LRect* clip = nullptr)
{
    typedef Traits<Pix> T;
    const A_long sw = src_r.right - src_r.left;
    ParallelRows(0, out->height, (double)out->width, [&](A_long yb, A_long ye) {
    for (A_long oy = yb; oy < ye; ++oy)
    {
        Pix* d = RowPtrW<Pix>(out, oy);
        const A_long ly = oy + out->origin_y;
        for (A_long ox = 0; ox < out->width; ++ox, ++d)
        {
            const A_long lx = ox + out->origin_x;
            if (lx < src_r.left || lx >= src_r.right || ly < src_r.top || ly >= src_r.bottom ||
                (clip && (lx < clip->left || lx >= clip->right || ly < clip->top || ly >= clip->bottom)))
            {
                d->alpha = d->red = d->green = d->blue = T::Out(0.f);
                continue;
            }
            const float* s = src + ((size_t)(ly - src_r.top) * sw + (lx - src_r.left)) * 4;
            float rr = s[0], gg = s[1], bb = s[2], a = s[3];
            // v2.2 (A7): coverage below 1e-6 is span-sum round-off, not content: fully transparent
            // (else the straight colour of such pixels is arbitrary and differs between tiles)
            if (!(a >= 1e-6f))
            {
                d->alpha = d->red = d->green = d->blue = T::Out(0.f);
                continue;
            }
            a = std::min(1.f, a);
            if (linear || !kHostPixelsPremultiplied)
            {
                const float ac = std::min(1.f, a);
                if (ac > 0.f)
                {
                    const float inv = 1.f / ac;
                    rr *= inv;
                    gg *= inv;
                    bb *= inv;
                    if (linear)
                    {
                        const float g = 1.f / kDisplayGamma;
                        rr = SignedPow(rr, g);
                        gg = SignedPow(gg, g);
                        bb = SignedPow(bb, g);
                    }
                    if (kHostPixelsPremultiplied)
                    {
                        rr *= ac;
                        gg *= ac;
                        bb *= ac;
                    }
                }
                else
                {
                    rr = gg = bb = 0.f;
                }
            }
            d->alpha = T::Out(a);
            d->red = T::Out(rr);
            d->green = T::Out(gg);
            d->blue = T::Out(bb);
        }
    }
    });
}

// One channel value of a layer-world pixel, as the layer composited over black (colour * alpha;
// host worlds are straight, see kHostPixelsPremultiplied).
template <typename Pix>
inline float LayerChannel(const Pix& p, int channel)
{
    typedef Traits<Pix> T;
    const float a = kHostPixelsPremultiplied ? 1.f : std::min(1.f, std::max(0.f, T::In(p.alpha)));
    switch (channel)
    {
    case CH_RED:
        return T::In(p.red) * a;
    case CH_GREEN:
        return T::In(p.green) * a;
    case CH_BLUE:
        return T::In(p.blue) * a;
    case CH_ALPHA:
        return T::In(p.alpha);
    default:
        return (0.2126f * T::In(p.red) + 0.7152f * T::In(p.green) + 0.0722f * T::In(p.blue)) * a;
    }
}

// Bilinear sample of a layer param at effect-layer pixel (lx, ly), stretching its bounds onto the
// effect layer bounds.
template <typename Pix>
float SampleLayer(const PF_EffectWorld* m, const RenderPlan& plan, const LayerSampler& ls, A_long lx, A_long ly)
{
    // pixel centers: layer coord u = (x + 0.5) * k - 0.5, clamped to the sampled layer bounds
    double u = (lx - plan.layerBounds.left + 0.5) * ls.kx - 0.5;
    double v = (ly - plan.layerBounds.top + 0.5) * ls.ky - 0.5;
    u = std::min((double)ls.w - 1, std::max(0.0, u));
    v = std::min((double)ls.h - 1, std::max(0.0, v));
    // to world pixel space, clamped to the buffer actually returned
    u -= m->origin_x;
    v -= m->origin_y;
    u = std::min((double)m->width - 1, std::max(0.0, u));
    v = std::min((double)m->height - 1, std::max(0.0, v));
    const A_long x0 = (A_long)u, y0 = (A_long)v;
    const A_long x1 = std::min(x0 + 1, m->width - 1), y1 = std::min(y0 + 1, m->height - 1);
    const float fx = (float)(u - x0), fy = (float)(v - y0);
    const Pix* r0 = RowPtr<Pix>(m, y0);
    const Pix* r1 = RowPtr<Pix>(m, y1);
    const int ch = ls.channel;
    const float c00 = SanitizeF(LayerChannel(r0[x0], ch)), c10 = SanitizeF(LayerChannel(r0[x1], ch));
    const float c01 = SanitizeF(LayerChannel(r1[x0], ch)), c11 = SanitizeF(LayerChannel(r1[x1], ch));
    const float a = c00 + (c10 - c00) * fx;
    const float b = c01 + (c11 - c01) * fx;
    return a + (b - a) * fy;
}

template <typename Pix>
void FillLayer(const PF_EffectWorld* m, const RenderPlan& plan, const LayerSampler& ls, const PF_LRect& r, float* dst)
{
    const A_long w = r.right - r.left;
    ParallelRows(r.top, r.bottom, (double)w, [&](A_long yb, A_long ye) {
        for (A_long y = yb; y < ye; ++y)
        {
            float* d = dst + (size_t)(y - r.top) * w;
            for (A_long x = r.left; x < r.right; ++x)
                *d++ = SampleLayer<Pix>(m, plan, ls, x, y);
        }
    });
}

// v2.3: the raw map over `r` (Map Blur applied: sampled Extent() further on every side, blurred, the
// inside kept, so the values do not depend on the tile).
template <typename Pix>
void RawMap(const PF_EffectWorld* m, const RenderPlan& plan, const LayerSampler& ls, const PF_LRect& r, float* dst)
{
    const int e = plan.mapExt;
    if (e <= 0)
    {
        FillLayer<Pix>(m, plan, ls, r, dst);
        return;
    }
    const PF_LRect g = Grow(r, e);
    const A_long gw = g.right - g.left, gh = g.bottom - g.top, w = r.right - r.left;
    std::vector<float> tmp((size_t)gw * gh);
    FillLayer<Pix>(m, plan, ls, g, tmp.data());
    obmap::Blur(tmp.data(), (int)gw, (int)gh, plan.mx.sigma);
    for (A_long y = r.top; y < r.bottom; ++y)
        std::memcpy(dst + (size_t)(y - r.top) * w, tmp.data() + (size_t)(y - g.top) * gw + (r.left - g.left), sizeof(float) * w);
}

// v2.3: the map the engine gets over `r`: classic = the sampled map; with the map features on, the
// remapped signed amount (ob_mapx.h) with `x.focus` already resolved (Focus at Point).
template <typename Pix>
void BuildMap(const PF_EffectWorld* m, const RenderPlan& plan, const obmap::Xform& x, const PF_LRect& r, float* dst)
{
    if (!x.active)
    {
        FillLayer<Pix>(m, plan, plan.map, r, dst);
        return;
    }
    RawMap<Pix>(m, plan, plan.map, r, dst);
    const size_t n = (size_t)(r.right - r.left) * (r.bottom - r.top);
    for (size_t i = 0; i < n; ++i)
        dst[i] = obmap::Apply(x, dst[i]);
}

// v2.3 Focus at Point: the (blurred) map value under the point, as the focus (Invert Map applied, like
// the engine applies it to the map before comparing with Focus)
template <typename Pix>
obmap::Xform ResolveFocus(const RenderPlan& plan, const PF_EffectWorld* mapPtW)
{
    obmap::Xform x = plan.mx;
    if (x.active && x.point && mapPtW)
    {
        const A_long px = (A_long)std::floor(x.px), py = (A_long)std::floor(x.py);
        float v = 0.f;
        RawMap<Pix>(mapPtW, plan, plan.mapPt, PF_LRect{px, py, px + 1, py + 1}, &v);
        const double m = (v == v) ? (double)v : 0.0;
        x.focus = x.invert ? 1.0 - m : m;
    }
    return x;
}

// "Blur Amount" view: opaque grayscale of the engine's per-pixel amount (v2.3: of the remapped map
// when the map features are on, so the view shows the resulting depth of field).
template <typename OutPix, typename MapPix>
void RenderAmountView(const PF_EffectWorld* m, const RenderPlan& plan, const obmap::Xform& x, PF_EffectWorld* out)
{
    typedef Traits<OutPix> T;
    std::vector<float> mrow;
    if (m)
        mrow.resize((size_t)out->width);
    for (A_long oy = 0; oy < out->height; ++oy)
    {
        OutPix* d = RowPtrW<OutPix>(out, oy);
        if (m)
            BuildMap<MapPix>(m, plan, x,
                             PF_LRect{out->origin_x, oy + out->origin_y, out->origin_x + out->width, oy + out->origin_y + 1},
                             mrow.data());
        for (A_long ox = 0; ox < out->width; ++ox, ++d)
        {
            const float* mvp = m ? &mrow[(size_t)ox] : nullptr;
            const float amt = EngineBlurAmount(plan.ep, mvp);
            d->alpha = T::Out(1.f);
            d->red = d->green = d->blue = T::Out(amt);
        }
    }
}

template <typename Pix>
void FillTransparent(PF_EffectWorld* out)
{
    for (A_long y = 0; y < out->height; ++y)
        memset(RowPtrW<Pix>(out, y), 0, sizeof(Pix) * (size_t)out->width);
}

// ------------------------------------------------------------- commands ---

void RegisterAegp(PF_InData* in_data);

// v2.3: the UI language, decided once per process from AE's UI language (PF_AppGetLanguage).
// Japanese only if every Japanese string fits the Windows ANSI code page and the SDK limits (obs::Init);
// otherwise English everywhere. Param names are fixed at PARAMS_SETUP: there is no runtime switch,
// because popup item strings cannot be changed after setup (PF_UpdateParamUI: name / flags only).
void EnsureLanguage(PF_InData* in_data)
{
    char tag[PF_APP_LANG_TAG_SIZE] = {0};
    const char* lang = nullptr;
    if (in_data && in_data->pica_basicP)
    {
        const PFAppSuite6* app = nullptr;
        if (!in_data->pica_basicP->AcquireSuite(kPFAppSuite, kPFAppSuiteVersion6, (const void**)&app) && app)
        {
            if (app->PF_AppGetLanguage && app->PF_AppGetLanguage(tag) == PF_Err_NONE)
            {
                tag[PF_APP_LANG_TAG_SIZE - 1] = 0;
                lang = tag;
            }
            in_data->pica_basicP->ReleaseSuite(kPFAppSuite, kPFAppSuiteVersion6);
        }
    }
    obs::Init(lang, 0, OB_HOST_STRING_ENCODING);
}

PF_Err About(PF_InData* in_data, PF_OutData* out_data)
{
    EnsureLanguage(in_data);
    // AE shows return_msg (256 bytes incl. the terminator): optional lines are left out when they do not fit
    char head[96];
    std::snprintf(head, sizeof(head), "%s %d.%d.%d (build %d, %s)", OB_NAME, OB_MAJOR_VERSION, OB_MINOR_VERSION, OB_BUG_VERSION,
                  OB_BUILD_VERSION, __DATE__);
    const std::string gpu = obgpuw::StatusLine(), lic = "Licences: THIRD_PARTY_LICENSES.txt";
    const std::string desc = obs::T("Camera-style lens blur with depth / blur map.");
    const std::string qual = obs::T("Quality: Fast / Standard / Best. GPU for Fast / Standard.");
    const size_t cap = sizeof(out_data->return_msg) - 1;
    std::string m = std::string(head) + "\r" + desc + "\r" + qual + "\r" + gpu + "\r" + lic;
    if (m.size() > cap)
        m = std::string(head) + "\r" + desc + "\r" + gpu + "\r" + lic;
    if (m.size() > cap)
        m = std::string(head) + "\r" + gpu + "\r" + lic;
    if (m.size() > cap)
        m = std::string(head) + "\r" + lic;  // whole lines only (never a cut multi-byte character)
    std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s", m.c_str());
    return PF_Err_NONE;
}


PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    // v2.2 (A1): only After Effects drives the SmartFX path this effect implements; other hosts that
    // load AE effects (Premiere Pro / Media Encoder via MediaCore) call PF_Cmd_RENDER, which would
    // render nothing. Refuse to load there, with a message.
    EnsureLanguage(in_data);  // v2.3: AE's UI language decides English / Japanese names (once per process)
    if (in_data->appl_id != 'FXTC')
    {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d %s", OB_NAME, OB_MAJOR_VERSION,
                      OB_MINOR_VERSION, obs::T("runs in After Effects only."));
        return PF_Err_INVALID_CALLBACK;
    }
    out_data->my_version =
        PF_VERSION(OB_MAJOR_VERSION, OB_MINOR_VERSION, OB_BUG_VERSION, OB_STAGE, OB_BUILD_VERSION);
    out_data->out_flags = kOutFlags;
    // No sequence data; the only mutable globals are the render counters (see RenderThreads /
    // BestSlot) and the AEGP id below: safe for multi-frame rendering.
    out_data->out_flags2 = kOutFlags2;
    RegisterAegp(in_data);
    obgpuw::Start();  // v2.4: background GPU initialisation (After Effects only: other hosts returned above)
    return PF_Err_NONE;
}

// AEGP plug-in id for the stream calls in QUERY_DYNAMIC_FLAGS (registered once in GLOBAL_SETUP, as in
// the SDK Supervisor sample). A write-once process-wide value; 0 = not available.
std::atomic<A_long> g_aegpId{0};
void RegisterAegp(PF_InData* in_data)
{
    if (g_aegpId.load() != 0 || !in_data || !in_data->pica_basicP)
        return;
    const AEGP_UtilitySuite6* u = nullptr;
    if (in_data->pica_basicP->AcquireSuite(kAEGPUtilitySuite, kAEGPUtilitySuiteVersion6, (const void**)&u) || !u)
        return;
    AEGP_PluginID id = 0;
    if (u->AEGP_RegisterWithAEGP(nullptr, OB_NAME, &id) == A_Err_NONE && id != 0)
        g_aegpId.store((A_long)id);
    in_data->pica_basicP->ReleaseSuite(kAEGPUtilitySuite, kAEGPUtilitySuiteVersion6);
}

// v2.2 (A6): can the param's value change over time? Keyframes (PF_ParamUtilsSuite3) or an enabled
// expression (AEGP stream suite). Unknown (a suite missing) counts as "yes" (no stale cache).
bool ParamMayVary(PF_InData* in_data, int index)
{
    if (!in_data->pica_basicP)  // no suites at all (not AE: GLOBAL_SETUP refuses other hosts)
        return false;
    bool vary = false, known = true;
    const PF_ParamUtilsSuite3* pu = nullptr;
    if (!in_data->pica_basicP->AcquireSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, (const void**)&pu) && pu)
    {
        PF_KeyIndex n = PF_KeyIndex_NONE;
        if (pu->PF_GetKeyframeCount(in_data->effect_ref, index, &n) == PF_Err_NONE)
            vary = n != PF_KeyIndex_NONE && n > 0;
        else
            known = false;
        in_data->pica_basicP->ReleaseSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
    }
    else
        known = false;
    if (vary)
        return true;
    const AEGP_PluginID id = (AEGP_PluginID)g_aegpId.load();
    const AEGP_PFInterfaceSuite1* pfi = nullptr;
    const AEGP_StreamSuite7* ss = nullptr;
    const AEGP_EffectSuite5* es = nullptr;
    bool haveExpr = false;
    if (id != 0 &&
        !in_data->pica_basicP->AcquireSuite(kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1, (const void**)&pfi) && pfi)
    {
        if (!in_data->pica_basicP->AcquireSuite(kAEGPStreamSuite, kAEGPStreamSuiteVersion7, (const void**)&ss) && ss)
        {
            if (!in_data->pica_basicP->AcquireSuite(kAEGPEffectSuite, kAEGPEffectSuiteVersion5, (const void**)&es) && es)
            {
                AEGP_EffectRefH eff = nullptr;
                if (pfi->AEGP_GetNewEffectForEffect(id, in_data->effect_ref, &eff) == A_Err_NONE && eff)
                {
                    AEGP_StreamRefH st = nullptr;
                    if (ss->AEGP_GetNewEffectStreamByIndex(id, eff, index, &st) == A_Err_NONE && st)
                    {
                        A_Boolean on = FALSE;
                        if (ss->AEGP_GetExpressionState(id, st, &on) == A_Err_NONE)
                        {
                            haveExpr = true;
                            vary = on != FALSE;
                        }
                        ss->AEGP_DisposeStream(st);
                    }
                    es->AEGP_DisposeEffect(eff);
                }
                in_data->pica_basicP->ReleaseSuite(kAEGPEffectSuite, kAEGPEffectSuiteVersion5);
            }
            in_data->pica_basicP->ReleaseSuite(kAEGPStreamSuite, kAEGPStreamSuiteVersion7);
        }
        in_data->pica_basicP->ReleaseSuite(kAEGPPFInterfaceSuite, kAEGPPFInterfaceSuiteVersion1);
    }
    // an unknown expression state (AEGP id / suites unavailable) is not counted: otherwise every
    // instance would lose frame caching; keyframes are always known in AE
    (void)haveExpr;
    return vary || !known;
}

// NON_PARAM_VARY while animated grain can be visible: (Grain Amount > 0 now, or keyframed /
// expression-driven) and (Animate Grain on, or keyframed / expression-driven). A still layer with
// grain whose amount is keyframed from 0 therefore keeps the flag at every time. AE_Effect.h: the params
// array is not valid here, so the values are checked out; layer params are not looked at.
PF_Err QueryDynamicFlags(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef amount, animate;
    AEFX_CLR_STRUCT(amount);
    AEFX_CLR_STRUCT(animate);
    bool outA = false, outB = false;
    ERR(PF_CHECKOUT_PARAM(in_data, OB_GRAIN_AMOUNT, in_data->current_time, in_data->time_step, in_data->time_scale, &amount));
    if (!err)
        outA = true;
    // Kantan: grain always animates (no Animate checkbox); Grain is 0..200 % of the stock's amount
    const int animIdx = OB_GRAIN_AMOUNT;
    ERR(PF_CHECKOUT_PARAM(in_data, animIdx, in_data->current_time, in_data->time_step, in_data->time_scale, &animate));
    if (!err)
        outB = true;
    if (!err)
    {
        const bool amountOn = amount.u.fs_d.value > 0 || ParamMayVary(in_data, OB_GRAIN_AMOUNT);
        const bool animOn = amountOn;
        if (amountOn && animOn)
            out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
        else
            out_data->out_flags &= ~PF_OutFlag_NON_PARAM_VARY;
    }
    else
    {
        out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;  // unknown: be safe (no stale cache)
        err = PF_Err_NONE;
    }
    if (outA)
        ERR2(PF_CHECKIN_PARAM(in_data, &amount));
    if (outB)
        ERR2(PF_CHECKIN_PARAM(in_data, &animate));
    return err;
}

// Adds a float slider. `dflt` is the value for a newly applied effect; `oldValue` is what a saved
// project that predates this param gets (PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS), so new
// instances start with a good look while old projects keep rendering exactly as before.
PF_Err AddFloat(
    PF_InData* in_data,
    const char* name,
    double vmin,
    double vmax,
    double smin,
    double smax,
    double dflt,
    double oldValue,
    int prec,
    PF_ValueDisplayFlags disp,
    A_long id,
    PF_ParamUIFlags ui = 0)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_FLOAT_SLIDER;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.fs_d.valid_min = (PF_FpShort)vmin;
    def.u.fs_d.slider_min = (PF_FpShort)smin;
    def.u.fs_d.valid_max = (PF_FpShort)vmax;
    def.u.fs_d.slider_max = (PF_FpShort)smax;
    def.u.fs_d.value = (PF_FpShort)oldValue;
    def.u.fs_d.dephault = (PF_FpShort)dflt;
    def.u.fs_d.precision = (PF_Precision)prec;
    def.u.fs_d.display_flags = disp;
    def.u.fs_d.curve_tolerance = AEFX_DEFAULT_CURVE_TOLERANCE;
    if (oldValue != dflt)
        def.flags |= PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
    def.flags |= SupFlag(id);
    def.ui_flags = ui;
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddPopup(PF_InData* in_data, const char* name, int choices, int dflt, const char* items, A_long id, PF_ParamUIFlags ui = 0,
                PF_ParamFlags flags = 0)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POPUP;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.pd.num_choices = (A_short)choices;
    def.u.pd.dephault = (A_short)dflt;
    def.u.pd.value = (A_short)dflt;
    def.u.pd.u.namesptr = items;
    def.flags = flags | SupFlag(id);
    def.ui_flags = ui;
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

// Popup whose saved projects that predate it get `oldValue` (USE_VALUE_FOR_OLD_PROJECTS).
PF_Err AddPopupOld(PF_InData* in_data, const char* name, int choices, int dflt, int oldValue, const char* items, A_long id,
                   PF_ParamFlags flags = 0)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POPUP;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.pd.num_choices = (A_short)choices;
    def.u.pd.dephault = (A_short)dflt;
    def.u.pd.value = (A_short)oldValue;
    def.u.pd.u.namesptr = items;
    def.flags = flags | (oldValue != dflt ? PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS : 0);
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

// v2.3: a thumbnail picker: custom-UI control area on a PF_Param_NO_DATA param (no value, nothing
// saved in the project, no flatten / unflatten, never part of the render). Its state is derived from
// the popup it belongs to and the params that popup writes.
const int kPickerNominalWidth = 300;
const int kPickerStatusLineH = 16;  // one text line under the grid ("Custom (edited)")
// status lines under the grid: Grain Stock adds "Preview exaggerated" above the Custom line
inline int PickerStatusH(obthumb::Kind kind)
{
    return kPickerStatusLineH * (kind == obthumb::STOCK ? 2 : 1);
}
PF_Err AddPicker(PF_InData* in_data, const char* name, obthumb::Kind kind, A_long id, bool startCollapsed)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_NO_DATA;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.ui_flags = PF_PUI_CONTROL | PF_PUI_DONT_ERASE_CONTROL;
    def.ui_width = (A_short)kPickerNominalWidth;
    def.ui_height = (A_short)(obthumb::PreferredHeight(kind, kPickerNominalWidth) + PickerStatusH(kind));
    def.flags = PF_ParamFlag_CANNOT_TIME_VARY | (startCollapsed ? PF_ParamFlag_START_COLLAPSED : 0);
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddInt(PF_InData* in_data, const char* name, int vmin, int vmax, int dflt, A_long id, PF_ParamUIFlags ui = 0)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_SLIDER;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.sd.valid_min = def.u.sd.slider_min = vmin;
    def.u.sd.valid_max = def.u.sd.slider_max = vmax;
    def.u.sd.value = def.u.sd.dephault = dflt;
    def.flags |= SupFlag(id);
    def.ui_flags = ui;
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

// v2.3 map-feature sliders, the same in both editions (ob_mapx.h). Defaults are the no-op values.
enum { MF_RANGE, MF_FRONT, MF_BACK, MF_BLUR };
PF_Err AddMapFeature(PF_InData* in_data, int which, A_long id)
{
    const PF_ValueDisplayFlags PCT = PF_ValueDisplayFlag_PERCENT;
    switch (which)
    {
    case MF_RANGE:  // % of the map's 0..1 range kept fully sharp around the focus value
        return AddFloat(in_data, obs::T("Focus Range"), 0, 100, 0, 50, 0, 0, 1, PCT, id);
    case MF_FRONT:  // blur in front of the focus (map below Focus; Invert Map swaps the sides)
        return AddFloat(in_data, obs::T("Front Blur"), 0, 400, 0, 200, 100, 100, 1, PCT, id);
    case MF_BACK:
        return AddFloat(in_data, obs::T("Back Blur"), 0, 400, 0, 200, 100, 100, 1, PCT, id);
    default:  // Gaussian sigma of a pre-blur of the map, full-res px (hides steps in 8-bit / painted maps)
        return AddFloat(in_data, obs::T("Map Blur"), 0, 200, 0, 30, 0, 0, 1, PF_ValueDisplayFlag_NONE, id);
    }
}

PF_Err AddColor(PF_InData* in_data, const char* name, const float lin[3], A_long id)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_COLOR;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    auto enc = [](float v) {
        return (A_u_char)std::lround(255.0 * std::pow(std::min(1.0f, std::max(0.0f, v)), 1.0 / kDisplayGamma));
    };
    def.u.cd.value.red = enc(lin[0]);
    def.u.cd.value.green = enc(lin[1]);
    def.u.cd.value.blue = enc(lin[2]);
    def.u.cd.value.alpha = 255;
    def.u.cd.dephault = def.u.cd.value;
    def.flags |= SupFlag(id);
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

#include "kantan_ui.inc"  // Kantan: ParamsSetup, UserChangedParam, UpdateParamsUI

// Probes a layer param's size, then checks out the part of it that maps onto `need` (effect-layer
// coords, stretched to fit). Leaves `ls.has` false if the param is empty.
PF_Err CheckoutSampledLayer(PF_InData* in_data, PF_PreRenderExtra* extra, const PF_RenderRequest& outReq, int paramIdx,
                            int probeId, int id, const RenderPlan& plan, double dsx, double dsy, const PF_LRect& need,
                            LayerSampler& ls)
{
    PF_Err err = PF_Err_NONE;
    ls.has = false;
    PF_RenderRequest probe = outReq;
    probe.rect.left = probe.rect.top = 0;
    probe.rect.right = probe.rect.bottom = 1;
    probe.channel_mask = PF_ChannelMask_ARGB;
    probe.preserve_rgb_of_zero_alpha = false;
    PF_CheckoutResult pr;
    AEFX_CLR_STRUCT(pr);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, paramIdx, probeId, &probe, in_data->current_time,
                                  in_data->time_step, in_data->time_scale, &pr));
    if (err)
        return err;
    const A_long lw = plan.layerBounds.right - plan.layerBounds.left;
    const A_long lh = plan.layerBounds.bottom - plan.layerBounds.top;
    if (!(pr.ref_width > 0 && pr.ref_height > 0 && lw > 0 && lh > 0 && !RectEmpty(pr.max_result_rect)))
        return PF_Err_NONE;
    ls.w = ScaledExtent(pr.ref_width, dsx);
    ls.h = ScaledExtent(pr.ref_height, dsy);
    ls.kx = (double)ls.w / (double)lw;
    ls.ky = (double)ls.h / (double)lh;
    PF_LRect mr;
    mr.left = (A_long)std::floor((need.left - plan.layerBounds.left + 0.5) * ls.kx - 0.5) - 1;
    mr.top = (A_long)std::floor((need.top - plan.layerBounds.top + 0.5) * ls.ky - 0.5) - 1;
    mr.right = (A_long)std::ceil((need.right - plan.layerBounds.left - 0.5) * ls.kx - 0.5) + 2;
    mr.bottom = (A_long)std::ceil((need.bottom - plan.layerBounds.top - 0.5) * ls.ky - 0.5) + 2;
    PF_LRect bounds = {0, 0, ls.w, ls.h};
    mr = Intersect(mr, bounds);
    if (RectEmpty(mr))
        mr = bounds;  // region is outside the layer; sample the nearest edge anyway
    PF_RenderRequest req = outReq;
    req.rect = mr;
    req.channel_mask = PF_ChannelMask_ARGB;
    req.preserve_rgb_of_zero_alpha = false;
    PF_CheckoutResult res;
    AEFX_CLR_STRUCT(res);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, paramIdx, id, &req, in_data->current_time, in_data->time_step,
                                  in_data->time_scale, &res));
    if (!err)
        ls.has = !RectEmpty(res.result_rect);
    return err;
}

// v2.3.1: one-pixel anti-aliasing ring of an input max rect -> the layer edge (see SmartPreRender)
inline PF_LRect SnapAaRing(PF_LRect r, const PF_LRect& layer)
{
    if (r.left < layer.left && layer.left - r.left <= 1)
        r.left = layer.left;
    if (r.top < layer.top && layer.top - r.top <= 1)
        r.top = layer.top;
    if (r.right > layer.right && r.right - layer.right <= 1)
        r.right = layer.right;
    if (r.bottom > layer.bottom && r.bottom - layer.bottom <= 1)
        r.bottom = layer.bottom;
    return r;
}

// v2.3 diagnosis (field data for the adjustment-layer / collapsed-precomp edge case): with the environment
// variable OB_DIAG_RECTS set to a file path, every SMART_PRE_RENDER appends the input geometry AE reports.
// Off (one getenv) unless the variable is set; never set in normal use.
void DiagRects(const PF_LRect& layer, const PF_LRect& inMax, const PF_LRect& content, const PF_LRect& req, int edge, double ds)
{
    std::string path;  // mac port: obp:: (Windows: the same _dupenv_s / fopen_s calls as v2.4.2)
    if (!obp::getenv("OB_DIAG_RECTS", path))
        return;
    static std::mutex m;
    {
        std::lock_guard<std::mutex> lk(m);
        FILE* f = obp::fopen(path.c_str(), "a");
        if (f)
        {
            std::fprintf(f, "%s edge %d ds %.3f layer %d %d %d %d inMax %d %d %d %d content %d %d %d %d request %d %d %d %d\n", OB_NAME, edge,
                         ds, (int)layer.left, (int)layer.top, (int)layer.right, (int)layer.bottom, (int)inMax.left, (int)inMax.top,
                         (int)inMax.right, (int)inMax.bottom, (int)content.left, (int)content.top, (int)content.right,
                         (int)content.bottom, (int)req.left, (int)req.top, (int)req.right, (int)req.bottom);
            fclose(f);
        }
    }
}

PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    (void)out_data;
    PF_Err err = PF_Err_NONE;

    RenderPlan* plan = new (std::nothrow) RenderPlan();
    if (!plan)
        return PF_Err_OUT_OF_MEMORY;
    // Hand ownership to AE immediately so every error path below frees it.
    extra->output->pre_render_data = plan;
    extra->output->delete_pre_render_data_func = DeletePlan;

    ERR(ReadSettings(in_data, plan->s));
    if (err)
        return err;

    const double dsx = Ratio(in_data->downsample_x);
    const double dsy = Ratio(in_data->downsample_y);
    const double par = Ratio(in_data->pixel_aspect_ratio);
    // engine px per full-res "square" px: x shrinks for wide pixels (par > 1)
    plan->ep = MakeEngineParams(plan->s, dsx / par, dsy);
    plan->hq = MakeHqParams(plan->s);
    // v2.3 map features: the adapter computes the signed amount itself (ob_mapx.h) and hands the
    // engine a remapped map with focus 0.5 / scale 2 / gamma 1. Inactive = untouched (bit-exact).
    plan->mx = obmap::Make(plan->s.invertMap, plan->s.focusPct, plan->s.mapScale, plan->s.mapGamma, plan->s.mapBlurPx,
                           plan->s.focusRangePct, plan->s.frontPct, plan->s.backPct, plan->s.focusPointOn,
                           plan->s.focusPointX, plan->s.focusPointY, std::sqrt((dsx / par) * dsy));
    if (plan->mx.active)
    {
        plan->ep.focus = 0.5f;
        plan->ep.mapScale = 2.f;
        plan->ep.mapGamma = 1.f;
        plan->ep.invertMap = false;
    }
    plan->mapExt = plan->mx.active ? obmap::Extent(plan->mx.sigma) : 0;
    // Grain: the pattern is a function of full-resolution layer pixels (downsample folded in here,
    // so half-res previews show the same grain) and of the frame number at the current time
    // (Animate off = one fixed pattern; Hold Frames = the pattern changes every N frames).
    // v2.2 (renderVersion 2): non-square pixels. Grain is generated on square pixels (not stretched on
    // DV / anamorphic footage) and the frame diagonal, vignetting and cat eye are measured in square
    // pixels (Params.pixelAspect = width / height of a buffer pixel). renderVersion 1 keeps v2.1.
    const bool v22 = plan->s.renderVersion >= 2;
    plan->ep.grainScaleX = (float)(v22 ? dsx / par : dsx);
    plan->ep.grainScaleY = (float)dsy;
    plan->ep.pixelAspect = v22 && dsx > 0 ? (float)std::min(20.0, std::max(0.05, par * dsy / dsx)) : 1.f;
    {
        // Grain frame index (v2.2): the frame of the LAYER's own comp (current_time / local_time_step),
        // so a nested comp rendered from a parent with another frame rate shows the same pattern at
        // the same layer time. renderVersion 1 keeps v2.1 (the render's time_step).
        long long frame = 0;
        const A_long stepL = v22 && in_data->local_time_step != 0 ? in_data->local_time_step : in_data->time_step;
        if (stepL != 0)
        {
            const double f = (double)in_data->current_time / std::fabs((double)stepL);
            frame = (long long)std::floor(f + 1e-6);
        }
        const int hold = std::max(1, plan->s.grainHold);
        const long long q = frame >= 0 ? frame / hold : -((-frame + hold - 1) / hold);
        plan->ep.grainFrame = plan->s.grainAnimate ? (int)(q & 0x7fffffff) : 0;
    }
    plan->map.channel = plan->s.mapChannel;
    plan->hl.channel = plan->s.hlLayerChannel;

    const bool amountView = plan->s.view == VIEW_BLUR_AMOUNT;
    const PF_RenderRequest& outReq = extra->input->output_request;

    // Probe the layer size first (1x1 request, pixels never checked out): cat eye, vignetting and
    // glow size are relative to the frame, so the engine gets the true layer geometry and we can
    // request only the output rect plus the engine's margin (no full-layer requests per tile).
    PF_CheckoutResult pr;
    AEFX_CLR_STRUCT(pr);
    {
        PF_RenderRequest probe = outReq;
        probe.rect.left = probe.rect.top = 0;
        probe.rect.right = probe.rect.bottom = 1;
        probe.channel_mask = PF_ChannelMask_ARGB;
        probe.preserve_rgb_of_zero_alpha = false;
        ERR(extra->cb->checkout_layer(in_data->effect_ref, OB_INPUT, CHECKOUT_INPUT_PROBE, &probe,
                                      in_data->current_time, in_data->time_step, in_data->time_scale, &pr));
        if (err)
            return err;
    }
    PF_LRect layer = {0, 0, ScaledExtent(pr.ref_width, dsx), ScaledExtent(pr.ref_height, dsy)};
    if (RectEmpty(layer))
        layer = pr.max_result_rect;
    // The layer rect is the lens FRAME (cat eye / vignetting centre, diagonal-relative sizes); the
    // pixels to process are wherever the input has content: an upstream effect (Grow Bounds,
    // Transform, Drop Shadow, another blur) or a shape / text layer can extend beyond it.
    plan->layerBounds = layer;
    // v2.3.1: AE reports the input of an ADJUSTMENT layer or a COLLAPSED / continuously rasterised layer
    // with max_result_rect = the layer grown by exactly one transparent anti-aliasing pixel on every side,
    // at every resolution (measured in AE 2026 and 2027: aetest\logs\rects_*.txt). Taken as content, that
    // ring made Repeat Edge repeat transparency and Clip to Layer clip to it (corner alpha 0.25 / 0.86).
    // Per side, an overhang of at most one pixel is that ring, not content: the side snaps to the layer
    // edge. Real expansion (Grow Bounds, Transform, text beyond the layer: more than one pixel) is kept
    // exactly as before. Render version 1 (v1.2 / v2.1 projects) keeps the old rule (bit-exact).
    PF_LRect inMax = pr.max_result_rect;
    if (!RectEmpty(inMax) && plan->s.renderVersion >= 2)
        inMax = SnapAaRing(inMax, layer);
    plan->content = RectEmpty(inMax) ? layer : Union(layer, inMax);
    DiagRects(layer, pr.max_result_rect, plan->content, outReq.rect, plan->s.edge, dsx);
    plan->ep.frameLeft = (float)layer.left;
    plan->ep.frameTop = (float)layer.top;
    plan->ep.frameWidth = (float)(layer.right - layer.left);
    plan->ep.frameHeight = (float)(layer.bottom - layer.top);

    // How far the result can extend beyond the content: blur spread + post effects (glow, halation,
    // aberration), the post part capped (see PostSpreadCap). Repeat Edge Pixels keeps the output
    // inside the content, like AE's own blurs.
    const A_long cap = PostSpreadCap(*plan);
    const A_long spread =
        (amountView || plan->s.edge != EDGE_SPREAD) ? 0 : (A_long)EngineSpread(plan->ep, plan->hq, plan->s.quality, cap);
    const PF_LRect maxRect = Grow(plan->content, spread);
    const PF_LRect result = Intersect(outReq.rect, maxRect);
    if (RectEmpty(result))
    {
        // v2.2 (A8): nothing of the request can be non-transparent: no input / layer checkouts at all
        // (Intersect returns {0,0,0,0}, which grown by the margin used to request pixels around the
        // layer origin)
        plan->emptyResult = true;
        extra->output->max_result_rect = maxRect;
        extra->output->result_rect = result;
        extra->output->solid = false;
        extra->output->flags = 0;
        return PF_Err_NONE;
    }
    plan->margin = amountView ? 0 : WindowMargin(*plan, result, cap);

    // Input: result rect grown by the engine margin so ROI/tile edges are exact (AE clips to the layer).
    PF_RenderRequest req = outReq;
    req.rect = Grow(result, plan->margin);
    req.channel_mask = PF_ChannelMask_ARGB;
    req.preserve_rgb_of_zero_alpha = false;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    if (!amountView)
    {
        ERR(extra->cb->checkout_layer(in_data->effect_ref, OB_INPUT, CHECKOUT_INPUT, &req, in_data->current_time,
                                      in_data->time_step, in_data->time_scale, &in_result));
        if (err)
            return err;
    }

    // Blur map / highlight layer: the part we will sample (same downsample, current time).
    const PF_LRect need0 = amountView ? result : Intersect(req.rect, Grow(plan->content, 2 * plan->margin + 2));
    const PF_LRect need = plan->mapExt > 0 ? Grow(need0, plan->mapExt + 1) : need0;  // Map Blur reads around
    ERR(CheckoutSampledLayer(in_data, extra, outReq, OB_MAP_LAYER, CHECKOUT_MAP_PROBE, CHECKOUT_MAP, *plan, dsx, dsy,
                             need, plan->map));
    if (!err && plan->map.has && plan->mx.active && plan->mx.point)
    {
        // Focus at Point: the map around the point, wherever the tile is (the focus value must not
        // depend on which part of the frame is rendered)
        const A_long px = (A_long)std::floor(plan->mx.px), py = (A_long)std::floor(plan->mx.py);
        const PF_LRect np = Grow(PF_LRect{px, py, px + 1, py + 1}, plan->mapExt + 2);
        ERR(CheckoutSampledLayer(in_data, extra, outReq, OB_MAP_LAYER, CHECKOUT_MAP_PT_PROBE, CHECKOUT_MAP_PT, *plan, dsx, dsy, np,
                                 plan->mapPt));
    }
    if (err)
        return err;

    extra->output->max_result_rect = maxRect;
    extra->output->result_rect = result;
    extra->output->solid = false;
    extra->output->flags = 0;
    return err;
}

template <typename Pix>
PF_Err RenderDepth(PF_InData* in_data, PF_OutData* out_data, const RenderPlan& plan, PF_EffectWorld* input,
                   PF_EffectWorld* mapW, PF_EffectWorld* hlW, PF_EffectWorld* output, PF_EffectWorld* mapPtW)
{
    if (plan.s.view == VIEW_BLUR_AMOUNT)
    {
        RenderAmountView<Pix, Pix>(mapW, plan, ResolveFocus<Pix>(plan, mapPtW), output);
        return PF_Err_NONE;
    }
    if (!input || !input->data || input->width <= 0 || input->height <= 0)
    {
        FillTransparent<Pix>(output);
        return PF_Err_NONE;
    }

    // Region the engine processes, in layer coords: the output rect grown by the engine margin,
    // limited to where anything can come from (input content + its spread + margin; the content is the
    // layer rect U the input's max_result_rect, so upstream expansion is kept). Pixels outside the
    // input world are transparent (blur spreads beyond the content) or repeated edge pixels.
    PF_LRect outR = {output->origin_x, output->origin_y, output->origin_x + output->width, output->origin_y + output->height};
    PF_LRect win = Grow(outR, plan.margin);
    if (plan.s.edge == EDGE_CLIP)  // v1.2: the engine buffer ends at the content (edge-normalised blur)
        win = Intersect(win, plan.content);
    else if (plan.s.edge == EDGE_REPEAT)  // repeated edge pixels up to the margin; output inside the content
        win = Intersect(win, Grow(plan.content, plan.margin));
    else
        win = Intersect(win, Grow(plan.content, 2 * plan.margin + 2));
    if (RectEmpty(win))
    {
        FillTransparent<Pix>(output);
        return PF_Err_NONE;
    }
    const A_long w = win.right - win.left, h = win.bottom - win.top;
    const size_t px = (size_t)w * (size_t)h;

    std::vector<float> src, dst, map, hl;
    try
    {
        src.resize(px * 4);
        dst.resize(px * 4);
        if (mapW)
            map.resize(px);
        if (hlW)
            hl.resize(px);
    }
    catch (const std::bad_alloc&)
    {
        return PF_Err_OUT_OF_MEMORY;
    }

    WorldToFloat<Pix>(input, win, plan.s.linear, plan.s.edge == EDGE_REPEAT, plan.content, src.data());
    if (mapW)
        BuildMap<Pix>(mapW, plan, ResolveFocus<Pix>(plan, mapPtW), win, map.data());
    if (hlW)
        FillLayer<Pix>(hlW, plan, plan.hl, win, hl.data());

    PF_Err err = PF_ABORT(in_data);
    if (err)
        return err;

    // frame (layer bounds) relative to the engine buffer origin
    lenscore::Params ep = plan.ep;
    ep.frameLeft = (float)(plan.layerBounds.left - win.left);
    ep.frameTop = (float)(plan.layerBounds.top - win.top);
    ep.frameWidth = (float)(plan.layerBounds.right - plan.layerBounds.left);
    ep.frameHeight = (float)(plan.layerBounds.bottom - plan.layerBounds.top);
    BestSlot bestSlot;  // Best: wait for a slot (A4), polling abort
    if (plan.s.quality == QUALITY_BEST && plan.s.view != VIEW_HL_SOURCE)
    {
        if ((err = bestSlot.Acquire(in_data)) != PF_Err_NONE)
            return err;
    }
    ActiveRenderGuard active;  // A3: split the CPU between concurrent renders
    ep.threads = RenderThreads();
    err = RunEngine(in_data, out_data, plan, ep, src.data(), (int)w, (int)h, mapW ? map.data() : nullptr,
                    hlW ? hl.data() : nullptr, dst.data(),
                    Intersect(PF_LRect{outR.left - win.left, outR.top - win.top, outR.right - win.left, outR.bottom - win.top},
                              PF_LRect{0, 0, w, h}));
    if (err)
        return err;

    // Repeat Edge Pixels: the result stays inside the layer (AE never asks for more, see PreRender)
    FloatToWorld<Pix>(dst.data(), win, plan.s.linear, output, plan.s.edge != EDGE_SPREAD ? &plan.content : nullptr);
    return PF_Err_NONE;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    (void)out_data;
    PF_Err err = PF_Err_NONE;
    const RenderPlan* plan = reinterpret_cast<const RenderPlan*>(extra->input->pre_render_data);
    if (!plan)
        return PF_Err_BAD_CALLBACK_PARAM;

    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    PF_EffectWorld* mapW = nullptr;
    PF_EffectWorld* hlW = nullptr;
    PF_EffectWorld* mapPtW = nullptr;  // v2.3 Focus at Point

    if (plan->emptyResult)  // nothing was checked out in PreRender (A8)
    {
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
        if (!err && output && output->data)
        {
            switch (extra->input->bitdepth)
            {
            case 32: FillTransparent<PF_PixelFloat>(output); break;
            case 16: FillTransparent<PF_Pixel16>(output); break;
            default: FillTransparent<PF_Pixel8>(output); break;
            }
        }
        return err;
    }
    if (plan->s.view != VIEW_BLUR_AMOUNT)
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_INPUT, &input));
    if (!err && plan->map.has)
    {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_MAP, &mapW));
        if (!err && mapW && (!mapW->data || mapW->width <= 0 || mapW->height <= 0))
            mapW = nullptr;
    }
    if (!err && plan->map.has && plan->mapPt.has)
    {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_MAP_PT, &mapPtW));
        if (!err && mapPtW && (!mapPtW->data || mapPtW->width <= 0 || mapPtW->height <= 0))
            mapPtW = nullptr;
    }
    if (!err && plan->hl.has)
    {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_HL, &hlW));
        if (!err && hlW && (!hlW->data || hlW->width <= 0 || hlW->height <= 0))
            hlW = nullptr;
    }
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (err)
        return err;
    if (!output || !output->data || output->width <= 0 || output->height <= 0)
        return PF_Err_NONE;

    switch (extra->input->bitdepth)
    {
    case 32:
        err = RenderDepth<PF_PixelFloat>(in_data, out_data, *plan, input, mapW, hlW, output, mapPtW);
        break;
    case 16:
        err = RenderDepth<PF_Pixel16>(in_data, out_data, *plan, input, mapW, hlW, output, mapPtW);
        break;
    case 8:
        err = RenderDepth<PF_Pixel8>(in_data, out_data, *plan, input, mapW, hlW, output, mapPtW);
        break;
    default:
        err = PF_Err_BAD_CALLBACK_PARAM;
        break;
    }
    return err;
}

// ------------------------------------------------------- v2.3 thumbnail pickers ---
// Custom UI (Drawbot) in the control area of three PF_Param_NO_DATA params, under Lens Preset,
// Aberration Style and Grain Stock. Clicking a thumbnail does exactly what choosing that popup item
// does (same Apply*ToParams code, one event = one undo step); the popups stay the source of truth,
// so the pickers keep no state of their own (nothing to save, flatten or keep in sync) except the
// hover cell below, which is cosmetic.

struct PickerInfo
{
    obthumb::Kind kind;
    int popup;  // param index of the popup it drives
    bool (*apply)(int, PF_ParamDef*[]);
    int first;  // popup value of the first thumbnail (2: the full effect's popups start with "Custom")
};

bool PickerFor(int paramIndex, PickerInfo& pi)
{
    const int first = 1;  // Kantan popups have no "Custom"; the popup is the whole state
    auto apLens = KantanPopupOnly, apStyle = KantanPopupOnly, apStock = KantanPopupOnly;
    if (paramIndex == OB_LENS_PICKER)
        pi = {obthumb::LENS, OB_PRESET, apLens, first};
    else if (paramIndex == OB_STYLE_PICKER)
        pi = {obthumb::STYLE, OB_LA_STYLE, apStyle, first};
    else if (paramIndex == OB_STOCK_PICKER)
        pi = {obthumb::STOCK, OB_GRAIN_STOCK, apStock, first};
    else
        return false;
    return true;
}

// Do the current param values equal what popup item `value` writes? (else the picker shows
// "Custom (edited)": e.g. a written param is keyframed or driven by an expression)
bool PickerValuesMatch(const PickerInfo& pi, int value, PF_ParamDef* params[])
{
    std::vector<PF_ParamDef> copy(OB_NUM_PARAMS);
    std::vector<PF_ParamDef*> ptr(OB_NUM_PARAMS);
    for (int i = 0; i < OB_NUM_PARAMS; ++i)
    {
        if (!params[i])
            return true;  // incomplete params array: do not claim a mismatch
        copy[i] = *params[i];
        copy[i].uu.change_flags = 0;
        ptr[i] = &copy[i];
    }
    if (!pi.apply(value, ptr.data()))
        return false;
    for (int i = 0; i < OB_NUM_PARAMS; ++i)
    {
        if (!(copy[i].uu.change_flags & PF_ChangeFlag_CHANGED_VALUE))
            continue;
        const PF_ParamDef& a = *params[i];
        const PF_ParamDef& b = copy[i];
        switch (a.param_type)
        {
        case PF_Param_FLOAT_SLIDER:
            if (std::fabs(a.u.fs_d.value - b.u.fs_d.value) > 1e-3 * std::max(1.0, std::fabs(b.u.fs_d.value)))
                return false;
            break;
        case PF_Param_SLIDER:
            if (a.u.sd.value != b.u.sd.value)
                return false;
            break;
        case PF_Param_POPUP:
            if (a.u.pd.value != b.u.pd.value)
                return false;
            break;
        case PF_Param_ANGLE:
            if (std::labs((long)a.u.ad.value - (long)b.u.ad.value) > 2)
                return false;
            break;
        case PF_Param_COLOR:
            if (a.u.cd.value.red != b.u.cd.value.red || a.u.cd.value.green != b.u.cd.value.green ||
                a.u.cd.value.blue != b.u.cd.value.blue)
                return false;
            break;
        default:
            break;
        }
    }
    return true;
}

// hover cell per (context, param): UI thread only, cosmetic
std::mutex g_hoverM;
std::map<std::pair<const void*, int>, int> g_hover;
int GetHover(PF_ContextH ctx, int idx)
{
    std::lock_guard<std::mutex> lk(g_hoverM);
    const auto it = g_hover.find(std::make_pair((const void*)ctx, idx));
    return it == g_hover.end() ? -1 : it->second;
}
bool SetHover(PF_ContextH ctx, int idx, int cell)
{
    std::lock_guard<std::mutex> lk(g_hoverM);
    int& h = g_hover[std::make_pair((const void*)ctx, idx)];
    const bool changed = h != cell;
    h = cell;
    return changed;
}

// Thumbnails are rendered by a background worker (not a render thread, not the UI thread beyond a
// short budget). It is joined on GLOBAL_SETDOWN.
std::mutex g_warmM;
std::thread g_warmThread;
std::atomic<bool> g_warmStarted{false};
// If the host never sends GLOBAL_SETDOWN, a still-joinable std::thread would call std::terminate when
// the DLL's statics are destroyed: detach it then (at process exit the thread is gone anyway).
struct WarmThreadGuard
{
    ~WarmThreadGuard()
    {
        if (g_warmThread.joinable())
            g_warmThread.detach();
    }
} g_warmThreadGuard;
void StartWarm(int scale)
{
    bool expected = false;
    if (!g_warmStarted.compare_exchange_strong(expected, true))
        return;
    std::lock_guard<std::mutex> lk(g_warmM);
    try
    {
        g_warmThread = std::thread([scale] {
            for (int s : {scale, scale == 1 ? 2 : 1})
                for (int k = 0; k < obthumb::kKinds; ++k)
                    obthumb::Warm((obthumb::Kind)k, s, 1e9);
        });
    }
    catch (...)
    {
        // no thread: the UI warms synchronously in small steps instead
    }
}
void JoinWarm()
{
    std::lock_guard<std::mutex> lk(g_warmM);
    if (g_warmThread.joinable())
        g_warmThread.join();
}

struct DrawbotAll
{
    const DRAWBOT_DrawbotSuite1* draw = nullptr;
    const DRAWBOT_SupplierSuite1* supplier = nullptr;
    const DRAWBOT_SurfaceSuite2* surface = nullptr;
    const DRAWBOT_PathSuite1* path = nullptr;
    const DRAWBOT_ImageSuite1* image = nullptr;
    SPBasicSuite* sp = nullptr;
    bool ok = false;
    explicit DrawbotAll(SPBasicSuite* b) : sp(b)
    {
        if (!sp)
            return;
        ok = !sp->AcquireSuite(kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_Version1, (const void**)&draw) && draw &&
             !sp->AcquireSuite(kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_Version1, (const void**)&supplier) && supplier &&
             !sp->AcquireSuite(kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_Version2, (const void**)&surface) && surface &&
             !sp->AcquireSuite(kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_Version1, (const void**)&path) && path &&
             !sp->AcquireSuite(kDRAWBOT_ImageSuite, kDRAWBOT_ImageSuite_Version1, (const void**)&image) && image;
    }
    ~DrawbotAll()
    {
        if (!sp)
            return;
        if (draw) sp->ReleaseSuite(kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_Version1);
        if (supplier) sp->ReleaseSuite(kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_Version1);
        if (surface) sp->ReleaseSuite(kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_Version2);
        if (path) sp->ReleaseSuite(kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_Version1);
        if (image) sp->ReleaseSuite(kDRAWBOT_ImageSuite, kDRAWBOT_ImageSuite_Version1);
    }
};

DRAWBOT_ColorRGBA ToDb(const PF_App_Color& c)
{
    DRAWBOT_ColorRGBA d;
    d.red = c.red / 65535.f;
    d.green = c.green / 65535.f;
    d.blue = c.blue / 65535.f;
    d.alpha = 1.f;
    return d;
}

struct PickerFrame
{
    float x = 0, y = 0, w = 0, h = 0;
};
PickerFrame FrameOf(const PF_EventExtra* e)
{
    PickerFrame f;
    f.x = (float)e->effect_win.current_frame.left;
    f.y = (float)e->effect_win.current_frame.top;
    f.w = (float)(e->effect_win.current_frame.right - e->effect_win.current_frame.left);
    f.h = (float)(e->effect_win.current_frame.bottom - e->effect_win.current_frame.top);
    return f;
}
obthumb::Grid PickerGrid(const PickerInfo& pi, const PickerFrame& f)
{
    return obthumb::Layout(obthumb::Count(pi.kind), f.w, std::max(1.f, f.h - PickerStatusH(pi.kind)));
}

void StrokeRect(DrawbotAll& db, DRAWBOT_SupplierRef sup, DRAWBOT_SurfaceRef surf, const DRAWBOT_ColorRGBA& c, float width,
                float x, float y, float w, float h)
{
    DRAWBOT_PenRef pen = nullptr;
    DRAWBOT_PathRef path = nullptr;
    if (!db.supplier->NewPen(sup, &c, width, &pen) && !db.supplier->NewPath(sup, &path))
    {
        DRAWBOT_RectF32 r = {x, y, w, h};
        db.path->AddRect(path, &r);
        db.surface->StrokePath(surf, pen, path);
    }
    if (path)
        db.supplier->ReleaseObject((DRAWBOT_ObjectRef)path);
    if (pen)
        db.supplier->ReleaseObject((DRAWBOT_ObjectRef)pen);
}

void DrawText(DrawbotAll& db, DRAWBOT_SupplierRef sup, DRAWBOT_SurfaceRef surf, DRAWBOT_FontRef font,
              const DRAWBOT_ColorRGBA& c, const std::wstring& s, float x, float y, float width, DRAWBOT_TextAlignment align)
{
    DRAWBOT_BrushRef brush = nullptr;
    if (!font || db.supplier->NewBrush(sup, &c, &brush) || !brush)
        return;
    std::vector<DRAWBOT_UTF16Char> u16(s.begin(), s.end());
    u16.push_back(0);
    DRAWBOT_PointF32 pt = {x, y};
    db.surface->DrawString(surf, brush, font, u16.data(), &pt, align, kDRAWBOT_TextTruncation_EndEllipsis, width);
    db.supplier->ReleaseObject((DRAWBOT_ObjectRef)brush);
}

PF_Err DrawPicker(PF_InData* in_data, PF_ParamDef* params[], PF_EventExtra* e, const PickerInfo& pi)
{
    DrawbotAll db(in_data->pica_basicP);
    const PFAppSuite6* app = nullptr;
    const PF_EffectCustomUISuite1* cui = nullptr;
    SPBasicSuite* sp = in_data->pica_basicP;
    if (!db.ok || !sp || sp->AcquireSuite(kPFAppSuite, kPFAppSuiteVersion6, (const void**)&app) || !app)
        return PF_Err_NONE;
    PF_Err err = PF_Err_NONE;
    if (!sp->AcquireSuite(kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, (const void**)&cui) && cui)
    {
        DRAWBOT_DrawRef dref = nullptr;
        DRAWBOT_SupplierRef sup = nullptr;
        DRAWBOT_SurfaceRef surf = nullptr;
        if (!cui->PF_GetDrawingReference(e->contextH, &dref) && dref && !db.draw->GetSupplier(dref, &sup) &&
            !db.draw->GetSurface(dref, &surf) && sup && surf)
        {
            PF_App_Color bgc = {0, 0, 0}, txc = {0, 0, 0}, hot = {0, 0, 0};
            app->PF_AppGetBgColor(&bgc);
            app->PF_AppGetColor(PF_App_Color_TEXT, &txc);
            app->PF_AppGetColor(PF_App_Color_HOT_TEXT, &hot);
            const DRAWBOT_ColorRGBA bg = ToDb(bgc), text = ToDb(txc), sel = ToDb(hot);
            DRAWBOT_ColorRGBA dim = text;
            dim.alpha = 0.45f;
            const PickerFrame f = FrameOf(e);
            DRAWBOT_RectF32 all = {f.x, f.y, f.w, f.h};
            db.surface->PaintRect(surf, &bg, &all);
            float screenScale = 1.f;
            db.surface->GetTransformToScreenScale(surf, &screenScale);
            const int scale = screenScale >= 1.5f ? 2 : 1;
            StartWarm(scale);
            obthumb::Warm(pi.kind, scale, 30.0);  // at most ~30 ms on the UI thread
            const int n = obthumb::Count(pi.kind);
            const obthumb::Grid g = PickerGrid(pi, f);
            const int value = params[pi.popup] ? params[pi.popup]->u.pd.value : 1;
            const int selIdx = value - pi.first;
            const bool matches = selIdx >= 0 && PickerValuesMatch(pi, value, params);
            const int hover = GetHover(e->contextH, e->effect_win.index);
            DRAWBOT_Boolean bgra = FALSE;
            db.supplier->SupportsPixelLayoutBGRA(sup, &bgra);
            float defSize = 11.f;
            db.supplier->GetDefaultFontSize(sup, &defSize);
            // label font: the panel font, scaled with the cells, never taller than a label line
            const float fontPx = std::max(7.f, std::min(defSize * 0.92f, obthumb::kLabelLineH * 0.86f) * g.s);
            DRAWBOT_FontRef font = nullptr;
            db.supplier->NewDefaultFont(sup, fontPx, &font);
            DRAWBOT_FontRef statusFont = nullptr;
            db.supplier->NewDefaultFont(sup, std::max(7.f, std::min(defSize * 0.92f, kPickerStatusLineH * 0.75f)), &statusFont);
            bool missing = false;
            for (int i = 0; i < n; ++i)
            {
                float cx, cy, tw, th;
                obthumb::CellRect(g, i, cx, cy, tw, th);
                cx += f.x;
                cy += f.y;
                const obthumb::Image* img = obthumb::Get(pi.kind, i, scale, false);
                if (img)
                {
                    std::vector<unsigned char> buf;
                    const unsigned char* data = img->bgra.data();
                    int layout = kDRAWBOT_PixelLayout_32BGRA_Straight;
                    if (!bgra)
                    {
                        buf.resize(img->bgra.size());
                        for (size_t k = 0; k + 3 < buf.size(); k += 4)
                        {
                            buf[k] = 255;
                            buf[k + 1] = img->bgra[k + 2];
                            buf[k + 2] = img->bgra[k + 1];
                            buf[k + 3] = img->bgra[k];
                        }
                        data = buf.data();
                        layout = kDRAWBOT_PixelLayout_32ARGB_Straight;
                    }
                    DRAWBOT_ImageRef im = nullptr;
                    if (!db.supplier->NewImageFromBuffer(sup, img->w, img->h, img->w * 4, layout, data, &im) && im)
                    {
                        // device px -> logical px, times the cell scale
                        db.image->SetScaleFactor(im, (float)scale / std::max(0.05f, g.s));
                        DRAWBOT_PointF32 o = {cx, cy};
                        db.surface->DrawImage(surf, im, &o, 1.f);
                        db.supplier->ReleaseObject((DRAWBOT_ObjectRef)im);
                    }
                }
                else
                {
                    missing = true;
                    DRAWBOT_ColorRGBA ph = dim;
                    ph.alpha = 0.18f;
                    DRAWBOT_RectF32 r = {cx, cy, tw, th};
                    db.surface->PaintRect(surf, &ph, &r);
                }
                const bool isSel = i == selIdx && matches;
                if (i == selIdx)
                    StrokeRect(db, sup, surf, sel, isSel ? 2.f : 1.f, cx - 2.f, cy - 2.f, tw + 4.f, th + 4.f);
                else if (i == hover)
                    StrokeRect(db, sup, surf, dim, 1.f, cx - 1.5f, cy - 1.5f, tw + 3.f, th + 3.f);
                // label: up to two centred lines (same wrap as tools\picker_sheet); line 2 ends with "..."
                float lx, ly, lw, lh;
                obthumb::LabelRect(g, i, lx, ly, lw, lh);
                const std::vector<std::wstring> lines = obthumb::WrapLabel(obs::W(obthumb::NameEn(pi.kind, i)), lw, fontPx);
                const float lineH = obthumb::kLabelLineH * g.s;
                for (size_t L = 0; L < lines.size() && L < 2; ++L)
                    DrawText(db, sup, surf, font, i == selIdx ? sel : text, lines[L], f.x + lx + lw * 0.5f,
                             f.y + ly + 2.f * g.s + lineH * (L + 0.82f), lw, kDRAWBOT_TextAlignment_Center);
            }
            // status line: Custom, or a preset whose written values were changed (keyframes / expressions)
            // (own lines below the grid: nothing else is drawn there, so it cannot overlap at any width)
            const float sy = f.y + f.h - 4.f;
            if (missing)
                DrawText(db, sup, surf, statusFont, dim, obs::W("Rendering previews..."), f.x + 4.f, sy, f.w - 8.f,
                         kDRAWBOT_TextAlignment_Left);
            else if (selIdx < 0 || !matches)
                DrawText(db, sup, surf, statusFont, sel, obs::W("Custom (edited)"), f.x + 4.f, sy, f.w - 8.f,
                         kDRAWBOT_TextAlignment_Left);
            if (pi.kind == obthumb::STOCK)
                DrawText(db, sup, surf, statusFont, dim, obs::W("Preview exaggerated"), f.x + 4.f, sy - kPickerStatusLineH,
                         f.w - 8.f, kDRAWBOT_TextAlignment_Left);
            if (font)
                db.supplier->ReleaseObject((DRAWBOT_ObjectRef)font);
            if (statusFont)
                db.supplier->ReleaseObject((DRAWBOT_ObjectRef)statusFont);
        }
        sp->ReleaseSuite(kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1);
    }
    sp->ReleaseSuite(kPFAppSuite, kPFAppSuiteVersion6);
    e->evt_out_flags = PF_EO_HANDLED_EVENT;
    return err;
}

void InvalidatePicker(PF_InData* in_data, PF_EventExtra* e)
{
    const PFAppSuite6* app = nullptr;
    SPBasicSuite* sp = in_data->pica_basicP;
    if (!sp || sp->AcquireSuite(kPFAppSuite, kPFAppSuiteVersion6, (const void**)&app) || !app)
        return;
    PF_Rect r;
    r.left = (A_short)e->effect_win.current_frame.left;
    r.top = (A_short)e->effect_win.current_frame.top;
    r.right = (A_short)e->effect_win.current_frame.right;
    r.bottom = (A_short)e->effect_win.current_frame.bottom;
    app->PF_InvalidateRect(e->contextH, &r);
    sp->ReleaseSuite(kPFAppSuite, kPFAppSuiteVersion6);
}

PF_Err HandleEvent(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_EventExtra* e)
{
    (void)out_data;
    if (!e || !e->contextH || !*e->contextH || (*e->contextH)->w_type != PF_Window_EFFECT || !params)
        return PF_Err_NONE;
    PickerInfo pi;
    if (!PickerFor(e->effect_win.index, pi) || e->effect_win.area != PF_EA_CONTROL)
        return PF_Err_NONE;
    const PickerFrame f = FrameOf(e);
    switch (e->e_type)
    {
    case PF_Event_DRAW:
        return DrawPicker(in_data, params, e, pi);
    case PF_Event_DO_CLICK:
    {
        const obthumb::Grid g = PickerGrid(pi, f);
        const int cell = obthumb::HitTest(g, obthumb::Count(pi.kind), e->u.do_click.screen_point.h - f.x,
                                          e->u.do_click.screen_point.v - f.y);
        if (cell >= 0 && params[pi.popup])
        {
            // exactly what choosing the popup item does (UserChangedParam): one event, one undo step
            const int value = cell + pi.first;
            params[pi.popup]->u.pd.value = (A_short)value;
            params[pi.popup]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
            pi.apply(value, params);
            InvalidatePicker(in_data, e);
            e->evt_out_flags = PF_EO_HANDLED_EVENT | PF_EO_UPDATE_NOW;
        }
        else
            e->evt_out_flags = PF_EO_HANDLED_EVENT;
        return PF_Err_NONE;
    }
    case PF_Event_ADJUST_CURSOR:
    {
        const obthumb::Grid g = PickerGrid(pi, f);
        const int cell = obthumb::HitTest(g, obthumb::Count(pi.kind), e->u.adjust_cursor.screen_point.h - f.x,
                                          e->u.adjust_cursor.screen_point.v - f.y);
        e->u.adjust_cursor.set_cursor = cell >= 0 ? PF_Cursor_FINGER_POINTER : PF_Cursor_NONE;
        // redraw for the hover outline, or when more thumbnails have been rendered meanwhile
        const bool hoverChanged = SetHover(e->contextH, e->effect_win.index, cell);
        const bool pending = !obthumb::Warm(pi.kind, 2, 0.0) && !obthumb::Warm(pi.kind, 1, 0.0);
        if (hoverChanged || pending)
            InvalidatePicker(in_data, e);
        e->evt_out_flags = PF_EO_HANDLED_EVENT;
        return PF_Err_NONE;
    }
    default:
        return PF_Err_NONE;
    }
}
} // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    (void)inSPBasicSuitePtr;
    (void)inHostName;
    (void)inHostVersion;
    PF_Err result = PF_Err_INVALID_CALLBACK;

    result = PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        OB_NAME,
        OB_MATCH_NAME,
        OB_CATEGORY,
        AE_RESERVED_INFO,
        "EffectMain",
        OB_SUPPORT_URL);

    return result;
}

PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra)
{
    (void)output;
    PF_Err err = PF_Err_NONE;
    try
    {
        switch (cmd)
        {
        case PF_Cmd_ABOUT:
            err = About(in_data, out_data);
            break;
        case PF_Cmd_GLOBAL_SETDOWN:
            // v2.2: drop lenshq's process-wide lens-table cache (up to 1.5 GB) when AE unloads us
            lenshq::purgeTableCache();
            JoinWarm();  // v2.3: thumbnail worker
            obgpuw::Shutdown();  // v2.4
            break;
        case PF_Cmd_EVENT:
            err = HandleEvent(in_data, out_data, params, reinterpret_cast<PF_EventExtra*>(extra));
            break;
        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(in_data, out_data);
            break;
        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_data, out_data);
            break;
        case PF_Cmd_SMART_PRE_RENDER:
            err = SmartPreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
            break;
        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
            break;
        case PF_Cmd_USER_CHANGED_PARAM:
            err = UserChangedParam(
                in_data, out_data, params, reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
            break;
        case PF_Cmd_UPDATE_PARAMS_UI:
            err = UpdateParamsUI(in_data, out_data, params);
            break;
        case PF_Cmd_QUERY_DYNAMIC_FLAGS:
            err = QueryDynamicFlags(in_data, out_data);
            break;
        default:
            break;
        }
    }
    catch (const std::bad_alloc&)
    {
        return PF_Err_OUT_OF_MEMORY;
    }
    catch (PF_Err& thrown_err)
    {
        err = thrown_err;
    }
    catch (...)
    {
        // v2.2 (A10): a message instead of PF_Err_INTERNAL_STRUCT_DAMAGED ("save and quit")
        err = ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "unexpected internal error. The command was not completed.");
    }
    return err;
}
