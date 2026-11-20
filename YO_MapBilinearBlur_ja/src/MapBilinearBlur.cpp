// Map Bilinear Blur (prototype) - After Effects SmartFX wrapper around mbb_core (pyramid blur driven by
// a depth / blur map, with a sharpen applied to the blurred result).
//
// Coordinates: "layer coords" = downsampled layer pixel coordinates (SmartFX request / result rects).
// A checked-out world's pixel (0,0) sits at layer coords (origin_x, origin_y).
// Pixels: AE worlds are ARGB and STRAIGHT (unpremultiplied) at 8/16/32 bpc - established by Optic Bokeh's
// host tests in AE 2026 (see LensBlur\plugin\OpticBokeh.cpp, kHostPixelsPremultiplied). We premultiply on
// input and unpremultiply on output; the engine works premultiplied in the layer's own (encoded) values,
// like AE's own blurs.
// Patterns (layer sampling, strings, MFR thread split, error reporting) are adapted from Optic Bokeh v2.3.

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectSuites.h"

#include "MapBilinearBlurConfig.h"
#include "mbb_core.h"
#include "mbb_strings.h"
#include "mbbgpu.h"  // v0.3: GPU engine (gpu\, Rust + wgpu static library); the CPU engine stays the fallback

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "platform/plat.h"  // mac_port: environment, log file, process / thread ids (was <windows.h> in v0.3.1)

extern "C" DllExport PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra);

// NON_PARAM_VARY: only while "Dither" + "Animate Dither" is on (QUERY_DYNAMIC_FLAGS clears it otherwise, as
// Optic Bokeh does for its grain), so ordinary use keeps AE's frame cache.
const A_long kOutFlags = PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_DEEP_COLOR_AWARE;
const A_long kOutFlags2 = PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                          PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
static_assert(MBB_OUTFLAGS == kOutFlags, "PiPL out_flags mismatch");
static_assert(MBB_OUTFLAGS2 == kOutFlags2, "PiPL out_flags2 mismatch");
#define MBB_STAGE PF_Stage_DEVELOP
static_assert(MBB_PIPL_VERSION ==
                  PF_VERSION(MBB_MAJOR_VERSION, MBB_MINOR_VERSION, MBB_BUG_VERSION, MBB_STAGE, MBB_BUILD_VERSION),
              "PiPL version mismatch");

namespace {

// ------------------------------------------------------------------ params ---
enum
{
    P_INPUT = 0,
    P_METHOD,        // v0.1 "Method" (hidden, ignored; its disk ID is kept)
    P_INTERP,        // v0.2
    P_DOWN,
    P_BLEND,
    P_ANTIRING,
    P_PROTECT,
    P_DITHER_SEED,
    P_DITHER_ANIM,
    P_RADIUS,
    P_MAP_START,
    P_MAP_LAYER,
    P_MAP_CHANNEL,
    P_MAP_INVERT,
    P_FOCUS,
    P_FOCUS_RANGE,
    P_DEPTH,
    P_MAP_END,
    P_SHARP_START,
    P_SHARP_AMOUNT,
    P_SHARP_TYPE,
    P_SHARP_RADIUS,
    P_SHARP_END,
    P_EDGE,
    P_VIEW,
    P_NUM
};
// disk IDs: never change once a build has been given out
enum
{
    D_METHOD = 1,
    D_RADIUS,
    D_MAP_START,
    D_MAP_LAYER,
    D_MAP_CHANNEL,
    D_MAP_INVERT,
    D_FOCUS,
    D_FOCUS_RANGE,
    D_DEPTH,
    D_MAP_END,
    D_SHARP_START,
    D_SHARP_AMOUNT,
    D_SHARP_TYPE,
    D_SHARP_RADIUS,
    D_SHARP_END,
    D_EDGE,
    D_VIEW,
    D_INTERP,        // v0.2 (append only)
    D_DOWN,
    D_BLEND,
    D_ANTIRING,
    D_PROTECT,
    D_DITHER_SEED,
    D_DITHER_ANIM
};
static_assert(D_VIEW == 17 && D_INTERP == 18 && D_DITHER_ANIM == 24, "disk IDs are append-only");

enum { CH_LUMINANCE = 1, CH_RED, CH_GREEN, CH_BLUE, CH_ALPHA };       // popup values (1-based)
// Interpolation popup. Mitchell and Lanczos 3 exist in the engine but are not offered: measured on sample1 they
// look like B-spline (Mitchell) and Catmull-Rom (Lanczos, mean |diff| 0.2-0.9 / 255) - tools\compare_interp.py
enum { INTERP_NEAREST = 1, INTERP_BILINEAR, INTERP_SMOOTHSTEP, INTERP_BSPLINE, INTERP_CATMULLROM, INTERP_HEX };
const int kInterpToEngine[7] = {mbb::IN_BSPLINE, mbb::IN_NEAREST, mbb::IN_BILINEAR, mbb::IN_SMOOTHSTEP, mbb::IN_BSPLINE, mbb::IN_CATMULLROM,
                                mbb::IN_HEX};
// Downsample: Auto = smooth levels for B-spline (the v0.1 default look), 2x2 average for the others (on smooth
// levels every interpolation looks almost the same - the cells must stay crisp for the kernel to show)
enum { DOWN_AUTO = 1, DOWN_AVERAGE, DOWN_SMOOTH };
enum { BLEND_SMOOTH = 1, BLEND_STEPS, BLEND_DITHER };
enum { SHARP_AE = 1, SHARP_UNSHARP };
enum { EDGE_REPEAT = 1, EDGE_SPREAD, EDGE_CLIP };
enum { VIEW_RESULT = 1, VIEW_AMOUNT };

enum { CO_INPUT_PROBE = 1, CO_INPUT = 2, CO_MAP_PROBE = 3, CO_MAP = 4 };

// How AE's Sharpen amount maps to our 3x3 kernel (centre 1 + 8a, neighbours -a): a = amount / 100 *
// kAeSharpenScale at full resolution. UNVERIFIED against AE (no AE run allowed): to be calibrated.
const double kAeSharpenScale = 1.0;
// At a downsample ds (< 1) the kernel strength is a * ds^kSharpenResExp, so a Half preview of
// "blur + sharpen" looks like the Full render shrunk (measured offline, see test\host.cpp "sharpres").
const double kSharpenResExp = 2.0;

struct Settings
{
    int interp = INTERP_BSPLINE;
    int down = DOWN_AUTO;
    int blend = BLEND_SMOOTH;
    bool antiRing = true, protect = true, ditherAnim = false;
    int ditherSeed = 0;
    double radius = 20;
    int mapChannel = CH_LUMINANCE;
    bool invert = false;
    double focusPct = 0, rangePct = 0, depth = 1;
    double sharpAmount = 0;
    int sharpType = SHARP_AE;
    double sharpRadius = 3;
    int edge = EDGE_REPEAT;
    int view = VIEW_RESULT;
};

struct LayerSampler
{
    bool has = false;
    double kx = 1, ky = 1;  // sampled-layer px per effect-layer px
    A_long w = 0, h = 0;
    int channel = CH_LUMINANCE;
};

struct RenderPlan
{
    Settings s;
    mbb::Settings es;
    double ds = 1;
    PF_LRect layerBounds{};
    PF_LRect content{};
    PF_LRect domain{};
    PF_LRect result{};
    PF_LRect R0{};
    LayerSampler map;
    bool passthrough = false;
    bool emptyResult = false;
};

void DeletePlan(void* p) { delete reinterpret_cast<RenderPlan*>(p); }

inline double Ratio(const PF_RationalScale& r) { return (r.den != 0 && r.num != 0) ? (double)r.num / (double)r.den : 1.0; }
inline bool RectEmpty(const PF_LRect& r) { return r.right <= r.left || r.bottom <= r.top; }
inline PF_LRect Intersect(const PF_LRect& a, const PF_LRect& b)
{
    PF_LRect r;
    r.left = std::max(a.left, b.left);
    r.top = std::max(a.top, b.top);
    r.right = std::min(a.right, b.right);
    r.bottom = std::min(a.bottom, b.bottom);
    if (RectEmpty(r))
        r.left = r.top = r.right = r.bottom = 0;
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
inline A_long ScaledExtent(A_long fullRes, double ds) { return (A_long)std::ceil(fullRes * ds - 1e-6); }
inline A_long FloorTo(A_long v, A_long a) { return (A_long)(std::floor((double)v / a) * a); }
inline A_long CeilTo(A_long v, A_long a) { return (A_long)(std::ceil((double)v / a) * a); }
inline mbb::Rect ToR(const PF_LRect& r) { return mbb::Rect{(int)r.left, (int)r.top, (int)r.right, (int)r.bottom}; }
inline PF_LRect ToL(const mbb::Rect& r) { return PF_LRect{r.l, r.t, r.r, r.b}; }

PF_Err ReadSettings(PF_InData* in_data, Settings& s)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef p[P_NUM];
    bool out[P_NUM] = {};
    const int idx[] = {P_INTERP, P_DOWN, P_BLEND, P_ANTIRING, P_PROTECT, P_DITHER_SEED, P_DITHER_ANIM, P_RADIUS, P_MAP_CHANNEL, P_MAP_INVERT, P_FOCUS, P_FOCUS_RANGE, P_DEPTH,
                       P_SHARP_AMOUNT, P_SHARP_TYPE, P_SHARP_RADIUS, P_EDGE, P_VIEW};
    for (int i : idx)
    {
        AEFX_CLR_STRUCT(p[i]);
        ERR(PF_CHECKOUT_PARAM(in_data, i, in_data->current_time, in_data->time_step, in_data->time_scale, &p[i]));
        if (!err)
            out[i] = true;
    }
    if (!err)
    {
        auto fin = [](double v, double lo, double hi, double dflt) { return std::isfinite(v) ? std::min(hi, std::max(lo, v)) : dflt; };
        s.interp = std::min((int)INTERP_HEX, std::max(1, (int)p[P_INTERP].u.pd.value));
        s.down = std::min(3, std::max(1, (int)p[P_DOWN].u.pd.value));
        s.blend = std::min(3, std::max(1, (int)p[P_BLEND].u.pd.value));
        s.antiRing = p[P_ANTIRING].u.bd.value != 0;
        s.protect = p[P_PROTECT].u.bd.value != 0;
        s.ditherSeed = (int)p[P_DITHER_SEED].u.sd.value;
        s.ditherAnim = p[P_DITHER_ANIM].u.bd.value != 0;
        s.radius = fin(p[P_RADIUS].u.fs_d.value, 0, 4000, 0);
        s.mapChannel = std::min(5, std::max(1, (int)p[P_MAP_CHANNEL].u.pd.value));
        s.invert = p[P_MAP_INVERT].u.bd.value != 0;
        s.focusPct = fin(p[P_FOCUS].u.fs_d.value, 0, 100, 0);
        s.rangePct = fin(p[P_FOCUS_RANGE].u.fs_d.value, 0, 100, 0);
        s.depth = fin(p[P_DEPTH].u.fs_d.value, 0, 20, 1);
        s.sharpAmount = fin(p[P_SHARP_AMOUNT].u.fs_d.value, 0, 500, 0);
        s.sharpType = std::min(2, std::max(1, (int)p[P_SHARP_TYPE].u.pd.value));
        s.sharpRadius = fin(p[P_SHARP_RADIUS].u.fs_d.value, 0.5, 200, 3);
        s.edge = std::min(3, std::max(1, (int)p[P_EDGE].u.pd.value));
        s.view = std::min(2, std::max(1, (int)p[P_VIEW].u.pd.value));
    }
    for (int i : idx)
        if (out[i])
            ERR2(PF_CHECKIN_PARAM(in_data, &p[i]));
    return err;
}

// --------------------------------------------------------- map -> amount ---
// Focus Range: 0 inside |d| <= half, a quadratic shoulder of width 2*s, then d - half - s (C1).
inline double Shoulder(double t, double half, double s)
{
    if (half <= 0)
        return t;
    if (t <= half)
        return 0;
    if (t >= half + 2 * s)
        return t - half - s;
    const double u = t - half;
    return u * u / (4 * s);
}
// signed blur amount in -1..1 from a raw map value (dark side of Focus = in front, like Optic Bokeh)
inline float MapToAmount(const Settings& s, float m)
{
    double v = std::isfinite(m) ? (double)m : 0.0;
    v = std::min(1.0, std::max(0.0, v));
    if (s.invert)
        v = 1.0 - v;
    const double d = v - s.focusPct / 100.0;
    const double half = s.rangePct / 200.0;
    double a = Shoulder(std::fabs(d), half, 0.5 * half) * s.depth;
    a = std::min(1.0, std::max(0.0, a));
    return (float)(d < 0 ? -a : a);
}

// ------------------------------------------------------- pixel conversion ---
template <typename Pix> struct Traits;
template <> struct Traits<PF_Pixel8>
{
    static inline float In(A_u_char v) { return v * (1.f / 255.f); }
    static inline A_u_char Out(float v)
    {
        v = v * 255.f + 0.5f;
        return (A_u_char)(!(v > 0.f) ? 0 : (v >= 255.f ? 255 : (int)v));
    }
};
template <> struct Traits<PF_Pixel16>
{
    static inline float In(A_u_short v) { return v * (1.f / PF_MAX_CHAN16); }
    static inline A_u_short Out(float v)
    {
        v = v * PF_MAX_CHAN16 + 0.5f;
        return (A_u_short)(!(v > 0.f) ? 0 : (v >= (float)PF_MAX_CHAN16 ? PF_MAX_CHAN16 : (int)v));
    }
};
template <> struct Traits<PF_PixelFloat>
{
    static inline float In(PF_FpShort v) { return v; }
    static inline PF_FpShort Out(float v) { return v; }
};

inline float SanitizeF(float v)
{
    const float kMax = 1.0e6f;
    return std::isfinite(v) ? std::min(kMax, std::max(-kMax, v)) : (v > 0 ? kMax : 0.f);
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

// R0 (layer coords) -> the engine's level 0: (q * premultiplied RGBA, q) per pixel, with the edge rule:
// inside the content: the input (transparent where the input world has no pixel); outside:
// Repeat = nearest content pixel, Spread = transparent, Clip = weight 0.
template <typename Pix>
void FillBase(const PF_EffectWorld* in, const PF_LRect& R0, const PF_LRect& content, int edge, float* base5, A_long yb, A_long ye)
{
    typedef Traits<Pix> T;
    const A_long w = R0.right - R0.left;
    const A_long ix0 = in ? in->origin_x : 0, iy0 = in ? in->origin_y : 0;
    const A_long ix1 = in ? in->origin_x + in->width : 0, iy1 = in ? in->origin_y + in->height : 0;
    for (A_long y = R0.top + yb; y < R0.top + ye; ++y)
        for (A_long x = R0.left; x < R0.right; ++x)
        {
            const size_t i = (size_t)(y - R0.top) * w + (x - R0.left);
            float* d = base5 + i * 5;
            d[0] = d[1] = d[2] = d[3] = 0.f;
            const bool inside = x >= content.left && x < content.right && y >= content.top && y < content.bottom;
            d[4] = (inside || edge != EDGE_CLIP) ? 1.f : 0.f;  // Clip to Layer: outside has no weight
            A_long sx = x, sy = y;
            if (!inside)
            {
                if (edge != EDGE_REPEAT)
                    continue;
                sx = std::min(content.right - 1, std::max(content.left, sx));
                sy = std::min(content.bottom - 1, std::max(content.top, sy));
            }
            if (!in || sx < ix0 || sx >= ix1 || sy < iy0 || sy >= iy1)
                continue;
            const Pix* s = RowPtr<Pix>(in, sy - iy0) + (sx - ix0);
            float a = std::min(1.f, std::max(0.f, SanitizeF(T::In(s->alpha))));
            d[0] = SanitizeF(T::In(s->red)) * a;
            d[1] = SanitizeF(T::In(s->green)) * a;
            d[2] = SanitizeF(T::In(s->blue)) * a;
            d[3] = a;
        }
}

// premultiplied float over O -> output world (straight); outside O or `clip` transparent
template <typename Pix>
void ToWorld(const float* src, const PF_LRect& O, PF_EffectWorld* out, const PF_LRect* clip, A_long yb, A_long ye)
{
    typedef Traits<Pix> T;
    const A_long sw = O.right - O.left;
    for (A_long oy = yb; oy < ye; ++oy)
    {
        Pix* d = RowPtrW<Pix>(out, oy);
        const A_long ly = oy + out->origin_y;
        for (A_long ox = 0; ox < out->width; ++ox, ++d)
        {
            const A_long lx = ox + out->origin_x;
            if (lx < O.left || lx >= O.right || ly < O.top || ly >= O.bottom ||
                (clip && (lx < clip->left || lx >= clip->right || ly < clip->top || ly >= clip->bottom)))
            {
                d->alpha = d->red = d->green = d->blue = T::Out(0.f);
                continue;
            }
            const float* s = src + ((size_t)(ly - O.top) * sw + (lx - O.left)) * 4;
            float a = s[3];
            if (!(a >= 1e-6f))
            {
                d->alpha = d->red = d->green = d->blue = T::Out(0.f);
                continue;
            }
            a = std::min(1.f, a);
            const float inv = 1.f / a;
            d->alpha = T::Out(a);
            d->red = T::Out(SanitizeF(s[0] * inv));
            d->green = T::Out(SanitizeF(s[1] * inv));
            d->blue = T::Out(SanitizeF(s[2] * inv));
        }
    }
}

template <typename Pix>
inline float LayerChannel(const Pix& p, int channel)
{
    typedef Traits<Pix> T;
    const float a = std::min(1.f, std::max(0.f, T::In(p.alpha)));
    switch (channel)
    {
    case CH_RED: return T::In(p.red) * a;
    case CH_GREEN: return T::In(p.green) * a;
    case CH_BLUE: return T::In(p.blue) * a;
    case CH_ALPHA: return T::In(p.alpha);
    default: return (0.2126f * T::In(p.red) + 0.7152f * T::In(p.green) + 0.0722f * T::In(p.blue)) * a;
    }
}

// bilinear sample of the map layer at effect-layer pixel (lx, ly), its bounds stretched onto the layer
template <typename Pix>
float SampleLayer(const PF_EffectWorld* m, const RenderPlan& plan, const LayerSampler& ls, A_long lx, A_long ly)
{
    double u = (lx - plan.layerBounds.left + 0.5) * ls.kx - 0.5;
    double v = (ly - plan.layerBounds.top + 0.5) * ls.ky - 0.5;
    u = std::min((double)ls.w - 1, std::max(0.0, u));
    v = std::min((double)ls.h - 1, std::max(0.0, v));
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
void FillAmount(const PF_EffectWorld* m, const RenderPlan& plan, const PF_LRect& r, float* dst, A_long yb = 0, A_long ye = -1)
{
    const A_long w = r.right - r.left;
    if (ye < 0)
        ye = r.bottom - r.top;
    for (A_long y = r.top + yb; y < r.top + ye; ++y)
        for (A_long x = r.left; x < r.right; ++x)
            dst[(size_t)(y - r.top) * w + (x - r.left)] = MapToAmount(plan.s, SampleLayer<Pix>(m, plan, plan.map, x, y));
}

template <typename Pix>
void FillTransparent(PF_EffectWorld* out)
{
    for (A_long y = 0; y < out->height; ++y)
        memset(RowPtrW<Pix>(out, y), 0, sizeof(Pix) * (size_t)out->width);
}

// rows [0, n) split over threads (pixel conversion; deterministic, each row written by one thread)
template <class Fn>
void ParallelRows(A_long n, int threads, const Fn& fn)
{
    const int T = std::max(1, std::min(threads, (int)(n / 16)));
    if (T <= 1)
    {
        fn((A_long)0, n);
        return;
    }
    std::vector<std::thread> pool;
    std::exception_ptr ex;
    std::mutex exm;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t] {
            try
            {
                fn((A_long)((long long)n * t / T), (A_long)((long long)n * (t + 1) / T));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(exm);
                if (!ex)
                    ex = std::current_exception();
            }
        });
    for (auto& th : pool)
        th.join();
    if (ex)
        std::rethrow_exception(ex);
}

// ------------------------------------------------------------- threads (MFR) ---
// Concurrent frames (multi-frame rendering) split the CPU: threads = hardware threads / active renders.
// The only mutable global of the plug-in (no per-instance data): MFR-safe.
std::atomic<int> g_active{0};
struct ActiveGuard
{
    ActiveGuard() { g_active.fetch_add(1); }
    ~ActiveGuard() { g_active.fetch_sub(1); }
};
int g_threadOverride = 0;  // tests only (0 = automatic)

// ---------------------------------------------------------------- GPU (v0.3) ---
// MAPBILINEARBLUR_GPU=0 (read once at GLOBAL_SETUP) forces the CPU. Frames smaller than kGpuMinPixels stay on
// the CPU (transfers and setup would dominate). Any GPU failure renders that frame on the CPU.
// v0.3.1: MAPBILINEARBLUR_GPU=2 forces the GPU for every frame (it was ignored in v0.3.0: only '0' was read).
// An eligible frame waits for the background device creation (bounded) instead of going to the CPU while it runs:
// in AE the render starts right after GLOBAL_SETUP and, under MFR, the CPU frames slowed the initialisation down
// further, so short renders never reached the GPU. MAPBILINEARBLUR_GPU_LOG=<path> appends one line per frame.
std::atomic<int> g_gpuMode{0};         // 0 auto, 1 off (env / tests), 2 on for every frame size (env / tests)
std::atomic<int> g_gpuLastStatus{-1};  // last GPU status (tests, About)
std::atomic<int> g_gpuFrames{0}, g_cpuFrames{0};
const long long kGpuMinPixels = 250000;
const uint32_t kGpuInitWaitAutoMs = 1500;    // auto: wait for the device at most this long after its creation began
const uint32_t kGpuInitWaitForcedMs = 30000;  // =2: (create_gpu's own steps give up after 10 s each)
const uint32_t kGpuSlotWaitForcedMs = 30000;  // =2: wait this long for a GPU slot under MFR (auto: the library's 2 s)

// ---- diagnostics log (off unless MAPBILINEARBLUR_GPU_LOG is set; read once at GLOBAL_SETUP)
std::atomic<bool> g_logOn{false};
plat::NativePath g_logPath;
std::mutex g_logMx;
std::chrono::steady_clock::time_point g_logT0;
std::atomic<bool> g_logInitNoted{false};
void LogLine(const char* line)
{
    std::lock_guard<std::mutex> lk(g_logMx);
    plat::AppendLine(g_logPath, line);
}
double LogMs(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(t - g_logT0).count(); }
const char* GpuReason(int s)
{
    switch (s)
    {
    case MBBGPU_OK: return "ok";
    case MBBGPU_NO_ADAPTER: return "no_adapter";
    case MBBGPU_DEVICE_LOST: return "device_lost";
    case MBBGPU_OUT_OF_VRAM: return "out_of_vram";
    case MBBGPU_BUSY: return "busy";
    case MBBGPU_TIMEOUT: return "timeout";
    case MBBGPU_ABORTED: return "aborted";
    case MBBGPU_BAD_ARGS: return "bad_args";
    case MBBGPU_UNSUPPORTED: return "unsupported_settings";
    case MBBGPU_DISABLED: return "disabled";
    case MBBGPU_NOT_READY: return "not_ready";
    default: return "internal";
    }
}
const char* ModeName(int m) { return m == 1 ? "off" : (m == 2 ? "forced" : "auto"); }

void GpuGlobalSetup()
{
    char buf[8] = {0};
#if defined(__APPLE__)
    // mac_port POLICY (mac_common/POLICY_macos.md): the GPU is OFF by default on macOS - an explicit gate on top of
    // MBB_GPU=OFF and the SHADER_F64 adapter check. Only MAPBILINEARBLUR_GPU=1 (auto, the Windows default) or =2
    // (forced) lets a GPU build try the GPU; unset, 0 or anything else = CPU. Windows keeps its default (#else).
    g_gpuMode = 1;
    if (plat::GetEnvShort("MAPBILINEARBLUR_GPU", buf, sizeof buf))
    {
        if (buf[0] == '1')
            g_gpuMode = 0;
        else if (buf[0] == '2')
            g_gpuMode = 2;
    }
#else
    if (plat::GetEnvShort("MAPBILINEARBLUR_GPU", buf, sizeof buf))
    {
        if (buf[0] == '0')
            g_gpuMode = 1;
        else if (buf[0] == '2')
            g_gpuMode = 2;
    }
#endif
    plat::NativePath lp;
    if (plat::GetEnvPath("MAPBILINEARBLUR_GPU_LOG", lp))
    {
        g_logPath = lp;
        g_logT0 = std::chrono::steady_clock::now();
        g_logOn = true;
        char line[256];
        std::snprintf(line, sizeof line, "# %s %s setup pid=%lu mode=%s", MBB_NAME, MBB_VERSION_STR, plat::ProcessId(),
                      ModeName(g_gpuMode));
        LogLine(line);
    }
    if (g_gpuMode == 2)
        mbbgpu_set_slot_wait_ms(kGpuSlotWaitForcedMs);
    if (g_gpuMode != 1)
        mbbgpu_init_async();
}

// one frame's record for the log
struct FrameRec
{
    bool on = false;
    std::chrono::steady_clock::time_point t0;
    const char* path = "CPU";
    const char* reason = "";
    int eligible = 0, status = -1;
    double waitInit = 0, waitSlot = 0, gpuMs = 0;
};
void LogFrame(const FrameRec& r, PF_InData* in_data, int w, int h, int bpc)
{
    const auto t1 = std::chrono::steady_clock::now();
    const long frame = (in_data && in_data->time_step > 0) ? (long)(in_data->current_time / in_data->time_step) : -1;
    const double ds = (in_data && in_data->downsample_x.den) ? (double)in_data->downsample_x.num / in_data->downsample_x.den : 1.0;
    char line[512];
    std::snprintf(line, sizeof line,
                  "t_ms=%.1f pid=%lu tid=%lu frame=%ld size=%dx%d bpc=%d ds=%.3g mode=%s eligible=%d path=%s reason=%s status=%d "
                  "wait_init_ms=%.1f wait_slot_ms=%.1f gpu_ms=%.1f render_ms=%.1f",
                  LogMs(r.t0), plat::ProcessId(), plat::ThreadId(), frame, w, h, bpc, ds,
                  ModeName(g_gpuMode), r.eligible, r.path, r.reason, r.status, r.waitInit, r.waitSlot, r.gpuMs,
                  std::chrono::duration<double, std::milli>(t1 - r.t0).count());
    LogLine(line);
    if (!g_logInitNoted && mbbgpu_init_ms() >= 0 && !g_logInitNoted.exchange(true))
    {
        std::snprintf(line, sizeof line, "# gpu init finished after %.0f ms (pid=%lu): %s", mbbgpu_init_ms(), plat::ProcessId(),
                      mbbgpu_info());
        LogLine(line);
    }
}
struct GpuAbortCtx
{
    PF_InData* in_data;
    PF_Err err;
};
int GpuAbort(void* u)
{
    GpuAbortCtx* c = reinterpret_cast<GpuAbortCtx*>(u);
    if (!c->err)
        c->err = PF_ABORT(c->in_data);
    return c->err != PF_Err_NONE ? 1 : 0;
}
int RenderThreads()
{
    if (g_threadOverride > 0)
        return g_threadOverride;
    const unsigned hw = std::thread::hardware_concurrency();
    const int n = hw > 0 ? (int)hw : 4;
    return std::max(1, n / std::max(1, g_active.load()));
}

PF_Err ReportError(PF_OutData* out_data, PF_Err err, const char* what)
{
    if (out_data && err != PF_Err_NONE && err != PF_Err_OUT_OF_MEMORY && err != PF_Interrupt_CANCEL)
    {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d: %s", MBB_NAME, MBB_MAJOR_VERSION,
                      MBB_MINOR_VERSION, obs::T(what));
    }
    return err;
}

// ------------------------------------------------------------- commands ---
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
    obs::Init(lang);
}

PF_Err About(PF_InData* in_data, PF_OutData* out_data)
{
    EnsureLanguage(in_data);
    std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d.%d (prototype, %s)\r%s\r%s", MBB_NAME,
                  MBB_MAJOR_VERSION, MBB_MINOR_VERSION, MBB_BUG_VERSION, __DATE__,
                  obs::T("Depth-map pyramid blur with sharpen (prototype)."),
                  obs::T("Interpolations from nearest to Catmull-Rom, hex mosaic. CPU."));
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    EnsureLanguage(in_data);
    if (in_data->appl_id != 'FXTC')
    {
        // Premiere Pro / Media Encoder also read MediaCore but would call the non-SmartFX path
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %s", MBB_NAME, obs::T("runs in After Effects only."));
        return PF_Err_INVALID_CALLBACK;
    }
    out_data->my_version = PF_VERSION(MBB_MAJOR_VERSION, MBB_MINOR_VERSION, MBB_BUG_VERSION, MBB_STAGE, MBB_BUILD_VERSION);
    out_data->out_flags = kOutFlags;
    out_data->out_flags2 = kOutFlags2;
    GpuGlobalSetup();
    return PF_Err_NONE;
}

PF_Err AddFloat(PF_InData* in_data, const char* name, double vmin, double vmax, double smin, double smax, double dflt,
                int prec, PF_ValueDisplayFlags disp, A_long id)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_FLOAT_SLIDER;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.fs_d.valid_min = (PF_FpShort)vmin;
    def.u.fs_d.slider_min = (PF_FpShort)smin;
    def.u.fs_d.valid_max = (PF_FpShort)vmax;
    def.u.fs_d.slider_max = (PF_FpShort)smax;
    def.u.fs_d.value = def.u.fs_d.dephault = (PF_FpShort)dflt;
    def.u.fs_d.precision = (PF_Precision)prec;
    def.u.fs_d.display_flags = disp;
    def.u.fs_d.curve_tolerance = AEFX_DEFAULT_CURVE_TOLERANCE;
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddPopup(PF_InData* in_data, const char* name, int choices, int dflt, const char* items, A_long id)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POPUP;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.pd.num_choices = (A_short)choices;
    def.u.pd.dephault = def.u.pd.value = (A_short)dflt;
    def.u.pd.u.namesptr = items;
    def.uu.id = id;
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;
    const PF_ValueDisplayFlags PCT = PF_ValueDisplayFlag_PERCENT, NONE = PF_ValueDisplayFlag_NONE;
    // v0.1 "Method" kept hidden so its disk ID stays reserved (prototype: old projects get the new defaults)
    {
        PF_ParamDef d;
        AEFX_CLR_STRUCT(d);
        d.param_type = PF_Param_POPUP;
        PF_STRNNCPY(d.PF_DEF_NAME, obs::T("Method"), sizeof(d.PF_DEF_NAME));
        d.u.pd.num_choices = 3;
        d.u.pd.dephault = d.u.pd.value = 3;
        d.u.pd.u.namesptr = obs::T("Plain Bilinear|Smooth|Protect Edges");
        d.ui_flags = PF_PUI_INVISIBLE;
        d.uu.id = D_METHOD;
        if ((err = PF_ADD_PARAM(in_data, -1, &d)))
            return err;
    }
    if ((err = AddPopup(in_data, obs::T("Interpolation"), 6, INTERP_BSPLINE,
                        obs::T("Nearest|Bilinear|Smoothstep|Smooth (B-Spline)|Catmull-Rom|Hex Mosaic"), D_INTERP)) ||
        (err = AddPopup(in_data, obs::T("Downsample"), 3, DOWN_AUTO, obs::T("Auto|Average (Crisp)|Smooth"), D_DOWN)) ||
        (err = AddPopup(in_data, obs::T("Level Blend"), 3, BLEND_SMOOTH, obs::T("Smooth|Steps|Dither"), D_BLEND)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Anti-Ringing"), "", true, 0, D_ANTIRING);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Protect Edges"), "", true, 0, D_PROTECT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(obs::T("Dither Seed"), 0, 10000, 0, 100, 0, D_DITHER_SEED);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Animate Dither"), "", false, 0, D_DITHER_ANIM);
    if ((err = AddFloat(in_data, obs::T("Blur Radius"), 0, 4000, 0, 300, 20, 1, NONE, D_RADIUS)))
        return err;

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPICX(obs::T("Blur Map"), 0, D_MAP_START);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER(obs::T("Map Layer"), PF_LayerDefault_NONE, D_MAP_LAYER);
    if ((err = AddPopup(in_data, obs::T("Map Channel"), 5, CH_LUMINANCE, obs::T("Luminance|Red|Green|Blue|Alpha"), D_MAP_CHANNEL)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Invert Map"), "", false, 0, D_MAP_INVERT);
    if ((err = AddFloat(in_data, obs::T("Focus"), 0, 100, 0, 100, 0, 1, PCT, D_FOCUS)) ||
        (err = AddFloat(in_data, obs::T("Focus Range"), 0, 100, 0, 50, 0, 1, PCT, D_FOCUS_RANGE)) ||
        (err = AddFloat(in_data, obs::T("Depth Strength"), 0, 20, 0, 4, 1, 2, NONE, D_DEPTH)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(D_MAP_END);

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPICX(obs::T("Sharpen"), 0, D_SHARP_START);
    if ((err = AddFloat(in_data, obs::T("Sharpen Amount"), 0, 500, 0, 100, 0, 1, NONE, D_SHARP_AMOUNT)) ||
        (err = AddPopup(in_data, obs::T("Sharpen Type"), 2, SHARP_AE, obs::T("AE Standard Style|Wide (Unsharp)"), D_SHARP_TYPE)) ||
        (err = AddFloat(in_data, obs::T("Sharpen Radius"), 0.5, 200, 0.5, 30, 3, 1, NONE, D_SHARP_RADIUS)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(D_SHARP_END);

    if ((err = AddPopup(in_data, obs::T("Layer Edges"), 3, EDGE_REPEAT, obs::T("Repeat Edge|Spread Outside|Clip to Layer"), D_EDGE)) ||
        (err = AddPopup(in_data, obs::T("View"), 2, VIEW_RESULT, obs::T("Result|Blur Amount"), D_VIEW)))
        return err;
    out_data->num_params = P_NUM;
    return err;
}

PF_Err CheckoutMap(PF_InData* in_data, PF_PreRenderExtra* extra, const PF_RenderRequest& outReq, const RenderPlan& plan,
                   double dsx, double dsy, const PF_LRect& need, LayerSampler& ls)
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
    ERR(extra->cb->checkout_layer(in_data->effect_ref, P_MAP_LAYER, CO_MAP_PROBE, &probe, in_data->current_time,
                                  in_data->time_step, in_data->time_scale, &pr));
    if (err)
        return err;
    const A_long lw = plan.layerBounds.right - plan.layerBounds.left, lh = plan.layerBounds.bottom - plan.layerBounds.top;
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
    const PF_LRect bounds = {0, 0, ls.w, ls.h};
    mr = Intersect(mr, bounds);
    if (RectEmpty(mr))
        mr = bounds;
    PF_RenderRequest req = outReq;
    req.rect = mr;
    req.channel_mask = PF_ChannelMask_ARGB;
    req.preserve_rgb_of_zero_alpha = false;
    PF_CheckoutResult res;
    AEFX_CLR_STRUCT(res);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, P_MAP_LAYER, CO_MAP, &req, in_data->current_time, in_data->time_step,
                                  in_data->time_scale, &res));
    if (!err)
        ls.has = !RectEmpty(res.result_rect);
    return err;
}

PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    (void)out_data;
    PF_Err err = PF_Err_NONE;
    RenderPlan* plan = new (std::nothrow) RenderPlan();
    if (!plan)
        return PF_Err_OUT_OF_MEMORY;
    extra->output->pre_render_data = plan;
    extra->output->delete_pre_render_data_func = DeletePlan;
    ERR(ReadSettings(in_data, plan->s));
    if (err)
        return err;
    const Settings& s = plan->s;

    const double dsx = Ratio(in_data->downsample_x), dsy = Ratio(in_data->downsample_y);
    const double ds = std::sqrt(dsx * dsy);  // isotropic in buffer px (non-square pixels: see README)
    plan->ds = ds;
    mbb::Settings& es = plan->es;
    es.interp = kInterpToEngine[std::min(6, std::max(1, s.interp))];
    es.downSmooth = s.down == DOWN_SMOOTH || (s.down == DOWN_AUTO && s.interp == INTERP_BSPLINE);
    es.blend = s.blend - 1;
    es.antiRing = s.antiRing;
    es.protect = s.protect;
    es.ditherSeed = (unsigned)s.ditherSeed;
    if (s.blend == BLEND_DITHER && s.ditherAnim)
    {
        // the layer's own frame number (as Optic Bokeh's grain): same pattern at the same layer time
        const A_long st = in_data->local_time_step != 0 ? in_data->local_time_step : in_data->time_step;
        es.ditherFrame = st != 0 ? (int)std::floor((double)in_data->current_time / std::fabs((double)st) + 1e-6) : 0;
    }
    es.radius = s.radius * ds;
    es.ds = ds;
    es.sharpType = s.sharpType == SHARP_UNSHARP ? mbb::SH_UNSHARP : mbb::SH_AE;
    if (es.sharpType == mbb::SH_AE)
        es.sharpAmount = s.sharpAmount / 100.0 * kAeSharpenScale * std::pow(std::min(1.0, ds), kSharpenResExp);
    else
        es.sharpAmount = s.sharpAmount / 100.0;
    es.sharpRadius = s.sharpRadius * ds;

    const PF_RenderRequest& outReq = extra->input->output_request;
    PF_CheckoutResult pr;
    AEFX_CLR_STRUCT(pr);
    {
        PF_RenderRequest probe = outReq;
        probe.rect.left = probe.rect.top = 0;
        probe.rect.right = probe.rect.bottom = 1;
        probe.channel_mask = PF_ChannelMask_ARGB;
        probe.preserve_rgb_of_zero_alpha = false;
        ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, CO_INPUT_PROBE, &probe, in_data->current_time,
                                      in_data->time_step, in_data->time_scale, &pr));
        if (err)
            return err;
    }
    PF_LRect layer = {0, 0, ScaledExtent(pr.ref_width, dsx), ScaledExtent(pr.ref_height, dsy)};
    if (RectEmpty(layer))
        layer = pr.max_result_rect;
    plan->layerBounds = layer;
    plan->content = RectEmpty(pr.max_result_rect) ? layer : Union(layer, pr.max_result_rect);

    const bool amountView = s.view == VIEW_AMOUNT;
    plan->passthrough = !amountView && s.radius <= 0 && s.sharpAmount <= 0;
    const bool spread = s.edge == EDGE_SPREAD && !plan->passthrough && !amountView;
    // Spread Outside: up to 4 sigma beyond the content, capped at a quarter of the layer's larger side
    // (memory: a 1000 px radius on UHD would otherwise make an 8k x 6k buffer; with the cap the blur
    // tail is cut where it is still ~15 % for radius 1000 on UHD - visible; ~0 for radius <= ~240 on UHD)
    const A_long capSp = std::max<A_long>(64, std::max(layer.right - layer.left, layer.bottom - layer.top) / 4);
    const A_long sp = spread ? std::min<A_long>((A_long)mbb::SpreadPx(es), capSp) : 0;
    const PF_LRect maxRect = Grow(plan->content, sp);
    const PF_LRect result = Intersect(outReq.rect, maxRect);
    plan->result = result;
    extra->output->max_result_rect = maxRect;
    extra->output->result_rect = result;
    extra->output->solid = false;
    extra->output->flags = 0;
    if (RectEmpty(result))
    {
        plan->emptyResult = true;
        return PF_Err_NONE;
    }
    // domain: the content (grown by the spread), aligned to the top pyramid level on the layer-origin grid
    const A_long A = mbb::Align(es);
    const PF_LRect c0 = Grow(plan->content, sp);
    plan->domain = PF_LRect{FloorTo(c0.left, A), FloorTo(c0.top, A), c0.right, c0.bottom};
    plan->R0 = plan->passthrough ? result : ToL(mbb::InputRect(es, ToR(plan->domain), ToR(result)));

    if (!amountView)
    {
        PF_RenderRequest req = outReq;
        req.rect = Intersect(plan->R0, plan->content);
        if (RectEmpty(req.rect))
            req.rect = PF_LRect{plan->content.left, plan->content.top, plan->content.left + 1, plan->content.top + 1};
        req.channel_mask = PF_ChannelMask_ARGB;
        req.preserve_rgb_of_zero_alpha = false;
        PF_CheckoutResult in_result;
        AEFX_CLR_STRUCT(in_result);
        ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, CO_INPUT, &req, in_data->current_time, in_data->time_step,
                                      in_data->time_scale, &in_result));
        if (err)
            return err;
    }
    plan->map.channel = s.mapChannel;
    if (!plan->passthrough)
        ERR(CheckoutMap(in_data, extra, outReq, *plan, dsx, dsy, amountView ? result : plan->R0, plan->map));
    return err;
}

template <typename Pix>
PF_Err RenderDepthImpl(PF_InData* in_data, PF_OutData* out_data, const RenderPlan& plan, PF_EffectWorld* input,
                       PF_EffectWorld* mapW, PF_EffectWorld* output, FrameRec& rec)
{
    typedef Traits<Pix> T;
    const PF_LRect outR = {output->origin_x, output->origin_y, output->origin_x + output->width, output->origin_y + output->height};
    if (plan.s.view == VIEW_AMOUNT)
    {
        rec.reason = "not_eligible:view_amount";
        // grey = how much blur (front of focus tinted blue), opaque
        std::vector<float> row((size_t)output->width);
        for (A_long oy = 0; oy < output->height; ++oy)
        {
            const PF_LRect r = {outR.left, outR.top + oy, outR.right, outR.top + oy + 1};
            if (mapW)
                FillAmount<Pix>(mapW, plan, r, row.data());
            else
                std::fill(row.begin(), row.end(), 1.f);
            Pix* d = RowPtrW<Pix>(output, oy);
            for (A_long ox = 0; ox < output->width; ++ox, ++d)
            {
                const float a = row[(size_t)ox], m = std::fabs(a);
                d->alpha = T::Out(1.f);
                d->red = T::Out(a < 0 ? 0.55f * m : m);
                d->green = T::Out(a < 0 ? 0.75f * m : m);
                d->blue = T::Out(m);
            }
        }
        return PF_Err_NONE;
    }
    if (plan.passthrough)
    {
        rec.reason = "not_eligible:passthrough";
        // radius 0 and no sharpen: the input pixels unchanged (bit exact)
        for (A_long oy = 0; oy < output->height; ++oy)
        {
            Pix* d = RowPtrW<Pix>(output, oy);
            const A_long ly = outR.top + oy;
            for (A_long ox = 0; ox < output->width; ++ox, ++d)
            {
                const A_long lx = outR.left + ox;
                if (input && input->data && lx >= input->origin_x && lx < input->origin_x + input->width &&
                    ly >= input->origin_y && ly < input->origin_y + input->height)
                    *d = RowPtr<Pix>(input, ly - input->origin_y)[lx - input->origin_x];
                else
                    std::memset(d, 0, sizeof(Pix));
            }
        }
        return PF_Err_NONE;
    }
    if (input && (!input->data || input->width <= 0 || input->height <= 0))
        input = nullptr;

    const PF_LRect O = Intersect(outR, plan.result);
    if (RectEmpty(O))
    {
        rec.reason = "not_eligible:empty";
        FillTransparent<Pix>(output);
        return PF_Err_NONE;
    }
    mbb::Job job;
    job.s = plan.es;
    job.D = ToR(plan.domain);
    job.O = ToR(O);
    job.R0 = mbb::InputRect(job.s, job.D, job.O);
    const PF_LRect R0 = ToL(job.R0);
    const size_t n0 = (size_t)job.R0.w() * job.R0.h();
    auto tp0 = std::chrono::steady_clock::now();
    auto plap = [&](const char* what) {  // tests: stage times (mbb::g_profile)
        if (!mbb::g_profile)
            return;
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  [plg] %-10s %7.1f ms\n", what, std::chrono::duration<double, std::milli>(t1 - tp0).count());
        tp0 = t1;
    };
    mbb::FVec base5, amount, out;  // v0.3: not zero-filled (fill / engine write every element)
    try
    {
        base5.resize(n0 * 5);
        if (mapW)
            amount.resize(n0);
        out.resize((size_t)job.O.w() * job.O.h() * 4);
    }
    catch (const std::bad_alloc&)
    {
        return PF_Err_OUT_OF_MEMORY;
    }
    plap("alloc");
    ActiveGuard g;
    const int threads = RenderThreads();
    ParallelRows(R0.bottom - R0.top, threads, [&](A_long yb, A_long ye) {
        FillBase<Pix>(input, R0, plan.content, plan.s.edge, base5.data(), yb, ye);
        if (mapW)
            FillAmount<Pix>(mapW, plan, R0, amount.data(), yb, ye);
    });
    if (mbb::g_profile)
        std::fprintf(stderr, "  [plg] R0 %dx%d  O %dx%d\n", job.R0.w(), job.R0.h(), job.O.w(), job.O.h());
    plap("fill");
    job.base5 = &base5;
    job.clip = plan.s.edge == EDGE_CLIP;
    job.amount = amount.empty() ? nullptr : amount.data();
    job.out = out.data();
    PF_Err abortErr = PF_Err_NONE;
    job.abort = [&]() {
        if (!abortErr)
            abortErr = PF_ABORT(in_data);
        return abortErr != PF_Err_NONE;
    };
    if ((abortErr = PF_ABORT(in_data)) != PF_Err_NONE)
        return abortErr;
    job.s.threads = threads;
    bool gpuDone = false;
    // Auto: the GPU only where it wins (measured, out\timings_v03.txt): a depth map with Protect Edges on (the
    // chains are the heavy part). Without a map the CPU is as fast or faster (transfers dominate on the GPU).
    const int mode = g_gpuMode;
    const char* notEligible = mode == 1 ? "not_eligible:gpu_off"
                              : mode == 2 ? nullptr
                              : (long long)job.O.w() * job.O.h() < kGpuMinPixels ? "not_eligible:small"
                              : !job.s.protect ? "not_eligible:no_protect_edges"
                              : !job.amount ? "not_eligible:no_map"
                                            : nullptr;
    rec.reason = notEligible ? notEligible : "";
    if (!notEligible)
    {
        rec.eligible = 1;
        mbbgpu_desc gd;
        if (mbb::DescribeGpu(job, gd))
        {
            GpuAbortCtx gac{in_data, PF_Err_NONE};
            // v0.3.1: wait (bounded) while the device is still being created instead of using the CPU meanwhile
            const auto tw = rec.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            int gs = mbbgpu_wait_init(mode == 2 ? kGpuInitWaitForcedMs : kGpuInitWaitAutoMs, GpuAbort, &gac);
            if (rec.on)
                rec.waitInit = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
            if (gs == MBBGPU_OK)
            {
                const auto tg = rec.on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                gs = mbbgpu_render(&gd, base5.data(), amount.empty() ? nullptr : amount.data(), out.data(), GpuAbort, &gac);
                if (rec.on)
                {
                    rec.waitSlot = mbbgpu_last_slot_wait_ms();
                    rec.gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg).count() - rec.waitSlot;
                }
            }
            g_gpuLastStatus = gs;
            rec.status = gs;
            if (gs == MBBGPU_OK)
                gpuDone = true;
            else if (gs == MBBGPU_ABORTED)
            {
                rec.reason = "aborted";
                return gac.err ? gac.err : PF_Interrupt_CANCEL;
            }
            // anything else: this frame on the CPU (below)
            rec.reason = GpuReason(gs);
        }
        else
            rec.reason = "not_describable";
        plap("gpu");
    }
    if (gpuDone)
    {
        ++g_gpuFrames;
        rec.path = "GPU";
    }
    else
    {
        ++g_cpuFrames;
        const mbb::Status st = mbb::Render(job);
        plap("engine");
        switch (st)
        {
        case mbb::ST_OK: break;
        case mbb::ST_ABORTED: return abortErr ? abortErr : PF_Interrupt_CANCEL;
        case mbb::ST_OOM: return PF_Err_OUT_OF_MEMORY;
        default: return ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render error. The frame was not rendered.");
        }
    }
    ParallelRows(output->height, threads, [&](A_long yb, A_long ye) {
        ToWorld<Pix>(out.data(), O, output, plan.s.edge == EDGE_SPREAD ? nullptr : &plan.content, yb, ye);
    });
    plap("toworld");
    return PF_Err_NONE;
}

template <typename Pix>
PF_Err RenderDepth(PF_InData* in_data, PF_OutData* out_data, const RenderPlan& plan, PF_EffectWorld* input,
                   PF_EffectWorld* mapW, PF_EffectWorld* output)
{
    FrameRec rec;
    if (!g_logOn)
        return RenderDepthImpl<Pix>(in_data, out_data, plan, input, mapW, output, rec);
    rec.on = true;
    rec.t0 = std::chrono::steady_clock::now();
    const PF_Err err = RenderDepthImpl<Pix>(in_data, out_data, plan, input, mapW, output, rec);
    if (err && rec.reason[0] == 0)
        rec.reason = "error";
    LogFrame(rec, in_data, output->width, output->height, sizeof(Pix) == 4 ? 8 : (sizeof(Pix) == 8 ? 16 : 32));
    return err;
}

// NON_PARAM_VARY only while Level Blend = Dither and Animate Dither is on (the params array is not valid here)
PF_Err QueryDynamicFlags(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef blend, anim;
    AEFX_CLR_STRUCT(blend);
    AEFX_CLR_STRUCT(anim);
    bool outB = false, outA = false;
    ERR(PF_CHECKOUT_PARAM(in_data, P_BLEND, in_data->current_time, in_data->time_step, in_data->time_scale, &blend));
    if (!err)
        outB = true;
    ERR(PF_CHECKOUT_PARAM(in_data, P_DITHER_ANIM, in_data->current_time, in_data->time_step, in_data->time_scale, &anim));
    if (!err)
        outA = true;
    if (!err && !(blend.u.pd.value == BLEND_DITHER && anim.u.bd.value))
        out_data->out_flags &= ~PF_OutFlag_NON_PARAM_VARY;
    else
        out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
    err = PF_Err_NONE;
    if (outB)
        ERR2(PF_CHECKIN_PARAM(in_data, &blend));
    if (outA)
        ERR2(PF_CHECKIN_PARAM(in_data, &anim));
    return err2;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    const RenderPlan* plan = reinterpret_cast<const RenderPlan*>(extra->input->pre_render_data);
    if (!plan)
        return PF_Err_BAD_CALLBACK_PARAM;
    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    PF_EffectWorld* mapW = nullptr;
    if (!plan->emptyResult && plan->s.view != VIEW_AMOUNT)
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CO_INPUT, &input));
    if (!err && !plan->emptyResult && plan->map.has)
    {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CO_MAP, &mapW));
        if (!err && mapW && (!mapW->data || mapW->width <= 0 || mapW->height <= 0))
            mapW = nullptr;
    }
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (err)
        return err;
    if (!output || !output->data || output->width <= 0 || output->height <= 0)
        return PF_Err_NONE;
    if (plan->emptyResult)
    {
        switch (extra->input->bitdepth)
        {
        case 32: FillTransparent<PF_PixelFloat>(output); break;
        case 16: FillTransparent<PF_Pixel16>(output); break;
        default: FillTransparent<PF_Pixel8>(output); break;
        }
        return PF_Err_NONE;
    }
    switch (extra->input->bitdepth)
    {
    case 32: err = RenderDepth<PF_PixelFloat>(in_data, out_data, *plan, input, mapW, output); break;
    case 16: err = RenderDepth<PF_Pixel16>(in_data, out_data, *plan, input, mapW, output); break;
    case 8: err = RenderDepth<PF_Pixel8>(in_data, out_data, *plan, input, mapW, output); break;
    default: err = PF_Err_BAD_CALLBACK_PARAM; break;
    }
    return err;
}

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                                     const char* inHostVersion)
{
    (void)inSPBasicSuitePtr;
    (void)inHostName;
    (void)inHostVersion;
    PF_Err result = PF_Err_INVALID_CALLBACK;  // the macro assigns `result`
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, MBB_NAME, MBB_MATCH_NAME, MBB_CATEGORY, AE_RESERVED_INFO,
                                     "EffectMain", MBB_SUPPORT_URL);
    return result;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra)
{
    (void)params;
    (void)output;
    PF_Err err = PF_Err_NONE;
    try
    {
        switch (cmd)
        {
        case PF_Cmd_ABOUT: err = About(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETDOWN: mbbgpu_shutdown(); break;
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data); break;
        case PF_Cmd_SMART_PRE_RENDER: err = SmartPreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)); break;
        case PF_Cmd_SMART_RENDER: err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)); break;
        case PF_Cmd_QUERY_DYNAMIC_FLAGS: err = QueryDynamicFlags(in_data, out_data); break;
        default: break;
        }
    }
    catch (const std::bad_alloc&)
    {
        return PF_Err_OUT_OF_MEMORY;
    }
    catch (PF_Err& e)
    {
        err = e;
    }
    catch (...)
    {
        err = ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render error. The frame was not rendered.");
    }
    return err;
}
