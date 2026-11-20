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
#ifdef OB_KANTAN
#include "kantan.h"             // v2.3 Kantan: the link curves
#endif
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
#ifndef OB_KANTAN
enum
{
    OB_INPUT = 0,
    OB_PRESET,
    OB_LENS_PICKER,     // v2.3 thumbnail picker (custom UI, no data)
    OB_QUALITY,
    OB_USE_GPU,         // v2.4
    OB_VIEW,
    OB_RADIUS,
    OB_MAP_GROUP_START,
    OB_MAP_LAYER,
    OB_MAP_CHANNEL,
    OB_MAP_INVERT,
    OB_FOCUS,
    OB_MAP_SCALE,
    OB_MAP_GAMMA,
    OB_FOCUS_RANGE,     // v2.3 map features (ob_mapx.h)
    OB_FRONT_BLUR,
    OB_BACK_BLUR,
    OB_MAP_BLUR,
    OB_FOCUS_POINT_ON,
    OB_FOCUS_POINT,
    OB_MAP_GROUP_END,
    OB_IRIS_GROUP_START,
    OB_BLADES,
    OB_CURVATURE,
    OB_CORNER,
    OB_ROTATION,
    OB_ANAMORPHIC,
    OB_CATEYE,
    OB_OBSTRUCTION,
    OB_IRIS_GROUP_END,
    OB_LENS_GROUP_START,
    OB_EDGE_RING,
    OB_RING_WIDTH,
    OB_SOFTNESS,
    OB_APOD,
    OB_CORE,
    OB_SA,
    OB_ONION,
    OB_ONION_COUNT,
    OB_CHROMA,
    OB_FRINGE_TINT,
    OB_LENS_GROUP_END,
    OB_FALLOFF_GROUP_START,
    OB_VIGNETTE,
    OB_GLOW,
    OB_GLOW_SIZE,
    OB_FALLOFF_GROUP_END,
    OB_HL_GROUP_START,
    OB_HL_BOOST,
    OB_HL_KNEE,
    OB_HL_CURVE,
    OB_HL_SEL,
    OB_HL_DETAIL,
    OB_HL_DETAIL_MIN,
    OB_HL_WHITE,
    OB_HL_DOTS,
    OB_HL_SAT,
    OB_HL_SOURCE,
    OB_HL_LAYER,
    OB_HL_LAYER_CHANNEL,
    OB_HL_ROLLOFF,
    OB_CLASSIC_GROUP_START,
    OB_HL_THRESHOLD,
    OB_HL_GAIN,
    OB_CLASSIC_GROUP_END,
    OB_HL_GROUP_END,
    OB_FILM_GROUP_START,
    OB_HAL_AMOUNT,
    OB_HAL_SIZE,
    OB_HAL_THRESHOLD,
    OB_HAL_RELATIVE,
    OB_HAL_SOFTNESS,
    OB_HAL_RIM,
    OB_HAL_TINT,
    OB_LA_STYLE,
    OB_STYLE_PICKER,    // v2.3
    OB_LA_AMOUNT,
    OB_LA_MODE,
    OB_LA_WIDTH,
    OB_LA_THRESHOLD,
    OB_LA_CURVE,
    OB_LA_HUE,
    OB_LA_TINT,
    OB_LA_SCALE,
    OB_LA_LINE,
    OB_LA_SHARP,
    OB_LA_BRIGHTCOL,
    OB_LA_BLUR,
    OB_LA_COMPLEMENT,
    OB_LA_FRINGE,
    OB_LA_DIR,
    OB_LA_MAPSOFT,
    OB_LA_PRESERVE,
    OB_FILM_GROUP_END,
    OB_GRAIN_GROUP_START,
    OB_GRAIN_STOCK,
    OB_STOCK_PICKER,    // v2.3
    OB_GRAIN_AMOUNT,
    OB_GRAIN_SIZE,
    OB_GRAIN_SOFT,
    OB_GRAIN_COLOR,
    OB_GRAIN_TONE,
    OB_GRAIN_ANIMATE,
    OB_GRAIN_HOLD,
    OB_GRAIN_SEED,
    OB_GRAIN_GROUP_END,
    OB_HQ_GROUP_START,
    OB_HQ_TEX_AMOUNT,
    OB_HQ_TEX_SCALE,
    OB_HQ_TEX_TYPE,
    OB_HQ_TEX_SEED,
    OB_HQ_SPECTRAL,
    OB_HQ_LATERAL_CA,
    OB_HQ_GROUP_END,
    OB_LINEAR,
    OB_INPUT_GAMMA,     // v2.2
    OB_EDGE_MODE,       // v2.2 (replaces the Repeat Edge Pixels checkbox in the UI)
    OB_REPEAT_EDGE,     // v2.1 checkbox, hidden: a v2.1 project with it on still renders Repeat Edge
    OB_RENDER_VERSION,  // v2.2, hidden
    OB_NUM_PARAMS
};

// Disk IDs: stable forever, never reuse or renumber (saved projects depend on them).
enum
{
    DISK_VIEW = 1,
    DISK_RADIUS,
    DISK_MAP_GROUP_START,
    DISK_MAP_LAYER,
    DISK_MAP_CHANNEL,
    DISK_MAP_INVERT,
    DISK_FOCUS,
    DISK_MAP_SCALE,
    DISK_MAP_GAMMA,
    DISK_MAP_GROUP_END,
    DISK_IRIS_GROUP_START,
    DISK_BLADES,
    DISK_CURVATURE,
    DISK_ROTATION,
    DISK_ANAMORPHIC,
    DISK_CATEYE,
    DISK_IRIS_GROUP_END,
    DISK_LENS_GROUP_START,
    DISK_EDGE_RING,
    DISK_RING_WIDTH,
    DISK_CHROMA,
    DISK_LENS_GROUP_END,
    DISK_HL_GROUP_START,
    DISK_HL_THRESHOLD,
    DISK_HL_GAIN,
    DISK_HL_GROUP_END,
    DISK_LINEAR,
    // v1.1
    DISK_PRESET = 28,
    DISK_CORNER,
    DISK_OBSTRUCTION,
    DISK_SA,
    DISK_SOFTNESS,
    DISK_APOD,
    DISK_CORE,
    DISK_ONION,
    DISK_ONION_COUNT,
    DISK_FALLOFF_GROUP_START,
    DISK_VIGNETTE,
    DISK_GLOW,
    DISK_GLOW_SIZE,
    DISK_FALLOFF_GROUP_END,
    DISK_UNUSED_42,  // v1.1 "Glints" group start (never released; not reused)
    DISK_HL_BOOST,   // 43 (was v1.1 Glint Gain)
    DISK_HL_DETAIL,  // 44 (was v1.1 Glint Size)
    DISK_UNUSED_45,  // v1.1 Glint Threshold (not reused)
    DISK_HL_SAT,     // 46 (was v1.1 Glint Saturation)
    DISK_UNUSED_47,  // v1.1 "Glints" group end (not reused)
    // v1.2
    DISK_HL_KNEE = 48,
    DISK_HL_CURVE,
    DISK_HL_SEL,
    DISK_HL_ROLLOFF,
    DISK_CLASSIC_GROUP_START,
    DISK_CLASSIC_GROUP_END,
    DISK_FRINGE_TINT,
    // v2.0
    DISK_HL_WHITE = 55,
    DISK_HL_DOTS,
    DISK_HL_DETAIL_MIN,
    DISK_HL_SOURCE,
    DISK_HL_LAYER,
    DISK_HL_LAYER_CHANNEL,
    DISK_FILM_GROUP_START,
    DISK_HAL_AMOUNT,
    DISK_HAL_SIZE,
    DISK_HAL_THRESHOLD,
    DISK_HAL_SOFTNESS,
    DISK_HAL_RIM,
    DISK_HAL_TINT,
    DISK_LA_AMOUNT,
    DISK_LA_WIDTH,
    DISK_LA_THRESHOLD,
    DISK_LA_CURVE,
    DISK_LA_HUE,
    DISK_LA_TINT,
    DISK_LA_MODE,
    DISK_FILM_GROUP_END,
    DISK_HQ_GROUP_START,
    DISK_HQ_TEX_AMOUNT,
    DISK_HQ_TEX_SCALE,
    DISK_HQ_TEX_TYPE,
    DISK_HQ_TEX_SEED,
    DISK_HQ_SPECTRAL,
    DISK_HQ_LATERAL_CA,
    DISK_HQ_GROUP_END,
    DISK_REPEAT_EDGE,
    // v2.1 (append only)
    DISK_QUALITY,       // 85
    DISK_LA_STYLE,      // 86
    DISK_HAL_RELATIVE,  // 87
    DISK_LA_SCALE,      // 88
    DISK_LA_LINE,       // 89
    DISK_LA_SHARP,      // 90
    DISK_LA_BRIGHTCOL,  // 91
    // v2.1b-d (append only)
    DISK_LA_BLUR,       // 92
    DISK_LA_COMPLEMENT, // 93
    DISK_LA_FRINGE,     // 94
    DISK_LA_DIR,        // 95
    DISK_LA_MAPSOFT,    // 96
    DISK_LA_PRESERVE,   // 97
    DISK_GRAIN_GROUP_START, // 98
    DISK_GRAIN_STOCK,   // 99
    DISK_GRAIN_AMOUNT,  // 100
    DISK_GRAIN_SIZE,    // 101
    DISK_GRAIN_SOFT,    // 102
    DISK_GRAIN_COLOR,   // 103
    DISK_GRAIN_TONE,    // 104
    DISK_GRAIN_ANIMATE, // 105
    DISK_GRAIN_HOLD,    // 106
    DISK_GRAIN_SEED,    // 107
    DISK_GRAIN_GROUP_END, // 108
    // v2.2 (append only)
    DISK_EDGE_MODE,     // 109
    DISK_INPUT_GAMMA,   // 110
    DISK_RENDER_VERSION, // 111
    // v2.3 (append only): the thumbnail pickers (PF_Param_NO_DATA: nothing is saved)
    DISK_LENS_PICKER,   // 112
    DISK_STYLE_PICKER,  // 113
    DISK_STOCK_PICKER,  // 114
    // v2.3 blur-map features (append only); defaults reproduce the earlier output exactly
    DISK_MAP_BLUR,      // 115
    DISK_FOCUS_RANGE,   // 116
    DISK_FRONT_BLUR,    // 117
    DISK_BACK_BLUR,     // 118
    DISK_FOCUS_POINT_ON, // 119
    DISK_FOCUS_POINT,   // 120
    // v2.4 (append only)
    DISK_USE_GPU        // 121
};
static_assert(DISK_LENS_PICKER == 112 && DISK_STOCK_PICKER == 114, "disk ID layout v2.3");
static_assert(DISK_MAP_BLUR == 115 && DISK_FOCUS_POINT == 120, "disk ID layout v2.3 map features");
static_assert(DISK_USE_GPU == 121, "disk ID layout v2.4");
static_assert(DISK_EDGE_MODE == 109 && DISK_RENDER_VERSION == 111, "disk ID layout v2.2");
static_assert(DISK_LINEAR == 27, "existing disk IDs must never change");
static_assert(DISK_HL_SAT == 46 && DISK_FRINGE_TINT == 54, "disk ID layout");
static_assert(DISK_HL_WHITE == 55 && DISK_REPEAT_EDGE == 84, "disk ID layout v2.0");
static_assert(DISK_QUALITY == 85 && DISK_LA_BRIGHTCOL == 91, "disk ID layout v2.1");
static_assert(DISK_LA_BLUR == 92 && DISK_GRAIN_GROUP_START == 98 && DISK_GRAIN_GROUP_END == 108, "disk ID layout v2.1d");
#else
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
#endif


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

#ifdef OB_KANTAN
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
#else
PF_Err ReadSettings(PF_InData* in_data, Settings& s)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef p[OB_NUM_PARAMS];
    bool out[OB_NUM_PARAMS] = {};
    const int idx[] = {OB_VIEW,         OB_RADIUS,      OB_MAP_CHANNEL,   OB_MAP_INVERT,  OB_FOCUS,
                       OB_MAP_SCALE,    OB_MAP_GAMMA,   OB_BLADES,        OB_CURVATURE,   OB_ROTATION,
                       OB_ANAMORPHIC,   OB_CATEYE,      OB_EDGE_RING,     OB_RING_WIDTH,  OB_CHROMA,
                       OB_HL_THRESHOLD, OB_HL_GAIN,     OB_LINEAR,        OB_CORNER,      OB_OBSTRUCTION,
                       OB_SA,           OB_SOFTNESS,    OB_APOD,          OB_CORE,        OB_ONION,
                       OB_ONION_COUNT,  OB_VIGNETTE,    OB_GLOW,          OB_GLOW_SIZE,   OB_FRINGE_TINT,
                       OB_HL_BOOST,     OB_HL_KNEE,     OB_HL_CURVE,      OB_HL_SEL,      OB_HL_DETAIL,
                       OB_HL_SAT,       OB_HL_ROLLOFF,  OB_HL_WHITE,      OB_HL_DOTS,     OB_HL_DETAIL_MIN,
                       OB_HL_SOURCE,    OB_HL_LAYER_CHANNEL, OB_HAL_AMOUNT, OB_HAL_SIZE,  OB_HAL_THRESHOLD,
                       OB_HAL_SOFTNESS, OB_HAL_RIM,     OB_HAL_TINT,      OB_LA_AMOUNT,   OB_LA_WIDTH,
                       OB_LA_THRESHOLD, OB_LA_CURVE,    OB_LA_HUE,        OB_LA_TINT,     OB_LA_MODE,
                       OB_HQ_TEX_AMOUNT, OB_HQ_TEX_SCALE, OB_HQ_TEX_TYPE, OB_HQ_TEX_SEED, OB_HQ_SPECTRAL,
                       OB_HQ_LATERAL_CA, OB_REPEAT_EDGE, OB_QUALITY,    OB_HAL_RELATIVE, OB_LA_SCALE,
                       OB_LA_LINE,      OB_LA_SHARP,    OB_LA_BRIGHTCOL, OB_LA_BLUR,   OB_LA_COMPLEMENT,
                       OB_LA_FRINGE,    OB_LA_DIR,      OB_LA_MAPSOFT,    OB_LA_PRESERVE, OB_GRAIN_AMOUNT,
                       OB_GRAIN_SIZE,   OB_GRAIN_SOFT,  OB_GRAIN_COLOR,   OB_GRAIN_TONE, OB_GRAIN_ANIMATE,
                       OB_GRAIN_HOLD,   OB_GRAIN_SEED,  OB_EDGE_MODE,     OB_INPUT_GAMMA, OB_RENDER_VERSION,
                       OB_MAP_BLUR,     OB_FOCUS_RANGE, OB_FRONT_BLUR,    OB_BACK_BLUR,   OB_FOCUS_POINT_ON,
                       OB_FOCUS_POINT,  OB_USE_GPU};
    for (int i : idx)
    {
        AEFX_CLR_STRUCT(p[i]);
        ERR(PF_CHECKOUT_PARAM(in_data, i, in_data->current_time, in_data->time_step, in_data->time_scale, &p[i]));
        if (!err)
            out[i] = true;
    }
    if (!err)
    {
        s.view = p[OB_VIEW].u.pd.value;
        s.radius = p[OB_RADIUS].u.fs_d.value;
        s.mapChannel = p[OB_MAP_CHANNEL].u.pd.value;
        s.invertMap = p[OB_MAP_INVERT].u.bd.value != 0;
        s.focusPct = p[OB_FOCUS].u.fs_d.value;
        s.mapScale = p[OB_MAP_SCALE].u.fs_d.value;
        s.mapGamma = p[OB_MAP_GAMMA].u.fs_d.value;
        s.blades = p[OB_BLADES].u.sd.value;
        s.curvaturePct = p[OB_CURVATURE].u.fs_d.value;
        s.rotationDeg = (double)p[OB_ROTATION].u.ad.value / 65536.0;
        s.anamorphic = p[OB_ANAMORPHIC].u.fs_d.value;
        s.catEyePct = p[OB_CATEYE].u.fs_d.value;
        s.edgeRingPct = p[OB_EDGE_RING].u.fs_d.value;
        s.ringWidthPct = p[OB_RING_WIDTH].u.fs_d.value;
        s.chromaPct = p[OB_CHROMA].u.fs_d.value;
        s.hlThreshold = p[OB_HL_THRESHOLD].u.fs_d.value;
        s.hlGain = p[OB_HL_GAIN].u.fs_d.value;
        s.linear = p[OB_LINEAR].u.bd.value != 0;
        s.cornerPct = p[OB_CORNER].u.fs_d.value;
        s.obstructionPct = p[OB_OBSTRUCTION].u.fs_d.value;
        s.saPct = p[OB_SA].u.fs_d.value;
        s.softnessPct = p[OB_SOFTNESS].u.fs_d.value;
        s.apodPct = p[OB_APOD].u.fs_d.value;
        s.corePct = p[OB_CORE].u.fs_d.value;
        s.onionPct = p[OB_ONION].u.fs_d.value;
        s.onionCount = p[OB_ONION_COUNT].u.fs_d.value;
        s.vignettePct = p[OB_VIGNETTE].u.fs_d.value;
        s.glowPct = p[OB_GLOW].u.fs_d.value;
        s.glowSizePct = p[OB_GLOW_SIZE].u.fs_d.value;
        s.fringeTintPct = p[OB_FRINGE_TINT].u.fs_d.value;
        s.hlBoost = p[OB_HL_BOOST].u.fs_d.value;
        s.hlKnee = p[OB_HL_KNEE].u.fs_d.value;
        s.hlCurve = p[OB_HL_CURVE].u.fs_d.value;
        s.hlSelPct = p[OB_HL_SEL].u.fs_d.value;
        s.hlDetail = p[OB_HL_DETAIL].u.fs_d.value;
        s.hlSatPct = p[OB_HL_SAT].u.fs_d.value;
        s.hlRolloffPct = p[OB_HL_ROLLOFF].u.fs_d.value;
        s.hlWhitePct = p[OB_HL_WHITE].u.fs_d.value;
        s.hlDotsPct = p[OB_HL_DOTS].u.fs_d.value;
        s.hlDetailMin = p[OB_HL_DETAIL_MIN].u.fs_d.value;
        s.hlSource = p[OB_HL_SOURCE].u.pd.value;
        s.hlLayerChannel = p[OB_HL_LAYER_CHANNEL].u.pd.value;
        s.halPct = p[OB_HAL_AMOUNT].u.fs_d.value;
        s.halSizePct = p[OB_HAL_SIZE].u.fs_d.value;
        s.halThreshold = p[OB_HAL_THRESHOLD].u.fs_d.value;
        s.halSoftPct = p[OB_HAL_SOFTNESS].u.fs_d.value;
        s.halRimPct = p[OB_HAL_RIM].u.fs_d.value;
        s.halTint[0] = SrgbToLinear(p[OB_HAL_TINT].u.cd.value.red);
        s.halTint[1] = SrgbToLinear(p[OB_HAL_TINT].u.cd.value.green);
        s.halTint[2] = SrgbToLinear(p[OB_HAL_TINT].u.cd.value.blue);
        s.laPct = p[OB_LA_AMOUNT].u.fs_d.value;
        s.laWidthPct = p[OB_LA_WIDTH].u.fs_d.value;
        s.laThreshold = p[OB_LA_THRESHOLD].u.fs_d.value;
        s.laCurve = p[OB_LA_CURVE].u.fs_d.value;
        s.laHuePct = p[OB_LA_HUE].u.fs_d.value;
        s.laTint[0] = SrgbToLinear(p[OB_LA_TINT].u.cd.value.red);
        s.laTint[1] = SrgbToLinear(p[OB_LA_TINT].u.cd.value.green);
        s.laTint[2] = SrgbToLinear(p[OB_LA_TINT].u.cd.value.blue);
        s.laMode = p[OB_LA_MODE].u.pd.value;
        s.hqTexPct = p[OB_HQ_TEX_AMOUNT].u.fs_d.value;
        s.hqTexScale = p[OB_HQ_TEX_SCALE].u.fs_d.value;
        s.hqTexType = p[OB_HQ_TEX_TYPE].u.pd.value;
        s.hqTexSeed = p[OB_HQ_TEX_SEED].u.sd.value;
        s.hqSpectral = p[OB_HQ_SPECTRAL].u.sd.value;
        s.hqLateralPct = p[OB_HQ_LATERAL_CA].u.fs_d.value;
        s.repeatEdge = p[OB_REPEAT_EDGE].u.bd.value != 0;
        s.quality = p[OB_QUALITY].u.pd.value;
        s.halRelative = p[OB_HAL_RELATIVE].u.fs_d.value;
        s.laScalePct = p[OB_LA_SCALE].u.fs_d.value;
        s.laLinePct = p[OB_LA_LINE].u.fs_d.value;
        s.laSharpPct = p[OB_LA_SHARP].u.fs_d.value;
        s.laBrightColPct = p[OB_LA_BRIGHTCOL].u.fs_d.value;
        s.laBlurPct = p[OB_LA_BLUR].u.fs_d.value;
        s.laComplementPct = p[OB_LA_COMPLEMENT].u.fs_d.value;
        s.laFringe = p[OB_LA_FRINGE].u.pd.value;
        s.laDir = p[OB_LA_DIR].u.pd.value;
        s.laMapSoftPct = p[OB_LA_MAPSOFT].u.fs_d.value;
        s.laPreservePct = p[OB_LA_PRESERVE].u.fs_d.value;
        s.grainPct = p[OB_GRAIN_AMOUNT].u.fs_d.value;
        s.grainSize = p[OB_GRAIN_SIZE].u.fs_d.value;
        s.grainSoftPct = p[OB_GRAIN_SOFT].u.fs_d.value;
        s.grainColorPct = p[OB_GRAIN_COLOR].u.fs_d.value;
        s.grainTonePct = p[OB_GRAIN_TONE].u.fs_d.value;
        s.grainAnimate = p[OB_GRAIN_ANIMATE].u.bd.value != 0;
        s.grainHold = p[OB_GRAIN_HOLD].u.sd.value;
        s.grainSeed = p[OB_GRAIN_SEED].u.sd.value;
        s.edgeMode = p[OB_EDGE_MODE].u.pd.value;
        s.inputGamma = p[OB_INPUT_GAMMA].u.pd.value;
        s.renderVersion = std::max(1, std::min(kRenderVersionCurrent, (int)p[OB_RENDER_VERSION].u.sd.value));
        s.edge = EffectiveEdge(s.edgeMode, s.repeatEdge);
        s.mapBlurPx = p[OB_MAP_BLUR].u.fs_d.value;
        s.focusRangePct = p[OB_FOCUS_RANGE].u.fs_d.value;
        s.frontPct = p[OB_FRONT_BLUR].u.fs_d.value;
        s.backPct = p[OB_BACK_BLUR].u.fs_d.value;
        s.focusPointOn = p[OB_FOCUS_POINT_ON].u.bd.value != 0;
        s.focusPointX = (double)p[OB_FOCUS_POINT].u.td.x_value / 65536.0;
        s.focusPointY = (double)p[OB_FOCUS_POINT].u.td.y_value / 65536.0;
        s.useGpu = p[OB_USE_GPU].u.pd.value;
        // Work in Linear Light with an already-linear input: no decode / encode
        if (InputIsLinear(s.inputGamma))
            s.linear = false;
    }
    for (int i : idx)
    {
        if (out[i])
            ERR2(PF_CHECKIN_PARAM(in_data, &p[i]));
    }
    return err;
}
#endif

// "Blur Amount" view: the engine's own per-pixel amount.
inline float EngineBlurAmount(const lenscore::Params& p, const float* mapValue)
{
    return mapValue ? lenscore::blurAmount(*mapValue, p) : 1.f;
}

#ifndef OB_KANTAN
// Lens Preset popup -> UI values. Popup value 1 = "Custom" (no change), 2.. = kLensPresets[v-2].
// Only Iris / Lens Character / Falloff & Glow / highlight params are written (never Radius, map,
// View, Highlight Rolloff, the highlight layer, film effects, the classic v1 gain or Linear Light).
bool ApplyPresetToParams(int popupValue, PF_ParamDef* params[])
{
    lenscore::Params q;
    if (!lenscore::applyLensPreset(popupValue - 2, q))
        return false;
    auto setF = [&](int i, double v) {
        params[i]->u.fs_d.value = v;
        params[i]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    };
    params[OB_BLADES]->u.sd.value = q.blades;
    params[OB_BLADES]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    params[OB_ROTATION]->u.ad.value = (PF_Fixed)std::lround(q.rotationDeg * 65536.0);
    params[OB_ROTATION]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    setF(OB_CURVATURE, q.bladeCurvature * 100.0);
    setF(OB_CORNER, q.cornerRounding * 100.0);
    setF(OB_ANAMORPHIC, q.anamorphic);
    setF(OB_CATEYE, q.catEye * 100.0);
    setF(OB_OBSTRUCTION, q.obstruction * 100.0);
    setF(OB_EDGE_RING, q.edgeRing * 100.0);
    setF(OB_RING_WIDTH, q.ringWidth * 100.0);
    setF(OB_SOFTNESS, q.edgeSoftness * 100.0);
    setF(OB_APOD, q.apodization * 100.0);
    setF(OB_CORE, q.coreWeight * 100.0);
    setF(OB_SA, q.sphericalAberration * 100.0);
    setF(OB_ONION, q.onionRings * 100.0);
    setF(OB_ONION_COUNT, q.onionCount);
    setF(OB_CHROMA, q.chromaFringe * 100.0);
    setF(OB_VIGNETTE, q.vignetting * 100.0);
    setF(OB_GLOW, q.glow * 100.0);
    setF(OB_GLOW_SIZE, q.glowSize * 100.0);
    setF(OB_FRINGE_TINT, q.fringeTint * 100.0);
    setF(OB_HL_BOOST, q.highlightBoost);
    setF(OB_HL_KNEE, q.highlightKnee);
    setF(OB_HL_CURVE, q.highlightCurve);
    setF(OB_HL_SEL, q.highlightSelectivity * 100.0);
    setF(OB_HL_DETAIL, q.detailSize);
    setF(OB_HL_DETAIL_MIN, q.detailMin);
    setF(OB_HL_WHITE, q.whiteProtection * 100.0);
    setF(OB_HL_DOTS, q.dotsOnly * 100.0);
    setF(OB_HL_SAT, q.highlightSaturation * 100.0);
    return true;
}

// Aberration Style popup -> LA shape/colour params (never the amount). Value 1 = "Custom".
bool ApplyAberrationStyleToParams(int popupValue, PF_ParamDef* params[])
{
    lenscore::Params q;
    if (!lenscore::applyAberrationStyle(popupValue - 2, q))
        return false;
    auto setF = [&](int i, double v) {
        params[i]->u.fs_d.value = v;
        params[i]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    };
    auto enc = [](float v) {
        return (A_u_char)std::lround(255.0 * std::pow(std::min(1.0f, std::max(0.0f, v)), 1.0 / kDisplayGamma));
    };
    params[OB_LA_MODE]->u.pd.value = (A_short)(q.laMode == 1   ? LA_SIDE_BRIGHT
                                               : q.laMode == 2 ? LA_SIDE_BOTH
                                               : q.laMode == 3 ? LA_SIDE_RIM
                                               : q.laMode == 4 ? LA_SIDE_DRIFT : LA_SIDE_DARK);
    params[OB_LA_MODE]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    setF(OB_LA_WIDTH, q.laWidth * 100.0);
    setF(OB_LA_THRESHOLD, q.laThreshold);
    setF(OB_LA_CURVE, q.laCurve);
    setF(OB_LA_HUE, q.laHueShift * 100.0);
    params[OB_LA_TINT]->u.cd.value.red = enc(q.laTint[0]);
    params[OB_LA_TINT]->u.cd.value.green = enc(q.laTint[1]);
    params[OB_LA_TINT]->u.cd.value.blue = enc(q.laTint[2]);
    params[OB_LA_TINT]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    setF(OB_LA_SCALE, q.laContrastScale * 100.0);
    setF(OB_LA_LINE, q.laLineProtect * 100.0);
    setF(OB_LA_SHARP, q.laSharpness * 100.0);
    setF(OB_LA_BRIGHTCOL, q.laBrightColor * 100.0);
    setF(OB_LA_BLUR, q.laBlur * 100.0);
    setF(OB_LA_COMPLEMENT, q.laComplement * 100.0);
    setF(OB_LA_MAPSOFT, q.laMapSoftness * 100.0);
    setF(OB_LA_PRESERVE, q.laPreserve * 100.0);
    int fringe = 1;  // the Fringe Color entry whose channel shifts the style uses
    for (int i = 0; i < lenscore::kFringeColorCount; ++i)
    {
        const float* s = lenscore::kFringeColors[i].shift;
        if (s[0] == q.laShift[0] && s[1] == q.laShift[1] && s[2] == q.laShift[2])
            fringe = i + 1;
    }
    params[OB_LA_FRINGE]->u.pd.value = (A_short)fringe;
    params[OB_LA_FRINGE]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    params[OB_LA_DIR]->u.pd.value = (A_short)(q.laShiftDir == 1 ? LA_DIR_ADVANCE : LA_DIR_RETREAT);
    params[OB_LA_DIR]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    return true;
}

// Grain Stock popup -> grain params (amount included). Value 1 = "Custom".
bool ApplyGrainStockToParams(int popupValue, PF_ParamDef* params[])
{
    lenscore::Params q;
    if (!lenscore::applyGrainStock(popupValue - 2, q))
        return false;
    auto setF = [&](int i, double v) {
        params[i]->u.fs_d.value = v;
        params[i]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    };
    setF(OB_GRAIN_AMOUNT, q.grain * 100.0);
    setF(OB_GRAIN_SIZE, q.grainSize);
    setF(OB_GRAIN_SOFT, q.grainSoftness * 100.0);
    setF(OB_GRAIN_COLOR, q.grainColor * 100.0);
    setF(OB_GRAIN_TONE, q.grainTone * 100.0);
    return true;
}

// v2.2 (A11): params written by the supervised popups. Editing one of them by hand resets its popup to
// "Custom", so the popup never claims a preset the controls no longer match (and choosing the same
// preset again re-applies it). Indices for USER_CHANGED_PARAM; disk IDs for PF_ParamFlag_SUPERVISE.
const int kPresetTargets[] = {OB_BLADES, OB_ROTATION, OB_CURVATURE, OB_CORNER, OB_ANAMORPHIC, OB_CATEYE, OB_OBSTRUCTION,
                              OB_EDGE_RING, OB_RING_WIDTH, OB_SOFTNESS, OB_APOD, OB_CORE, OB_SA, OB_ONION, OB_ONION_COUNT,
                              OB_CHROMA, OB_VIGNETTE, OB_GLOW, OB_GLOW_SIZE, OB_FRINGE_TINT, OB_HL_BOOST, OB_HL_KNEE,
                              OB_HL_CURVE, OB_HL_SEL, OB_HL_DETAIL, OB_HL_DETAIL_MIN, OB_HL_WHITE, OB_HL_DOTS, OB_HL_SAT};
const int kStyleTargets[] = {OB_LA_MODE, OB_LA_WIDTH, OB_LA_THRESHOLD, OB_LA_CURVE, OB_LA_HUE, OB_LA_TINT, OB_LA_SCALE,
                             OB_LA_LINE, OB_LA_SHARP, OB_LA_BRIGHTCOL, OB_LA_BLUR, OB_LA_COMPLEMENT, OB_LA_MAPSOFT,
                             OB_LA_PRESERVE, OB_LA_FRINGE, OB_LA_DIR};
const int kStockTargets[] = {OB_GRAIN_AMOUNT, OB_GRAIN_SIZE, OB_GRAIN_SOFT, OB_GRAIN_COLOR, OB_GRAIN_TONE};
const A_long kSupervisedDisk[] = {
    DISK_BLADES, DISK_ROTATION, DISK_CURVATURE, DISK_CORNER, DISK_ANAMORPHIC, DISK_CATEYE, DISK_OBSTRUCTION,
    DISK_EDGE_RING, DISK_RING_WIDTH, DISK_SOFTNESS, DISK_APOD, DISK_CORE, DISK_SA, DISK_ONION, DISK_ONION_COUNT,
    DISK_CHROMA, DISK_VIGNETTE, DISK_GLOW, DISK_GLOW_SIZE, DISK_FRINGE_TINT, DISK_HL_BOOST, DISK_HL_KNEE,
    DISK_HL_CURVE, DISK_HL_SEL, DISK_HL_DETAIL, DISK_HL_DETAIL_MIN, DISK_HL_WHITE, DISK_HL_DOTS, DISK_HL_SAT,
    DISK_LA_MODE, DISK_LA_WIDTH, DISK_LA_THRESHOLD, DISK_LA_CURVE, DISK_LA_HUE, DISK_LA_TINT, DISK_LA_SCALE,
    DISK_LA_LINE, DISK_LA_SHARP, DISK_LA_BRIGHTCOL, DISK_LA_BLUR, DISK_LA_COMPLEMENT, DISK_LA_MAPSOFT,
    DISK_LA_PRESERVE, DISK_LA_FRINGE, DISK_LA_DIR,
    DISK_GRAIN_AMOUNT, DISK_GRAIN_SIZE, DISK_GRAIN_SOFT, DISK_GRAIN_COLOR, DISK_GRAIN_TONE};
template <size_t N>
inline bool Contains(const int (&a)[N], int v)
{
    return std::find(a, a + N, v) != a + N;
}
inline PF_ParamFlags SupFlag(A_long diskId)
{
    const size_t n = sizeof(kSupervisedDisk) / sizeof(kSupervisedDisk[0]);
    return std::find(kSupervisedDisk, kSupervisedDisk + n, diskId) != kSupervisedDisk + n ? PF_ParamFlag_SUPERVISE : 0;
}
#else
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
#endif

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
#ifdef OB_KANTAN
    // Kantan: grain always animates (no Animate checkbox); Grain is 0..200 % of the stock's amount
    const int animIdx = OB_GRAIN_AMOUNT;
#else
    const int animIdx = OB_GRAIN_ANIMATE;
#endif
    ERR(PF_CHECKOUT_PARAM(in_data, animIdx, in_data->current_time, in_data->time_step, in_data->time_scale, &animate));
    if (!err)
        outB = true;
    if (!err)
    {
        const bool amountOn = amount.u.fs_d.value > 0 || ParamMayVary(in_data, OB_GRAIN_AMOUNT);
#ifdef OB_KANTAN
        const bool animOn = amountOn;
#else
        const bool animOn = amountOn && (animate.u.bd.value || ParamMayVary(in_data, OB_GRAIN_ANIMATE));
#endif
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

#ifndef OB_KANTAN
PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_ParamDef def;
    PF_Err err = PF_Err_NONE;
    const PF_ValueDisplayFlags PCT = PF_ValueDisplayFlag_PERCENT, NONE = PF_ValueDisplayFlag_NONE;
    const lenscore::Params def0;  // engine defaults (film effects etc.)

    // New instances start as the first preset (the flagship), so the effect looks right
    // before any preset is chosen. Defaults are taken from the shared preset table.
    lenscore::Params dp;
    lenscore::applyLensPreset(0, dp);

    // Popup names built once from the shared preset table (immutable after first use).
    static const std::string presetNames = [] {
        std::string s = obs::T("Custom");
        for (int i = 0; i < lenscore::kLensPresetCount; ++i)
        {
            s += "|";
            s += obs::T(lenscore::kLensPresets[i].ui);
        }
        return s;
    }();
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POPUP;
    PF_STRNNCPY(def.PF_DEF_NAME, obs::T("Lens Preset"), sizeof(def.PF_DEF_NAME));
    def.u.pd.num_choices = (A_short)(lenscore::kLensPresetCount + 1);
    def.u.pd.dephault = 2;  // first preset for new instances
    def.u.pd.value = 1;     // "Custom" for projects saved before this param existed
    def.u.pd.u.namesptr = presetNames.c_str();
    def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
    def.uu.id = DISK_PRESET;
    if ((err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE)
        return err;
    if ((err = AddPicker(in_data, obs::T("Preset Thumbnails"), obthumb::LENS, DISK_LENS_PICKER, false)))
        return err;

    // Quality (v2.1, replaces the v2.0 alpha/beta editions). Old projects get Standard (= v1.2 engine).
    // supervised: the Best group is enabled / greyed immediately (USER_CHANGED_PARAM), not only on
    // the next UPDATE_PARAMS_UI
    if ((err = AddPopup(in_data, obs::T("Quality"), 3, QUALITY_STANDARD, obs::T("Fast (Alpha)|Standard|Best (Beta)"), DISK_QUALITY, 0,
                        PF_ParamFlag_SUPERVISE)))
        return err;
    // v2.4: Auto = the GPU blur for Fast / Standard when worth it (CPU otherwise, and on any GPU error);
    // Off = CPU only. Part of the render (AE's cache sees it). Old projects get Auto (the new default).
    if ((err = AddPopup(in_data, obs::T("Use GPU"), 2, obgpuw::USE_AUTO, obs::T("Auto|Off (CPU only)"), DISK_USE_GPU)))
        return err;

    // View: the two highlight views are the tuning tools for "what becomes a ball".
    if ((err = AddPopup(in_data, obs::T("View"), 4, VIEW_RESULT, obs::T("Result|Blur Amount|Highlight Source|Highlights Only"), DISK_VIEW)))
        return err;

    PF_ADD_FLOAT_SLIDERX(obs::T("Radius"), 0, 500, 0, 100, 20, 1, PF_ValueDisplayFlag_NONE, 0, DISK_RADIUS);

    PF_ADD_TOPICX(obs::T("Blur Map"), 0, DISK_MAP_GROUP_START);
    AEFX_CLR_STRUCT(def);
    // Not "Blur Map": the group is already called that, and duplicate names make
    // effect("Optic Bokeh")("Blur Map") in expressions/scripts resolve to the group.
    PF_ADD_LAYER(obs::T("Map Layer"), PF_LayerDefault_NONE, DISK_MAP_LAYER);
    PF_ADD_POPUPX(obs::T("Map Channel"), 5, CH_LUMINANCE, obs::T("Luminance|Red|Green|Blue|Alpha"), 0, DISK_MAP_CHANNEL);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Invert Map"), "", false, 0, DISK_MAP_INVERT);
    PF_ADD_FLOAT_SLIDERX(obs::T("Focus"), 0, 100, 0, 100, 0, 1, PF_ValueDisplayFlag_PERCENT, 0, DISK_FOCUS);
    PF_ADD_FLOAT_SLIDERX(obs::T("Map Scale"), 0, 20, 0, 4, 1, 2, PF_ValueDisplayFlag_NONE, 0, DISK_MAP_SCALE);
    PF_ADD_FLOAT_SLIDERX(obs::T("Map Gamma"), 0.1, 10, 0.2, 5, 1, 2, PF_ValueDisplayFlag_NONE, 0, DISK_MAP_GAMMA);
    // v2.3 map features (ob_mapx.h). Defaults = the earlier output exactly (old projects get them too).
    if ((err = AddMapFeature(in_data, MF_RANGE, DISK_FOCUS_RANGE)) || (err = AddMapFeature(in_data, MF_FRONT, DISK_FRONT_BLUR)) ||
        (err = AddMapFeature(in_data, MF_BACK, DISK_BACK_BLUR)) || (err = AddMapFeature(in_data, MF_BLUR, DISK_MAP_BLUR)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Focus at Point"), "", false, 0, DISK_FOCUS_POINT_ON);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT(obs::T("Focus Point"), 50, 50, 0, DISK_FOCUS_POINT);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_MAP_GROUP_END);

    // Existing (v1) params: changing a default only affects newly applied effects; saved projects
    // always carry their own values.
    PF_ADD_TOPICX(obs::T("Iris"), 0, DISK_IRIS_GROUP_START);
    AEFX_CLR_STRUCT(def);
    // 0 = circle; 1 and 2 are also treated as circle by the engine.
    def.flags = SupFlag(DISK_BLADES);
    PF_ADD_SLIDER(obs::T("Blades"), 0, 16, 0, 16, dp.blades, DISK_BLADES);
    PF_ADD_FLOAT_SLIDERX(obs::T("Blade Curvature"), 0, 100, 0, 100, dp.bladeCurvature * 100.0, 1, PCT, SupFlag(DISK_CURVATURE), DISK_CURVATURE);
    if ((err = AddFloat(in_data, obs::T("Corner Rounding"), 0, 100, 0, 100, dp.cornerRounding * 100.0, 0, 1, PCT, DISK_CORNER)))
        return err;
    AEFX_CLR_STRUCT(def);
    def.flags = SupFlag(DISK_ROTATION);
    PF_ADD_ANGLE(obs::T("Rotation"), 0, DISK_ROTATION);
    PF_ADD_FLOAT_SLIDERX(obs::T("Anamorphic"), 0.25, 4, 0.25, 4, 1, 2, NONE, SupFlag(DISK_ANAMORPHIC), DISK_ANAMORPHIC);
    PF_ADD_FLOAT_SLIDERX(obs::T("Cat Eye"), 0, 100, 0, 100, dp.catEye * 100.0, 1, PCT, SupFlag(DISK_CATEYE), DISK_CATEYE);
    if ((err = AddFloat(in_data, obs::T("Obstruction"), 0, 90, 0, 90, dp.obstruction * 100.0, 0, 1, PCT, DISK_OBSTRUCTION)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_IRIS_GROUP_END);

    PF_ADD_TOPICX(obs::T("Lens Character"), 0, DISK_LENS_GROUP_START);
    PF_ADD_FLOAT_SLIDERX(obs::T("Edge Ring"), -100, 100, -100, 100, dp.edgeRing * 100.0, 1, PCT, SupFlag(DISK_EDGE_RING), DISK_EDGE_RING);
    PF_ADD_FLOAT_SLIDERX(obs::T("Ring Width"), 0, 100, 0, 100, dp.ringWidth * 100.0, 1, PCT, SupFlag(DISK_RING_WIDTH), DISK_RING_WIDTH);
    if ((err = AddFloat(in_data, obs::T("Edge Softness"), 0, 100, 0, 100, dp.edgeSoftness * 100.0, 0, 1, PCT, DISK_SOFTNESS)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Apodization"), 0, 100, 0, 100, dp.apodization * 100.0, 0, 1, PCT, DISK_APOD)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Core Brightness"), -100, 100, -100, 100, dp.coreWeight * 100.0, 0, 1, PCT, DISK_CORE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Spherical Aberration"), -100, 100, -100, 100, dp.sphericalAberration * 100.0, 0, 1, PCT, DISK_SA)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Onion Rings"), 0, 100, 0, 100, dp.onionRings * 100.0, 0, 1, PCT, DISK_ONION)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Onion Ring Count"), 1, 10, 1, 10, dp.onionCount, 4, 1, NONE, DISK_ONION_COUNT)))
        return err;
    // negative values swap the fringe colours
    PF_ADD_FLOAT_SLIDERX(obs::T("Chroma Fringe"), -100, 100, -100, 100, dp.chromaFringe * 100.0, 1, PCT, SupFlag(DISK_CHROMA), DISK_CHROMA);
    // 0% = red/blue (v1), 100% = green behind / magenta in front (classic longitudinal CA)
    if ((err = AddFloat(in_data, obs::T("Fringe Tint"), 0, 100, 0, 100, dp.fringeTint * 100.0, 0, 1, PCT, DISK_FRINGE_TINT)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_LENS_GROUP_END);

    PF_ADD_TOPICX(obs::T("Falloff & Glow"), PF_ParamFlag_START_COLLAPSED, DISK_FALLOFF_GROUP_START);
    if ((err = AddFloat(in_data, obs::T("Vignetting"), 0, 100, 0, 100, dp.vignetting * 100.0, 0, 1, PCT, DISK_VIGNETTE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Veiling Glow"), 0, 100, 0, 50, dp.glow * 100.0, 0, 1, PCT, DISK_GLOW)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Glow Size"), 0.5, 25, 0.5, 15, dp.glowSize * 100.0, 4, 1, PCT, DISK_GLOW_SIZE)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_FALLOFF_GROUP_END);

    // One highlight model: luminance-true expansion of bright pixels (inverse tone map), controls
    // for WHAT becomes a ball (size window, white protection, dots vs lines, a highlight layer) and
    // an output rolloff. The v1 gain lives in a sub-group. Tune with View = Highlight Source.
    PF_ADD_TOPICX(obs::T("Highlights"), 0, DISK_HL_GROUP_START);
    if ((err = AddFloat(in_data, obs::T("Highlight Boost"), 0, 20, 0, 4, dp.highlightBoost, 0, 2, NONE, DISK_HL_BOOST)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Highlight Knee"), 0, 0.97, 0, 0.95, dp.highlightKnee, 0.45, 2, NONE, DISK_HL_KNEE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Highlight Curve"), 1, 8, 1, 4, dp.highlightCurve, 2, 2, NONE, DISK_HL_CURVE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Size Selectivity"), 0, 100, 0, 100, dp.highlightSelectivity * 100.0, 85, 1, PCT, DISK_HL_SEL)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Detail Size"), 1, 200, 2, 40, dp.detailSize, 8, 1, NONE, DISK_HL_DETAIL)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Min Detail Size"), 0, 100, 0, 20, dp.detailMin, 0, 1, NONE, DISK_HL_DETAIL_MIN)))
        return err;
    if ((err = AddFloat(in_data, obs::T("White Protection"), 0, 100, 0, 100, dp.whiteProtection * 100.0, 0, 1, PCT, DISK_HL_WHITE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Dots vs Lines"), 0, 100, 0, 100, dp.dotsOnly * 100.0, 0, 1, PCT, DISK_HL_DOTS)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Highlight Saturation"), 0, 100, 0, 100, dp.highlightSaturation * 100.0, 0, 1, PCT, DISK_HL_SAT)))
        return err;
    if ((err = AddPopup(in_data, obs::T("Highlight Source"), 3, SRC_AUTO, obs::T("Automatic|Highlight Layer Only|Automatic x Highlight Layer"),
                        DISK_HL_SOURCE)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER(obs::T("Highlight Layer"), PF_LayerDefault_NONE, DISK_HL_LAYER);
    if ((err = AddPopup(in_data, obs::T("Highlight Layer Channel"), 5, CH_LUMINANCE, obs::T("Luminance|Red|Green|Blue|Alpha"),
                        DISK_HL_LAYER_CHANNEL)))
        return err;
    // Output shoulder for 8/16-bpc delivery; 0 = linear (HDR / 32-bpc workflows). Not set by presets.
    if ((err = AddFloat(in_data, obs::T("Highlight Rolloff"), 0, 100, 0, 100, 50, 0, 1, PCT, DISK_HL_ROLLOFF)))
        return err;
    PF_ADD_TOPICX(obs::T("Classic Gain (v1)"), PF_ParamFlag_START_COLLAPSED, DISK_CLASSIC_GROUP_START);
    PF_ADD_FLOAT_SLIDERX(obs::T("Classic Threshold"), 0, 10, 0, 2, 1, 2, NONE, 0, DISK_HL_THRESHOLD);
    PF_ADD_FLOAT_SLIDERX(obs::T("Classic Gain"), 0, 100, 0, 10, 0, 2, NONE, 0, DISK_HL_GAIN);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_CLASSIC_GROUP_END);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_HL_GROUP_END);

    // Film effects on the final image (in-focus pixels included). Off by default.
    PF_ADD_TOPICX(obs::T("Halation & Aberration"), PF_ParamFlag_START_COLLAPSED, DISK_FILM_GROUP_START);
    if ((err = AddFloat(in_data, obs::T("Halation"), 0, 200, 0, 100, 0, 0, 1, PCT, DISK_HAL_AMOUNT)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Halation Size"), 0.1, 20, 0.2, 5, def0.halationSize * 100.0, def0.halationSize * 100.0, 2, PCT,
                        DISK_HAL_SIZE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Halation Threshold"), 0, 4, 0, 1.5, def0.halationThreshold, def0.halationThreshold, 2, NONE,
                        DISK_HAL_THRESHOLD)))
        return err;
    // stops above the local scene level: only lights that are bright for their surroundings halate
    if ((err = AddFloat(in_data, obs::T("Halation Above Scene"), -2, 6, 0, 4, def0.halationRelative, def0.halationRelative, 2, NONE,
                        DISK_HAL_RELATIVE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Halation Softness"), 0, 100, 0, 100, def0.halationSoftness * 100.0,
                        def0.halationSoftness * 100.0, 1, PCT, DISK_HAL_SOFTNESS)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Halation Rim"), 0, 100, 0, 100, def0.halationRim * 100.0, def0.halationRim * 100.0, 1, PCT,
                        DISK_HAL_RIM)))
        return err;
    if ((err = AddColor(in_data, obs::T("Halation Tint"), def0.halationTint, DISK_HAL_TINT)))
        return err;
    // Aberration Style: a starting point that writes every shape/colour param below (not the amount),
    // supervised like Lens Preset. 1 = Custom (no change).
    static const std::string styleNames = [] {
        std::string s = obs::T("Custom");
        for (int i = 0; i < lenscore::kAberrationStyleCount; ++i)
        {
            s += "|";
            s += obs::T(lenscore::kAberrationStyles[i].ui);
        }
        return s;
    }();
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POPUP;
    PF_STRNNCPY(def.PF_DEF_NAME, obs::T("Aberration Style"), sizeof(def.PF_DEF_NAME));
    def.u.pd.num_choices = (A_short)(lenscore::kAberrationStyleCount + 1);
    def.u.pd.dephault = 1;
    def.u.pd.value = 1;
    def.u.pd.u.namesptr = styleNames.c_str();
    def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
    def.uu.id = DISK_LA_STYLE;
    if ((err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE)
        return err;
    if ((err = AddPicker(in_data, obs::T("Style Thumbnails"), obthumb::STYLE, DISK_STYLE_PICKER, true)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Luminance Aberration"), 0, 200, 0, 100, 0, 0, 1, PCT, DISK_LA_AMOUNT)))
        return err;
    if ((err = AddPopup(in_data, obs::T("Aberration Side"), 5, LA_SIDE_DARK,
                        obs::T("Bleed into Dark Side|Tint Bright Side|Both|Drawn Rim|Channel Drift"), DISK_LA_MODE)))
        return err;
    // Aberration Width: bleed sigma (sides 1-3) / rim width (Contrast Rim), % of the diagonal
    if ((err = AddFloat(in_data, obs::T("Aberration Width"), 0.01, 10, 0.02, 2, def0.laWidth * 100.0, def0.laWidth * 100.0, 3, PCT,
                        DISK_LA_WIDTH)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Aberration Threshold"), 0, 8, 0, 5, def0.laThreshold, def0.laThreshold, 2, NONE,
                        DISK_LA_THRESHOLD)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Aberration Curve"), 0.25, 4, 0.25, 3, def0.laCurve, def0.laCurve, 2, NONE, DISK_LA_CURVE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Hue by Contrast"), 0, 100, 0, 100, def0.laHueShift * 100.0, def0.laHueShift * 100.0, 1, PCT,
                        DISK_LA_HUE)))
        return err;
    if ((err = AddColor(in_data, obs::T("Aberration Color"), def0.laTint, DISK_LA_TINT)))
        return err;
    // Contrast Rim only
    if ((err = AddFloat(in_data, obs::T("Rim Contrast Scale"), 0.05, 5, 0.1, 2, def0.laContrastScale * 100.0,
                        def0.laContrastScale * 100.0, 2, PCT, DISK_LA_SCALE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Rim Line Protection"), 0, 2, 0, 0.6, def0.laLineProtect * 100.0, def0.laLineProtect * 100.0,
                        3, PCT, DISK_LA_LINE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Rim Sharpness"), 0, 100, 0, 100, def0.laSharpness * 100.0, def0.laSharpness * 100.0, 1, PCT,
                        DISK_LA_SHARP)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Rim Color from Bright Side"), 0, 100, 0, 100, def0.laBrightColor * 100.0,
                        def0.laBrightColor * 100.0, 1, PCT, DISK_LA_BRIGHTCOL)))
        return err;
    // Drawn Rim: 0 = the crisp line; > 0 = a soft tinted spill (optical fringe)
    if ((err = AddFloat(in_data, obs::T("Rim Blur"), 0, 2, 0, 0.5, def0.laBlur * 100.0, def0.laBlur * 100.0, 3, PCT, DISK_LA_BLUR)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Rim Complement"), 0, 100, 0, 100, def0.laComplement * 100.0, def0.laComplement * 100.0, 1,
                        PCT, DISK_LA_COMPLEMENT)))
        return err;
    // Channel Drift (the per-channel vector-blur recipe)
    {
        static const std::string fringeNames = [] {
            std::string s;
            for (int i = 0; i < lenscore::kFringeColorCount; ++i)
            {
                if (i)
                    s += "|";
                s += lenscore::kFringeColors[i].ui;
            }
            return s;
        }();
        if ((err = AddPopup(in_data, obs::T("Fringe Color"), lenscore::kFringeColorCount, 1, obs::T(fringeNames.c_str()), DISK_LA_FRINGE)))
            return err;
    }
    if ((err = AddPopup(in_data, obs::T("Fringe Side"), 2, LA_DIR_RETREAT, obs::T("Bright Edge (channels retreat)|Dark Edge (channels advance)"),
                        DISK_LA_DIR)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Map Softness"), 0, 2, 0, 0.6, def0.laMapSoftness * 100.0, def0.laMapSoftness * 100.0, 3, PCT,
                        DISK_LA_MAPSOFT)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Preserve Detail"), 0, 100, 0, 100, def0.laPreserve * 100.0, def0.laPreserve * 100.0, 1, PCT,
                        DISK_LA_PRESERVE)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_FILM_GROUP_END);

    // Film grain on the final image (in-focus pixels included). Off by default.
    PF_ADD_TOPICX(obs::T("Grain"), PF_ParamFlag_START_COLLAPSED, DISK_GRAIN_GROUP_START);
    {
        static const std::string stockNames = [] {
            std::string s = obs::T("Custom");
            for (int i = 0; i < lenscore::kGrainStockCount; ++i)
            {
                s += "|";
                s += obs::T(lenscore::kGrainStocks[i].ui);
            }
            return s;
        }();
        AEFX_CLR_STRUCT(def);
        def.param_type = PF_Param_POPUP;
        PF_STRNNCPY(def.PF_DEF_NAME, obs::T("Grain Stock"), sizeof(def.PF_DEF_NAME));
        def.u.pd.num_choices = (A_short)(lenscore::kGrainStockCount + 1);
        def.u.pd.dephault = 1;
        def.u.pd.value = 1;
        def.u.pd.u.namesptr = stockNames.c_str();
        def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
        def.uu.id = DISK_GRAIN_STOCK;
        if ((err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE)
            return err;
    }
    if ((err = AddPicker(in_data, obs::T("Stock Thumbnails"), obthumb::STOCK, DISK_STOCK_PICKER, true)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Grain Amount"), 0, 200, 0, 100, 0, 0, 1, PCT, DISK_GRAIN_AMOUNT)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Grain Size"), 0.3, 8, 0.3, 4, def0.grainSize, def0.grainSize, 2, NONE, DISK_GRAIN_SIZE)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Grain Softness"), 0, 100, 0, 100, def0.grainSoftness * 100.0, def0.grainSoftness * 100.0, 1,
                        PCT, DISK_GRAIN_SOFT)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Grain Color"), 0, 100, 0, 100, def0.grainColor * 100.0, def0.grainColor * 100.0, 1, PCT,
                        DISK_GRAIN_COLOR)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Grain Tone Response"), 0, 100, 0, 100, def0.grainTone * 100.0, def0.grainTone * 100.0, 1, PCT,
                        DISK_GRAIN_TONE)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Animate Grain"), "", true, 0, DISK_GRAIN_ANIMATE);
    if ((err = AddInt(in_data, obs::T("Grain Hold Frames"), 1, 24, 1, DISK_GRAIN_HOLD)))
        return err;
    if ((err = AddInt(in_data, obs::T("Grain Seed"), 0, 9999, 0, DISK_GRAIN_SEED)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_GRAIN_GROUP_END);

    // Reserved for the Best quality engine (see BlurStage; no effect yet). Enabled only while
    // Quality = Best (UpdateParamsUI); disabled at setup because the default is Standard.
    const PF_ParamUIFlags hqUi = PF_PUI_DISABLED;
    PF_ADD_TOPICX(obs::T("Best Quality (Beta)"), PF_ParamFlag_START_COLLAPSED, DISK_HQ_GROUP_START);
    if ((err = AddFloat(in_data, obs::T("Disc Texture Amount"), 0, 200, 0, 100, 0, 0, 1, PCT, DISK_HQ_TEX_AMOUNT, hqUi)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Texture Scale"), 0.25, 4, 0.25, 4, 1, 1, 2, NONE, DISK_HQ_TEX_SCALE, hqUi)))
        return err;
    // order = lenshq::TextureType
    if ((err = AddPopup(in_data, obs::T("Texture Type"), 6, 1, obs::T("Mixed|Fine Grain|Dust|Wipe Streaks|Polishing Marks|Soap Bubble"),
                        DISK_HQ_TEX_TYPE, hqUi)))
        return err;
    if ((err = AddInt(in_data, obs::T("Texture Seed"), 0, 9999, 0, DISK_HQ_TEX_SEED, hqUi)))
        return err;
    // 0 = Auto (spectral), 1-2 = RGB three-band (like Standard), >= 3 = that many wavelengths
    if ((err = AddInt(in_data, obs::T("Spectral Samples (0=Auto)"), 0, 32, 0, DISK_HQ_SPECTRAL, hqUi)))
        return err;
    if ((err = AddFloat(in_data, obs::T("Lateral Chromatic Aberration"), -100, 100, -100, 100, 0, 0, 1, PCT, DISK_HQ_LATERAL_CA, hqUi)))
        return err;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DISK_HQ_GROUP_END);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(obs::T("Work in Linear Light"), "", true, 0, DISK_LINEAR);
    // v2.2: what the input pixels are when Work in Linear Light is on. 2.2 = decode with gamma 2.2
    // (v2.1 behaviour, old projects); Linear = the working space is already linear (Linearize Working
    // Space, OCIO scene-linear): no decode / encode; Auto = see InputIsLinear (currently = 2.2).
    if ((err = AddPopupOld(in_data, obs::T("Input Gamma"), 3, GAMMA_AUTO, GAMMA_22, obs::T("Auto|Gamma 2.2|Linear"), DISK_INPUT_GAMMA)))
        return err;
    // v2.2 Edge Behavior (like AE's own blurs): Spread = the blur spreads beyond the layer bounds;
    // Repeat Edge = edge pixels are repeated and the output stays inside the layer; Clip to Layer =
    // transparent outside, output clipped to the layer (v1.2; old projects).
    // v2.3 (owner decision): NEW instances and Reset default to Repeat Edge, so a layer that fills the
    // frame never fades out at its edges. Only the default (dephault) changes: AE stores the popup value
    // of every saved instance, so v2.2.x instances saved with Spread stay Spread; projects that predate
    // this param still get Clip to Layer through USE_VALUE_FOR_OLD_PROJECTS (u.pd.value), and a v2.1
    // instance with the hidden Repeat Edge Pixels checkbox on still renders Repeat Edge.
    if ((err = AddPopupOld(in_data, obs::T("Edge Behavior"), 3, EDGE_REPEAT, EDGE_CLIP, obs::T("Spread|Repeat Edge|Clip to Layer"),
                           DISK_EDGE_MODE, PF_ParamFlag_SUPERVISE)))
        return err;
    // v2.1 checkbox, kept (hidden) so v2.1 projects with it on still render Repeat Edge. Changing
    // Edge Behavior clears it (UserChangedParam).
    AEFX_CLR_STRUCT(def);
    def.ui_flags = PF_PUI_INVISIBLE;
    PF_ADD_CHECKBOX(obs::T("Repeat Edge Pixels"), "", false, 0, DISK_REPEAT_EDGE);
    // Hidden render version (lenscore Params.renderVersion): 1 for projects saved before v2.2 (their
    // highlights / widths / drift / rolloff render exactly as in v2.1), 2 for new instances.
    {
        AEFX_CLR_STRUCT(def);
        def.param_type = PF_Param_SLIDER;
        PF_STRNNCPY(def.PF_DEF_NAME, obs::T("Render Version"), sizeof(def.PF_DEF_NAME));
        def.u.sd.valid_min = def.u.sd.slider_min = 1;
        def.u.sd.valid_max = def.u.sd.slider_max = 99;
        def.u.sd.value = 1;
        def.u.sd.dephault = kRenderVersionCurrent;
        def.flags = PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS | PF_ParamFlag_CANNOT_TIME_VARY;
        def.ui_flags = PF_PUI_INVISIBLE;
        def.uu.id = DISK_RENDER_VERSION;
        if ((err = PF_ADD_PARAM(in_data, -1, &def)) != PF_Err_NONE)
            return err;
    }

    // v2.3: custom UI events for the thumbnail pickers (Effect Controls only)
    {
        PF_CustomUIInfo ci;
        AEFX_CLR_STRUCT(ci);
        ci.events = PF_CustomEFlag_EFFECT;
        if (in_data->inter.register_ui && (err = PF_REGISTER_UI(in_data, &ci)) != PF_Err_NONE)
            return err;
    }

    out_data->num_params = OB_NUM_PARAMS;
    return PF_Err_NONE;
}

// Lens Preset popup changed: write the preset into the character params (undoable, the user can
// tweak freely afterwards). "Custom" writes nothing. Modelled on the SDK Supervisor sample:
// set the new values + PF_ChangeFlag_CHANGED_VALUE on each changed param.
PF_Err UpdateParamsUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]);

PF_Err UserChangedParam(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], const PF_UserChangedParamExtra* which)
{
    if (!which || !params)
        return PF_Err_NONE;
    const int i = which->param_index;
    if (i == OB_QUALITY)  // Best group enabled state, immediately
        return UpdateParamsUI(in_data, out_data, params);
    if (i == OB_PRESET)
        ApplyPresetToParams(params[OB_PRESET]->u.pd.value, params);
    else if (i == OB_LA_STYLE)
        ApplyAberrationStyleToParams(params[OB_LA_STYLE]->u.pd.value, params);
    else if (i == OB_GRAIN_STOCK)
        ApplyGrainStockToParams(params[OB_GRAIN_STOCK]->u.pd.value, params);
    else if (i == OB_EDGE_MODE)
    {
        // the user chose an edge behaviour: the hidden v2.1 checkbox no longer overrides it
        if (params[OB_REPEAT_EDGE]->u.bd.value)
        {
            params[OB_REPEAT_EDGE]->u.bd.value = 0;
            params[OB_REPEAT_EDGE]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
        }
    }
    else
    {
        // a hand edit of a param a popup wrote: that popup now says Custom (value 1). AE does not send
        // USER_CHANGED_PARAM for values the effect itself changes, so this cannot loop.
        auto toCustom = [&](int popup) {
            if (params[popup]->u.pd.value != 1)
            {
                params[popup]->u.pd.value = 1;
                params[popup]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
            }
        };
        if (Contains(kPresetTargets, i))
            toCustom(OB_PRESET);
        if (Contains(kStyleTargets, i))
            toCustom(OB_LA_STYLE);
        if (Contains(kStockTargets, i))
            toCustom(OB_GRAIN_STOCK);
    }
    return PF_Err_NONE;
}

// Beta Quality group: enabled only while Quality = Best. Sent on every param change and when a
// project opens (PF_OutFlag_SEND_UPDATE_PARAMS_UI); only the UI flag changes, never a value, so
// renders are unaffected (and nothing here is shared between threads).
PF_Err UpdateParamsUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[])
{
    (void)out_data;
    if (!params || !in_data->pica_basicP)
        return PF_Err_NONE;
    const bool best = params[OB_QUALITY]->u.pd.value == QUALITY_BEST;
    const PF_ParamUtilsSuite3* pu = nullptr;
    if (in_data->pica_basicP->AcquireSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, (const void**)&pu) || !pu)
        return PF_Err_NONE;
    PF_Err err = PF_Err_NONE;
    const int hq[] = {OB_HQ_TEX_AMOUNT, OB_HQ_TEX_SCALE, OB_HQ_TEX_TYPE, OB_HQ_TEX_SEED, OB_HQ_SPECTRAL, OB_HQ_LATERAL_CA};
    for (int i : hq)
    {
        const bool disabled = (params[i]->ui_flags & PF_PUI_DISABLED) != 0;
        if (disabled != best)
            continue;  // already in the right state
        PF_ParamDef d = *params[i];
        if (best)
            d.ui_flags &= ~PF_PUI_DISABLED;
        else
            d.ui_flags |= PF_PUI_DISABLED;
        ERR(pu->PF_UpdateParamUI(in_data->effect_ref, i, &d));
    }
    in_data->pica_basicP->ReleaseSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
    return err;
}
#else
#include "kantan_ui.inc"  // Kantan: ParamsSetup, UserChangedParam, UpdateParamsUI
#endif

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
#ifndef OB_KANTAN  // Kantan: Highlight Source is always Automatic (no highlight layer)
    if (!err && !amountView && plan->ep.highlightSource != 0)
        ERR(CheckoutSampledLayer(in_data, extra, outReq, OB_HL_LAYER, CHECKOUT_HL_PROBE, CHECKOUT_HL, *plan, dsx, dsy,
                                 need, plan->hl));
#endif
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
#ifdef OB_KANTAN
    const int first = 1;  // Kantan popups have no "Custom"; the popup is the whole state
    auto apLens = KantanPopupOnly, apStyle = KantanPopupOnly, apStock = KantanPopupOnly;
#else
    const int first = 2;
    auto apLens = ApplyPresetToParams, apStyle = ApplyAberrationStyleToParams, apStock = ApplyGrainStockToParams;
#endif
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
