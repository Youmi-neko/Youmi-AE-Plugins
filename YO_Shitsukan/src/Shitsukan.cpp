// Shitsukan (prototype) - After Effects SmartFX wrapper around sk_core (texture control, line-keeping blur and
// watercolour in one effect). Structure adapted from MapBilinearBlur (strings, MFR thread split, map layer
// sampling) and VectorNobi 0.2.0.2 (SmartFX checkout order: input pixels BEFORE the output, always).
//
// Coordinates: "layer coords" = downsampled layer pixel coordinates (SmartFX request / result rects).
// AE worlds are ARGB and STRAIGHT (unpremultiplied) at 8/16/32 bpc (Optic Bokeh's host tests).
//
// Edges: the effect never changes alpha and never grows the layer. Its working domain is the input's
// max_result_rect (adjustment layers / collapsed precomps report one larger than the layer: the extra area is
// transparent and, because every filter is normalised by alpha, contributes nothing - no darkening at the comp
// edge). The whole domain is computed for every request (tile == full exactly); only a huge domain (> 40 Mpx,
// e.g. a layer grown by an upstream effect) is cut to the request plus a margin.
#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectSuites.h"

#include "ShitsukanConfig.h"
#include "sk_core.h"
#include "sk_strings.h"
#include "sk_styles.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

extern "C" DllExport PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra);

const A_long kOutFlags = PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE;
const A_long kOutFlags2 = PF_OutFlag2_SUPPORTS_QUERY_DYNAMIC_FLAGS | PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
                          PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                          PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
static_assert(SK_OUTFLAGS == kOutFlags, "PiPL out_flags mismatch");
static_assert(SK_OUTFLAGS2 == kOutFlags2, "PiPL out_flags2 mismatch");
#define SK_STAGE PF_Stage_DEVELOP
static_assert(SK_PIPL_VERSION == PF_VERSION(SK_MAJOR_VERSION, SK_MINOR_VERSION, SK_BUG_VERSION, SK_STAGE, SK_BUILD_VERSION),
              "PiPL version mismatch");

namespace {

// ------------------------------------------------------------------ params ---
// Index = UI order (what callbacks, PF_CHECKOUT_PARAM and params[] use). Disk IDs (PF_ParamDef uu.id, what AE
// stores in projects) come from kDiskId below and are frozen: the 0.6 params keep their 0.6 numbers (= their 0.6
// indices); a new param may be inserted anywhere in this enum but gets a NEW disk ID at the end. Never reuse one.
enum
{
    P_INPUT = 0,
    P_STYLE,
    P_AMOUNT,
    P_TEX_START,
    P_FINE,
    P_MID,
    P_LARGE,
    P_DETAIL,
    P_HALO,
    P_EDGEAWARE,
    P_COLORDETAIL,
    P_TEX_END,
    P_LINE_START,
    P_PROTECT,
    P_LINE_SENS,
    P_LINE_WIDTH,
    P_LINE_COLOR,
    P_BLACK_WALLS,
    P_BG_DETAIL,
    P_FILL_BLUR,
    P_KEEP_EDGES,
    P_LINE_THICK,
    P_LINE_DARK,
    P_LINE_SOFT,
    P_LINE_STYLE,
    P_LINE_END,
    P_WC_START,
    P_WC,
    P_WC_MODE,
    P_PAINT_SCALE,
    P_MERGE,
    P_LOOSE,
    P_LIFT_DARKS,
    P_CHARGE,
    P_PALETTE,
    P_PIGSET,
    P_WARP,
    P_ENGINE,  // 0.6.0.3: Painting Engine (disk ID 88): Current | Round 6
    P_MARGIN,
    P_FLATTEN,
    P_FLATTEN_SIZE,
    P_STROKES,
    P_STROKE_LEN,
    P_STROKE_ANGLE,
    P_GLAZE,
    P_RIM,
    P_RIM_WIDTH,
    P_GRAVITY,
    P_BLEED,
    P_BLEED_WIDTH,
    P_BACKRUN,
    P_WOBBLE,
    P_WOBBLE_SCALE,
    P_UNEVEN,
    P_GRAN,
    P_PAPER,
    P_PAPER_SCALE,
    P_KEEP_WHITE,
    P_DILUTION,
    P_PROTECT_DETAIL,
    P_PAPER_LAYER,
    P_WC_END,
    P_MAP_START,
    P_MAP_LAYER,
    P_MAP_TARGET,
    P_MAP_CHANNEL,
    P_MAP_INVERT,
    P_MAP_USE,       // depth matte (round 10): Strength Mask | Depth Matte
    P_MAP_BLUR,
    P_DEPTH_REF,
    P_DEPTH_RANGE,
    P_DEPTH_PICK,
    P_DEPTH_POINT,
    P_DEPTH_MASTER,
    P_D_AMOUNT,
    P_D_SCALE,
    P_D_WET,
    P_D_LINES,
    P_D_PALE,
    P_D_TEXTURE,
    P_D_BLUR,
    P_MAP_END,
    P_EDGE,
    P_SPACE,
    P_VIEW,
    P_SEED,
    P_ANIMATE,
    P_HOLD,
    P_RENDER_VERSION,
    P_NUM
};
static_assert(P_NUM == 89, "param list changed: update kDiskId, the fake host and aetest\\build_sk.jsx");

// disk ID of each index, in enum order (frozen; see above). Next new disk ID: 89.
constexpr int kDiskId[P_NUM] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 72, 73, 88, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,  // round 10: Map Use .. Depth: Blur
    64, 65, 66, 67, 68, 69, 70, 71,
};
constexpr bool DiskIdsUnique()
{
    for (int i = 0; i < P_NUM; ++i)
        for (int j = i + 1; j < P_NUM; ++j)
            if (kDiskId[i] == kDiskId[j])
                return false;
    return true;
}
static_assert(DiskIdsUnique(), "disk IDs must be unique");
static_assert(kDiskId[P_INPUT] == 0 && kDiskId[P_PALETTE] == 34 && kDiskId[P_MARGIN] == 35 && kDiskId[P_FLATTEN] == 36 &&
                  kDiskId[P_PAPER_LAYER] == 57 && kDiskId[P_MAP_LAYER] == 60 && kDiskId[P_VIEW] == 67 &&
                  kDiskId[P_RENDER_VERSION] == 71 && kDiskId[P_PIGSET] == 72 && kDiskId[P_WARP] == 73 &&
                  kDiskId[P_MAP_INVERT] == 63 && kDiskId[P_MAP_END] == 64 && kDiskId[P_MAP_USE] == 74 &&
                  kDiskId[P_DEPTH_POINT] == 79 && kDiskId[P_D_BLUR] == 87 && kDiskId[P_ENGINE] == 88,
              "frozen disk IDs changed");

enum { STYLE_CUSTOM = 1 };  // popup value s + 2 = sk::kStyles[s]
enum { CH_LUMA = 1, CH_ALPHA, CH_RED, CH_GREEN, CH_BLUE };  // Map Channel popup values (1, 2 = the 0.6 meaning)
enum { MU_STRENGTH = 1, MU_DEPTH };                          // Map Use popup values

// the depth-matte sliders (NOT style-written: styles never touch the map group, so no SUPERVISE and no Custom
// switch - they are deliberately not in kFloats)
struct DepthCtl
{
    int idx;
    double sk::Params::*m;
    const char* name;
    double vmin, vmax, smin, smax, dflt;
    bool pct;
};
const DepthCtl kDepthFloats[] = {
    {P_MAP_BLUR, &sk::Params::mapBlur, "Map Blur", 0, 200, 0, 50, 0, false},
    {P_DEPTH_REF, &sk::Params::depthRef, "Reference Depth", 0, 100, 0, 100, 50, true},
    {P_DEPTH_RANGE, &sk::Params::depthRange, "Depth Range", 1, 100, 1, 100, 30, true},
    {P_DEPTH_MASTER, &sk::Params::dMaster, "Depth Strength", 0, 100, 0, 100, 100, true},
    {P_D_AMOUNT, &sk::Params::dAmount, "Depth: Amount", -100, 100, -100, 100, 0, true},
    {P_D_SCALE, &sk::Params::dScale, "Depth: Wash Size", -100, 100, -100, 100, 0, true},
    {P_D_WET, &sk::Params::dWet, "Depth: Wetness", -100, 100, -100, 100, 0, true},
    {P_D_LINES, &sk::Params::dLines, "Depth: Lines", -100, 100, -100, 100, 0, true},
    {P_D_PALE, &sk::Params::dPale, "Depth: Paleness", -100, 100, -100, 100, 0, true},
    {P_D_TEXTURE, &sk::Params::dTexture, "Depth: Texture", -100, 100, -100, 100, 0, true},
    {P_D_BLUR, &sk::Params::dBlur, "Depth: Blur", -100, 100, -100, 100, 0, true},
};
enum { CO_INPUT_PROBE = 1, CO_INPUT, CO_MAP_PROBE, CO_MAP, CO_PAPER_PROBE, CO_PAPER };

// The style-written controls, in ONE table: ParamsSetup (defaults = the default style), ReadSettings, the Style
// popup and "edit -> Custom" all use it, so they cannot drift apart.
struct FloatCtl
{
    int idx;
    double sk::Params::*m;
    const char* name;
    double vmin, vmax, smin, smax;
    int prec;
    bool pct;
};
const FloatCtl kFloats[] = {
    {P_FINE, &sk::Params::fine, "Fine Detail", -100, 200, -100, 200, 1, true},
    {P_MID, &sk::Params::mid, "Mid Detail", -100, 200, -100, 200, 1, true},
    {P_LARGE, &sk::Params::large, "Large Light/Dark", -100, 200, -100, 200, 1, true},
    {P_DETAIL, &sk::Params::detailSize, "Detail Size", 0.3, 50, 0.5, 10, 1, false},
    {P_HALO, &sk::Params::halo, "Halo Control", 0, 100, 0, 100, 1, true},
    {P_EDGEAWARE, &sk::Params::edgeAware, "Edge Aware", 0, 100, 0, 100, 1, true},
    {P_COLORDETAIL, &sk::Params::colorDetail, "Color Detail", 0, 100, 0, 100, 1, true},
    {P_LINE_SENS, &sk::Params::lineSens, "Line Detection", 0, 100, 0, 100, 1, true},
    {P_LINE_WIDTH, &sk::Params::lineWidth, "Line Width", 0.5, 32, 1, 12, 1, false},
    {P_FILL_BLUR, &sk::Params::fillBlur, "Fill Blur", 0, 500, 0, 100, 1, false},
    {P_KEEP_EDGES, &sk::Params::keepEdges, "Keep Color Edges", 0, 100, 0, 100, 1, true},
    {P_LINE_THICK, &sk::Params::lineThick, "Line Thickness", -10, 10, -3, 3, 2, false},
    {P_LINE_DARK, &sk::Params::lineDark, "Line Darkness", -100, 200, -100, 100, 1, true},
    {P_LINE_SOFT, &sk::Params::lineSoft, "Line Softness", 0, 100, 0, 10, 2, false},
    {P_WC, &sk::Params::wc, "Watercolor Amount", 0, 100, 0, 100, 1, true},
    {P_PAINT_SCALE, &sk::Params::paintScale, "Wash Size", 2, 400, 4, 80, 1, false},
    {P_MERGE, &sk::Params::merge, "Merge Colors", 0, 100, 0, 100, 1, true},
    {P_LOOSE, &sk::Params::loose, "Overshoot and Gaps", 0, 100, 0, 100, 1, true},
    {P_LIFT_DARKS, &sk::Params::liftDarks, "Lift Darks", 0, 100, 0, 100, 1, true},
    {P_CHARGE, &sk::Params::charge, "Color Drift", 0, 100, 0, 100, 1, true},
    {P_PALETTE, &sk::Params::palette, "Pigments", 0, 66, 0, 24, 0, false},
    {P_WARP, &sk::Params::warp, "Distortion", 0, 100, 0, 100, 1, true},
    {P_MARGIN, &sk::Params::margin, "Paper Margin", 0, 100, 0, 100, 1, true},
    {P_FLATTEN, &sk::Params::flatten, "Flatten Washes", 0, 100, 0, 100, 1, true},
    {P_FLATTEN_SIZE, &sk::Params::flattenSize, "Flatten Size", 0.5, 200, 1, 40, 1, false},
    {P_STROKES, &sk::Params::strokes, "Brush Strokes", 0, 100, 0, 100, 1, true},
    {P_STROKE_LEN, &sk::Params::strokeLen, "Stroke Length", 2, 200, 4, 60, 1, false},
    {P_STROKE_ANGLE, &sk::Params::strokeAngle, "Stroke Angle", -180, 180, -90, 90, 0, false},
    {P_GLAZE, &sk::Params::glaze, "Glazing", 0, 100, 0, 100, 1, true},
    {P_RIM, &sk::Params::rim, "Edge Darkening", 0, 200, 0, 200, 1, true},
    {P_RIM_WIDTH, &sk::Params::rimWidth, "Edge Width", 0.3, 100, 0.5, 20, 1, false},
    {P_GRAVITY, &sk::Params::gravity, "Pigment Runs Down", 0, 100, 0, 100, 1, true},
    {P_BLEED, &sk::Params::bleed, "Bleed", 0, 100, 0, 100, 1, true},
    {P_BLEED_WIDTH, &sk::Params::bleedWidth, "Bleed Width", 0.5, 300, 1, 60, 1, false},
    {P_BACKRUN, &sk::Params::backrun, "Backruns", 0, 100, 0, 100, 1, true},
    {P_WOBBLE, &sk::Params::wobble, "Wobble", 0, 50, 0, 10, 1, false},
    {P_WOBBLE_SCALE, &sk::Params::wobbleScale, "Wobble Scale", 2, 1000, 5, 200, 1, false},
    {P_UNEVEN, &sk::Params::uneven, "Pigment Unevenness", 0, 100, 0, 100, 1, true},
    {P_GRAN, &sk::Params::gran, "Granulation", 0, 100, 0, 100, 1, true},
    {P_PAPER, &sk::Params::paper, "Paper Texture", 0, 100, 0, 100, 1, true},
    {P_PAPER_SCALE, &sk::Params::paperScale, "Paper Scale", 25, 800, 25, 400, 0, true},
    {P_KEEP_WHITE, &sk::Params::keepWhite, "Keep Whites", 0, 100, 0, 100, 1, true},
    {P_DILUTION, &sk::Params::dilution, "Dilution", 0, 100, 0, 100, 1, true},
    {P_PROTECT_DETAIL, &sk::Params::protectDetail, "Protect Small Features", 0, 100, 0, 100, 1, true},
};
struct BoolCtl
{
    int idx;
    bool sk::Params::*m;
    const char* name;
};
const BoolCtl kBools[] = {{P_PROTECT, &sk::Params::protectLines, "Protect Lines"},
                          {P_BLACK_WALLS, &sk::Params::blackWalls, "Black Fills as Walls"},
                          {P_BG_DETAIL, &sk::Params::bgDetailLines, "Background Detail as Lines"}};
struct PopupCtl
{
    int idx;
    int sk::Params::*m;  // popup value v <-> field v - 1, or fields[v - 1] when fields is set
    const char* name;
    int choices;
    const char* items;
    const int* fields;
};
const int kWcModeFields[] = {sk::WCM_PAINT, sk::WCM_FILTER};  // "Paint in Washes|Filter (Old)"
const PopupCtl kPopups[] = {
    {P_LINE_COLOR, &sk::Params::lineColor, "Line Color", 3, "Dark Lines|Dark and Colored Lines|Also Light Lines", nullptr},
    {P_LINE_STYLE, &sk::Params::lineStyle, "Line Style", 3, "As Is|Pencil|Ink", nullptr},
    {P_WC_MODE, &sk::Params::wcMode, "Painting Method", 2, "Paint in Washes|Filter (Old)", kWcModeFields},
    {P_PIGSET, &sk::Params::pigSet, "Pigment Set", 4,
     "Watercolor Pigments|Storybook (Soft)|Storybook (Bright)|Source Colors (Simplified)", nullptr},
    {P_ENGINE, &sk::Params::paintEngine, "Painting Engine", 2, "Current (Stable in Video)|Round 6 (Look First)", nullptr}};
// popup value (1-based, clamped) -> field value, and back (an unknown field value -> 1)
int PopupToField(const PopupCtl& c, int v)
{
    v = std::min(c.choices, std::max(1, v));
    return c.fields ? c.fields[v - 1] : v - 1;
}
int FieldToPopup(const PopupCtl& c, int f)
{
    if (!c.fields)
        return f + 1;
    for (int v = 1; v <= c.choices; ++v)
        if (c.fields[v - 1] == f)
            return v;
    return 1;
}

bool IsStyleTarget(int i)
{
    for (const FloatCtl& c : kFloats)
        if (c.idx == i)
            return true;
    for (const BoolCtl& c : kBools)
        if (c.idx == i)
            return true;
    for (const PopupCtl& c : kPopups)
        if (c.idx == i)
            return true;
    return false;
}

struct Settings
{
    sk::Params p;
    int mapChannel = CH_LUMA;
    bool mapInvert = false;
    bool animate = false;
    int hold = 2;
};

struct LayerSampler
{
    bool has = false;
    double kx = 1, ky = 1;  // sampled-layer px per effect-layer px
    A_long w = 0, h = 0;
};

struct RenderPlan
{
    Settings s;
    double ds = 1;
    PF_LRect layerBounds{};
    PF_LRect domain{};
    PF_LRect result{};
    LayerSampler map, paper;
    bool passthrough = false;
    bool emptyResult = false;
    bool inputChecked = false;
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
inline PF_LRect Grow(PF_LRect r, A_long m)
{
    r.left -= m;
    r.top -= m;
    r.right += m;
    r.bottom += m;
    return r;
}
inline A_long ScaledExtent(A_long fullRes, double ds) { return (A_long)std::ceil(fullRes * ds - 1e-6); }
inline double Area(const PF_LRect& r) { return RectEmpty(r) ? 0.0 : (double)(r.right - r.left) * (double)(r.bottom - r.top); }

PF_Err ReadSettings(PF_InData* in_data, Settings& s)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef p[P_NUM];
    bool out[P_NUM] = {};
    for (int i = 1; i < P_NUM; ++i)
    {
        if (i == P_PAPER_LAYER || i == P_MAP_LAYER || i == P_TEX_START || i == P_TEX_END || i == P_LINE_START || i == P_LINE_END ||
            i == P_WC_START || i == P_WC_END || i == P_MAP_START || i == P_MAP_END)
            continue;
        AEFX_CLR_STRUCT(p[i]);
        ERR(PF_CHECKOUT_PARAM(in_data, i, in_data->current_time, in_data->time_step, in_data->time_scale, &p[i]));
        if (!err)
            out[i] = true;
    }
    if (!err)
    {
        sk::Params& q = s.p;
        sk::ApplyStyle(sk::kStyles[sk::kDefaultStyle], q);  // fallbacks for non-finite values
        for (const FloatCtl& c : kFloats)
        {
            const double v = p[c.idx].u.fs_d.value;
            if (std::isfinite(v))
                q.*c.m = std::min(c.vmax, std::max(c.vmin, v));
        }
        for (const BoolCtl& c : kBools)
            q.*c.m = p[c.idx].u.bd.value != 0;
        for (const PopupCtl& c : kPopups)
            q.*c.m = PopupToField(c, (int)p[c.idx].u.pd.value);
        const double am = p[P_AMOUNT].u.fs_d.value;
        q.amount = std::isfinite(am) ? std::min(100.0, std::max(0.0, am)) : 100.0;
        q.mapTarget = std::min((int)sk::MT_COUNT - 1, std::max(0, (int)p[P_MAP_TARGET].u.pd.value - 1));
        s.mapChannel = std::min((int)CH_BLUE, std::max((int)CH_LUMA, (int)p[P_MAP_CHANNEL].u.pd.value));
        s.mapInvert = p[P_MAP_INVERT].u.bd.value != 0;
        // depth matte (Map Use 2). With no map the engine gets map = nullptr and treats the depth matte as off.
        q.depth = p[P_MAP_USE].u.pd.value == MU_DEPTH;
        for (const DepthCtl& c : kDepthFloats)
        {
            const double v = p[c.idx].u.fs_d.value;
            q.*c.m = std::isfinite(v) ? std::min(c.vmax, std::max(c.vmin, v)) : c.dflt;
        }
        q.depthPick = p[P_DEPTH_PICK].u.bd.value != 0;
        {
            // Depth Point: AE delivers a PF_Param_POINT value as 16.16 fixed in the effect layer's own pixel
            // coordinates (origin = the layer's top-left, the same origin as the SmartFX layer coords here, even
            // when the input's max_result_rect is larger), ALREADY SCALED BY THE CURRENT DOWNSAMPLE (as Optic Bokeh
            // relies on: "buffer coords at the current downsample"). The engine wants FULL-RES layer px, so divide by
            // downsample_x / _y (ratios <= 1). Layer pixels are not square-corrected: no pixel aspect ratio here.
            const double dsx = Ratio(in_data->downsample_x), dsy = Ratio(in_data->downsample_y);
            const double px = (double)p[P_DEPTH_POINT].u.td.x_value / 65536.0 / dsx;
            const double py = (double)p[P_DEPTH_POINT].u.td.y_value / 65536.0 / dsy;
            q.pickX = std::isfinite(px) ? px : 0.0;
            q.pickY = std::isfinite(py) ? py : 0.0;
        }
        q.edgeMode = std::min(1, std::max(0, (int)p[P_EDGE].u.pd.value - 1));
        q.linear = p[P_SPACE].u.pd.value == 2;
        q.view = std::min((int)sk::V_COUNT - 1, std::max(0, (int)p[P_VIEW].u.pd.value - 1));
        q.seed = (int)p[P_SEED].u.sd.value;
        q.par = Ratio(in_data->pixel_aspect_ratio);
        s.animate = p[P_ANIMATE].u.bd.value != 0;
        s.hold = std::max(1, (int)p[P_HOLD].u.sd.value);
        if (s.animate)
        {
            // the layer's own frame number: the same pattern at the same layer time, boiling every `hold` frames
            const A_long st = in_data->local_time_step != 0 ? in_data->local_time_step : in_data->time_step;
            const long long fr = st != 0 ? (long long)std::floor((double)in_data->current_time / std::fabs((double)st) + 1e-6) : 0;
            const long long k = fr >= 0 ? fr / s.hold : -((-fr + s.hold - 1) / s.hold);
            q.paperFrame = (int)(k + 1);
        }
    }
    for (int i = 1; i < P_NUM; ++i)
        if (out[i])
            ERR2(PF_CHECKIN_PARAM(in_data, &p[i]));
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
    const float kMax = 65504.f;
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

template <typename Pix>
inline float LayerValue(const Pix& p, int channel)
{
    typedef Traits<Pix> T;
    const float a = std::min(1.f, std::max(0.f, SanitizeF(T::In(p.alpha))));
    if (channel == CH_ALPHA)
        return a;
    // R / G / B: the straight channel value (not weighted by alpha), 0..1
    if (channel == CH_RED)
        return std::min(1.f, std::max(0.f, SanitizeF(T::In(p.red))));
    if (channel == CH_GREEN)
        return std::min(1.f, std::max(0.f, SanitizeF(T::In(p.green))));
    if (channel == CH_BLUE)
        return std::min(1.f, std::max(0.f, SanitizeF(T::In(p.blue))));
    return (0.2126f * SanitizeF(T::In(p.red)) + 0.7152f * SanitizeF(T::In(p.green)) + 0.0722f * SanitizeF(T::In(p.blue))) * a;
}

// bilinear sample of a layer param at effect-layer pixel (lx, ly), its bounds stretched onto the layer
template <typename Pix>
float SampleLayer(const PF_EffectWorld* m, const RenderPlan& plan, const LayerSampler& ls, A_long lx, A_long ly, int channel)
{
    double u = (lx - plan.layerBounds.left + 0.5) * ls.kx - 0.5;
    double v = (ly - plan.layerBounds.top + 0.5) * ls.ky - 0.5;
    u = std::min((double)ls.w - 1, std::max(0.0, u)) - m->origin_x;
    v = std::min((double)ls.h - 1, std::max(0.0, v)) - m->origin_y;
    u = std::min((double)m->width - 1, std::max(0.0, u));
    v = std::min((double)m->height - 1, std::max(0.0, v));
    const A_long x0 = (A_long)u, y0 = (A_long)v;
    const A_long x1 = std::min(x0 + 1, m->width - 1), y1 = std::min(y0 + 1, m->height - 1);
    const float fx = (float)(u - x0), fy = (float)(v - y0);
    const Pix* r0 = RowPtr<Pix>(m, y0);
    const Pix* r1 = RowPtr<Pix>(m, y1);
    const float a = LayerValue(r0[x0], channel) + (LayerValue(r0[x1], channel) - LayerValue(r0[x0], channel)) * fx;
    const float b = LayerValue(r1[x0], channel) + (LayerValue(r1[x1], channel) - LayerValue(r1[x0], channel)) * fx;
    return std::min(1.f, std::max(0.f, a + (b - a) * fy));
}

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
std::atomic<int> g_active{0};
struct ActiveGuard
{
    ActiveGuard() { g_active.fetch_add(1); }
    ~ActiveGuard() { g_active.fetch_sub(1); }
};
int RenderThreads()
{
    const unsigned hw = std::thread::hardware_concurrency();
    const int n = hw > 0 ? (int)hw : 4;
    return std::max(1, n / std::max(1, g_active.load()));
}

PF_Err ReportError(PF_OutData* out_data, PF_Err err, const char* what)
{
    if (out_data && err != PF_Err_NONE && err != PF_Err_OUT_OF_MEMORY && err != PF_Interrupt_CANCEL)
    {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d: %s", SK_NAME, SK_MAJOR_VERSION,
                      SK_MINOR_VERSION, obs::T(what));
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
    std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d.%d (%s)\r%s\r%s", SK_NAME, SK_MAJOR_VERSION,
                  SK_MINOR_VERSION, SK_BUG_VERSION, __DATE__,
                  obs::T("Texture, line-keeping blur and watercolor in one effect (prototype)."),
                  obs::T("Lines are detected as ink; fills are processed under them. CPU."));
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    EnsureLanguage(in_data);
    if (in_data->appl_id != 'FXTC')
    {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %s", SK_NAME, obs::T("runs in After Effects only."));
        return PF_Err_INVALID_CALLBACK;
    }
    out_data->my_version = PF_VERSION(SK_MAJOR_VERSION, SK_MINOR_VERSION, SK_BUG_VERSION, SK_STAGE, SK_BUILD_VERSION);
    out_data->out_flags = kOutFlags;
    out_data->out_flags2 = kOutFlags2;
    return PF_Err_NONE;
}

PF_Err AddFloat(PF_InData* in_data, int idx, const char* name, double vmin, double vmax, double smin, double smax, double dflt, int prec,
                bool pct)
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
    def.u.fs_d.display_flags = pct ? PF_ValueDisplayFlag_PERCENT : PF_ValueDisplayFlag_NONE;
    def.u.fs_d.curve_tolerance = AEFX_DEFAULT_CURVE_TOLERANCE;
    def.flags = IsStyleTarget(idx) ? PF_ParamFlag_SUPERVISE : 0;
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddPopup(PF_InData* in_data, int idx, const char* name, int choices, int dflt, const char* items, PF_ParamFlags flags = 0)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POPUP;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.pd.num_choices = (A_short)choices;
    def.u.pd.dephault = def.u.pd.value = (A_short)dflt;
    def.u.pd.u.namesptr = items;
    def.flags = flags | (IsStyleTarget(idx) ? PF_ParamFlag_SUPERVISE : 0);
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddCheck(PF_InData* in_data, int idx, const char* name, bool dflt)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_CHECKBOX;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.bd.value = def.u.bd.dephault = dflt ? 1 : 0;
    def.u.bd.u.nameptr = "";
    def.flags = IsStyleTarget(idx) ? PF_ParamFlag_SUPERVISE : 0;
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddTopic(PF_InData* in_data, int idx, const char* name)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.flags = PF_ParamFlag_START_COLLAPSED;
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}
PF_Err EndTopic(PF_InData* in_data, int idx)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}
PF_Err AddLayer(PF_InData* in_data, int idx, const char* name)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_LAYER;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.ld.dephault = PF_LayerDefault_NONE;
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}
PF_Err AddInt(PF_InData* in_data, int idx, const char* name, int vmin, int vmax, int smax, int dflt, PF_ParamUIFlags ui = 0)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_SLIDER;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.sd.valid_min = def.u.sd.slider_min = vmin;
    def.u.sd.valid_max = vmax;
    def.u.sd.slider_max = smax;
    def.u.sd.value = def.u.sd.dephault = dflt;
    def.ui_flags = ui;
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}

// a point (defaults in % of the layer, 16.16 fixed; not restricted to the layer)
PF_Err AddPoint(PF_InData* in_data, int idx, const char* name, double xPct, double yPct)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POINT;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.td.restrict_bounds = FALSE;
    def.u.td.x_dephault = (PF_Fixed)(xPct * 65536.0);
    def.u.td.y_dephault = (PF_Fixed)(yPct * 65536.0);
    def.uu.id = kDiskId[idx];
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddDepthCtl(PF_InData* in_data, int idx)
{
    for (const DepthCtl& c : kDepthFloats)
        if (c.idx == idx)
            return AddFloat(in_data, idx, obs::T(c.name), c.vmin, c.vmax, c.smin, c.smax, c.dflt, 1, c.pct);
    return PF_Err_BAD_CALLBACK_PARAM;
}

// one style-written control by its index (defaults = the default style)
PF_Err AddStyleCtl(PF_InData* in_data, int idx)
{
    const sk::Params& d = sk::kStyles[sk::kDefaultStyle].v;
    for (const FloatCtl& c : kFloats)
        if (c.idx == idx)
            return AddFloat(in_data, idx, obs::T(c.name), c.vmin, c.vmax, c.smin, c.smax, d.*c.m, c.prec, c.pct);
    for (const BoolCtl& c : kBools)
        if (c.idx == idx)
            return AddCheck(in_data, idx, obs::T(c.name), d.*c.m);
    for (const PopupCtl& c : kPopups)
        if (c.idx == idx)
            return AddPopup(in_data, idx, obs::T(c.name), c.choices, FieldToPopup(c, d.*c.m), obs::T(c.items));
    return PF_Err_BAD_CALLBACK_PARAM;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    EnsureLanguage(in_data);
#define TRY(x) \
    if ((err = (x)) != PF_Err_NONE) \
        return err;
    TRY(AddPopup(in_data, P_STYLE, obs::T("Style"), sk::kStyleCount + 1, sk::kDefaultStyle + 2,
                 obs::T("Custom|Texture Only|Blur Fills, Keep Lines|Watercolor (Loose)|Watercolor (Careful)|Line and Wash Sketch|"
                        "Picture Book Watercolor|BG Art (Transparent Watercolor)|Wet-in-Wet|Watercolor (Old Filter)|"
                        "Background Art Boost|Smooth Skin|Soft Lines|Line and Wash (Round 6)|Picture Book (Round 6)|BG Art (Round 6)"),
                 PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY));
    TRY(AddFloat(in_data, P_AMOUNT, obs::T("Amount"), 0, 100, 0, 100, 100, 1, true));
    TRY(AddTopic(in_data, P_TEX_START, obs::T("Texture")));
    for (int i = P_FINE; i <= P_COLORDETAIL; ++i)
        TRY(AddStyleCtl(in_data, i));
    TRY(EndTopic(in_data, P_TEX_END));
    TRY(AddTopic(in_data, P_LINE_START, obs::T("Lines")));
    for (int i = P_PROTECT; i <= P_LINE_STYLE; ++i)
        TRY(AddStyleCtl(in_data, i));
    TRY(EndTopic(in_data, P_LINE_END));
    TRY(AddTopic(in_data, P_WC_START, obs::T("Watercolor")));
    for (int i = P_WC; i <= P_PROTECT_DETAIL; ++i)
        TRY(AddStyleCtl(in_data, i));
    TRY(AddLayer(in_data, P_PAPER_LAYER, obs::T("Paper Layer")));
    TRY(EndTopic(in_data, P_WC_END));
    TRY(AddTopic(in_data, P_MAP_START, obs::T("Map")));
    TRY(AddLayer(in_data, P_MAP_LAYER, obs::T("Map Layer")));
    TRY(AddPopup(in_data, P_MAP_TARGET, obs::T("Map Affects"), 5, 1, obs::T("Whole Effect|Texture|Fill Blur|Watercolor|Lines")));
    TRY(AddPopup(in_data, P_MAP_CHANNEL, obs::T("Map Channel"), 5, CH_LUMA, obs::T("Luminance|Alpha|Red|Green|Blue")));
    TRY(AddCheck(in_data, P_MAP_INVERT, obs::T("Invert Map"), false));
    TRY(AddPopup(in_data, P_MAP_USE, obs::T("Map Use"), 2, MU_STRENGTH, obs::T("Strength Mask|Depth Matte")));
    TRY(AddDepthCtl(in_data, P_MAP_BLUR));
    TRY(AddDepthCtl(in_data, P_DEPTH_REF));
    TRY(AddDepthCtl(in_data, P_DEPTH_RANGE));
    TRY(AddCheck(in_data, P_DEPTH_PICK, obs::T("Pick Depth at Point"), false));
    TRY(AddPoint(in_data, P_DEPTH_POINT, obs::T("Depth Point"), 50, 50));
    for (int i = P_DEPTH_MASTER; i <= P_D_BLUR; ++i)
        TRY(AddDepthCtl(in_data, i));
    TRY(EndTopic(in_data, P_MAP_END));
    TRY(AddPopup(in_data, P_EDGE, obs::T("Layer Edges"), 2, 1, obs::T("Repeat Edge|Paper Outside Layer")));
    TRY(AddPopup(in_data, P_SPACE, obs::T("Working Space"), 2, 1, obs::T("Gamma (Normal)|Linear")));
    static_assert(sk::V_COUNT == 11, "View popup items: one per sk::View");
    TRY(AddPopup(in_data, P_VIEW, obs::T("View"), 11, 1,
                 obs::T("Result|Line Mask|Fine Detail|Mid Detail|Large Light/Dark|Edge Darkening|Fill (Lines Removed)|Brush Strokes|Glazes|"
                        "Map|Depth Weight")));
    TRY(AddInt(in_data, P_SEED, obs::T("Seed"), 0, 100000, 1000, 0));
    TRY(AddCheck(in_data, P_ANIMATE, obs::T("Move Paper"), false));
    TRY(AddInt(in_data, P_HOLD, obs::T("Every N Frames"), 1, 120, 12, 2));
    TRY(AddInt(in_data, P_RENDER_VERSION, obs::T("Render Version"), 1, 100, 10, 5, PF_PUI_INVISIBLE));
#undef TRY
    out_data->num_params = P_NUM;
    return err;
}

// Style popup: write its values (one undo step); a hand edit of a written value turns the popup to Custom.
// AE does not send USER_CHANGED_PARAM for values the effect itself changes, so this cannot loop.
PF_Err UserChangedParam(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], const PF_UserChangedParamExtra* which)
{
    (void)in_data;
    (void)out_data;
    if (!which || !params)
        return PF_Err_NONE;
    const int i = which->param_index;
    if (i == P_STYLE)
    {
        const int v = params[P_STYLE]->u.pd.value;
        if (v < 2 || v - 2 >= sk::kStyleCount)
            return PF_Err_NONE;
        const sk::Params& s = sk::kStyles[v - 2].v;
        for (const FloatCtl& c : kFloats)
            if (params[c.idx]->u.fs_d.value != (PF_FpShort)(s.*c.m))
            {
                params[c.idx]->u.fs_d.value = (PF_FpShort)(s.*c.m);
                params[c.idx]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
            }
        for (const BoolCtl& c : kBools)
            if ((params[c.idx]->u.bd.value != 0) != s.*c.m)
            {
                params[c.idx]->u.bd.value = s.*c.m ? 1 : 0;
                params[c.idx]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
            }
        for (const PopupCtl& c : kPopups)
            if (params[c.idx]->u.pd.value != FieldToPopup(c, s.*c.m))
            {
                params[c.idx]->u.pd.value = (A_short)FieldToPopup(c, s.*c.m);
                params[c.idx]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
            }
    }
    else if (IsStyleTarget(i) && params[P_STYLE]->u.pd.value != STYLE_CUSTOM)
    {
        params[P_STYLE]->u.pd.value = STYLE_CUSTOM;
        params[P_STYLE]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    }
    return PF_Err_NONE;
}

PF_Err CheckoutSampled(PF_InData* in_data, PF_PreRenderExtra* extra, const PF_RenderRequest& outReq, int param, int probeId, int id,
                       const RenderPlan& plan, double dsx, double dsy, const PF_LRect& need, LayerSampler& ls)
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
    ERR(extra->cb->checkout_layer(in_data->effect_ref, param, probeId, &probe, in_data->current_time, in_data->time_step,
                                  in_data->time_scale, &pr));
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
    ERR(extra->cb->checkout_layer(in_data->effect_ref, param, id, &req, in_data->current_time, in_data->time_step, in_data->time_scale, &res));
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
    const double dsx = Ratio(in_data->downsample_x), dsy = Ratio(in_data->downsample_y);
    const double ds = std::sqrt(dsx * dsy);
    plan->ds = ds;
    const PF_RenderRequest& outReq = extra->input->output_request;

    // the input, always checked out first (probe: its size and max rect)
    PF_CheckoutResult pr;
    AEFX_CLR_STRUCT(pr);
    {
        PF_RenderRequest probe = outReq;
        probe.rect.left = probe.rect.top = 0;
        probe.rect.right = probe.rect.bottom = 1;
        probe.channel_mask = PF_ChannelMask_ARGB;
        probe.preserve_rgb_of_zero_alpha = false;
        ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, CO_INPUT_PROBE, &probe, in_data->current_time, in_data->time_step,
                                      in_data->time_scale, &pr));
        if (err)
            return err;
    }
    PF_LRect layer = {0, 0, ScaledExtent(pr.ref_width, dsx), ScaledExtent(pr.ref_height, dsy)};
    if (RectEmpty(layer))
        layer = pr.max_result_rect;
    plan->layerBounds = layer;
    const PF_LRect content = RectEmpty(pr.max_result_rect) ? layer : pr.max_result_rect;
    const PF_LRect result = Intersect(outReq.rect, content);
    plan->result = result;
    extra->output->max_result_rect = content;
    extra->output->result_rect = result;
    extra->output->solid = false;
    extra->output->flags = 0;
    plan->passthrough = sk::IsIdentity(plan->s.p);
    plan->emptyResult = RectEmpty(result);
    if (plan->emptyResult)
        return PF_Err_NONE;  // the probe above is the PreRender input checkout
    PF_LRect domain = content;
    if (plan->passthrough)
        domain = result;
    else if (Area(domain) > 40.0e6)
    {
        // huge upstream bounds: the request plus a generous margin (filters here reach far less)
        const A_long m = (A_long)std::ceil(std::max(256.0, 768.0 * ds));
        domain = Intersect(Grow(result, m), content);
    }
    plan->domain = domain;
    PF_RenderRequest req = outReq;
    req.rect = domain;
    req.channel_mask = PF_ChannelMask_ARGB;
    req.preserve_rgb_of_zero_alpha = false;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, CO_INPUT, &req, in_data->current_time, in_data->time_step,
                                  in_data->time_scale, &in_result));
    if (err)
        return err;
    plan->inputChecked = true;
    if (!plan->passthrough)
    {
        ERR(CheckoutSampled(in_data, extra, outReq, P_MAP_LAYER, CO_MAP_PROBE, CO_MAP, *plan, dsx, dsy, domain, plan->map));
        if (!err && plan->s.p.wc > 0)
            ERR(CheckoutSampled(in_data, extra, outReq, P_PAPER_LAYER, CO_PAPER_PROBE, CO_PAPER, *plan, dsx, dsy, domain, plan->paper));
    }
    return err;
}

template <typename Pix>
PF_Err RenderDepth(PF_InData* in_data, PF_OutData* out_data, const RenderPlan& plan, PF_EffectWorld* input, PF_EffectWorld* mapW,
                   PF_EffectWorld* paperW, PF_EffectWorld* output)
{
    typedef Traits<Pix> T;
    const PF_LRect outR = {output->origin_x, output->origin_y, output->origin_x + output->width, output->origin_y + output->height};
    if (input && (!input->data || input->width <= 0 || input->height <= 0))
        input = nullptr;
    auto inputAt = [&](A_long lx, A_long ly) -> const Pix* {
        if (input && lx >= input->origin_x && lx < input->origin_x + input->width && ly >= input->origin_y &&
            ly < input->origin_y + input->height)
            return RowPtr<Pix>(input, ly - input->origin_y) + (lx - input->origin_x);
        return nullptr;
    };
    if (plan.passthrough)
    {
        // nothing would change: the input pixels, bit exact
        for (A_long oy = 0; oy < output->height; ++oy)
        {
            Pix* d = RowPtrW<Pix>(output, oy);
            for (A_long ox = 0; ox < output->width; ++ox, ++d)
            {
                const Pix* s = inputAt(outR.left + ox, outR.top + oy);
                if (s)
                    *d = *s;
                else
                    std::memset(d, 0, sizeof(Pix));
            }
        }
        return PF_Err_NONE;
    }
    const PF_LRect D = plan.domain;
    const A_long W = D.right - D.left, H = D.bottom - D.top;
    const size_t N = (size_t)W * H;
    std::vector<float> rgba, map, paper;  // the engine writes its result into rgba (in place)
    try
    {
        rgba.resize(N * 4);
        if (mapW)
            map.resize(N);
        if (paperW)
            paper.resize(N);
    }
    catch (const std::bad_alloc&)
    {
        return PF_Err_OUT_OF_MEMORY;
    }
    ActiveGuard g;
    const int threads = RenderThreads();
    ParallelRows(H, threads, [&](A_long y0, A_long y1) {
        for (A_long y = y0; y < y1; ++y)
            for (A_long x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                const A_long lx = D.left + x, ly = D.top + y;
                float* q = &rgba[i * 4];
                const Pix* s = inputAt(lx, ly);
                if (s)
                {
                    q[0] = SanitizeF(T::In(s->red));
                    q[1] = SanitizeF(T::In(s->green));
                    q[2] = SanitizeF(T::In(s->blue));
                    q[3] = std::min(1.f, std::max(0.f, SanitizeF(T::In(s->alpha))));
                }
                else
                    q[0] = q[1] = q[2] = q[3] = 0.f;
                if (mapW)
                {
                    float m = SampleLayer<Pix>(mapW, plan, plan.map, lx, ly, plan.s.mapChannel);
                    map[i] = plan.s.mapInvert ? 1.f - m : m;
                }
                if (paperW)
                    paper[i] = SampleLayer<Pix>(paperW, plan, plan.paper, lx, ly, CH_LUMA);
            }
    });
    sk::Frame f;
    f.w = W;
    f.h = H;
    f.ox = D.left;
    f.oy = D.top;
    f.ds = plan.ds;
    f.rgba = rgba.data();
    f.map = mapW ? map.data() : nullptr;
    f.paperTex = paperW ? paper.data() : nullptr;
    f.layer = sk::Rect{(int)(plan.layerBounds.left - D.left), (int)(plan.layerBounds.top - D.top), (int)(plan.layerBounds.right - D.left),
                       (int)(plan.layerBounds.bottom - D.top)};
    f.out = rgba.data();
    f.threads = threads;
    PF_Err abortErr = PF_Err_NONE;
    f.abort = [&]() {
        if (!abortErr)
            abortErr = PF_ABORT(in_data);
        return abortErr != PF_Err_NONE;
    };
    if ((abortErr = PF_ABORT(in_data)) != PF_Err_NONE)
        return abortErr;
    const sk::Status st = sk::Render(plan.s.p, f);
    switch (st)
    {
    case sk::ST_OK: break;
    case sk::ST_ABORTED: return abortErr ? abortErr : PF_Interrupt_CANCEL;
    case sk::ST_OOM: return PF_Err_OUT_OF_MEMORY;
    default: return ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render error. The frame was not rendered.");
    }
    ParallelRows(output->height, threads, [&](A_long y0, A_long y1) {
        for (A_long oy = y0; oy < y1; ++oy)
        {
            Pix* d = RowPtrW<Pix>(output, oy);
            const A_long ly = outR.top + oy;
            for (A_long ox = 0; ox < output->width; ++ox, ++d)
            {
                const A_long lx = outR.left + ox;
                if (lx < D.left || lx >= D.right || ly < D.top || ly >= D.bottom)
                {
                    std::memset(d, 0, sizeof(Pix));
                    continue;
                }
                const float* s = &rgba[((size_t)(ly - D.top) * W + (lx - D.left)) * 4];
                d->red = T::Out(SanitizeF(s[0]));
                d->green = T::Out(SanitizeF(s[1]));
                d->blue = T::Out(SanitizeF(s[2]));
                d->alpha = T::Out(std::min(1.f, std::max(0.f, s[3])));
            }
        }
    });
    return PF_Err_NONE;
}

// NON_PARAM_VARY only while "Move Paper" is on
PF_Err QueryDynamicFlags(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_ParamDef anim;
    AEFX_CLR_STRUCT(anim);
    ERR(PF_CHECKOUT_PARAM(in_data, P_ANIMATE, in_data->current_time, in_data->time_step, in_data->time_scale, &anim));
    const bool outA = !err;
    if (!err && !anim.u.bd.value)
        out_data->out_flags &= ~PF_OutFlag_NON_PARAM_VARY;
    else
        out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
    if (outA)
        ERR2(PF_CHECKIN_PARAM(in_data, &anim));
    return err2;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const RenderPlan* plan = reinterpret_cast<const RenderPlan*>(extra->input->pre_render_data);
    if (!plan)
        return PF_Err_BAD_CALLBACK_PARAM;
    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    PF_EffectWorld* mapW = nullptr;
    PF_EffectWorld* paperW = nullptr;
    // AE 2026 rule (25 :: 244): the input pixels BEFORE the output, on every path (the probe id when the
    // result is empty: it is the input checkout PreRender registered)
    const A_long inId = plan->inputChecked ? CO_INPUT : CO_INPUT_PROBE;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, inId, &input));
    const bool inOut = !err;
    bool mapOut = false, paperOut = false;
    if (!err && !plan->emptyResult && plan->map.has)
    {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CO_MAP, &mapW));
        mapOut = !err;
        if (!err && mapW && (!mapW->data || mapW->width <= 0 || mapW->height <= 0))
            mapW = nullptr;
    }
    if (!err && !plan->emptyResult && plan->paper.has)
    {
        ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CO_PAPER, &paperW));
        paperOut = !err;
        if (!err && paperW && (!paperW->data || paperW->width <= 0 || paperW->height <= 0))
            paperW = nullptr;
    }
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && output && output->data && output->width > 0 && output->height > 0)
    {
        if (plan->emptyResult)
        {
            const size_t ps = extra->input->bitdepth == 32 ? sizeof(PF_PixelFloat) : extra->input->bitdepth == 16 ? sizeof(PF_Pixel16) : sizeof(PF_Pixel8);
            for (A_long y = 0; y < output->height; ++y)
                std::memset((char*)output->data + (size_t)y * output->rowbytes, 0, ps * (size_t)output->width);
        }
        else
        {
            try
            {
                switch (extra->input->bitdepth)
                {
                case 32: err = RenderDepth<PF_PixelFloat>(in_data, out_data, *plan, input, mapW, paperW, output); break;
                case 16: err = RenderDepth<PF_Pixel16>(in_data, out_data, *plan, input, mapW, paperW, output); break;
                case 8: err = RenderDepth<PF_Pixel8>(in_data, out_data, *plan, input, mapW, paperW, output); break;
                default: err = PF_Err_BAD_CALLBACK_PARAM; break;
                }
            }
            catch (const std::bad_alloc&)
            {
                err = PF_Err_OUT_OF_MEMORY;
            }
            catch (...)
            {
                err = ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render error. The frame was not rendered.");
            }
        }
    }
    if (paperOut)
        ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, CO_PAPER));
    if (mapOut)
        ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, CO_MAP));
    if (inOut)
        ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, inId));
    return err ? err : err2;
}

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* inSPBasicSuitePtr, const char* inHostName, const char* inHostVersion)
{
    (void)inSPBasicSuitePtr;
    (void)inHostName;
    (void)inHostVersion;
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, SK_NAME, SK_MATCH_NAME, SK_CATEGORY, AE_RESERVED_INFO, "EffectMain",
                                     SK_SUPPORT_URL);
    return result;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra)
{
    (void)output;
    PF_Err err = PF_Err_NONE;
    try
    {
        switch (cmd)
        {
        case PF_Cmd_ABOUT: err = About(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETDOWN: sk::GpuShutdown(); break;
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data); break;
        case PF_Cmd_USER_CHANGED_PARAM:
            err = UserChangedParam(in_data, out_data, params, reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
            break;
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
    catch (...)
    {
        err = ReportError(out_data, PF_Err_BAD_CALLBACK_PARAM, "internal render error. The frame was not rendered.");
    }
    return err;
}
