// Tebure Shake (prototype) - After Effects SmartFX wrapper around the camera-shake engine (tb_model / tb_plan /
// tb_warp). Skeleton from Shitsukan / MapBilinearBlur (strings by AE version, MFR thread split, style popup that
// writes values, SmartFX checkout order: input pixels BEFORE the output on every path).
//
// Time: every value is a function of (layer time, params, seed). Params are read at OTHER times (speed / rhythm
// integration, the impact keyframes, the shot start = last keyframe of Preset / Cut) through PF_CHECKOUT_PARAM
// and the PF Param Utils suite (keyframe count / times); PF_OutFlag_WIDE_TIME_INPUT +
// PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT tell AE about it. Nothing is cached between frames (MFR safe).
// Coordinates: buffer px = downsampled layer px (SmartFX rects); point params arrive in buffer px.
// AE worlds are straight (unpremultiplied) ARGB at 8/16/32 bpc.
#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"
#include "platform/plat_math.h"  // tbm:: = the libm calls that feed the picture (forward to std::)
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectSuites.h"

#include "TebureConfig.h"
#include "tb_model.h"
#include "tb_plan.h"
#include "tb_presets.h"
#include "tb_strings.h"
#include "tb_thumbs.h"
#include "adobesdk/DrawbotSuite.h"
#include "tb_warp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <string>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

extern "C" DllExport PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra);

const A_long kOutFlags = PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_DEEP_COLOR_AWARE |
                         PF_OutFlag_CUSTOM_UI;
const A_long kOutFlags2 = PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG | PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |
                          PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;
static_assert(TB_OUTFLAGS == kOutFlags, "PiPL out_flags mismatch");
static_assert(TB_OUTFLAGS2 == kOutFlags2, "PiPL out_flags2 mismatch");
#define TB_STAGE PF_Stage_DEVELOP
static_assert(TB_PIPL_VERSION == PF_VERSION(TB_MAJOR_VERSION, TB_MINOR_VERSION, TB_BUG_VERSION, TB_STAGE, TB_BUILD_VERSION),
              "PiPL version mismatch");

namespace {

// ------------------------------------------------------------------ profiling (TB_PROFILE builds only, never shipped)
#ifdef TB_PROFILE
std::mutex g_profM;
std::map<std::string, std::pair<double, long long>> g_prof;
struct ProfScope
{
    const char* n;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ProfScope(const char* name) : n(name) {}
    ~ProfScope()
    {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lk(g_profM);
        auto& e = g_prof[n];
        e.first += ms;
        e.second++;
    }
};
#define TB_CAT2(a, b) a##b
#define TB_CAT(a, b) TB_CAT2(a, b)
#define TB_PROF(name) ProfScope TB_CAT(prof_, __LINE__)(name)
#else
#define TB_PROF(name) (void)0
#endif

// ------------------------------------------------------------------ params (index == disk ID; append only)
enum
{
    // UI order (0.3). Disk ids (kDiskId) keep the 0.2 meaning; 105-108 are new. Append new controls with new ids.
    P_INPUT = 0,  // id 0
    P_PRESET,  // id 1
    P_PICKER,  // id 2
    P_AMOUNT,  // id 3
    P_PERIOD,  // id 105
    P_FOCAL,  // id 5
    P_FRRATE,  // id 6
    P_TRIGGER,  // id 7
    P_EDGE,  // id 8
    P_ZOOM_MODE,  // id 106
    P_SEED,  // id 9
    P_ADJ_START,  // id 10
    P_MULX,  // id 11
    P_MULY,  // id 12
    P_BALPOS,  // id 13
    P_MULROT,  // id 14
    P_BALROT,  // id 15
    P_MULZOOM,  // id 16
    P_BALZOOM,  // id 17
    P_ADJ_END,  // id 18
    P_HAND_START,  // id 19
    P_HAND_DEG,  // id 20
    P_HAND_HZ,  // id 21
    P_HAND_Q,  // id 22
    P_HAND_ROUGH,  // id 23
    P_HAND_IRR,  // id 24
    P_HAND_YHZ,  // id 25
    P_HAND_YX,  // id 26
    P_HAND_ROLL,  // id 27
    P_BODY_PX,  // id 28
    P_BODY_HZ,  // id 29
    P_TREMOR,  // id 30
    P_BREATH,  // id 31
    P_BREATH_HZ,  // id 32
    P_DRIFT,  // id 33
    P_CORR_INT,  // id 34
    P_HAND_END,  // id 35
    P_WALK_START,  // id 36
    P_CADENCE,  // id 37
    P_BOUNCE,  // id 38
    P_SWAY,  // id 39
    P_WALK_ROLL,  // id 40
    P_WALK_PAN,  // id 41
    P_HEEL,  // id 42
    P_HEEL_SHARP,  // id 43
    P_GAIT_VAR,  // id 44
    P_PULSE,  // id 45
    P_WALK_END,  // id 46
    P_VIB_START,  // id 47
    P_VIB_PX,  // id 48
    P_VIB_HZ,  // id 49
    P_VIB_BAND,  // id 50
    P_BUMP_PX,  // id 51
    P_BUMP_RATE,  // id 52
    P_BIG_PX,  // id 53
    P_BIG_HZ,  // id 54
    P_BIG_ROT,  // id 55
    P_BIG_ZOOM,  // id 56
    P_SWELL,  // id 57
    P_QUAKE_ENV,  // id 58
    P_VIB_END,  // id 59
    P_IMP_START,  // id 60
    P_FIRST_HIT,  // id 61
    P_HIT_EVERY,  // id 62
    P_DECAY,  // id 63
    P_IMP_TYPE,  // id 64
    P_IMP_PX,  // id 65
    P_IMP_ROT,  // id 66
    P_IMP_ZOOM,  // id 67
    P_IMP_HZ,  // id 68
    P_IMP_GRIT,  // id 69
    P_IMP_ANGLE,  // id 70
    P_IMP_RANDOM,  // id 71
    P_IMP_END,  // id 72
    P_JOLT_START,  // id 73
    P_JOLT_PX,  // id 74
    P_JOLT_INT,  // id 75
    P_JOLT_DUR,  // id 76
    P_JOLT_SETTLE,  // id 77
    P_JOLT_MODE,  // id 78
    P_JOLT_YX,  // id 79
    P_JOLT_END,  // id 80
    P_FILM_START,  // id 81
    P_WEAVE,  // id 82
    P_JITTER,  // id 83
    P_JITTER_YX,  // id 84
    P_FILM_END,  // id 85
    P_MB_START,  // id 86
    P_SHUTTER,  // id 87
    P_SAMPLES,  // id 88
    P_LINEAR,  // id 89
    P_MB_END,  // id 90
    P_CUT_START,  // id 91
    P_CUT,  // id 92
    P_MOOD,  // id 93
    P_LOOP,  // id 94
    P_CUT_END,  // id 95
    P_MISC_START,  // id 96
    P_VIEW,  // id 97
    P_HOLDN,  // id 98
    P_INTERP,  // id 99
    P_SNAP,  // id 100
    P_CENTER,  // id 101
    P_REFW,  // id 102
    P_ZOOM_FIX,  // id 107
    P_ZOOM_SMOOTH,  // id 108
    P_MISC_END,  // id 103
    P_SPEED,  // id 4
    P_RENDER_VERSION,  // id 104
    P_NUM
};
static_assert(P_NUM == 109, "param list changed: update tests\\fakehost.cpp and aetest\\build_tb.jsx");
const int kDiskId[P_NUM] = {0, 1, 2, 3, 105, 5, 6, 7, 8, 106, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 107, 108, 103, 4, 104};
inline int DiskId(int idx) { return kDiskId[idx]; }

enum { PRESET_CUSTOM = 1 };
enum { ZOOM_AUTO = 1, ZOOM_FIXED, ZOOM_NONE, ZOOM_LEGACY, ZOOM_COUNT = 4 };
enum { VIEW_RESULT = 1, VIEW_GRAPH };
enum { CO_INPUT_PROBE = 1, CO_INPUT };

// preset-written controls in ONE table: ParamsSetup (defaults), ReadShape, the Preset popup and "edit -> Custom"
enum CtlKind { K_FLOAT = 0, K_INT, K_POPUP };
struct Ctl
{
    int idx;
    int kind;
    double tb::Shape::*md;
    int tb::Shape::*mi;
    const char* name;
    double vmin, vmax, smin, smax;
    int prec;
    double scale;  // UI value = shape value * scale
    bool pct;      // percent display
    int choices;
    const char* items;
};
#define FD(I, M, N, A, B, C, D, P, S, PC) {I, K_FLOAT, &tb::Shape::M, nullptr, N, A, B, C, D, P, S, PC, 0, nullptr}
const Ctl kCtls[] = {
    FD(P_FOCAL, focal, "Focal Length (mm)", 8, 1200, 12, 300, 1, 1, false),
    {P_FRRATE, K_POPUP, nullptr, &tb::Shape::frRate, "Shake Frame Rate", 0, 0, 0, 0, 0, 1, false, tb::FR_COUNT,
     "Same as Comp|24 fps|18 fps|16 fps|12 fps|8 fps|Hold Frames"},
    {P_HOLDN, K_INT, nullptr, &tb::Shape::holdN, "Hold Frames", 1, 24, 1, 8, 0, 1, false, 0, nullptr},
    FD(P_HAND_DEG, handDeg, "Hand Sway (deg)", 0, 20, 0, 1, 3, 1, false),
    FD(P_HAND_HZ, handHz, "Sway Frequency (Hz)", 0.02, 20, 0.1, 5, 2, 1, false),
    FD(P_HAND_Q, handQ, "Sway Resonance", 0.3, 20, 0.3, 10, 2, 1, false),
    FD(P_HAND_ROUGH, handRough, "Roughness", -100, 100, -100, 100, 0, 100, true),
    FD(P_HAND_IRR, handIrr, "Irregularity", 0, 100, 0, 100, 0, 100, true),
    FD(P_HAND_YHZ, handYHz, "Vertical Frequency", 12.5, 800, 25, 300, 0, 100, true),
    FD(P_HAND_YX, handYX, "Vertical Amount", 0, 1000, 0, 200, 0, 100, true),
    FD(P_HAND_ROLL, handRoll, "Roll Share", 0, 1000, 0, 200, 0, 100, true),
    FD(P_BODY_PX, bodyPx, "Body Sway (px)", 0, 2000, 0, 50, 2, 1, false),
    FD(P_BODY_HZ, bodyHz, "Body Sway Freq (Hz)", 0.01, 10, 0.05, 2, 2, 1, false),
    FD(P_TREMOR, tremorDeg, "Tremor (deg)", 0, 2, 0, 0.05, 4, 1, false),
    FD(P_BREATH, breathDeg, "Breathing (deg)", 0, 5, 0, 0.5, 3, 1, false),
    FD(P_BREATH_HZ, breathHz, "Breathing Rate (Hz)", 0.02, 3, 0.05, 1, 2, 1, false),
    FD(P_DRIFT, driftDeg, "Drift (deg)", 0, 20, 0, 1, 3, 1, false),
    FD(P_CORR_INT, corrInt, "Correction Interval (s)", 0.2, 60, 0.5, 10, 2, 1, false),
    FD(P_CADENCE, cadence, "Steps / Beats per s", 0.1, 12, 0.5, 4, 2, 1, false),
    FD(P_BOUNCE, bouncePx, "Bounce (px)", 0, 1000, 0, 60, 1, 1, false),
    FD(P_SWAY, swayPx, "Side Sway (px)", 0, 1000, 0, 60, 1, 1, false),
    FD(P_WALK_ROLL, walkRoll, "Walk Roll (deg)", 0, 45, 0, 3, 2, 1, false),
    FD(P_WALK_PAN, walkPan, "Walk Pan (deg)", 0, 45, 0, 2, 2, 1, false),
    FD(P_HEEL, heelPx, "Heel Strike (px)", 0, 1000, 0, 40, 1, 1, false),
    FD(P_HEEL_SHARP, heelSharp, "Running Shape", 0, 100, 0, 100, 0, 100, true),
    FD(P_GAIT_VAR, gaitVar, "Step Variation", 0, 100, 0, 100, 0, 100, true),
    FD(P_PULSE, pulsePct, "Heartbeat Pulse (%)", 0, 50, 0, 3, 2, 1, false),
    FD(P_FIRST_HIT, firstHit, "Impact at (s)", -1, 3600, -1, 10, 2, 1, false),
    FD(P_HIT_EVERY, hitEvery, "Random Impacts Every (s)", 0, 600, 0, 20, 2, 1, false),
    FD(P_DECAY, decay, "Decay (s)", 0, 60, 0, 3, 2, 1, false),
    {P_IMP_TYPE, K_POPUP, nullptr, &tb::Shape::impType, "Impact Type", 0, 0, 0, 0, 0, 1, false, tb::IMP_COUNT,
     "Damped Shake|Gamen-do (Alternate)|Whip"},
    FD(P_IMP_PX, impPx, "Impact Size (px)", 0, 4000, 0, 200, 1, 1, false),
    FD(P_IMP_ROT, impRot, "Impact Rotation (deg)", 0, 90, 0, 10, 2, 1, false),
    FD(P_IMP_ZOOM, impZoom, "Impact Zoom (%)", 0, 100, 0, 10, 2, 1, false),
    FD(P_IMP_HZ, impHz, "Impact Frequency (Hz)", 0.1, 60, 0.5, 20, 2, 1, false),
    FD(P_IMP_GRIT, impGrit, "Grit", 0, 300, 0, 100, 0, 100, true),
    FD(P_IMP_ANGLE, impAngle, "Impact Direction (deg)", -180, 180, -180, 180, 1, 1, false),
    FD(P_IMP_RANDOM, impRandom, "Randomness", 0, 100, 0, 100, 0, 100, true),
    FD(P_JOLT_PX, joltPx, "Jolt Size (px)", 0, 4000, 0, 200, 1, 1, false),
    FD(P_JOLT_INT, joltInt, "Jolt Interval (s)", 0.05, 120, 0.2, 10, 2, 1, false),
    FD(P_JOLT_DUR, joltDur, "Jolt Duration (s)", 0.01, 5, 0.02, 1, 3, 1, false),
    FD(P_JOLT_SETTLE, joltSettle, "Jolt Overshoot", 0, 60, 0, 60, 0, 100, true),
    {P_JOLT_MODE, K_POPUP, nullptr, &tb::Shape::joltMode, "Jolt Direction", 0, 0, 0, 0, 0, 1, false, tb::JOLT_COUNT,
     "Free|Horizontal|Vertical|Alternate Axes"},
    FD(P_JOLT_YX, joltYX, "Jolt Vertical Ratio", 0, 1000, 0, 200, 0, 100, true),
    FD(P_VIB_PX, vibPx, "Vibration (px)", 0, 500, 0, 10, 2, 1, false),
    FD(P_VIB_HZ, vibHz, "Vibration Freq (Hz)", 0.2, 120, 1, 40, 1, 1, false),
    FD(P_VIB_BAND, vibBand, "Vibration Bandwidth", 1, 300, 1, 100, 0, 100, true),
    FD(P_BUMP_PX, bumpPx, "Bumps (px)", 0, 1000, 0, 60, 1, 1, false),
    FD(P_BUMP_RATE, bumpRate, "Bumps per Second", 0.01, 20, 0.05, 3, 2, 1, false),
    FD(P_BIG_PX, bigPx, "Large Motion (px)", 0, 4000, 0, 200, 1, 1, false),
    FD(P_BIG_HZ, bigHz, "Large Motion Freq (Hz)", 0.005, 20, 0.02, 3, 3, 1, false),
    FD(P_BIG_ROT, bigRot, "Large Rotation (deg)", 0, 90, 0, 10, 2, 1, false),
    FD(P_BIG_ZOOM, bigZoom, "Large Zoom (%)", 0, 100, 0, 10, 2, 1, false),
    FD(P_SWELL, swell, "Swell", 0, 100, 0, 100, 0, 100, true),
    FD(P_QUAKE_ENV, quakeEnv, "Quake Waves (P then S)", 0, 100, 0, 100, 0, 100, true),
    FD(P_WEAVE, weavePx, "Gate Weave (px)", 0, 200, 0, 10, 2, 1, false),
    FD(P_JITTER, jitterPx, "Frame Jitter (px)", 0, 200, 0, 10, 2, 1, false),
    FD(P_JITTER_YX, jitterYX, "Jitter Vertical Ratio", 0, 2000, 0, 400, 0, 100, true),
    FD(P_SHUTTER, shutter, "Shutter Angle", 0, 720, 0, 360, 0, 1, false),
};
#undef FD

const Ctl* CtlOf(int idx)
{
    for (const Ctl& c : kCtls)
        if (c.idx == idx)
            return &c;
    return nullptr;
}
bool IsPresetTarget(int idx) { return CtlOf(idx) != nullptr; }

double UiValue(const Ctl& c, const tb::Shape& s)
{
    if (c.kind == K_FLOAT)
        return s.*(c.md) * c.scale;
    if (c.kind == K_POPUP)
        return s.*(c.mi) + 1;
    return s.*(c.mi);
}

// ------------------------------------------------------------------ helpers
inline double Ratio(const PF_RationalScale& r) { return (r.den != 0 && r.num != 0) ? (double)r.num / (double)r.den : 1.0; }
inline A_long ScaledExtent(A_long fullRes, double ds) { return (A_long)std::ceil(fullRes * ds - 1e-6); }
inline tb::RectI ToR(const PF_LRect& r) { return tb::RectI{(int)r.left, (int)r.top, (int)r.right, (int)r.bottom}; }
inline PF_LRect ToL(const tb::RectI& r)
{
    PF_LRect o;
    o.left = r.l;
    o.top = r.t;
    o.right = r.r;
    o.bottom = r.b;
    return o;
}

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
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d: %s", TB_NAME, TB_MAJOR_VERSION, TB_MINOR_VERSION,
                      obs::T(what));
    }
    return err;
}

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

// ------------------------------------------------------------------ reading params at any time
struct TimeReader
{
    PF_InData* in = nullptr;
    const PF_ParamUtilsSuite3* pu = nullptr;
    double fps = 24;
    PF_Err err = PF_Err_NONE;

    // value of a slider / popup / checkbox at layer time `sec` (s)
    double At(int idx, double sec)
    {
        if (err)
            return 0;
        double scale = (double)in->time_scale;
        if (!(scale > 0))
            scale = 24;
        A_u_long K = 1;
        while (K < 256 && std::fabs(sec) * scale * K * 2 < 1.0e9)
            K *= 2;
        const A_long t = (A_long)std::llround(sec * scale * K);
        const A_long st = (A_long)((in->time_step != 0 ? std::labs(in->time_step) : 1) * K);
        return Checkout(idx, t, st, (A_u_long)(scale * K));
    }
    double Now(int idx) { return Checkout(idx, in->current_time, in->time_step, in->time_scale); }
    double Checkout(int idx, A_long t, A_long st, A_u_long sc)
    {
        PF_ParamDef p;
        AEFX_CLR_STRUCT(p);
        PF_Err e = PF_CHECKOUT_PARAM(in, idx, t, st, sc, &p);
        if (e)
        {
            err = e;
            return 0;
        }
        double v = 0;
        switch (p.param_type)
        {
        case PF_Param_FLOAT_SLIDER: v = p.u.fs_d.value; break;
        case PF_Param_SLIDER: v = p.u.sd.value; break;
        case PF_Param_POPUP: v = p.u.pd.value; break;
        case PF_Param_CHECKBOX: v = p.u.bd.value ? 1 : 0; break;
        default: break;
        }
        PF_CHECKIN_PARAM(in, &p);
        return std::isfinite(v) ? v : 0;
    }
    // keyframe times (s); empty if constant or the suite is missing
    std::vector<double> Keys(int idx)
    {
        std::vector<double> k;
        if (!pu || !pu->PF_GetKeyframeCount || !pu->PF_KeyIndexToTime)
            return k;
        PF_KeyIndex n = 0;
        if (pu->PF_GetKeyframeCount(in->effect_ref, idx, &n) != PF_Err_NONE || n <= 0)
            return k;
        for (PF_KeyIndex i = 0; i < n && i < 100000; ++i)
        {
            A_long kt = 0;
            A_u_long ks = 0;
            if (pu->PF_KeyIndexToTime(in->effect_ref, idx, i, &kt, &ks) == PF_Err_NONE && ks > 0)
                k.push_back((double)kt / (double)ks);
        }
        std::sort(k.begin(), k.end());
        return k;
    }
    // a curve over [a, b]: keys, constant detection (expressions without keys are probed)
    tb::Curve MakeCurve(int idx, double a, double b, double mul, double lo, double hi)
    {
        tb::Curve c;
        c.keys = Keys(idx);
        const double now = std::min(hi, std::max(lo, Now(idx) * mul));
        c.constant = now;
        if (c.keys.size() >= 2)
            c.animated = true;
        else
        {
            // one key (or none): constant unless an expression animates it - probe a few instants
            const double probes[7] = {a, b, 0.5 * (a + b), a + 1 / fps, b - 1 / fps, a + 0.25 * (b - a), a + 0.75 * (b - a)};
            for (double p : probes)
                if (std::min(hi, std::max(lo, At(idx, p) * mul)) != now)
                    c.animated = true;
            c.keys.clear();
        }
        if (c.animated)
            c.at = [this, idx, mul, lo, hi](double t) { return std::min(hi, std::max(lo, At(idx, t) * mul)); };
        return c;
    }
    double MaxOver(const tb::Curve& c, double a, double b)
    {
        if (!c.animated)
            return c.constant;
        double m = std::max(c.at(a), c.at(b));
        std::vector<double> pts{a, b};
        for (double k : c.keys)
            if (k > a && k < b)
                pts.push_back(k);
        std::sort(pts.begin(), pts.end());
        const int sub = c.keys.empty() ? 256 : 12;
        for (size_t i = 0; i + 1 < pts.size(); ++i)
            for (int j = 0; j <= sub; ++j)
                m = std::max(m, c.at(pts[i] + (pts[i + 1] - pts[i]) * j / sub));
        return m;
    }
};

// ------------------------------------------------------------------ the render plan
struct RenderPlan
{
    bool emptyResult = false, passthrough = false, inputChecked = false;
    tb::RectI content, result, src;
    std::vector<tb::Mat3> maps;
    int edge = tb::EDGE_AUTO, interp = tb::INTERP_BICUBIC;
    bool linear = false;
    int view = VIEW_RESULT;
    double ds = 1, focal = 35, widthRef = 1920;
    tb::Pose bound;
    std::vector<double> graphT;
    std::vector<tb::Pose> graph;
};
void DeletePlan(void* p) { delete reinterpret_cast<RenderPlan*>(p); }

PF_Err About(PF_InData* in_data, PF_OutData* out_data)
{
    EnsureLanguage(in_data);
    std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %d.%d.%d (%s)\r%s\r%s", TB_NAME, TB_MAJOR_VERSION,
                  TB_MINOR_VERSION, TB_BUG_VERSION, __DATE__,
                  obs::T("Camera shake with physically modelled hand-held, walking, impact, vehicle and film motion (prototype)."),
                  obs::T("Presets write the detail values; editing one turns the preset to Custom. CPU."));
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data)
{
    EnsureLanguage(in_data);
    if (in_data->appl_id != 'FXTC')
    {
        out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
        std::snprintf(out_data->return_msg, sizeof(out_data->return_msg), "%s %s", TB_NAME, obs::T("runs in After Effects only."));
        return PF_Err_INVALID_CALLBACK;
    }
    out_data->my_version = PF_VERSION(TB_MAJOR_VERSION, TB_MINOR_VERSION, TB_BUG_VERSION, TB_STAGE, TB_BUILD_VERSION);
    out_data->out_flags = kOutFlags;
    out_data->out_flags2 = kOutFlags2;
    return PF_Err_NONE;
}

PF_Err AddFloat(PF_InData* in_data, int idx, const char* name, double vmin, double vmax, double smin, double smax, double dflt, int prec,
                bool pct, PF_ParamFlags flags = 0, PF_ParamUIFlags ui = 0)
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
    def.flags = flags | (IsPresetTarget(idx) ? PF_ParamFlag_SUPERVISE : 0);
    def.ui_flags = ui;
    def.uu.id = DiskId(idx);
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
    def.flags = flags | (IsPresetTarget(idx) ? PF_ParamFlag_SUPERVISE : 0);
    def.uu.id = DiskId(idx);
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
    def.uu.id = DiskId(idx);
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
    def.flags = IsPresetTarget(idx) ? PF_ParamFlag_SUPERVISE : 0;
    def.uu.id = DiskId(idx);
    return PF_ADD_PARAM(in_data, -1, &def);
}
PF_Err AddTopic(PF_InData* in_data, int idx, const char* name)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.flags = PF_ParamFlag_START_COLLAPSED;
    def.uu.id = DiskId(idx);
    return PF_ADD_PARAM(in_data, -1, &def);
}
PF_Err EndTopic(PF_InData* in_data, int idx)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    def.uu.id = DiskId(idx);
    return PF_ADD_PARAM(in_data, -1, &def);
}
PF_Err AddPoint(PF_InData* in_data, int idx, const char* name)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_POINT;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.u.td.restrict_bounds = 0;
    def.u.td.x_value = def.u.td.x_dephault = (PF_Fixed)(50L << 16);  // percent of the layer
    def.u.td.y_value = def.u.td.y_dephault = (PF_Fixed)(50L << 16);
    def.uu.id = DiskId(idx);
    return PF_ADD_PARAM(in_data, -1, &def);
}
// the preset picker: a custom-UI control area on a PF_Param_NO_DATA param (no value, nothing is saved)
const int kPickerNominalWidth = 300, kPickerStatusH = 18;
PF_Err AddPicker(PF_InData* in_data, int idx, const char* name, int n)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_NO_DATA;
    PF_STRNNCPY(def.PF_DEF_NAME, name, sizeof(def.PF_DEF_NAME));
    def.ui_flags = PF_PUI_CONTROL | PF_PUI_DONT_ERASE_CONTROL;
    def.ui_width = (A_short)kPickerNominalWidth;
    def.ui_height = (A_short)(tbthumb::PreferredHeight(n, (float)kPickerNominalWidth) + kPickerStatusH);
    def.uu.id = DiskId(idx);
    return PF_ADD_PARAM(in_data, -1, &def);
}

PF_Err AddCtl(PF_InData* in_data, int idx)
{
    const Ctl* c = CtlOf(idx);
    if (!c)
        return PF_Err_BAD_CALLBACK_PARAM;
    const tb::Shape& d = tb::Presets(nullptr)[tb::DefaultPreset()].s;
    const double v = UiValue(*c, d);
    switch (c->kind)
    {
    case K_FLOAT: return AddFloat(in_data, idx, obs::T(c->name), c->vmin, c->vmax, c->smin, c->smax, v, c->prec, c->pct);
    case K_POPUP: return AddPopup(in_data, idx, obs::T(c->name), c->choices, (int)v, obs::T(c->items));
    default: return AddInt(in_data, idx, obs::T(c->name), (int)c->vmin, (int)c->vmax, (int)c->smax, (int)v);
    }
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    EnsureLanguage(in_data);
#define TRY(x) \
    if ((err = (x)) != PF_Err_NONE) \
        return err;
    int np = 0;
    tb::Presets(&np);
    // ---- the short top level
    TRY(AddPopup(in_data, P_PRESET, obs::T("Preset"), np + 1, tb::DefaultPreset() + 2, obs::T(TB_PRESET_ITEMS), PF_ParamFlag_SUPERVISE));
    TRY(AddPicker(in_data, P_PICKER, obs::T("Preset Paths"), np));
    // the user's two master dials (never written by presets, never turn the preset to Custom): amplitude (id 3, the 0.2
    // "Amount") and period (new id 105: 100 % = the preset's own timing, larger = slower; slider 10..200 so the fast
    // side has travel, typed values up to 400 %)
    TRY(AddFloat(in_data, P_AMOUNT, obs::T("Shake Amplitude"), 0, 1000, 0, 200, 100, 1, true));
    TRY(AddFloat(in_data, P_PERIOD, obs::T("Period"), 10, 400, 10, 200, 100, 0, true));
    TRY(AddCtl(in_data, P_FOCAL));
    TRY(AddCtl(in_data, P_FRRATE));
    TRY(AddFloat(in_data, P_TRIGGER, obs::T("Impact Trigger"), 0, 1000, 0, 100, 0, 1, true));
    TRY(AddPopup(in_data, P_EDGE, obs::T("Edges"), tb::EDGE_COUNT, 1,
                 obs::T("Auto Scale to Hide|Repeat Edge|Mirror|Transparent (Overflow)|Transparent (In Layer)")));
    {
        // zoom used by "Auto Scale to Hide": new instances follow the amplitude over time; projects saved by 0.2 (which
        // lack this param) get "Constant per Shot (0.2)" = exactly the 0.2 behaviour (USE_VALUE_FOR_OLD_PROJECTS)
        PF_ParamDef def;
        AEFX_CLR_STRUCT(def);
        def.param_type = PF_Param_POPUP;
        PF_STRNNCPY(def.PF_DEF_NAME, obs::T("Zoom to Hide Edges"), sizeof(def.PF_DEF_NAME));
        def.u.pd.num_choices = ZOOM_COUNT;
        def.u.pd.dephault = ZOOM_AUTO;
        def.u.pd.value = ZOOM_LEGACY;
        def.u.pd.u.namesptr = obs::T("Auto (Follow Amplitude)|Fixed %|None|Constant per Shot (0.2)");
        def.flags = PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
        def.uu.id = DiskId(P_ZOOM_MODE);
        TRY(PF_ADD_PARAM(in_data, -1, &def));
    }
    TRY(AddInt(in_data, P_SEED, obs::T("Seed"), 0, 100000, 1000, 0));
    // ---- collapsed groups
    TRY(AddTopic(in_data, P_ADJ_START, obs::T("Direction Balance")));
    TRY(AddFloat(in_data, P_MULX, obs::T("Horizontal"), 0, 1000, 0, 200, 100, 1, true));
    TRY(AddFloat(in_data, P_MULY, obs::T("Vertical"), 0, 1000, 0, 200, 100, 1, true));
    TRY(AddFloat(in_data, P_BALPOS, obs::T("Position Low/High"), -100, 100, -100, 100, 0, 0, false));
    TRY(AddFloat(in_data, P_MULROT, obs::T("Roll"), 0, 1000, 0, 200, 100, 1, true));
    TRY(AddFloat(in_data, P_BALROT, obs::T("Rotation Low/High"), -100, 100, -100, 100, 0, 0, false));
    TRY(AddFloat(in_data, P_MULZOOM, obs::T("Zoom Amount"), 0, 1000, 0, 200, 100, 1, true));
    TRY(AddFloat(in_data, P_BALZOOM, obs::T("Zoom Low/High"), -100, 100, -100, 100, 0, 0, false));
    TRY(EndTopic(in_data, P_ADJ_END));
    TRY(AddTopic(in_data, P_HAND_START, obs::T("Detail: Hand / Body")));
    for (int i = P_HAND_DEG; i <= P_CORR_INT; ++i)
        TRY(AddCtl(in_data, i));
    TRY(EndTopic(in_data, P_HAND_END));
    TRY(AddTopic(in_data, P_WALK_START, obs::T("Walking / Heartbeat")));
    for (int i = P_CADENCE; i <= P_PULSE; ++i)
        TRY(AddCtl(in_data, i));
    TRY(EndTopic(in_data, P_WALK_END));
    TRY(AddTopic(in_data, P_VIB_START, obs::T("Vehicle / Earthquake")));
    for (int i = P_VIB_PX; i <= P_QUAKE_ENV; ++i)
        TRY(AddCtl(in_data, i));
    TRY(EndTopic(in_data, P_VIB_END));
    TRY(AddTopic(in_data, P_IMP_START, obs::T("Impact")));
    for (int i = P_FIRST_HIT; i <= P_IMP_RANDOM; ++i)
        TRY(AddCtl(in_data, i));
    TRY(EndTopic(in_data, P_IMP_END));
    TRY(AddTopic(in_data, P_JOLT_START, obs::T("Jolts / Searching")));
    for (int i = P_JOLT_PX; i <= P_JOLT_YX; ++i)
        TRY(AddCtl(in_data, i));
    TRY(EndTopic(in_data, P_JOLT_END));
    TRY(AddTopic(in_data, P_FILM_START, obs::T("Film")));
    for (int i = P_WEAVE; i <= P_JITTER_YX; ++i)
        TRY(AddCtl(in_data, i));
    TRY(EndTopic(in_data, P_FILM_END));
    TRY(AddTopic(in_data, P_MB_START, obs::T("Motion Blur")));
    TRY(AddCtl(in_data, P_SHUTTER));
    TRY(AddInt(in_data, P_SAMPLES, obs::T("Samples"), 2, 64, 32, 16));
    TRY(AddCheck(in_data, P_LINEAR, obs::T("Blend in Linear Light"), false));
    TRY(EndTopic(in_data, P_MB_END));
    TRY(AddTopic(in_data, P_CUT_START, obs::T("Cuts / Auto")));
    TRY(AddInt(in_data, P_CUT, obs::T("Cut"), 0, 100000, 100, 0));
    TRY(AddFloat(in_data, P_MOOD, obs::T("Auto: Calm / Lively"), -100, 100, -100, 100, 0, 0, false));
    TRY(AddFloat(in_data, P_LOOP, obs::T("Loop (s)"), 0, 3600, 0, 20, 0, 2, false));
    TRY(EndTopic(in_data, P_CUT_END));
    TRY(AddTopic(in_data, P_MISC_START, obs::T("View / Center / Other")));
    TRY(AddPopup(in_data, P_VIEW, obs::T("View"), 2, VIEW_RESULT, obs::T("Result|Motion Graph")));
    TRY(AddCtl(in_data, P_HOLDN));
    TRY(AddPopup(in_data, P_INTERP, obs::T("Interpolation"), tb::INTERP_COUNT, 1, obs::T("Bicubic|Lanczos 3")));
    TRY(AddCheck(in_data, P_SNAP, obs::T("Whole Pixels"), false));
    TRY(AddPoint(in_data, P_CENTER, obs::T("Center")));
    TRY(AddFloat(in_data, P_REFW, obs::T("Reference Width (px)"), 0, 16384, 0, 3840, 0, 0, false));
    TRY(AddFloat(in_data, P_ZOOM_FIX, obs::T("Fixed Zoom (%)"), 100, 400, 100, 200, 110, 1, false));
    TRY(AddFloat(in_data, P_ZOOM_SMOOTH, obs::T("Zoom Smoothing (s)"), 0, 10, 0, 4, 1, 2, false));
    TRY(EndTopic(in_data, P_MISC_END));
    // 0.2's Speed (id 4): hidden, still applied (multiplies with Period) so 0.2 projects render the same
    TRY(AddFloat(in_data, P_SPEED, obs::T("Speed"), 0, 1000, 0, 400, 100, 1, true, 0, PF_PUI_INVISIBLE));
    // 3 for new instances; saved projects keep their stored value (0.1 / 0.2: 1, 0.3.0: 2) = their exact rules
    TRY(AddInt(in_data, P_RENDER_VERSION, obs::T("Render Version"), 1, 100, 10, 3, PF_PUI_INVISIBLE));
    // the picker's custom UI (Effect Controls only)
    {
        PF_CustomUIInfo ci;
        AEFX_CLR_STRUCT(ci);
        ci.events = PF_CustomEFlag_EFFECT;
        if (in_data->inter.register_ui && (err = PF_REGISTER_UI(in_data, &ci)) != PF_Err_NONE)
            return err;
    }
#undef TRY
    out_data->num_params = P_NUM;
    return err;
}

// write preset v (popup value) into the detail params (+ change flags); false for Custom / out of range
bool ApplyPreset(int v, PF_ParamDef* params[])
{
    {
        int n = 0;
        const tb::Preset* P = tb::Presets(&n);
        if (v < 2 || v - 2 >= n)
            return false;
        const tb::Shape& s = P[v - 2].s;
        for (const Ctl& c : kCtls)
        {
            PF_ParamDef* p = params[c.idx];
            const double u = UiValue(c, s);
            if (c.kind == K_FLOAT)
            {
                if (p->u.fs_d.value != (PF_FpShort)u)
                {
                    p->u.fs_d.value = (PF_FpShort)u;
                    p->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
                }
            }
            else if (c.kind == K_POPUP)
            {
                if (p->u.pd.value != (A_short)u)
                {
                    p->u.pd.value = (A_short)u;
                    p->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
                }
            }
            else if (p->u.sd.value != (A_long)u)
            {
                p->u.sd.value = (A_long)u;
                p->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
            }
        }
    }
    return true;
}


// Preset popup: write its values (one undo step); a hand edit of a written value turns the popup to Custom.
// AE does not send USER_CHANGED_PARAM for values the effect itself changes, so this cannot loop.
PF_Err UserChangedParam(PF_InData*, PF_OutData*, PF_ParamDef* params[], const PF_UserChangedParamExtra* which)
{
    if (!which || !params)
        return PF_Err_NONE;
    const int i = which->param_index;
    if (i == P_PRESET)
        ApplyPreset(params[P_PRESET]->u.pd.value, params);
    else if (IsPresetTarget(i) && params[P_PRESET]->u.pd.value != PRESET_CUSTOM)
    {
        params[P_PRESET]->u.pd.value = PRESET_CUSTOM;
        params[P_PRESET]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
    }
    return PF_Err_NONE;
}

// the shape from the detail sliders (Custom)
tb::Shape ShapeFromSliders(TimeReader& R)
{
    tb::Shape s;
    for (const Ctl& c : kCtls)
    {
        double v = R.Now(c.idx);
        if (c.kind == K_FLOAT)
        {
            v = std::min(c.vmax, std::max(c.vmin, v));
            s.*(c.md) = v / c.scale;
        }
        else if (c.kind == K_POPUP)
            s.*(c.mi) = std::min(c.choices, std::max(1, (int)v)) - 1;
        else
            s.*(c.mi) = std::min((int)c.vmax, std::max((int)c.vmin, (int)v));
    }
    return s;
}

// a preset's shape exactly as the sliders would hold it (float UI values / scale): a preset and "Custom with the
// preset's values" render bit-identically
tb::Shape Canonical(const tb::Shape& in)
{
    tb::Shape s = in;
    for (const Ctl& c : kCtls)
        if (c.kind == K_FLOAT)
        {
            const double v = std::min(c.vmax, std::max(c.vmin, (double)(PF_FpShort)(s.*(c.md) * c.scale)));
            s.*(c.md) = v / c.scale;
        }
    return s;
}

// shot boundaries = keyframes of Preset / Cut EXCEPT the first one (the value before the first key is that key's
// value: no change there). before = last boundary <= t (or -inf), after = next one (or +inf), count = boundaries <= t
void Around(const std::vector<double>& keys, double t, double& before, double& after, int& countBefore)
{
    before = -1e300;
    after = 1e300;
    countBefore = 0;
    for (size_t i = 1; i < keys.size(); ++i)
    {
        const double k = keys[i];
        if (k <= t + 1e-9)
        {
            before = std::max(before, k);
            ++countBefore;
        }
        else
            after = std::min(after, k);
    }
}

PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    TB_PROF("prerender total");
    (void)out_data;
    PF_Err err = PF_Err_NONE;
    RenderPlan* plan = new (std::nothrow) RenderPlan();
    if (!plan)
        return PF_Err_OUT_OF_MEMORY;
    extra->output->pre_render_data = plan;
    extra->output->delete_pre_render_data_func = DeletePlan;
    const PF_RenderRequest& outReq = extra->input->output_request;
    const double dsx = Ratio(in_data->downsample_x), dsy = Ratio(in_data->downsample_y);
    const double ds = dsx;  // horizontal; the vertical factor and the pixel aspect ratio go into the projection
    const double par = Ratio(in_data->pixel_aspect_ratio);
    plan->ds = ds;

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
    // content: the layer rect; a max rect that exceeds it by <= 1 px on a side (adjustment layers / collapsed
    // precomps report a 1 px transparent ring) keeps the layer edge there; a real expansion is content
    const tb::RectI layer{0, 0, (int)ScaledExtent(pr.ref_width, dsx), (int)ScaledExtent(pr.ref_height, dsy)};
    const tb::RectI maxr = ToR(pr.max_result_rect);
    tb::RectI content = layer;
    if (!maxr.empty() && !layer.empty())
    {
        content.l = maxr.l < layer.l - 1 ? maxr.l : layer.l;
        content.t = maxr.t < layer.t - 1 ? maxr.t : layer.t;
        content.r = maxr.r > layer.r + 1 ? maxr.r : layer.r;
        content.b = maxr.b > layer.b + 1 ? maxr.b : layer.b;
    }
    else if (layer.empty())
        content = maxr;
    plan->content = content;

    // ---- params
    TimeReader R;
    R.in = in_data;
    {
        const A_long st = in_data->local_time_step != 0 ? std::labs(in_data->local_time_step) : std::labs(in_data->time_step);
        R.fps = (st > 0 && in_data->time_scale > 0) ? (double)in_data->time_scale / (double)st : 24.0;
    }
    const PF_ParamUtilsSuite3* pu = nullptr;
    const bool havePu = in_data->pica_basicP &&
                        !in_data->pica_basicP->AcquireSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, (const void**)&pu) && pu;
    R.pu = havePu ? pu : nullptr;
    struct Release
    {
        PF_InData* in;
        bool on;
        ~Release()
        {
            if (on)
                in->pica_basicP->ReleaseSuite(kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
        }
    } rel{in_data, havePu};

    const double t = (double)in_data->current_time / (double)std::max<A_u_long>(1, in_data->time_scale);
    // shots: the last keyframe of Preset / Cut at or before t starts the shot
    const std::vector<double> kPreset = R.Keys(P_PRESET), kCut = R.Keys(P_CUT);
    double b1, a1, b2, a2;
    int n1, n2;
    Around(kPreset, t, b1, a1, n1);
    Around(kCut, t, b2, a2, n2);
    double t0 = std::max(0.0, std::max(b1, b2));
    if (t < 0)
        t0 = std::min(t0, t);
    const double tNext = std::min(a1, a2);
    const int cut = (int)R.Now(P_CUT);
    const int64_t shot = (int64_t)n1 * 1000003 + cut;

    int np = 0;
    const tb::Preset* P = tb::Presets(&np);
    const int pv = (int)R.Now(P_PRESET);
    tb::Shape shape;
    double autoFactor = 1;
    const bool custom = !(pv >= 2 && pv - 2 < np);
    const uint32_t seedUser = (uint32_t)std::max(0.0, R.Now(P_SEED));
    if (custom)
        shape = ShapeFromSliders(R);
    else if (P[pv - 2].autoPerCut)
        shape = Canonical(P[tb::AutoPick(seedUser, shot, &autoFactor, std::min(1.0, std::max(-1.0, R.Now(P_MOOD) / 100.0)))].s);
    else
        shape = Canonical(P[pv - 2].s);

    const double tEnd = std::min(tNext, t0 + 3600.0);
    tb::PlanInputs in;
    in.shape = shape;
    in.ctl.seed = (uint32_t)(tb::Mix64((uint64_t)seedUser * 0x9E3779B97F4A7C15ull + (uint64_t)shot) & 0xFFFFFFFFu);
    in.ctl.balPos = std::min(1.0, std::max(-1.0, R.Now(P_BALPOS) / 100.0));
    in.ctl.balRot = std::min(1.0, std::max(-1.0, R.Now(P_BALROT) / 100.0));
    in.ctl.balZoom = std::min(1.0, std::max(-1.0, R.Now(P_BALZOOM) / 100.0));
    {
        // effective time speed = 0.2's hidden Speed (id 4) / Period (id 105): both keyframable, integrated together
        const tb::Curve sp = R.MakeCurve(P_SPEED, t0, std::max(t, t0 + 1 / R.fps), 0.01, 0, 10);
        const tb::Curve pe = R.MakeCurve(P_PERIOD, t0, std::max(t, t0 + 1 / R.fps), 0.01, 0.1, 4);
        in.speed.animated = sp.animated || pe.animated;
        in.speed.constant = sp.constant / pe.constant;
        in.speed.keys = sp.keys;
        in.speed.keys.insert(in.speed.keys.end(), pe.keys.begin(), pe.keys.end());
        std::sort(in.speed.keys.begin(), in.speed.keys.end());
        if (in.speed.animated)
            in.speed.at = [sp, pe](double u) { return (sp.animated ? sp.at(u) : sp.constant) / (pe.animated ? pe.at(u) : pe.constant); };
    }
    in.renderVersion = std::max(1, (int)R.Now(P_RENDER_VERSION));
    const int zoomMode = std::min((int)ZOOM_COUNT, std::max(1, (int)R.Now(P_ZOOM_MODE)));
    const double zoomFix = std::max(100.0, R.Now(P_ZOOM_FIX)) / 100.0, zoomSmooth = std::max(0.0, R.Now(P_ZOOM_SMOOTH));
    if (custom)
        in.cadence = R.MakeCurve(P_CADENCE, t0, std::max(t, t0 + 1 / R.fps), 1, 0.1, 12);
    else
    {
        in.cadence.constant = shape.cadence;
        in.cadence.animated = false;
    }
    {
        tb::Curve am = R.MakeCurve(P_AMOUNT, t0, std::max(t, t0 + 1 / R.fps), 0.01, 0, 10);
        in.amountNow = am.constant * autoFactor;
        in.amountMax = R.MaxOver(am, t0, std::min(tEnd, t0 + 3600)) * autoFactor;
        if (zoomMode == ZOOM_AUTO && am.animated && in.amountMax > 0)
        {
            // amplitude envelope for the auto zoom: E(t) = mean over [t-w, t+w] of M(s), M(s) = max of the amplitude over
            // [s-w, s+w] (+ the shutter span). E >= amplitude everywhere in the frame, so no edge shows, and it changes
            // gently (a slow "breath", not pumping). w = Zoom Smoothing (s).
            const double w = std::max(zoomSmooth, 1.0 / R.fps), pad = 1.0 / R.fps;
            std::vector<double> ts, vs;
            for (int j = 0; j <= 32; ++j)
                ts.push_back(t + (j / 16.0 - 1.0) * 2 * w);
            for (double k : am.keys)
                if (k > t - 2 * w - pad && k < t + 2 * w + pad)
                    ts.push_back(k);
            ts.push_back(t - pad);
            ts.push_back(t + pad);
            for (double u : ts)
                vs.push_back(am.at(std::max(t0, std::min(tEnd, u))));
            double E = 0;
            for (int j = 0; j <= 16; ++j)
            {
                const double s = t + (j / 8.0 - 1.0) * w;
                double M = 0;
                for (size_t i = 0; i < ts.size(); ++i)
                    if (ts[i] >= s - w - pad && ts[i] <= s + w + pad)
                        M = std::max(M, vs[i]);
                E += M / 17.0;
            }
            in.envScale = std::min(1.0, 1.02 * E * autoFactor / in.amountMax);
        }
        const int mi[4] = {P_MULX, P_MULY, P_MULROT, P_MULZOOM};
        for (int k = 0; k < 4; ++k)
        {
            tb::Curve c = R.MakeCurve(mi[k], t0, std::max(t, t0 + 1 / R.fps), 0.01, 0, 10);
            in.mulNow[k] = c.constant;
            in.mulMax[k] = R.MaxOver(c, t0, std::min(tEnd, t0 + 3600));
        }
    }
    // impact triggers: every keyframe whose value is higher than the previous key fires (strength = rise / 100)
    {
        const std::vector<double> kt = R.Keys(P_TRIGGER);
        double prev = 0;
        bool havePrev = false;
        in.trigMax = 0;
        for (double k : kt)
        {
            const double v = R.At(P_TRIGGER, k);
            if (havePrev && v > prev + 1e-6 && k >= t0 && k < tEnd)
            {
                in.hits.push_back(tb::Hit{k, (v - prev) / 100.0});
                in.trigMax = std::max(in.trigMax, (v - prev) / 100.0);
            }
            prev = v;
            havePrev = true;
        }
        if (kt.size() < 2)
        {
            // no keys: an expression could still drive it - scan the comp frames of the response window
            const double now = R.Now(P_TRIGGER);
            const double win = std::min(30.0, shape.decay > 0 ? 12 * shape.decay : 30.0);
            const double a = std::max(t0, t - win);
            if (R.At(P_TRIGGER, a) != now || R.At(P_TRIGGER, 0.5 * (a + t)) != now)
            {
                double pv2 = R.At(P_TRIGGER, a);
                for (double u = a + 1 / R.fps; u <= t + 1e-9; u += 1 / R.fps)
                {
                    const double v = R.At(P_TRIGGER, u);
                    if (v > pv2 + 1e-6)
                    {
                        in.hits.push_back(tb::Hit{u, (v - pv2) / 100.0});
                        in.trigMax = std::max(in.trigMax, std::max(1.0, (v - pv2) / 100.0));
                    }
                    pv2 = v;
                }
            }
        }
    }
    in.t = t;
    in.t0 = t0;
    in.fps = R.fps;
    in.loop = std::max(0.0, R.Now(P_LOOP));
    in.shutterDeg = std::max(0.0, shape.shutter);  // preset-written (Custom: the slider)
    in.maxSamples = (int)R.Now(P_SAMPLES);
    in.snap = R.Now(P_SNAP) != 0;
    const double refW = R.Now(P_REFW);
    in.widthRef = refW > 0 ? refW : (double)std::max<A_long>(1, pr.ref_width);
    plan->view = (int)R.Now(P_VIEW) == VIEW_GRAPH ? VIEW_GRAPH : VIEW_RESULT;
    if (plan->view == VIEW_GRAPH)
        in.graphSamples = 600;
    plan->edge = std::min(tb::EDGE_COUNT - 1, std::max(0, (int)R.Now(P_EDGE) - 1));
    plan->interp = std::min(tb::INTERP_COUNT - 1, std::max(0, (int)R.Now(P_INTERP) - 1));
    plan->linear = R.Now(P_LINEAR) != 0;
    // pivot (buffer px)
    double cx = (content.l + content.r) * 0.5, cy = (content.t + content.b) * 0.5;
    {
        PF_ParamDef pc;
        AEFX_CLR_STRUCT(pc);
        if (PF_CHECKOUT_PARAM(in_data, P_CENTER, in_data->current_time, in_data->time_step, in_data->time_scale, &pc) == PF_Err_NONE)
        {
            const double x = pc.u.td.x_value / 65536.0, y = pc.u.td.y_value / 65536.0;
            if (std::isfinite(x) && std::isfinite(y))
                cx = x, cy = y;
            PF_CHECKIN_PARAM(in_data, &pc);
        }
    }
    if (R.err)
        return R.err;

    tb::PlanOutput po;
    {
        TB_PROF("prerender MakePlan");
        po = tb::MakePlan(in);
    }
    plan->focal = shape.focal;
    plan->widthRef = in.widthRef;
    plan->bound = po.bound;
    plan->graphT = po.graphT;
    plan->graph = po.graph;
    tb::Geo g;
    g.widthRef = in.widthRef;
    g.ds = ds;
    g.dsy = dsy;
    g.par = par;
    g.cx = cx;
    g.cy = cy;
    g.focal = shape.focal;
    if (plan->edge == tb::EDGE_AUTO)
    {
        // whip events are excluded (boundAuto): during a whip the edge pixels are repeated
        const tb::RectI cf{(int)std::floor(content.l / ds), (int)std::floor(content.t / dsy), (int)std::ceil(content.r / ds),
                           (int)std::ceil(content.b / dsy)};
        TB_PROF("prerender AutoScale");
        if (zoomMode == ZOOM_FIXED)
            g.autoScale = zoomFix;
        else if (zoomMode == ZOOM_NONE)
            g.autoScale = 1.0;  // (edge pixels are repeated where the frame would be empty)
        else  // Auto: the envelope-scaled bound; Constant per Shot (0.2): the shot's maximum (envScale = 1)
            g.autoScale = tb::AutoScale(po.boundAuto, cf, cx / ds, cy / dsy, in.widthRef, shape.focal, 2.0, par);
    }
    for (const tb::Pose& p : po.poses)
        plan->maps.push_back(tb::InverseMap(p, g));
    bool ident = plan->view == VIEW_RESULT;
    for (const tb::Mat3& M : plan->maps)
        ident = ident && tb::IsIdentity(M);
    plan->passthrough = ident;

    tb::RectI maxOut = content;
    if (plan->edge == tb::EDGE_TRANSPARENT && !ident)
        maxOut = tb::ForwardBounds(plan->maps, content);
    if (plan->view == VIEW_GRAPH)
        maxOut = tb::Union(maxOut, content);
    const tb::RectI result = tb::Intersect(ToR(outReq.rect), maxOut);
    plan->result = result;
    extra->output->max_result_rect = ToL(maxOut);
    extra->output->result_rect = ToL(result);
    extra->output->solid = false;
    extra->output->flags = 0;
    plan->emptyResult = result.empty();
    if (plan->emptyResult)
        return PF_Err_NONE;  // the probe above is the PreRender input checkout
    tb::RectI srcR = ident ? tb::Intersect(result, maxr.empty() ? content : tb::Union(content, maxr))
                           : tb::SourceNeeded(plan->maps, result, content, plan->edge, plan->interp);
    if (srcR.empty())
        srcR = tb::RectI{content.l, content.t, content.l + 1, content.t + 1};
    plan->src = srcR;
    PF_RenderRequest req = outReq;
    req.rect = ToL(srcR);
    req.channel_mask = PF_ChannelMask_ARGB;
    req.preserve_rgb_of_zero_alpha = false;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, CO_INPUT, &req, in_data->current_time, in_data->time_step,
                                  in_data->time_scale, &in_result));
    if (!err)
        plan->inputChecked = true;
    return err;
}

// ------------------------------------------------------------------ pixels
template <typename Pix> struct Traits;
template <> struct Traits<PF_Pixel8>
{
    static inline float In(A_u_char v) { return v * (1.f / 255.f); }
    static inline A_u_char Out(float v)
    {
        v = v * 255.f + 0.5f;
        return (A_u_char)(!(v > 0.f) ? 0 : (v >= 255.f ? 255 : (int)v));
    }
    static const bool kFloat = false;
};
template <> struct Traits<PF_Pixel16>
{
    static inline float In(A_u_short v) { return v * (1.f / PF_MAX_CHAN16); }
    static inline A_u_short Out(float v)
    {
        v = v * PF_MAX_CHAN16 + 0.5f;
        return (A_u_short)(!(v > 0.f) ? 0 : (v >= (float)PF_MAX_CHAN16 ? PF_MAX_CHAN16 : (int)v));
    }
    static const bool kFloat = false;
};
template <> struct Traits<PF_PixelFloat>
{
    static inline float In(PF_FpShort v) { return v; }
    static inline PF_FpShort Out(float v) { return v; }
    static const bool kFloat = true;
};
template <typename Pix> struct s_dummyPix
{
    Pix p;
};
inline int ChanIndex(A_u_char v) { return v; }
inline int ChanIndex(A_u_short v) { return std::min<int>(v, PF_MAX_CHAN16); }
inline int ChanIndex(PF_FpShort) { return 0; }
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
template <class Fn>
void ParallelRows(int n, int threads, const Fn& fn)
{
    const int T = std::max(1, std::min(threads, n / 16));
    if (T <= 1)
    {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    std::exception_ptr ex;
    std::mutex exm;
    for (int k = 0; k < T; ++k)
        pool.emplace_back([&, k] {
            try
            {
                fn((int)((long long)n * k / T), (int)((long long)n * (k + 1) / T));
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

template <typename Pix>
PF_Err RenderDepth(PF_InData* in_data, const RenderPlan& plan, PF_EffectWorld* input, PF_EffectWorld* output)
{
    typedef Traits<Pix> T;
    const tb::RectI outR{(int)output->origin_x, (int)output->origin_y, (int)(output->origin_x + output->width),
                         (int)(output->origin_y + output->height)};
    if (input && (!input->data || input->width <= 0 || input->height <= 0))
        input = nullptr;
    const tb::RectI inR = input ? tb::RectI{(int)input->origin_x, (int)input->origin_y, (int)(input->origin_x + input->width),
                                            (int)(input->origin_y + input->height)}
                                : tb::RectI{0, 0, 0, 0};
    ActiveGuard guard;
    const int threads = RenderThreads();
    if (plan.passthrough)
    {
        // no motion: the input pixels, bit exact
        ParallelRows(output->height, threads, [&](int y0, int y1) {
            for (int oy = y0; oy < y1; ++oy)
            {
                Pix* d = RowPtrW<Pix>(output, oy);
                const int ly = outR.t + oy;
                for (int ox = 0; ox < output->width; ++ox, ++d)
                {
                    const int lx = outR.l + ox;
                    if (input && lx >= inR.l && lx < inR.r && ly >= inR.t && ly < inR.b)
                        *d = RowPtr<Pix>(input, ly - inR.t)[lx - inR.l];
                    else
                        std::memset(d, 0, sizeof(Pix));
                }
            }
        });
        return PF_Err_NONE;
    }
    // source -> premultiplied float (optionally linear light), only inside the content (the ring stays out)
    TB_PROF("render total (no passthrough)");
    tb::Image src;
    src.rect = tb::Intersect(tb::Intersect(plan.src, inR), plan.content);
    if (src.rect.empty())
        src.rect = tb::RectI{0, 0, 0, 0};
    src.px.resize((size_t)src.rect.w() * src.rect.h() * 4);  // every pixel is written below
    const bool lin = plan.linear;
    // linear light: 8 / 16 bpc inputs through an exact table (same std::pow values, computed once)
    const float* linLut = nullptr;
    if (lin && !T::kFloat)
    {
        static std::vector<float> lut8, lut16;
        static std::once_flag o8, o16;
        if (sizeof(s_dummyPix<Pix>) == sizeof(PF_Pixel8))
        {
            std::call_once(o8, [] { lut8.resize(256); for (int i = 0; i < 256; ++i) lut8[i] = tbm::powf(i * (1.f / 255.f), 2.2f); });
            linLut = lut8.data();
        }
        else
        {
            std::call_once(o16, [] { lut16.resize(PF_MAX_CHAN16 + 1); for (int i = 0; i <= PF_MAX_CHAN16; ++i) lut16[i] = tbm::powf(i * (1.f / PF_MAX_CHAN16), 2.2f); });
            linLut = lut16.data();
        }
    }
    ParallelRows(src.rect.h(), threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
        {
            const int ly = src.rect.t + y;
            const Pix* s = RowPtr<Pix>(input, ly - inR.t) + (src.rect.l - inR.l);
            float* q = &src.px[(size_t)y * src.rect.w() * 4];
            for (int x = 0; x < src.rect.w(); ++x, ++s, q += 4)
            {
                const float a = std::min(1.f, std::max(0.f, SanitizeF(T::In(s->alpha))));
                if (linLut)
                {
                    q[0] = linLut[ChanIndex(s->red)] * a;
                    q[1] = linLut[ChanIndex(s->green)] * a;
                    q[2] = linLut[ChanIndex(s->blue)] * a;
                }
                else
                {
                    float c[3] = {SanitizeF(T::In(s->red)), SanitizeF(T::In(s->green)), SanitizeF(T::In(s->blue))};
                    for (int k = 0; k < 3; ++k)
                    {
                        float v = std::max(0.f, c[k]);
                        if (lin)
                            v = tbm::powf(v, 2.2f);
                        q[k] = v * a;
                    }
                }
                q[3] = a;
            }
        }
    });
    if (PF_ABORT(in_data))
        return PF_Interrupt_CANCEL;
    TB_PROF("render after convert-in");
    const tb::RectI outRect = outR;
    auto writeRow = [&](int oy, const float* q) {
        Pix* d = RowPtrW<Pix>(output, oy);
        for (int ox = 0; ox < output->width; ++ox, ++d, q += 4)
        {
            const float a = std::min(1.f, std::max(0.f, q[3]));
            if (!(a > 1e-6f))
            {
                std::memset(d, 0, sizeof(Pix));
                continue;
            }
            float c[3];
            for (int k = 0; k < 3; ++k)
            {
                float v = std::max(0.f, q[k]) / a;
                if (lin)
                    v = tbm::powf(v, 1.f / 2.2f);
                if (!T::kFloat)
                    v = std::min(1.f, v);
                c[k] = SanitizeF(v);
            }
            d->red = T::Out(c[0]);
            d->green = T::Out(c[1]);
            d->blue = T::Out(c[2]);
            d->alpha = T::Out(a);
        }
    };
    (void)outRect;
    tb::WarpJob job;
    job.src = &src;
    job.content = plan.content;
    job.edge = plan.edge;
    job.interp = plan.interp;
    job.maps = plan.maps;
    job.out = outR;
    job.threads = threads;
    if (plan.view == VIEW_GRAPH)
    {
        // the overlay needs the whole picture first
        tb::Image dst;
        tb::Warp(job, &dst);
        tb::DrawGraph(&dst, plan.content, plan.ds, plan.graphT, plan.graph, plan.bound, plan.focal, plan.widthRef);
        ParallelRows(output->height, threads, [&](int y0, int y1) {
            for (int oy = y0; oy < y1; ++oy)
                writeRow(oy, &dst.px[(size_t)oy * dst.rect.w() * 4]);
        });
    }
    else
    {
        // warp and convert in one pass: each finished row goes straight into the output world (0.3)
        TB_PROF("render Warp+write");
        job.rowOut = [&](int y, const float* row) { writeRow(y, row); };
        tb::Warp(job, nullptr);
    }
    return PF_Err_NONE;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const RenderPlan* plan = reinterpret_cast<const RenderPlan*>(extra->input->pre_render_data);
    if (!plan)
        return PF_Err_BAD_CALLBACK_PARAM;
    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    // AE 2026 rule (25 :: 244): the input pixels BEFORE the output, on every path
    const A_long inId = plan->inputChecked ? CO_INPUT : CO_INPUT_PROBE;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, inId, &input));
    const bool inOut = !err;
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
                case 32: err = RenderDepth<PF_PixelFloat>(in_data, *plan, input, output); break;
                case 16: err = RenderDepth<PF_Pixel16>(in_data, *plan, input, output); break;
                case 8: err = RenderDepth<PF_Pixel8>(in_data, *plan, input, output); break;
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
    if (inOut)
        ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, inId));
    return err ? err : err2;
}

// ------------------------------------------------------------------ preset picker (custom UI, Drawbot)
// Each cell shows the preset's own 5 s motion path (computed by the engine with the preset's values, tb_thumbs)
// and an amplitude bar. Clicking a cell does exactly what choosing that popup item does (ApplyPreset, one undo
// step); the popup stays the source of truth, the picker keeps no state except the cosmetic hover cell.
struct DrawbotAll
{
    const DRAWBOT_DrawbotSuite1* draw = nullptr;
    const DRAWBOT_SupplierSuite1* supplier = nullptr;
    const DRAWBOT_SurfaceSuite2* surface = nullptr;
    const DRAWBOT_PathSuite1* path = nullptr;
    SPBasicSuite* sp = nullptr;
    bool ok = false;
    explicit DrawbotAll(SPBasicSuite* b) : sp(b)
    {
        if (!sp)
            return;
        ok = !sp->AcquireSuite(kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_Version1, (const void**)&draw) && draw &&
             !sp->AcquireSuite(kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_Version1, (const void**)&supplier) && supplier &&
             !sp->AcquireSuite(kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_Version2, (const void**)&surface) && surface &&
             !sp->AcquireSuite(kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_Version1, (const void**)&path) && path;
    }
    ~DrawbotAll()
    {
        if (!sp)
            return;
        if (draw) sp->ReleaseSuite(kDRAWBOT_DrawSuite, kDRAWBOT_DrawSuite_Version1);
        if (supplier) sp->ReleaseSuite(kDRAWBOT_SupplierSuite, kDRAWBOT_SupplierSuite_Version1);
        if (surface) sp->ReleaseSuite(kDRAWBOT_SurfaceSuite, kDRAWBOT_SurfaceSuite_Version2);
        if (path) sp->ReleaseSuite(kDRAWBOT_PathSuite, kDRAWBOT_PathSuite_Version1);
    }
};
DRAWBOT_ColorRGBA ToDb(const PF_App_Color& c, float a = 1.f)
{
    DRAWBOT_ColorRGBA d;
    d.red = c.red / 65535.f;
    d.green = c.green / 65535.f;
    d.blue = c.blue / 65535.f;
    d.alpha = a;
    return d;
}
DRAWBOT_ColorRGBA Rgb(float r, float g, float b, float a = 1.f)
{
    DRAWBOT_ColorRGBA d;
    d.red = r, d.green = g, d.blue = b, d.alpha = a;
    return d;
}
void StrokeRect(DrawbotAll& db, DRAWBOT_SupplierRef sup, DRAWBOT_SurfaceRef surf, const DRAWBOT_ColorRGBA& c, float width, float x,
                float y, float w, float h)
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
void DrawText(DrawbotAll& db, DRAWBOT_SupplierRef sup, DRAWBOT_SurfaceRef surf, DRAWBOT_FontRef font, const DRAWBOT_ColorRGBA& c,
              const std::wstring& s, float x, float y, float width, DRAWBOT_TextAlignment align)
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
// preset item labels in the UI language (the popup string split at '|', "Custom" dropped)
std::vector<std::wstring> PresetLabels()
{
    const std::wstring all = obs::W(TB_PRESET_ITEMS);
    std::vector<std::wstring> v;
    size_t st = 0;
    for (;;)
    {
        const size_t bar = all.find(L'|', st);
        v.push_back(all.substr(st, bar == std::wstring::npos ? std::wstring::npos : bar - st));
        if (bar == std::wstring::npos)
            break;
        st = bar + 1;
    }
    if (!v.empty())
        v.erase(v.begin());
    return v;
}
// two lines: break before a full-width bracket / after a colon or a nakaguro, else in the middle
void SplitLabel(const std::wstring& s, std::wstring& a, std::wstring& b)
{
    size_t k = s.find(L'\xFF08');  // (
    if (k == std::wstring::npos || k == 0)
    {
        k = s.find(L'\xFF1A');  // :
        if (k != std::wstring::npos)
            ++k;
    }
    if (k == std::wstring::npos)
    {
        k = s.find(L'\x30FB');  // nakaguro
        if (k != std::wstring::npos)
            ++k;
    }
    if (k == std::wstring::npos || k >= s.size())
    {
        if (s.size() <= 7)
        {
            a = s;
            b.clear();
            return;
        }
        k = s.size() / 2;
    }
    a = s.substr(0, k);
    b = s.substr(k);
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

PF_Err DrawPicker(PF_InData* in_data, PF_ParamDef* params[], PF_EventExtra* e)
{
    DrawbotAll db(in_data->pica_basicP);
    const PFAppSuite6* app = nullptr;
    const PF_EffectCustomUISuite1* cui = nullptr;
    SPBasicSuite* sp = in_data->pica_basicP;
    if (!db.ok || !sp || sp->AcquireSuite(kPFAppSuite, kPFAppSuiteVersion6, (const void**)&app) || !app)
        return PF_Err_NONE;
    if (!sp->AcquireSuite(kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1, (const void**)&cui) && cui)
    {
        DRAWBOT_DrawRef dref = nullptr;
        DRAWBOT_SupplierRef sup = nullptr;
        DRAWBOT_SurfaceRef surf = nullptr;
        if (!cui->PF_GetDrawingReference(e->contextH, &dref) && dref && !db.draw->GetSupplier(dref, &sup) && !db.draw->GetSurface(dref, &surf) &&
            sup && surf)
        {
            PF_App_Color bgc = {0, 0, 0}, txc = {0, 0, 0}, hot = {0, 0, 0};
            app->PF_AppGetBgColor(&bgc);
            app->PF_AppGetColor(PF_App_Color_TEXT, &txc);
            app->PF_AppGetColor(PF_App_Color_HOT_TEXT, &hot);
            const DRAWBOT_ColorRGBA bg = ToDb(bgc), text = ToDb(txc), sel = ToDb(hot), dim = ToDb(txc, 0.45f);
            const PickerFrame f = FrameOf(e);
            DRAWBOT_RectF32 all = {f.x, f.y, f.w, f.h};
            db.surface->PaintRect(surf, &bg, &all);
            const int n = tbthumb::Count();
            const tbthumb::Grid g = tbthumb::Layout(n, f.w);
            const int value = params[P_PRESET] ? params[P_PRESET]->u.pd.value : 1;
            const int selIdx = value - 2;
            const int hover = GetHover(e->contextH, e->effect_win.index);
            float defSize = 11.f;
            db.supplier->GetDefaultFontSize(sup, &defSize);
            const float fontPx = std::max(7.f, std::min(defSize * 0.85f, 11.f) * g.s);
            DRAWBOT_FontRef font = nullptr;
            db.supplier->NewDefaultFont(sup, fontPx, &font);
            const std::vector<std::wstring> labels = PresetLabels();
            static const float famCol[5][3] = {{0.95f, 0.55f, 0.45f}, {0.5f, 0.85f, 0.5f}, {0.95f, 0.8f, 0.35f}, {0.55f, 0.75f, 1.f}, {0.95f, 0.65f, 0.9f}};
            for (int i = 0; i < n; ++i)
            {
                float cx, cy, tw, th;
                tbthumb::CellRect(g, i, cx, cy, tw, th);
                cx += f.x;
                cy += f.y;
                const DRAWBOT_ColorRGBA cellBg = Rgb(0.11f, 0.11f, 0.12f);
                DRAWBOT_RectF32 r = {cx, cy, tw, th};
                db.surface->PaintRect(surf, &cellBg, &r);
                const tbthumb::Path* p = tbthumb::Get(i);
                if (p && p->x.size() > 1)
                {
                    const float* fc = famCol[std::min(4, std::max(0, p->family))];
                    const DRAWBOT_ColorRGBA pc = Rgb(fc[0], fc[1], fc[2], 0.95f);
                    DRAWBOT_PenRef pen = nullptr;
                    DRAWBOT_PathRef path = nullptr;
                    if (!db.supplier->NewPen(sup, &pc, std::max(0.8f, 1.1f * g.s), &pen) && !db.supplier->NewPath(sup, &path))
                    {
                        const float mx = cx + tw * 0.5f, my = cy + th * 0.45f, sc = std::min(tw, th) * 0.4f;
                        db.path->MoveTo(path, mx + p->x[0] * sc, my + p->y[0] * sc);
                        for (size_t k = 1; k < p->x.size(); ++k)
                            db.path->LineTo(path, mx + p->x[k] * sc, my + p->y[k] * sc);
                        db.surface->StrokePath(surf, pen, path);
                    }
                    if (path)
                        db.supplier->ReleaseObject((DRAWBOT_ObjectRef)path);
                    if (pen)
                        db.supplier->ReleaseObject((DRAWBOT_ObjectRef)pen);
                    // amplitude bar (log scale over all presets): how big this preset moves at 100 %
                    const DRAWBOT_ColorRGBA bc = Rgb(fc[0], fc[1], fc[2], 0.55f);
                    DRAWBOT_RectF32 br = {cx + 3.f, cy + th - 5.f * g.s, std::max(1.f, (tw - 6.f) * p->barFrac), 2.5f * g.s};
                    db.surface->PaintRect(surf, &bc, &br);
                }
                if (i == selIdx)
                    StrokeRect(db, sup, surf, sel, 2.f, cx - 2.f, cy - 2.f, tw + 4.f, th + 4.f);
                else if (i == hover)
                    StrokeRect(db, sup, surf, dim, 1.f, cx - 1.5f, cy - 1.5f, tw + 3.f, th + 3.f);
                float lx, ly, lw, lh;
                tbthumb::LabelRect(g, i, lx, ly, lw, lh);
                std::wstring a, b;
                if (i < (int)labels.size())
                    SplitLabel(labels[i], a, b);
                const float lineH = lh * 0.5f;
                DrawText(db, sup, surf, font, i == selIdx ? sel : text, a, f.x + lx + lw * 0.5f, f.y + ly + lineH * 0.85f, lw, kDRAWBOT_TextAlignment_Center);
                if (!b.empty())
                    DrawText(db, sup, surf, font, i == selIdx ? sel : text, b, f.x + lx + lw * 0.5f, f.y + ly + lineH * 1.85f, lw,
                             kDRAWBOT_TextAlignment_Center);
            }
            if (selIdx < 0)
                DrawText(db, sup, surf, font, sel, obs::W("Custom (edited)"), f.x + 4.f, f.y + f.h - 5.f, f.w - 8.f, kDRAWBOT_TextAlignment_Left);
            if (font)
                db.supplier->ReleaseObject((DRAWBOT_ObjectRef)font);
        }
        sp->ReleaseSuite(kPFEffectCustomUISuite, kPFEffectCustomUISuiteVersion1);
    }
    sp->ReleaseSuite(kPFAppSuite, kPFAppSuiteVersion6);
    e->evt_out_flags = PF_EO_HANDLED_EVENT;
    return PF_Err_NONE;
}

void InvalidatePicker(PF_InData* in_data, PF_EventExtra* e)
{
    const PFAppSuite6* app = nullptr;
    SPBasicSuite* sp = in_data->pica_basicP;
    if (!sp || sp->AcquireSuite(kPFAppSuite, kPFAppSuiteVersion6, (const void**)&app) || !app)
        return;
    if (app->PF_InvalidateRect)
    {
        PF_Rect r;
        r.left = (A_short)e->effect_win.current_frame.left;
        r.top = (A_short)e->effect_win.current_frame.top;
        r.right = (A_short)e->effect_win.current_frame.right;
        r.bottom = (A_short)e->effect_win.current_frame.bottom;
        app->PF_InvalidateRect(e->contextH, &r);
    }
    sp->ReleaseSuite(kPFAppSuite, kPFAppSuiteVersion6);
}

PF_Err HandleEvent(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_EventExtra* e)
{
    (void)out_data;
    if (!e || !e->contextH || !*e->contextH || (*e->contextH)->w_type != PF_Window_EFFECT || !params)
        return PF_Err_NONE;
    if (e->effect_win.index != P_PICKER || e->effect_win.area != PF_EA_CONTROL)
        return PF_Err_NONE;
    const PickerFrame f = FrameOf(e);
    const int n = tbthumb::Count();
    switch (e->e_type)
    {
    case PF_Event_DRAW:
        return DrawPicker(in_data, params, e);
    case PF_Event_DO_CLICK:
    {
        const tbthumb::Grid g = tbthumb::Layout(n, f.w);
        const int cell = tbthumb::HitTest(g, n, e->u.do_click.screen_point.h - f.x, e->u.do_click.screen_point.v - f.y);
        if (cell >= 0 && params[P_PRESET])
        {
            // exactly what choosing the popup item does: one event, one undo step
            params[P_PRESET]->u.pd.value = (A_short)(cell + 2);
            params[P_PRESET]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
            ApplyPreset(cell + 2, params);
            InvalidatePicker(in_data, e);
            e->evt_out_flags = PF_EO_HANDLED_EVENT | PF_EO_UPDATE_NOW;
        }
        else
            e->evt_out_flags = PF_EO_HANDLED_EVENT;
        return PF_Err_NONE;
    }
    case PF_Event_ADJUST_CURSOR:
    {
        const tbthumb::Grid g = tbthumb::Layout(n, f.w);
        const int cell = tbthumb::HitTest(g, n, e->u.adjust_cursor.screen_point.h - f.x, e->u.adjust_cursor.screen_point.v - f.y);
        e->u.adjust_cursor.set_cursor = cell >= 0 ? PF_Cursor_FINGER_POINTER : PF_Cursor_NONE;
        if (SetHover(e->contextH, e->effect_win.index, cell))
            InvalidatePicker(in_data, e);
        e->evt_out_flags = PF_EO_HANDLED_EVENT;
        return PF_Err_NONE;
    }
    default:
        return PF_Err_NONE;
    }
}

}  // namespace

#ifdef TB_PROFILE
extern "C" DllExport int TbProfileDump(char* buf, int n)
{
    std::lock_guard<std::mutex> lk(g_profM);
    std::string s;
    for (auto& kv : g_prof)
    {
        char line[160];
        std::snprintf(line, sizeof(line), "%-34s %9.2f ms / %lld = %7.3f ms\n", kv.first.c_str(), kv.second.first, kv.second.second,
                      kv.second.first / std::max(1LL, kv.second.second));
        s += line;
    }
    g_prof.clear();
    std::snprintf(buf, (size_t)n, "%s", s.c_str());
    return (int)s.size();
}
#endif

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* inSPBasicSuitePtr, const char* inHostName, const char* inHostVersion)
{
    (void)inSPBasicSuitePtr;
    (void)inHostName;
    (void)inHostVersion;
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, TB_NAME, TB_MATCH_NAME, TB_CATEGORY, AE_RESERVED_INFO, "EffectMain",
                                     TB_SUPPORT_URL);
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
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data); break;
        case PF_Cmd_USER_CHANGED_PARAM:
            err = UserChangedParam(in_data, out_data, params, reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
            break;
        case PF_Cmd_EVENT: err = HandleEvent(in_data, out_data, params, reinterpret_cast<PF_EventExtra*>(extra)); break;
        case PF_Cmd_SMART_PRE_RENDER: err = SmartPreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra)); break;
        case PF_Cmd_SMART_RENDER: err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra)); break;
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
