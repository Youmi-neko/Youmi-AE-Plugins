// lens_presets.h - lens character presets shared by the AE plugin and tools/apply.
// Values are in lenscore::Params units. A preset sets ONLY the iris / lens character / falloff &
// glow / highlight fields (incl. White Protection, Dots, Min Size); it never touches radius, the
// Highlight Source / highlight layer,
// map settings, film effects (halation / luminance aberration), the v1 classic
// highlight gain, the output rolloff, threads or frame geometry.
// Encoding: UTF-8 (Japanese labels are used by the preview tools only; the plugin uses `ui`).
#pragma once

#include "lenscore.h"
#include <cstring>

namespace lenscore {

struct LensPresetInfo {
    const char* key;  // ASCII id for tools (preset=<key>)
    const char* ui;   // AE popup label (ASCII, no '|')
    const char* jp;   // Japanese label for preview sheets
    const char* desc; // short Japanese description
};

static const LensPresetInfo kLensPresets[] = {
    {"premium85", "Premium Portrait f/1.2", "高級ポートレート f/1.2",
     "平坦な円・なめらかな縁、隅に少しレモン形"},
    {"ultra58", "Ultra-Fast Normal f/0.95", "超大口径標準 f/0.95",
     "より大きく明るい真円、ごく薄い縁"},
    {"outlined", "Fast Portrait Outlined", "大口径ポートレート・輪郭",
     "細い明るい輪郭と年輪、隅は強いレモン形"},
    {"chromatic", "Chromatic Fast Prime", "色収差の大口径", "緑とマゼンタの縁どり（前後で反転）"},
    {"aspherical", "Modern Aspherical", "非球面モダン", "11枚曲線絞り、うっすら年輪と縁"},
    {"creamy", "Creamy Apodized", "とろけるアポダイズ", "縁が溶けて消える、輪郭のないボケ"},
    {"sparkle", "Stopped-Down Sparkle", "絞り込みキラキラ", "8角形・くっきり明るい縁"},
    {"hexagon", "Hexagon Classic", "六角形クラシック", "6枚直線羽根の六角形"},
    {"pentagon", "Pentagon Vintage", "五角形オールド", "5枚羽根・縁あり・赤青の色にじみ"},
    {"heptaglow", "Vintage Heptagon Glow", "七角形グロー", "角の丸い7角形とにじみ光"},
    {"bubble", "Soap Bubble Triplet", "シャボン玉トリプレット", "後ボケに強い輪郭、前ボケはやわらか"},
    {"swirly", "Swirly Vintage", "ぐるぐるボケ", "強いレモン形で渦を巻く、周辺減光"},
    {"dreamy", "Dreamy Soft Focus", "ドリーミー・ソフト", "にじみ光と中心の明るいやわらかボケ"},
    {"mirror", "Mirror Lens Donut", "ミラーレンズ・リング", "中心が抜けたドーナツ形"},
    {"anamorphic", "Anamorphic Cine", "アナモルフィック", "縦長の楕円、青い縁どり"},
    {"vintagefast", "Vintage Fast Prime", "オールド大口径開放", "芯のあるやわらかボケ・にじみ・周辺減光"},
};
static const int kLensPresetCount = (int)(sizeof(kLensPresets) / sizeof(kLensPresets[0]));

inline int findLensPreset(const char* key) {
    for (int i = 0; i < kLensPresetCount; ++i)
        if (std::strcmp(kLensPresets[i].key, key) == 0) return i;
    return -1;
}

// Writes preset `idx` (0-based in kLensPresets) into the character fields of p.
inline bool applyLensPreset(int idx, Params& p) {
    if (idx < 0 || idx >= kLensPresetCount) return false;
    const Params d;  // reset every preset-owned field to its default first
    p.blades = 0; p.bladeCurvature = 0.f; p.rotationDeg = 0.f; p.anamorphic = 1.f;
    p.catEye = 0.f; p.cornerRounding = 0.f; p.obstruction = 0.f;
    p.edgeRing = 0.f; p.ringWidth = d.ringWidth; p.chromaFringe = 0.f; p.fringeTint = 0.f;
    p.sphericalAberration = 0.f; p.edgeSoftness = 0.f; p.apodization = 0.f; p.coreWeight = 0.f;
    p.onionRings = 0.f; p.onionCount = d.onionCount;
    p.vignetting = 0.f; p.glow = 0.f; p.glowSize = d.glowSize;
    // v2.0: what becomes a ball. White Protection and Dots keep paper-white, text and hair-gap lines
    // of anime frames from turning into balls; night lights are unaffected (rare and compact).
    // (Highlight Source / the highlight layer are never written: same rule in the plugin and the CLI)
    p.whiteProtection = 0.8f; p.dotsOnly = 0.5f; p.detailMin = 0.f;
    auto hl = [&p](float boost, float knee, float curve, float sel, float detail, float sat) {
        p.highlightBoost = boost; p.highlightKnee = knee; p.highlightCurve = curve;
        p.highlightSelectivity = sel; p.detailSize = detail; p.highlightSaturation = sat;
    };
    switch (idx) {
    case 0:  // premium85: flat disc, gentle 6% roll-off, trace of LoCA, moderate cat eye
        p.catEye = 0.35f; p.edgeSoftness = 0.06f; p.sphericalAberration = -0.08f;
        p.ringWidth = 0.08f; p.chromaFringe = 0.06f; p.fringeTint = 0.7f; p.vignetting = 0.15f;
        hl(1.4f, 0.45f, 2.f, 0.85f, 8.f, 0.35f);
        break;
    case 1:  // ultra58: larger-feeling, more luminous, cleaner; very faint rim, little cat eye
        p.catEye = 0.15f; p.edgeSoftness = 0.04f; p.edgeRing = 0.06f; p.ringWidth = 0.05f;
        p.sphericalAberration = -0.04f; p.chromaFringe = 0.05f; p.fringeTint = 0.7f; p.vignetting = 0.2f;
        hl(2.0f, 0.42f, 2.f, 0.85f, 9.f, 0.4f);
        break;
    case 2:  // outlined: thin bright rim, onion rings, strong cat eye
        p.catEye = 0.6f; p.edgeRing = 0.35f; p.ringWidth = 0.06f; p.edgeSoftness = 0.015f;
        p.onionRings = 0.25f; p.onionCount = 6.f; p.chromaFringe = 0.06f; p.fringeTint = 0.5f;
        p.sphericalAberration = 0.2f; p.vignetting = 0.15f;
        hl(1.2f, 0.45f, 2.f, 0.85f, 8.f, 0.4f);
        break;
    case 3:  // chromatic: longitudinal CA as a feature - green rims behind focus, magenta in front
        p.catEye = 0.3f; p.edgeSoftness = 0.045f; p.edgeRing = 0.15f; p.ringWidth = 0.06f;
        p.chromaFringe = 0.35f; p.fringeTint = 1.f; p.sphericalAberration = 0.25f; p.vignetting = 0.2f;
        hl(1.1f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 4:  // aspherical
        p.blades = 11; p.bladeCurvature = 0.85f; p.catEye = 0.3f;
        p.edgeRing = 0.15f; p.ringWidth = 0.1f; p.edgeSoftness = 0.035f;
        p.onionRings = 0.3f; p.onionCount = 5.f; p.chromaFringe = 0.12f; p.fringeTint = 0.5f;
        hl(0.9f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 5:  // creamy
        p.catEye = 0.2f; p.apodization = 0.25f; p.edgeSoftness = 0.55f;
        hl(0.6f, 0.5f, 2.f, 0.9f, 8.f, 0.3f);
        break;
    case 6:  // sparkle
        p.blades = 8; p.bladeCurvature = 0.f; p.rotationDeg = 22.5f; p.catEye = 0.1f;
        p.edgeRing = 0.6f; p.ringWidth = 0.08f; p.edgeSoftness = 0.01f; p.chromaFringe = 0.15f;
        hl(1.6f, 0.45f, 2.f, 0.9f, 6.f, 0.3f);
        p.whiteProtection = 0.6f; p.dotsOnly = 0.3f;  // sparkle: more glints
        break;
    case 7:  // hexagon
        p.blades = 6; p.bladeCurvature = 0.f; p.edgeRing = 0.15f; p.ringWidth = 0.15f; p.chromaFringe = 0.1f;
        hl(1.1f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 8:  // pentagon
        p.blades = 5; p.bladeCurvature = 0.1f; p.rotationDeg = 18.f; p.catEye = 0.3f;
        p.edgeRing = 0.35f; p.ringWidth = 0.15f; p.chromaFringe = 0.35f; p.vignetting = 0.25f;
        hl(1.0f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 9:  // heptaglow
        p.blades = 7; p.bladeCurvature = 0.f; p.cornerRounding = 0.4f; p.rotationDeg = 10.f;
        p.edgeRing = 0.2f; p.ringWidth = 0.1f; p.chromaFringe = 0.25f;
        p.glow = 0.1f; p.glowSize = 0.03f;
        hl(1.1f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 10:  // bubble
        p.catEye = 0.35f; p.sphericalAberration = 0.8f; p.edgeRing = 0.1f; p.ringWidth = 0.08f;
        p.chromaFringe = 0.2f;
        hl(1.1f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 11:  // swirly
        p.catEye = 0.95f; p.edgeRing = 0.4f; p.ringWidth = 0.12f; p.glow = 0.05f; p.vignetting = 0.35f;
        p.chromaFringe = 0.2f;
        hl(0.9f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 12:  // dreamy
        p.sphericalAberration = -0.6f; p.coreWeight = 0.5f; p.edgeSoftness = 0.25f;
        p.glow = 0.12f; p.glowSize = 0.04f;
        hl(0.7f, 0.5f, 2.f, 0.6f, 8.f, 0.2f);
        break;
    case 13:  // mirror
        p.obstruction = 0.45f; p.edgeRing = 0.25f; p.ringWidth = 0.08f;
        hl(1.4f, 0.45f, 2.f, 0.85f, 8.f, 0.3f);
        break;
    case 14:  // anamorphic
        p.anamorphic = 2.f; p.edgeRing = 0.25f; p.ringWidth = 0.1f; p.edgeSoftness = 0.03f;
        p.chromaFringe = -0.3f;
        hl(1.1f, 0.45f, 2.f, 0.85f, 8.f, 0.4f);
        break;
    case 15:  // vintagefast: under-corrected wide-open prime - soft glowing core, harder front bokeh
        p.catEye = 0.45f; p.sphericalAberration = -0.35f; p.coreWeight = 0.25f; p.glow = 0.08f; p.edgeSoftness = 0.04f;
        p.chromaFringe = 0.3f; p.fringeTint = 0.5f; p.vignetting = 0.3f;
        hl(0.9f, 0.45f, 2.f, 0.8f, 8.f, 0.3f);
        break;
    }
    return true;
}


// ---- Quality modes (v2.1; one effect, chosen in the "Quality" popup / CLI quality=).
//   0 Fast     : blur-stage shortcuts (Params.fastBlur): earlier multi-resolution, shared fringe
//                layers, tabulated cat eye. Measured budget vs the exact path, see the memo.
//   1 Standard : the exact span engine with the v1.2 multi-resolution rule (big discs on 2x/4x grids,
//                level base 34). Identical to v1.2 for v1.2 parameters (projects keep their look).
//   2 Best     : everything at full resolution (no coarse levels). The quality engine (core/hq) is
//                meant to replace the blur stage here; until it is wired the host runs the exact
//                span engine (see BlurStage in plugin/OpticBokeh.cpp, TODO(lenshq)).
enum { kQualityFast = 0, kQualityStandard = 1, kQualityBest = 2 };
inline void applyQuality(int q, Params& p) {
    p.fastBlur = q == kQualityFast;
    p.fullResolution = q == kQualityBest;
}

// ---- Luminance Aberration styles (v2.1). Each sets every LA shape/colour field, never the amount.
// The plugin's "Aberration Style" popup: 1 = Custom (no change), 2.. = kAberrationStyles[v-2].
struct AberrationStyle { const char* key; const char* ui; };
static const AberrationStyle kAberrationStyles[] = {
    {"softbleed",    "Soft Bleed"},     // A: thin warm bleed into the dark side
    {"wideglow",     "Wide Glow"},      // B: wide orange glow into the dark side
    {"brighttint",   "Bright Tint"},    // C: the bright side of strong edges turns warm
    {"contrastrim",  "Contrast Rim"},   // D: channel drift (the AE recipe): soft orange fringe on
                                        //    strong bright/dark boundaries, colour from the picture
    {"redcyan",      "Red / Cyan Fringe"},  // drift with red moving the other way: red one side, cyan the other
    {"softfringe",   "Soft Glow Fringe"},   // v2.1b laMode 3 + Rim Blur: tinted spill into the dark side
    {"hardrim",      "Hard Rim (v2.1)"},    // v2.1 laMode 3: the crisp drawn line
};
static const int kAberrationStyleCount = (int)(sizeof(kAberrationStyles) / sizeof(kAberrationStyles[0]));

inline int findAberrationStyle(const char* key) {
    for (int i = 0; i < kAberrationStyleCount; ++i)
        if (std::strcmp(kAberrationStyles[i].key, key) == 0) return i;
    return -1;
}

inline bool applyAberrationStyle(int idx, Params& p) {
    if (idx < 0 || idx >= kAberrationStyleCount) return false;
    const Params d;
    p.laWidth = d.laWidth; p.laThreshold = d.laThreshold; p.laCurve = d.laCurve; p.laHueShift = d.laHueShift;
    for (int i = 0; i < 3; ++i) p.laTint[i] = d.laTint[i];
    p.laContrastScale = d.laContrastScale; p.laLineProtect = d.laLineProtect;
    p.laSharpness = d.laSharpness; p.laBrightColor = d.laBrightColor;
    p.laBlur = d.laBlur; p.laComplement = d.laComplement;
    p.laMapSoftness = d.laMapSoftness; p.laShiftDir = d.laShiftDir; p.laPreserve = d.laPreserve;
    for (int i = 0; i < 3; ++i) p.laShift[i] = d.laShift[i];
    auto drift = [&p](int fringe) {  // fringe: index into kFringeColors (defined below)
        p.laMode = 4; p.laWidth = 0.002f; p.laThreshold = 1.2f; p.laCurve = 1.f;
        p.laMapSoftness = 0.002f; p.laLineProtect = 0.003f; p.laPreserve = 0.5f; p.laShiftDir = 0;
        static const float sh[2][3] = {{-0.15f, 0.5f, 1.3f}, {-0.8f, 0.6f, 1.0f}};  // = kFringeColors[0], [1]
        for (int c = 0; c < 3; ++c) p.laShift[c] = sh[fringe][c];
    };
    switch (idx) {
    case 0: p.laMode = 0; p.laWidth = 0.003f; break;
    case 1: p.laMode = 0; p.laWidth = 0.012f; break;
    case 2: p.laMode = 1; p.laWidth = 0.004f; break;
    case 3: drift(0); break;
    case 4: drift(1); break;
    case 5:  // v2.1b: soft tinted spill (Rim Blur > 0), salmon-orange, faint complement
        p.laMode = 3; p.laWidth = 0.0004f; p.laThreshold = 2.2f; p.laCurve = 0.8f; p.laHueShift = 0.7f;
        p.laContrastScale = 0.006f; p.laLineProtect = 0.003f; p.laSharpness = 0.85f; p.laBrightColor = 0.5f;
        p.laBlur = 0.001f; p.laComplement = 0.06f;
        p.laTint[0] = 1.0f; p.laTint[1] = 0.40f; p.laTint[2] = 0.18f;
        break;
    case 6:  // v2.1 drawn rim, exactly as first delivered
        p.laMode = 3; p.laWidth = 0.0007f; p.laThreshold = 2.2f; p.laCurve = 0.8f; p.laHueShift = 0.7f;
        p.laContrastScale = 0.006f; p.laLineProtect = 0.003f; p.laSharpness = 0.85f; p.laBrightColor = 0.45f;
        break;
    }
    return true;
}

// ---- Fringe Color (laMode 4): which channels drift. Popup 1.. = kFringeColors[v-1].
struct FringeColor { const char* key; const char* ui; float shift[3]; };
static const FringeColor kFringeColors[] = {
    {"orange",  "Orange (G+B move)",        {-0.15f, 0.5f, 1.3f}},
    {"redcyan", "Red / Cyan (R opposite)",  {-0.8f, 0.6f, 1.0f}},
    {"blue",    "Blue (R+G move)",          {1.3f, 0.7f, -0.2f}},
    {"magenta", "Magenta (G moves)",        {0.f, 1.f, 0.f}},
    {"green",   "Green (R+B move)",         {1.f, 0.f, 1.f}},
};
static const int kFringeColorCount = (int)(sizeof(kFringeColors) / sizeof(kFringeColors[0]));
inline int findFringeColor(const char* key) {
    for (int i = 0; i < kFringeColorCount; ++i)
        if (std::strcmp(kFringeColors[i].key, key) == 0) return i;
    return -1;
}

// ---- Grain stocks (v2.1c). Generic names. Each writes size / softness / colour / tone / amount.
struct GrainStock { const char* key; const char* ui; float amount, size, softness, color, tone; };
static const GrainStock kGrainStocks[] = {
    // v2.1e: retuned for the Gaussian-filtered noise (smoother than the old lattice noise)
    {"fine35",   "Fine 35mm",         0.50f, 0.8f, 0.50f, 0.12f, 0.80f},
    {"medium35", "Medium 35mm",       0.75f, 1.2f, 0.45f, 0.18f, 0.75f},
    {"coarse16", "Coarse 16mm",       1.00f, 1.9f, 0.35f, 0.18f, 0.70f},
    {"pushed",   "Pushed High-ISO",   1.35f, 1.5f, 0.20f, 0.25f, 0.55f},
    {"home8",    "8mm Home Movie",    1.15f, 2.6f, 0.45f, 0.22f, 0.60f},
};
static const int kGrainStockCount = (int)(sizeof(kGrainStocks) / sizeof(kGrainStocks[0]));
inline int findGrainStock(const char* key) {
    for (int i = 0; i < kGrainStockCount; ++i)
        if (std::strcmp(kGrainStocks[i].key, key) == 0) return i;
    return -1;
}
inline bool applyGrainStock(int idx, Params& p) {
    if (idx < 0 || idx >= kGrainStockCount) return false;
    const GrainStock& g = kGrainStocks[idx];
    p.grain = g.amount; p.grainSize = g.size; p.grainSoftness = g.softness; p.grainColor = g.color; p.grainTone = g.tone;
    return true;
}

} // namespace lenscore
