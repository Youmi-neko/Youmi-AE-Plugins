// Optic Bokeh: the full effect's UI values (Settings) and their mapping to the engine (lenscore /
// lenshq Params). Shared by the effect (OpticBokeh.cpp, both editions: full and Kantan), the Kantan
// front end (kantan.cpp) and the offline tools. Moved verbatim out of OpticBokeh.cpp in v2.3 (Kantan):
// no behaviour change.
#pragma once
#include "lenscore.h"
#include "lens_presets.h"
#include "hq/lenshq_seam.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace obfx {

const float kDisplayGamma = 2.2f;

enum { VIEW_RESULT = 1, VIEW_BLUR_AMOUNT, VIEW_HL_SOURCE, VIEW_HL_ONLY };
enum { CH_LUMINANCE = 1, CH_RED, CH_GREEN, CH_BLUE, CH_ALPHA };
enum { SRC_AUTO = 1, SRC_LAYER, SRC_AUTO_X_LAYER };
// Quality popup (1-based) -> lenscore::kQuality* (0-based), see lens_presets.h
enum { QUALITY_FAST = 1, QUALITY_STANDARD, QUALITY_BEST };
// Aberration Side popup: 4 = Contrast Rim (v2.1) = lenscore laMode 3
enum { LA_SIDE_DARK = 1, LA_SIDE_BRIGHT, LA_SIDE_BOTH, LA_SIDE_RIM, LA_SIDE_DRIFT };
// Fringe Side popup (Channel Drift): 1 = retreat from the bright edge, 2 = advance into the dark side
enum { LA_DIR_RETREAT = 1, LA_DIR_ADVANCE };
// Edge Behavior popup (v2.2). Projects saved before v2.2 get Clip to Layer (= v1.2: output clipped to
// the input's content, transparent outside); new instances Repeat Edge (v2.3; v2.2.x instances keep the
// stored value, e.g. Spread). The hidden v2.1 checkbox, when on,
// forces Repeat Edge (EffectiveEdge).
enum { EDGE_SPREAD = 1, EDGE_REPEAT, EDGE_CLIP };
// Input Gamma popup (v2.2), used while Work in Linear Light is on. Auto: see InputIsLinear.
enum { GAMMA_AUTO = 1, GAMMA_22, GAMMA_LINEAR };
// Render Version (hidden, v2.2): lenscore Params.renderVersion. Old projects 1, new instances 2.
const int kRenderVersionCurrent = 2;
// Texture Type popup (UI order, append only) -> lenshq::TextureType values. The popup order is frozen
// for saved projects; the engine enum may change, so the mapping lives here.
const int kTextureTypeMap[] = {lenshq::TexMixed, lenshq::TexGrain, lenshq::TexDust,
                               lenshq::TexStreaks, lenshq::TexPolish, lenshq::TexBusy};
const int kTextureTypeCount = (int)(sizeof(kTextureTypeMap) / sizeof(kTextureTypeMap[0]));

// Raw UI values (units as shown in the UI).
struct Settings
{
    int view;
    double radius;     // px at full resolution
    int mapChannel;
    bool invertMap;
    double focusPct;
    double mapScale;
    double mapGamma;
    int blades;
    double curvaturePct;
    double rotationDeg;
    double anamorphic; // bokeh height / width
    double catEyePct;
    double edgeRingPct;
    double ringWidthPct;
    double chromaPct;
    double hlThreshold;
    double hlGain;
    bool linear;
    // v1.1
    double cornerPct;
    double obstructionPct;
    double saPct;
    double softnessPct;
    double apodPct;
    double corePct;
    double onionPct;
    double onionCount;
    double vignettePct;
    double glowPct;
    double glowSizePct; // % of frame diagonal
    // v1.2
    double fringeTintPct;
    double hlBoost;
    double hlKnee;
    double hlCurve;
    double hlSelPct;
    double hlDetail;    // px at full resolution
    double hlSatPct;
    double hlRolloffPct;
    // v2.0
    double hlWhitePct;
    double hlDotsPct;
    double hlDetailMin; // px at full resolution
    int hlSource;
    int hlLayerChannel;
    double halPct;
    double halSizePct;  // % of frame diagonal
    double halThreshold;
    double halSoftPct;
    double halRimPct;
    float halTint[3];   // linear
    double laPct;
    double laWidthPct;  // % of frame diagonal
    double laThreshold; // stops
    double laCurve;
    double laHuePct;
    float laTint[3];    // linear
    int laMode;
    double hqTexPct;
    double hqTexScale;
    int hqTexType;
    int hqTexSeed;
    int hqSpectral;
    double hqLateralPct;
    bool repeatEdge;
    // v2.1
    int quality;
    double halRelative; // stops
    double laScalePct;  // % of frame diagonal
    double laLinePct;   // % of frame diagonal
    double laSharpPct;
    double laBrightColPct;
    // v2.1b-d
    double laBlurPct;   // % of frame diagonal
    double laComplementPct;
    int laFringe;       // popup, 1.. = kFringeColors[v-1]
    int laDir;
    double laMapSoftPct; // % of frame diagonal
    double laPreservePct;
    double grainPct;
    double grainSize;   // full-res px
    double grainSoftPct;
    double grainColorPct;
    double grainTonePct;
    bool grainAnimate;
    int grainHold;
    int grainSeed;
    // v2.2
    int edgeMode;       // popup EDGE_*; see EffectiveEdge (the hidden v2.1 checkbox can force Repeat)
    int inputGamma;     // popup GAMMA_*
    int renderVersion;  // hidden: 1 = v2.1 looks (old projects), 2 = current
    int edge;           // effective edge behaviour (EDGE_*), set by ReadSettings
    // v2.3 blur-map features (ob_mapx.h). Defaults = no change (the classic map path).
    double mapBlurPx;     // Map Blur: Gaussian sigma, full-res px
    double focusRangePct; // Focus Range: % of the map range kept fully sharp around Focus
    double frontPct;      // Front Blur: scale of the blur in front of focus, %
    double backPct;       // Back Blur: behind focus, %
    bool focusPointOn;    // Focus at Point
    double focusPointX;   // Focus Point, effect-layer px as the point param gives them
    double focusPointY;
    // v2.4
    int useGpu;           // Use GPU popup: 1 Auto, 2 Off (CPU only)
};

inline int EffectiveEdge(int popup, bool legacyRepeat)
{
    if (legacyRepeat)
        return EDGE_REPEAT;
    return popup == EDGE_REPEAT || popup == EDGE_CLIP ? popup : EDGE_SPREAD;
}

// Whether the input pixels are already linear (Work in Linear Light then skips the 2.2 decode).
// Auto: the host's colour settings would decide, but AE exposes them to an effect only through AEGP
// calls that need a registered AEGP plug-in id and the comp handle, and are not documented as safe on
// MFR render threads; v2.2 therefore treats Auto as 2.2 (the v2.1 behaviour) and lets the user pick
// Linear for linearized / OCIO-linear working spaces. Documented in the manual notes (覚書.md).
inline bool InputIsLinear(int inputGamma)
{
    return inputGamma == GAMMA_LINEAR;
}

inline float SrgbToLinear(unsigned char v)
{
    return std::pow(v / 255.f, kDisplayGamma);
}

// ------------------------------------------------------- engine adapter ---
// Every lenscore-specific assumption lives in this section.

// UI settings + render geometry -> engine params. sx/sy = engine px per full-res square px.
// lenscore's anamorphic is height/width, area preserving (rx = r/sqrt(a), ry = r*sqrt(a)), so
// non-square scaling (downsample x != y, pixel aspect) folds into radius and anamorphic.
inline lenscore::Params MakeEngineParams(const Settings& s, double sx, double sy)
{
    lenscore::Params p;
    const double a = std::min(10.0, std::max(0.1, s.anamorphic * sy / sx));
    const double sc = std::sqrt(sx * sy);
    p.maxRadius = (float)std::min(4000.0, std::max(0.0, s.radius * sc));
    // area of a buffer pixel in full-res square px: full-res-defined quantities (highlight disc area,
    // Min Detail Size ramp) stay the same at Half / Quarter resolution
    p.pixelScale = (float)std::max(1e-4, sx * sy);
    p.anamorphic = (float)a;
    p.focus = (float)(s.focusPct / 100.0);
    p.mapScale = (float)s.mapScale;
    p.mapGamma = (float)std::max(0.01, s.mapGamma);
    p.invertMap = s.invertMap;
    p.blades = s.blades < 3 ? 0 : s.blades;
    p.bladeCurvature = (float)(s.curvaturePct / 100.0);
    p.rotationDeg = (float)s.rotationDeg;
    p.catEye = (float)(s.catEyePct / 100.0);
    p.edgeRing = (float)(s.edgeRingPct / 100.0);
    p.ringWidth = (float)(s.ringWidthPct / 100.0);
    p.chromaFringe = (float)(s.chromaPct / 100.0);
    p.highlightThreshold = (float)s.hlThreshold;
    p.highlightGain = (float)s.hlGain;
    p.cornerRounding = (float)(s.cornerPct / 100.0);
    p.obstruction = (float)(s.obstructionPct / 100.0);
    p.sphericalAberration = (float)(s.saPct / 100.0);
    p.edgeSoftness = (float)(s.softnessPct / 100.0);
    p.apodization = (float)(s.apodPct / 100.0);
    p.coreWeight = (float)(s.corePct / 100.0);
    p.onionRings = (float)(s.onionPct / 100.0);
    p.onionCount = (float)s.onionCount;
    p.vignetting = (float)(s.vignettePct / 100.0);
    p.glow = (float)(s.glowPct / 100.0);
    p.glowSize = (float)(s.glowSizePct / 100.0); // relative to the frame: no downsample scaling
    p.fringeTint = (float)(s.fringeTintPct / 100.0);
    p.highlightBoost = (float)s.hlBoost;
    p.highlightKnee = (float)s.hlKnee;
    p.highlightCurve = (float)s.hlCurve;
    p.highlightSelectivity = (float)(s.hlSelPct / 100.0);
    p.detailSize = (float)std::max(1.0, s.hlDetail * sc); // px, scales like Radius
    p.highlightSaturation = (float)(s.hlSatPct / 100.0);
    p.highlightRolloff = (float)(s.hlRolloffPct / 100.0);
    p.whiteProtection = (float)(s.hlWhitePct / 100.0);
    p.dotsOnly = (float)(s.hlDotsPct / 100.0);
    p.detailMin = (float)std::max(0.0, s.hlDetailMin * sc);
    p.highlightSource = s.hlSource == SRC_LAYER ? 1 : (s.hlSource == SRC_AUTO_X_LAYER ? 2 : 0);
    p.halation = (float)(s.halPct / 100.0);
    p.halationSize = (float)(s.halSizePct / 100.0);
    p.halationThreshold = (float)s.halThreshold;
    p.halationSoftness = (float)(s.halSoftPct / 100.0);
    p.halationRim = (float)(s.halRimPct / 100.0);
    for (int i = 0; i < 3; ++i)
    {
        p.halationTint[i] = s.halTint[i];
        p.laTint[i] = s.laTint[i];
    }
    p.lumAberration = (float)(s.laPct / 100.0);
    p.laWidth = (float)(s.laWidthPct / 100.0);
    p.laThreshold = (float)s.laThreshold;
    p.laCurve = (float)s.laCurve;
    p.laHueShift = (float)(s.laHuePct / 100.0);
    p.laMode = s.laMode == LA_SIDE_BRIGHT ? 1
             : s.laMode == LA_SIDE_BOTH   ? 2
             : s.laMode == LA_SIDE_RIM    ? 3
             : s.laMode == LA_SIDE_DRIFT  ? 4 : 0;
    p.laBlur = (float)(s.laBlurPct / 100.0);
    p.laComplement = (float)(s.laComplementPct / 100.0);
    {
        const int fi = std::min(lenscore::kFringeColorCount, std::max(1, s.laFringe)) - 1;
        for (int c = 0; c < 3; ++c)
            p.laShift[c] = lenscore::kFringeColors[fi].shift[c];
    }
    p.laShiftDir = s.laDir == LA_DIR_ADVANCE ? 1 : 0;
    p.laMapSoftness = (float)(s.laMapSoftPct / 100.0);
    p.laPreserve = (float)(s.laPreservePct / 100.0);
    p.grain = (float)(s.grainPct / 100.0);
    p.grainSize = (float)s.grainSize;  // full-res px; the downsample enters via grainScaleX/Y (PreRender)
    p.grainSoftness = (float)(s.grainSoftPct / 100.0);
    p.grainColor = (float)(s.grainColorPct / 100.0);
    p.grainTone = (float)(s.grainTonePct / 100.0);
    p.grainSeed = s.grainSeed;
    p.halationRelative = (float)s.halRelative;
    p.laContrastScale = (float)(s.laScalePct / 100.0);
    p.laLineProtect = (float)(s.laLinePct / 100.0);
    p.laSharpness = (float)(s.laSharpPct / 100.0);
    p.laBrightColor = (float)(s.laBrightColPct / 100.0);
    // Quality: Fast = blur shortcuts, Standard = exact span engine (v1.2 multi-resolution rule),
    // Best = full resolution (+ the quality engine once wired, see BlurStage). The margins
    // (blurMargin/inputMargin) follow from these flags, so PreRender always uses the selected mode's.
    lenscore::applyQuality(s.quality == QUALITY_FAST ? lenscore::kQualityFast
                           : s.quality == QUALITY_BEST ? lenscore::kQualityBest : lenscore::kQualityStandard, p);
    // Worker threads: set per render call from the number of renders running at once (RenderThreads).
    const unsigned hw = std::thread::hardware_concurrency();
    p.threads = hw > 0 ? (int)hw : 4;
    p.renderVersion = s.renderVersion;
    // frame geometry is set per render call (RenderDepth), relative to the engine buffer
    return p;
}

inline lenscore::HqParams MakeHqParams(const Settings& s)
{
    lenscore::HqParams h;
    h.textureAmount = (float)(s.hqTexPct / 100.0);
    h.textureScale = (float)std::min(4.0, std::max(0.25, s.hqTexScale));
    h.textureType = kTextureTypeMap[std::min(kTextureTypeCount, std::max(1, s.hqTexType)) - 1];  // UI order -> engine enum
    h.textureSeed = s.hqTexSeed;
    // UI: 0 = Auto (spectral, recommended), 1-2 = RGB three-band, >= 3 = that many wavelengths
    h.spectralSamples = s.hqSpectral <= 0 ? -1 : (s.hqSpectral < 3 ? 0 : s.hqSpectral);
    h.lateralCA = (float)(s.hqLateralPct / 100.0);
    return h;
}

}  // namespace obfx
