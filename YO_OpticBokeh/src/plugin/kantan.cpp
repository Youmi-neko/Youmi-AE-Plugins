// Optic Bokeh Kantan v2.3: Kantan controls -> the full effect's UI values (see kantan.h and
// docs\kantan_design.md for the reasoning behind every link).
//
// Exactness rules (the anchors must equal the full effect bit for bit):
//  * values a popup of the full effect writes (Lens Preset, Aberration Style, Grain Stock) are computed
//    exactly as Apply*ToParams computes them (double, e.g. q.catEye * 100.0; rotation through PF_Fixed);
//  * values a new full instance gets from a slider default go through float (PF_FpShort dephault): F();
//  * colours go through the 8-bit colour param (Enc8 -> SrgbToLinear), as AddColor / ReadSettings do;
//  * every link is written so that at its anchor it multiplies by exactly 1.0 or adds exactly 0.0.
#include "kantan.h"

#include <algorithm>
#include <cmath>

namespace kantan {
namespace {

using namespace obfx;

inline double F(double v)  // a slider default as AE stores it (PF_FpShort = float)
{
    return (double)(float)v;
}
inline double Clamp(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}
inline unsigned char Enc8(float v)  // = AddColor's / ApplyAberrationStyleToParams' enc
{
    return (unsigned char)std::lround(255.0 * std::pow(std::min(1.0f, std::max(0.0f, v)), 1.0 / kDisplayGamma));
}
inline double Smooth(double t)  // smoothstep: C1, 0 -> 0, 1 -> 1, zero slope at both ends
{
    t = Clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Character link for one signed "imperfection" (UI %). c <= 1: linear toward 0 (v * c, exactly v at
// c = 1). c > 1: grows with the same slope at c = 1 (C1) and saturates toward `limit` (never past it):
// v + (L - v)(1 - exp(-k (c - 1))), k = v / (L - v). A zero stays zero (a clean preset stays clean).
double CharacterScale(double v, double c, double limit)
{
    if (c <= 1.0)
        return v * c;
    const double a = std::fabs(v);
    if (a <= 0.0)
        return v;
    const double L = std::max(limit, a);
    if (L - a < 1e-9)
        return v;
    const double k = a / (L - a);
    const double r = a + (L - a) * (1.0 - std::exp(-k * (c - 1.0)));
    return v < 0 ? -r : r;
}

// --- link version 1 ----------------------------------------------------------------------------

void FullDefaults(Settings& s)
{
    // a new instance of the full effect (ParamsSetup dephaults), before any popup writes
    const lenscore::Params def0;
    s.view = VIEW_RESULT;
    s.radius = F(20);
    s.mapChannel = CH_LUMINANCE;
    s.invertMap = false;
    s.focusPct = F(0);
    s.mapScale = F(1);
    s.mapGamma = F(1);
    s.hlThreshold = F(1);  // Classic Gain (v1)
    s.hlGain = F(0);
    s.linear = true;
    s.hlRolloffPct = F(50);
    s.hlSource = SRC_AUTO;
    s.hlLayerChannel = CH_LUMINANCE;
    s.halPct = F(0);
    s.halSizePct = F(def0.halationSize * 100.0);
    s.halThreshold = F(def0.halationThreshold);
    s.halRelative = F(def0.halationRelative);
    s.halSoftPct = F(def0.halationSoftness * 100.0);
    s.halRimPct = F(def0.halationRim * 100.0);
    for (int i = 0; i < 3; ++i)
        s.halTint[i] = SrgbToLinear(Enc8(def0.halationTint[i]));
    s.laPct = F(0);
    s.grainAnimate = true;
    s.grainHold = 1;
    s.grainSeed = 0;
    s.hqTexPct = F(0);
    s.hqTexScale = F(1);
    s.hqTexType = 1;
    s.hqTexSeed = 0;
    s.hqSpectral = 0;
    s.hqLateralPct = F(0);
    s.repeatEdge = false;
    s.quality = QUALITY_STANDARD;
    s.edgeMode = EDGE_REPEAT;
    s.inputGamma = GAMMA_AUTO;
    s.renderVersion = kRenderVersionCurrent;
    s.mapBlurPx = F(0);
    s.focusRangePct = F(0);
    s.frontPct = F(100);
    s.backPct = F(100);
    s.focusPointOn = false;
    s.focusPointX = s.focusPointY = 0;  // the point param's default is the layer centre (host-dependent)
    s.useGpu = 1;
}

// = ApplyPresetToParams (OpticBokeh.cpp) on Settings
void ApplyPreset(int idx, Settings& s)
{
    lenscore::Params q;
    lenscore::applyLensPreset(idx, q);
    s.blades = q.blades;
    s.rotationDeg = (double)(int)std::lround(q.rotationDeg * 65536.0) / 65536.0;  // PF_Fixed angle
    s.curvaturePct = q.bladeCurvature * 100.0;
    s.cornerPct = q.cornerRounding * 100.0;
    s.anamorphic = q.anamorphic;
    s.catEyePct = q.catEye * 100.0;
    s.obstructionPct = q.obstruction * 100.0;
    s.edgeRingPct = q.edgeRing * 100.0;
    s.ringWidthPct = q.ringWidth * 100.0;
    s.softnessPct = q.edgeSoftness * 100.0;
    s.apodPct = q.apodization * 100.0;
    s.corePct = q.coreWeight * 100.0;
    s.saPct = q.sphericalAberration * 100.0;
    s.onionPct = q.onionRings * 100.0;
    s.onionCount = q.onionCount;
    s.chromaPct = q.chromaFringe * 100.0;
    s.vignettePct = q.vignetting * 100.0;
    s.glowPct = q.glow * 100.0;
    s.glowSizePct = q.glowSize * 100.0;
    s.fringeTintPct = q.fringeTint * 100.0;
    s.hlBoost = q.highlightBoost;
    s.hlKnee = q.highlightKnee;
    s.hlCurve = q.highlightCurve;
    s.hlSelPct = q.highlightSelectivity * 100.0;
    s.hlDetail = q.detailSize;
    s.hlDetailMin = q.detailMin;
    s.hlWhitePct = q.whiteProtection * 100.0;
    s.hlDotsPct = q.dotsOnly * 100.0;
    s.hlSatPct = q.highlightSaturation * 100.0;
}

// = ApplyAberrationStyleToParams
void ApplyStyle(int idx, Settings& s)
{
    lenscore::Params q;
    lenscore::applyAberrationStyle(idx, q);
    s.laMode = q.laMode == 1 ? LA_SIDE_BRIGHT : q.laMode == 2 ? LA_SIDE_BOTH : q.laMode == 3 ? LA_SIDE_RIM
             : q.laMode == 4 ? LA_SIDE_DRIFT : LA_SIDE_DARK;
    s.laWidthPct = q.laWidth * 100.0;
    s.laThreshold = q.laThreshold;
    s.laCurve = q.laCurve;
    s.laHuePct = q.laHueShift * 100.0;
    for (int i = 0; i < 3; ++i)
        s.laTint[i] = SrgbToLinear(Enc8(q.laTint[i]));
    s.laScalePct = q.laContrastScale * 100.0;
    s.laLinePct = q.laLineProtect * 100.0;
    s.laSharpPct = q.laSharpness * 100.0;
    s.laBrightColPct = q.laBrightColor * 100.0;
    s.laBlurPct = q.laBlur * 100.0;
    s.laComplementPct = q.laComplement * 100.0;
    s.laMapSoftPct = q.laMapSoftness * 100.0;
    s.laPreservePct = q.laPreserve * 100.0;
    int fringe = 1;
    for (int i = 0; i < lenscore::kFringeColorCount; ++i)
    {
        const float* f = lenscore::kFringeColors[i].shift;
        if (f[0] == q.laShift[0] && f[1] == q.laShift[1] && f[2] == q.laShift[2])
            fringe = i + 1;
    }
    s.laFringe = fringe;
    s.laDir = q.laShiftDir == 1 ? LA_DIR_ADVANCE : LA_DIR_RETREAT;
}

// = ApplyGrainStockToParams
void ApplyStock(int idx, Settings& s)
{
    lenscore::Params q;
    lenscore::applyGrainStock(idx, q);
    s.grainPct = q.grain * 100.0;
    s.grainSize = q.grainSize;
    s.grainSoftPct = q.grainSoftness * 100.0;
    s.grainColorPct = q.grainColor * 100.0;
    s.grainTonePct = q.grainTone * 100.0;
}

void LinksV1(const Values& v, Settings& s)
{
    // ---- Character: the preset's imperfections, coherently (0 = clean, 1 = as designed, 2 = more)
    const double c = Clamp(v.characterPct, 0.0, kCharacterMax) / 100.0;
    s.edgeRingPct = CharacterScale(s.edgeRingPct, c, 90.0);
    s.saPct = CharacterScale(s.saPct, c, 95.0);
    s.onionPct = CharacterScale(s.onionPct, c, 70.0);
    s.catEyePct = CharacterScale(s.catEyePct, c, 100.0);
    s.vignettePct = CharacterScale(s.vignettePct, c, 70.0);
    s.glowPct = CharacterScale(s.glowPct, c, 35.0);
    s.corePct = CharacterScale(s.corePct, c, 90.0);
    if (c > 1.0)
    {
        // Best only: disc texture (dust / polishing marks) of an old, worn lens. Zero with zero slope
        // at 100 % (C1), so the anchor is untouched. (Lateral CA belongs to Chromatic Aberration.)
        const double e = (c - 1.0) * (c - 1.0);
        s.hqTexPct = 40.0 * e;
    }

    // ---- Chromatic Aberration (own control; Character does not scale the fringe, so the two never
    // double up). k <= 1: the preset's own fringe x k (exactly the preset at 100 %). k > 1: grows with the
    // same slope at 100 % (C1) and bends up toward preset + max(preset, 10 %) at 200 %, so a lens
    // designed without longitudinal CA (Creamy, Dreamy, Mirror: 0) still gets a fringe above 100 %.
    // The sign (which colour is outside) is the preset's. Best: lateral CA follows from 100 % up.
    {
        const double k = Clamp(v.chromaPct, 0.0, kChromaMax) / 100.0;
        const double p = s.chromaPct, a = std::fabs(p);
        if (k <= 1.0)
            s.chromaPct = p * k;
        else
        {
            const double e = k - 1.0, ref = std::max(a, 10.0);
            const double mag = a + a * e + (ref - a) * e * e;
            s.chromaPct = p < 0 ? -mag : mag;
            s.hqLateralPct = 30.0 * e * e;
        }
    }

    // ---- Aperture: what a real iris changes together when it closes (0 = wide open = as designed)
    const double A = Clamp(v.aperturePct, 0.0, kApertureMax) / 100.0;
    const double sA = Smooth(A);
    if (A > 0.0)
    {
        if (s.blades < 3)
        {
            // a round-looking fast lens has a (typically 9-blade) iris behind it: round wide open
            // (curvature >= 99.99 % is an exact circle in both engines), polygonal when stopped down
            s.blades = 9;
            s.rotationDeg = 0.0;
            s.curvaturePct = 100.0 * (1.0 - 0.75 * sA);
        }
        else
            s.curvaturePct *= 1.0 - 0.7 * sA;  // curved blades straighten
    }
    s.cornerPct *= 1.0 - 0.6 * sA;
    s.catEyePct *= (1.0 - A) * (1.0 - A);  // the iris becomes the limiting stop: no clipping
    s.vignettePct *= 1.0 - 0.6 * sA;       // optical vignetting goes, natural falloff stays
    s.saPct *= 1.0 - 0.8 * sA;             // spherical aberration ~ aperture^2
    s.edgeRingPct *= 1.0 - 0.5 * sA;
    s.onionPct *= 1.0 - 0.3 * sA;          // asphere mould marks: only slightly
    s.chromaPct *= 1.0 - 0.5 * sA;         // LoCA shrinks with the depth of focus
    s.glowPct *= 1.0 - 0.8 * sA;           // veiling glow from SA
    s.corePct *= 1.0 - 0.6 * sA;
    s.softnessPct *= 1.0 - 0.6 * sA;       // crisper disc edges

    // ---- Bokeh Highlights: how readily and how strongly lights become discs (0.5 = as designed)
    const double h = Clamp(v.highlightsPct, 0.0, kHighlightsMax) / 100.0;
    const double u = 2.0 * h - 1.0;  // -1 .. 0 (anchor) .. 1
    s.hlBoost *= (2.0 * h) * (1.0 + 0.5 * u);
    s.hlKnee = Clamp(s.hlKnee - 0.15 * u, 0.05, 0.95);
    s.hlCurve = Clamp(s.hlCurve * (1.0 - 0.25 * u), 1.0, 8.0);
    s.hlSelPct = Clamp(s.hlSelPct - 10.0 * u, 0.0, 100.0);
    s.hlSatPct = Clamp(s.hlSatPct * (1.0 + 0.3 * u), 0.0, 100.0);

    // ---- Film Look
    const double ha = Clamp(v.halationPct, 0.0, kHalationMax) / 100.0;
    s.halPct = Clamp(v.halationPct, 0.0, kHalationMax);
    s.halSizePct *= 0.7 + 0.3 * ha;        // brighter halation reads wider
    s.halRelative -= 0.5 * (ha - 1.0);     // stronger: lights less far above the scene halate too
    s.halRimPct += 15.0 * (1.0 - ha);      // faint: the tight red rim; strong: the wide glow
    const double la = Clamp(v.laPct, 0.0, kLaMax) / 100.0;
    s.laPct = Clamp(v.laPct, 0.0, kLaMax);
    s.laWidthPct *= 0.75 + 0.25 * la;      // a stronger aberration also spreads further
    s.laBlurPct *= 0.75 + 0.25 * la;
    // 0..100 %: linear up to the stock's own amount; above, the same smooth saturation as Character so
    // the strongest stock (135 %) bends toward the full effect's 200 % limit instead of hitting it
    const double g = Clamp(v.grainPct, 0.0, kGrainMax) / 100.0;
    s.grainPct = CharacterScale(s.grainPct, g, 200.0);
}

}  // namespace

void ToSettings(const Values& v, Settings& s)
{
    FullDefaults(s);
    ApplyPreset(std::min(lenscore::kLensPresetCount, std::max(1, v.lens)) - 1, s);
    ApplyStyle(std::min(lenscore::kAberrationStyleCount, std::max(1, v.laStyle)) - 1, s);
    ApplyStock(std::min(lenscore::kGrainStockCount, std::max(1, v.grainStock)) - 1, s);
    s.view = std::min(4, std::max(1, v.view));
    s.quality = std::min(3, std::max(1, v.quality));
    s.radius = Clamp(v.radius, 0.0, kRadiusMax);
    s.invertMap = v.invertMap;
    s.focusPct = Clamp(v.focusPct, 0.0, 100.0);
    s.mapChannel = std::min(5, std::max(1, v.mapChannel));
    s.mapScale = Clamp(v.depthStrengthPct, 0.0, kDepthStrengthMax) / 100.0;  // exactly 1 at 100 %
    s.mapGamma = Clamp(v.depthCurve, F(0.1), 10.0);  // F: the slider's own (float) minimum
    s.focusRangePct = Clamp(v.focusRangePct, 0.0, 100.0);
    s.frontPct = Clamp(v.frontPct, 0.0, 400.0);
    s.backPct = Clamp(v.backPct, 0.0, 400.0);
    s.mapBlurPx = Clamp(v.mapBlurPx, 0.0, 200.0);
    s.focusPointOn = v.focusPointOn;
    s.focusPointX = v.focusPointX;
    s.focusPointY = v.focusPointY;
    s.useGpu = v.useGpu == 2 ? 2 : 1;
    s.edgeMode = v.edgeMode == EDGE_SPREAD || v.edgeMode == EDGE_CLIP ? v.edgeMode : EDGE_REPEAT;
    // link versions: 1 (v2.3). Unknown (newer) versions render with the newest links this build has.
    LinksV1(v, s);
    s.renderVersion = kRenderVersionCurrent;  // engine algorithms of link version 1
    s.edge = EffectiveEdge(s.edgeMode, s.repeatEdge);
}

}  // namespace kantan
