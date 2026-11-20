// lenshq: "beta" (quality-first) blur stage of Optic Bokeh.
// C++17, standard library only. Shares lenscore::Params (lens controls and presets) with the fast
// "alpha" engine; beta-only controls live in HqOptions.
//
// PIPELINE SEAM. The full effect is
//     [shared pre-pass ] highlight estimation: adds HDR energy to bright details (values >> 1)
//     [this engine     ] lenshq::render  -> scatter blur (+ optional vignetting)
//     [shared post-pass] veiling glow, output highlight roll-off
// render() treats its input as already scene-referred HDR and NEVER reads highlightThreshold,
// highlightGain, highlightBoost, highlightKnee, highlightCurve, highlightSelectivity, detailSize,
// highlightSaturation, highlightRolloff, glow, glowSize or fullResolution.
// To keep the shared post-pass exact, render() can carry one auxiliary scalar plane through the
// same kernels (HqOptions::auxSrc/auxDst, e.g. the luminance of the energy the pre-pass added,
// which the roll-off needs), and vignetting can be switched off here and applied after the glow
// with lenshq::applyVignetting (alpha's order is blur -> glow -> vignetting -> roll-off).
#pragma once

#include "lenscore.h"

#include <vector>

namespace lenshq {

using lenscore::AbortFn;
using lenscore::Params;
using lenscore::Status;

enum TextureType {
    TexMixed = 0,    // grain + a few dust specks + faint wipe streaks + polishing marks
    TexGrain = 1,    // fine "orange peel" grain only (aspherical surface roughness)
    TexDust = 2,     // dust shadows with faint Fresnel rings (+ a little grain)
    TexStreaks = 3,  // parallel cleaning / wipe streaks (+ a little grain)
    TexPolish = 4,   // concentric + radial polishing marks (irregular onion texture)
    TexBusy = 5,     // "soap bubble" irregularity: uneven disc and uneven rim
};

enum FringeModel {
    FringeSpectral = 0,   // N wavelengths, smooth rainbow-graded rims (default)
    FringeThreeBand = 1,  // alpha-compatible: one kernel scale per R/G/B channel
};

struct HqStats {
    double secModel = 0, secScatter = 0, secResolve = 0;  // wall time of the phases
    double tableMB = 0, accumMB = 0, workMB = 0;          // memory of tables / accumulators / per-thread work
    int tableRes = 0, spectralSamplesA = 0, spectralSamplesB = 0;  // B == A (one wavelength set since v2.2)
    bool pathB = false;   // position-dependent kernels (cat eye / lateral CA / disc falloff)
    bool tableCacheHit = false;  // the lens tables came from the process-wide cache
    long long splats = 0, stampsComputed = 0;
    long long stampPixels = 0, wavelengthPixels = 0;  // pixels integrated per wavelength (clip edges / lateral CA)
    long long latticeSplats = 0;                      // splats blended from lattice stamps
    int accumScaleLog2 = 40;                          // fixed-point scale chosen for this render
};

struct HqOptions {
    // ---- in-disc aperture texture (fixed per lens; the same pattern appears in every disc) ----
    float textureAmount = 0.f;   // 0..2   0 = off, 1 = clearly visible
    float textureScale = 1.f;    // 0.25..4 feature size multiplier (aperture coordinates)
    int   textureSeed = 1;       // pattern seed
    int   textureType = TexMixed;

    // ---- chromatic aberration ----
    int   fringeModel = FringeSpectral;
    int   spectralSamples = 0;   // wavelengths for position-dependent kernels; 0 = auto (3..15)
    float lateralCA = 0.f;       // -1..1 beta-only: wavelength-dependent disc shift growing toward
                                 // the frame edges, proportional to the blur radius (0 at focus)

    // ---- cat eye (mechanical vignetting) ----
    float catEyeSoftness = 0.02f;  // 0..0.3 width of the clip edge, fraction of the aperture radius
    float catEyeClipRim = 1.f;     // 0..1 edge character (rim / soft edge) along the clipped edge
                                   //   1 = like the aperture edge (alpha, matches outlined real
                                   //   lenses), 0 = plain mechanical cut
    float catEyeLightLoss = 0.f;   // 0 = each disc normalised (alpha); 1 = physically correct: the
                                   //   clipped light is lost (all pixels, as real optical vignetting)
    float catEyeClipRadius = 1.f;  // 0.7..1.5 clip circle radius relative to the aperture
    float catEyeCurve = 1.f;       // 0.5..3 growth of the clip offset with frame radius (r^curve);
                                   //   >1 keeps the centre round longer and swirls harder at the edges

    // ---- radial profile ----
    float rimShape = 0.f;          // 0..1 0 = flat rim band (alpha), 1 = caustic rim (brightest at
                                   //   the very edge, same energy)
    bool  invertFrontBokeh = false;// true: front-of-focus discs are the pupil image rotated 180
                                   //   degrees relative to behind-focus discs (physically correct;
                                   //   visible with odd blade counts and textures). false = lenscore
                                   //   (Fast/Standard) convention, so all qualities agree.

    // ---- optional extra realism (default off) ----
    float diffraction = 0.f;       // 0..1 slight edge softening, sigma = 0.15*d*sqrt(R) px
    float discFalloff = 0.f;       // 0..1 cos^4-like brightness tilt across off-axis discs

    // ---- quality ----
    int   tableResolution = 0;     // texels across the aperture (multiple of 64, 256..4096); 0 = auto
    float minTexelsPerPixel = 6.f; // table mip selection (higher = closer to the exact box integral)
    int   tableBudgetMB = 1024;    // max memory of one lens table set (resolution is reduced to fit)
    float latticeTolerance = 1.f;  // stamp lattice: 0 = exact kernel for every source pixel (slow);
                                   //   1 = default (kernels blended from exact stamps on a lattice in
                                   //   radius and frame position, L1 error <~ 1e-3 vs exact); >1 coarser

    // ---- pipeline seam ----
    bool  applyVignetting = true;  // Params::vignetting applied after the blur
    bool  normalizeByWeight = true;// false = raw scatter (energy test / debugging)
    const float* auxSrc = nullptr; // optional scalar plane blurred with the luminance kernel
    int   auxSrcStride = 0;        //   (floats per row)
    float* auxDst = nullptr;
    int   auxDstStride = 0;

    // per-pixel radius (px) / side (+1 behind, -1 in front) overriding the map (used by the
    // lenscore::Prepared adapter, lenshq_seam.h); stride = w (elements)
    const double* radiusIn = nullptr;
    const signed char* sideIn = nullptr;

    HqStats* stats = nullptr;      // optional output
};

// src : premultiplied linear float RGBA (may contain HDR energy >> 1), row stride in floats.
// map : single-channel float (blur map), stride in floats; nullptr => amount 1 everywhere.
// dst : premultiplied linear float RGBA. Must not overlap src or map.
// abort: polled only from the calling thread (about every 10 ms of work, every phase).
// Concurrent calls are fine. The only shared state is a process-wide cache of immutable lens tables
// (mutex-protected, shared_ptr, LRU, at most 8 sets / 1536 MB); a cached table
// set is never modified after it is built, so sharing it across threads / MFR frames is safe.
// Output is clamped to >= 0 and is bit-identical for any thread count and any tiling that uses
// inputMargin.
Status render(const float* src, int w, int h, int srcStride,
              const float* map, int mapStride,
              float* dst, int dstStride,
              const Params& p, const HqOptions& q,
              AbortFn abort = nullptr, void* user = nullptr);

// Input margin (px) a tiled host must add around an output rect for an exact result.
int inputMargin(const Params& p, const HqOptions& q);

// Drop every cached lens table set (e.g. on plugin sequence teardown / low memory).
void purgeTableCache();

// Params::vignetting on an RGBA buffer (frame-relative, same law as alpha: rgb *= 1 - v * rn^2).
void applyVignetting(float* rgba, int w, int h, int stride, const Params& p);

// Per-pixel blur amount (0..1) for a map value, sign = +1 behind focus, -1 in front.
float blurAmount(float mapValue, const Params& p, int* sign = nullptr);

// Debug / validation: the normalised per-channel kernel of ONE source pixel at buffer position
// (px, py) (pixel centre = integer + 0.5) with radius R and side (+1 behind, -1 in front).
// Returns planes r, g, b of size kw*kh; plane[(j*kw)+i] is the weight landing on output pixel
// (x0 + i, y0 + j) relative to the source pixel. Frame = buffer w x h unless Params sets it.
Status kernelStamp(const Params& p, const HqOptions& q, int w, int h, double R, int side,
                   int px, int py, int& x0, int& y0, int& kw, int& kh,
                   std::vector<float>& planes /* 3*kw*kh */);   // exact kernel; R <= 4000

} // namespace lenshq
