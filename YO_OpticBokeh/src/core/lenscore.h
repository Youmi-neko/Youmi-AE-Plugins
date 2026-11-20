// lenscore: SDK-independent scatter-based camera lens blur engine.
// C++17, standard library only. No host/OS dependencies.
//
// ============================== STAGE SEAM (v2.0) ==============================
// render() is exactly   prepareHighlights() -> blurAlpha() -> postProcess().
//
//  Stage 1  prepareHighlights(src, map, highlight layer, Params) -> Prepared
//           Everything that decides WHAT is blurred: per pixel the working colour (source +
//           added highlight energy; premultiplied linear RGBA), the blur radius in px
//           (= blurAmount(map) * maxRadius, double), the side (+1 behind focus / -1 in front;
//           +1 without a map), and eL = luminance of the added highlight energy.
//           Optional: the added energy itself (for the "Highlight Source"/"Highlights Only" views).
//
//  Stage 2  BLUR. Prepared + the lens-character fields of Params -> blurred premultiplied RGBA,
//           plus (only if postNeedsEL(p, prep)) the blur of Prepared.eL normalised the same way
//           as the colour (sum w*eL / sum w, weights of the G kernel), one float per pixel.
//           Fast/Standard/Best quality: blurAlpha() below (span engine; Params.fastBlur /
//           fullResolution select the variant). A quality engine (core/hq) replaces only this
//           stage; it must keep: radius/side semantics, energy conservation, clamping of
//           negative output to 0, identity where radius*... <= 0.5 px, and the frame rect
//           (Params.frame*) for position-dependent effects (cat eye).
//
//  Stage 3  postProcess(Prepared, Params, rgba in/out, eL) -> veiling glow (from Prepared.rgba),
//           vignetting, halation, luminance aberration, output highlight rolloff, film grain.
//           Order fixed.
//
//  Tiles:   inputMargin(p) = prepareMargin(p) + blurMargin(p) + postMargin(p) (+2). A host that
//           pads every output rect by inputMargin and passes the frame rect gets results
//           identical to a full-frame render. All stages are independent of the thread count.
// ===============================================================================
#pragma once

#include <vector>

namespace lenscore {

struct Params {
    float maxRadius = 20.f;          // px, blur radius where amount == 1
    float focus = 0.f;               // map value that is in focus
    float mapScale = 1.f;            // amount = clamp(|map - focus| * mapScale, 0, 1)
    float mapGamma = 1.f;            // amount = amount^mapGamma
    bool  invertMap = false;         // map = 1 - map (applied before focus)
    int   blades = 0;                // 0 (or <3) = circle, 3..16 = polygon
    float bladeCurvature = 0.f;      // 0 = straight edges, 1 = round (edges are circular arcs)
    float rotationDeg = 0.f;
    float anamorphic = 1.f;          // height/width ratio of the bokeh (area preserving), 0.1..10
    float catEye = 0.f;              // 0..1 optical vignetting (aperture clipping) toward the frame edges
    float edgeRing = 0.f;            // -1..1, >0 bright rim, <0 bright core / soft edge
    float ringWidth = 0.3f;          // 0..1 fraction of radius occupied by the rim
    float chromaFringe = 0.f;        // -1..1 longitudinal CA (R/B kernel scale +-20% at 1); <0 swaps colours
    float highlightThreshold = 1.f;  // v1 "classic" gain: luminance above which highlights are boosted
    float highlightGain = 0.f;       // v1 classic: boost = 1 + gain * amount * (L - thr) / L (none where in focus)
    int   threads = 0;               // 0 = std::thread::hardware_concurrency()
    bool  fullResolution = false;    // v1.2: false = splats with R >~ 55 px are rendered on 2x/4x/..
                                     // coarser grids (much faster; edges soften by ~R/34 px).
                                     // true = everything at full resolution (v1 behaviour, slow).
    bool  fastBlur = false;          // v2.0 Fast quality: speed shortcuts in the blur stage (earlier
                                     // multi-resolution, shared inner layers for the colour fringe).
                                     // Approximations within a measured quality budget; false = exact.

    // ---- lens character (v2). Every one of these at its default reproduces v1 output exactly.
    float sphericalAberration = 0.f; // -1..1. >0 over-corrected: edgeRing + SA behind focus (bright rim),
                                     // edgeRing - SA in front (soft edge); <0 the reverse. No map = behind.
    float cornerRounding = 0.f;      // 0..1 rounds polygon vertices (1 = circle inscribed in the polygon)
    float edgeSoftness = 0.f;        // 0..1 width of a smooth roll-off at the disc edge (fraction of radius)
    float apodization = 0.f;         // 0..1 gaussian-like radial falloff over the whole disc ("melting" look)
    float coreWeight = 0.f;          // -1..1 >0 brighter centre, <0 dimmer centre
    float obstruction = 0.f;         // 0..0.9 hollow centre radius (mirror lens donut)
    float onionRings = 0.f;          // 0..1 concentric ring modulation inside the disc
    float onionCount = 4.f;          // 1..10 number of rings
    float vignetting = 0.f;          // 0..1 brightness falloff toward the frame corners (all pixels)
    float glow = 0.f;                // 0..1 veiling glow: fraction of light spread into a wide halo (all pixels)
    float glowSize = 0.04f;          // glow sigma as a fraction of the frame diagonal (0.005..0.25)

    float fringeTint = 0.f;          // 0..1 chroma fringe colours: 0 = red/blue (v1), 1 = green behind /
                                     // magenta in front (classic longitudinal CA)

    // ---- highlights (v1.2): luminance-true expansion of bright pixels before the blur (an inverse
    // tone map). Added energy = colour * boost * W * [(1-sel)*amount + sel*disc(R)], W = E(L) * detail
    // (auto source) with E(L) = ((L-knee)/(1-knee))^curve above the knee. Disc brightness is monotonic
    // in the source's luminance and size; nothing is added where amount == 0.
    float highlightBoost = 0.f;       // 0 = off. ~ brightness of the disc from a 4x4 px white detail
    float highlightKnee = 0.45f;      // linear luminance where expansion starts (0..0.97); the engine raises it to ~1.3x the local mean brightness
    float highlightCurve = 2.f;        // >= 1, how steeply expansion rises above the knee
    float highlightSelectivity = 0.8f;// 0 = expand every bright pixel (classic gain), 1 = small details only
    float detailSize = 8.f;           // px, largest detail treated as point-like (size window maximum)
    float highlightSaturation = 0.f;  // 0..1 saturates the added energy (coloured discs instead of white)
    float highlightRolloff = 0.f;     // 0..1 output shoulder on the added highlight energy (0 = linear/HDR)

    // ---- highlight appearance controls (v2.0). Defaults = off (v1.2 output unchanged).
    float whiteProtection = 0.f;      // 0..1: brightness that is common in a wide neighbourhood is
                                      // diffuse white (paper, sky, clothes), not a light -> no balls
    float dotsOnly = 0.f;             // 0..1: 0 = any small bright detail, 1 = compact dots only
                                      // (elongated strokes - text, hair gaps, hatching - are rejected)
    float detailMin = 0.f;            // px, smallest detail that becomes a ball (0 = no minimum)
    int   highlightSource = 0;        // 0 = automatic, 1 = highlight layer only, 2 = automatic x layer

    // ---- film effects (v2.0, post stage). Defaults = off. They change in-focus pixels too.
    float halation = 0.f;             // 0..2 amount of film halation (red-orange glow from highlights)
    float halationSize = 0.015f;      // wide glow sigma, fraction of the frame diagonal
    float halationThreshold = 0.25f;  // absolute floor: linear luminance below which nothing halates
    float halationRelative = 1.0f;    // v2.1: stops above the local scene level (wide neighbourhood
                                      // mean) where halation starts. Source threshold per pixel =
                                      // max(halationThreshold, sceneLevel * 2^halationRelative), so
                                      // only lights that are bright FOR THEIR SURROUNDINGS halate
                                      // (no overall cast on high-key frames, strong on night lights)
    float halationSoftness = 0.5f;    // 0..1 width of the threshold knee
    float halationRim = 0.5f;         // 0..1 balance: 0 = only the wide glow, 1 = only the tight rim
    float halationTint[3] = {1.0f, 0.22f, 0.06f};  // linear RGB of the halation light
    float lumAberration = 0.f;        // 0..2 "luminance aberration": warm bleed at bright/dark edges,
                                      // stronger and deeper (orange -> red) for larger differences
    float laWidth = 0.004f;           // bleed sigma (modes 0-2) / rim width (mode 3), fraction of the diagonal
    float laThreshold = 1.5f;         // luminance difference (stops) where it starts
    float laCurve = 1.f;              // response exponent above the threshold
    float laHueShift = 0.6f;          // 0 = one colour, 1 = colour follows the difference (modes 0-2:
                                      // yellow-orange -> laTint; mode 3: red-brown -> laTint)
    float laTint[3] = {1.0f, 0.30f, 0.04f};        // linear RGB for large differences
    int   laMode = 0;                 // 0 = bleed into the dark side, 1 = tint the bright side, 2 = both,
                                      // 3 = contrast rim (v2.1): a crisp coloured line ON the boundary
                                      // between a bright region and a much darker one, replacing the
                                      // line art there; laWidth = rim width (dark-side extent)
    // v2.1, used by laMode 3 only
    float laContrastScale = 0.006f;   // region size (fraction of the diagonal) over which the
                                      // bright/dark levels are measured (larger than line width)
    float laLineProtect = 0.003f;     // dark lines up to ~2x this (fraction of the diagonal) that lie
                                      // inside a bright area are ignored (eyelashes, mouth lines)
    float laSharpness = 0.8f;         // 0..1: 1 = hard edge on the bright side and a short falloff
                                      // into the dark side; 0 = soft on both sides
    float laBrightColor = 0.3f;       // 0..1: rim colour taken from the bright side (warm-shifted)
                                      // instead of the fixed laTint
    float laBlur = 0.f;               // v2.1b, fraction of the diagonal. 0 = the crisp drawn rim
                                      // (laSharpness shapes it). > 0 = optical fringe: the light of the
                                      // bright region is defocused per channel (red spreads furthest,
                                      // then green, blue least) and spills into the dark side, tinted
                                      // toward laTint - a soft orange->red gradient that tints what is
                                      // underneath instead of painting over it
    float laComplement = 0.f;         // 0..1 (fringe mode): faint complementary (cyan) tint just inside
                                      // the bright side, where the red has spread away
    // v2.1d, laMode 4 = "channel drift" (the AE recipe: per-channel vector blur along the luminance
    // gradient). Each channel is smeared along the gradient of a softened luminance map by
    // laShift[c] x laWidth x diagonal x lumAberration x response(edge contrast), so at a strong
    // bright/dark boundary e.g. green+blue retreat from the bright edge while red stays: a soft,
    // physically coloured fringe, nothing painted. Flat areas and weak edges are untouched.
    float laMapSoftness = 0.002f;     // smoothing of the luminance map (fraction of the diagonal)
    float laShift[3] = {-0.15f, 0.5f, 1.3f};  // per-channel drift (signed; 0 = channel stays;
                                      // negative = moves the other way). Default = "Orange" fringe
    int   laShiftDir = 0;             // 0 = channels retreat into the bright side (fringe on the bright
                                      // edge), 1 = they advance into the dark side
    float laPreserve = 0.5f;          // 0..1: restrict the drift to edges of genuinely bright regions
                                      // (keeps faces / mid-tone line art from distorting)

    // ---- film grain (v2.1c, last step of the post stage; covers in-focus pixels too). Off by default.
    // A pure function of the absolute position in FULL-RESOLUTION layer pixels (via the frame rect and
    // grainScaleX/Y), grainSeed and grainFrame: tile-exact, thread- and render-order independent, and a
    // half-resolution render shows the same pattern (area-averaged), not a different one.
    float grain = 0.f;                // 0..2 amount (0 = exact no-op)
    float grainSize = 1.2f;           // clump size, full-resolution px (0.3..8)
    float grainSoftness = 0.5f;       // 0..1: 0 = crisp, nearly binary clumps; 1 = soft
    float grainColor = 0.3f;          // 0 = monochrome grain, 1 = independent per channel (blue coarser)
    float grainTone = 0.7f;           // 0 = same in every tone (except black); 1 = film-like: strong in
                                      // shadows / mid-tones, fading out in the highlights
    int   grainSeed = 0;
    int   grainFrame = 0;             // pattern index (host: frame number / hold frames, or 0 = static)
    float grainScaleX = 1.f, grainScaleY = 1.f;  // buffer px per full-resolution layer px (downsampling)

    // ---- resolution (v2.1f). Area of one buffer pixel relative to a full-resolution pixel, squared
    // scale: 1 = full resolution, 0.25 = half (AE Half), 0.0625 = quarter. Quantities defined in
    // full-resolution pixels (the highlight disc-area term, the Min Detail Size ramp, the Highlight
    // Source view units) use it so a downsampled preview looks like the final render.
    float pixelScale = 1.f;

    // ---- render version (v2.2). 2 = current algorithms; 1 = v2.1 / v1.2 behaviour where v2.2
    // changed a look: highlight energy per pixel (not coverage-linear), integer size-window radii,
    // 3-box blurs for halation / aberration widths, Channel Drift without band spreading, Highlight
    // Rolloff on the whole pixel. Hosts store 1 for projects saved before v2.2 (bit-exact renders).
    int   renderVersion = 2;

    // ---- frame geometry (buffer px): cat eye, vignetting and glow size are relative to this rect.
    // frameWidth <= 0 => the buffer itself. Lets a host process tiles while keeping frame-relative effects.
    float frameLeft = 0.f, frameTop = 0.f, frameWidth = 0.f, frameHeight = 0.f;

    // ---- pixel aspect (v2.2, appended): width / height of one BUFFER pixel (non-square pixel
    // footage, or downsample x != y). The frame diagonal, vignetting and cat-eye distances are
    // measured in square pixels with it (1 = square, bit-identical to v2.1). Diagonal-relative
    // widths (glow, halation, aberration) stay isotropic in buffer pixels.
    float pixelAspect = 1.f;

    // ---- output rows (v2.3, appended). Buffer rows [outRowBegin, outRowEnd) of the FINAL output are
    // the only ones the host reads (its output rect inside the padded engine buffer). blurAlpha then
    // computes only the rows postProcess needs for them (that range grown by postMargin) and sets the
    // other rows (and their eLOut) to 0; postProcess's result inside the range is unchanged (bit-
    // identical to a full render: rows are independent in the blur stage). Both 0 (default) = all rows.
    int outRowBegin = 0, outRowEnd = 0;
};

// Controls reserved for the Best-quality blur engine (core/hq). Carried through the blur seam; the
// span engine ignores them.
struct HqParams {
    float textureAmount = 0.f;   // in-disc texture (dust / streaks)
    float textureScale = 1.f;
    int   textureType = 0;
    int   textureSeed = 0;
    int   spectralSamples = 0;   // 0 = RGB only
    float lateralCA = 0.f;       // lateral (transverse) chromatic aberration
};

enum class Status { Ok = 0, BadArgs, Aborted, OutOfMemory, Error };  // Error (v2.2): any other exception

// Return true to abort. It is called ONLY from the thread that called render()
// (between row bands and while waiting for workers), so host abort checks that
// must run on the render thread are safe.
typedef bool (*AbortFn)(void* user);

// Output of stage 1 (see the seam description above). All vectors are w*h (rgba: w*h*4).
struct Prepared {
    int w = 0, h = 0;
    std::vector<float> rgba;          // working colour, premultiplied linear
    std::vector<double> radius;       // blur radius, px
    std::vector<signed char> side;    // +1 behind focus, -1 in front
    std::vector<float> eL;            // luminance of the added highlight energy (0 = none)
    std::vector<float> extra;         // optional (keepExtra): added energy normalised to "disc
                                      // brightness of a 4x4 px source" units, RGB, w*h*3
    bool rollRelevant = false;        // some highlight energy was added (rolloff can matter)
};

struct PrepareOptions {
    bool keepExtra = false;           // fill Prepared.extra (views)
    const float* highlightLayer = nullptr;  // single channel, >= 0, stride highlightLayerStride;
    int highlightLayerStride = 0;           // used by highlightSource 1/2 (nullptr => treated as 0)
};

// src : premultiplied linear float RGBA, row stride in floats (>= 4*w).
// map : single-channel float, row stride in floats (>= w); nullptr => amount 1 everywhere.
// dst : premultiplied linear float RGBA, row stride in floats (>= 4*w). Must not overlap src or map.
// No global state; concurrent calls are fine.
Status render(const float* src, int w, int h, int srcStrideFloats,
              const float* map, int mapStrideFloats,
              float* dst, int dstStrideFloats,
              const Params& p, AbortFn abort = nullptr, void* user = nullptr);

// Same, with the highlight layer (highlightSource 1/2).
Status render(const float* src, int w, int h, int srcStrideFloats,
              const float* map, int mapStrideFloats,
              const float* highlightLayer, int highlightLayerStride,
              float* dst, int dstStrideFloats,
              const Params& p, AbortFn abort = nullptr, void* user = nullptr);

// ---- the three stages ----
Status prepareHighlights(const float* src, int w, int h, int srcStrideFloats,
                         const float* map, int mapStrideFloats, const Params& p,
                         const PrepareOptions& opt, Prepared& out,
                         AbortFn abort = nullptr, void* user = nullptr);

// span-engine blur stage. eLOut: w*h floats, written only if postNeedsEL(p, prep) (may be nullptr otherwise).
Status blurAlpha(const Prepared& prep, const Params& p, float* dst, int dstStrideFloats,
                 float* eLOut, AbortFn abort = nullptr, void* user = nullptr);

// does postProcess need the blurred eL (output rolloff active)?
bool postNeedsEL(const Params& p, const Prepared& prep);

// in place on the blurred image. eL: w*h (only read if postNeedsEL).
Status postProcess(const Prepared& prep, const Params& p, float* rgba, int strideFloats,
                   const float* eL, AbortFn abort = nullptr, void* user = nullptr);

// v2.3: a sub-rectangle [x0, x0+w) x [y0, y0+h) of `prep` (clipped to it) as its own Prepared, so
// a host can run the blur and post stages on a smaller window than the prepare stage needed (the
// prepare stage needs prepareMargin more input on every side; the later stages do not). The caller
// shifts the frame rect by (-x0, -y0) (Params.frameLeft / frameTop) for the cropped stages and any
// row range (outRowBegin / outRowEnd) by -y0. Results inside the output rect then equal those of
// the uncropped buffer up to double round-off of the row prefix sums (they start at the new left
// edge), i.e. what a tiled host gets today; see build\perf\roi_check.txt.
Prepared cropPrepared(const Prepared& prep, int x0, int y0, int w, int h);

// v2.3: the blur stage's setup (what blurAlpha builds before it scatters: shape tables, layer profiles,
// cat-eye normaliser table, multi-resolution levels, computed row range) for an external implementation
// of the same blur (the GPU library gpu\obgpu). Computed by the same code blurAlpha runs, so it cannot
// drift. Returns Ok, BadArgs (invalid params / size), Aborted, OutOfMemory or Error.
struct BlurSetup {
    static const int kMaxLayers = 32, kTiers = 8;
    struct Layers { int nl = 1; bool share = false; int shareFrom = 1; double s2 = 1, s2out = 0, s2in = 0;
                    double ls[kMaxLayers] = {1}; double lb[kMaxLayers] = {1}; };
    struct Level { int f = 1, w = 0, h = 0, gox = 0, goy = 0, maxReach = 0, rowLo = 0, rowHi = 0; };
    struct Shape { int n = 0; double ymin = -1, ymax = 1, invStep = 0, area = 0; std::vector<double> table; };  // table: IL, IR interleaved, 2*(n+1)
    int w = 0, h = 0, nk = 1, nTiers = 1, nLevels = 1, catN = 0;
    bool cat = false, clampNeg = false, fringeShare = false;
    int maxReach = 0, rowBegin = 0, rowEnd = 0;   // fine rows [rowBegin, rowEnd) are computed, the others are 0
    double ax = 1, ay = 1, xext = 1, yext = 1, catK = 0, catLim = 1, catInv = 1, levelBase = 34;
    double fcx = 0, fcy = 0, halfDiag = 1, pa = 1, maxRadius = 0, smaxG = 1;
    double ks[2][3] = {{1, 1, 1}, {1, 1, 1}};   // [front 0 / behind 1][R, G, B] kernel scale
    Layers prof[2][kTiers];                     // [side][tier]
    std::vector<Level> levels;                  // [1, nLevels) used
    Shape ap, circ;
    std::vector<double> catA;                   // catN*catN (row = oy index) or empty
};
Status describeBlur(const Params& p, int w, int h, BlurSetup& out, AbortFn abort = nullptr, void* user = nullptr);
// "Highlights Only" helper: a Prepared whose colour is only the added highlight energy (alpha 1),
// so blurring it shows the bokeh contribution of the highlights on black. Needs prep.extra.
Prepared highlightsOnly(const Prepared& prep);

// Per-pixel blur amount (0..1) the engine uses for a map value (radius = amount * maxRadius).
float blurAmount(float mapValue, const Params& p);

// Margins (px). inputMargin = all stages; a tiled host pads every output rect by inputMargin.
// The frame-relative terms (glow, halation, adaptive knee, ...) use the FRAME RECT of p
// (Params.frame*): with frameWidth <= 0 these one-argument forms assume a 1x1 frame and return far
// too small margins. Hosts without a frame rect must use the (p, w, h) forms, which take the
// w x h buffer as the frame when p has none.
int prepareMargin(const Params& p);
int blurMargin(const Params& p);
int postMargin(const Params& p);
int inputMargin(const Params& p);
int prepareMargin(const Params& p, int w, int h);
int blurMargin(const Params& p, int w, int h);
int postMargin(const Params& p, int w, int h);
int inputMargin(const Params& p, int w, int h);
} // namespace lenscore
