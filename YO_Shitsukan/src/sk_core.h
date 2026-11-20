// Shitsukan (working title 「質感・線・水彩」) - render engine, no After Effects dependency.  Round 2 (0.2).
//
// One effect for three ideas of the owner:
//   (12) texture control  - fine / mid / large bands boosted or removed
//   (7)  line-keeping blur - fills blurred, line art kept crisp; colour never crosses a line
//   (6)  watercolour       - washes laid in strokes, glazes, pooled edges, blooms, bleeding, paper
//
// Shared machinery (computed once per frame, used by all three):
//   * Line analysis: dark thin structures = closing(s) - s (black top-hat) divided by the local line contrast,
//     on luma and (option) on each colour channel (brown / red outlines) and as a white top-hat (light lines on
//     dark). Dense clusters of thin dark specks (foliage, brush texture) are not lines unless asked for. Lines are
//     INK: a per channel optical density delta = ln((fill + eps) / (pixel + eps)) over an in-painted fill:
//         out = (processedFill + eps) * exp(-delta') - eps     (== the input when nothing changes)
//   * One smoother: the recursive domain-transform filter (Gastal & Oliveira 2011), O(1) per pixel for any
//     radius, normalised by a weight channel (alpha, 0 on lines), with an edge cost from colour differences and
//     an infinite cost on line pixels (barrier).
//   * Watercolour (see docs\watercolor_model.md) works in optical density (D = -ln colour, paper white = 0).
//
// Coordinates: buffer px = AE layer px at the current downsample ds. All sizes in Params are FULL-RES px.
// Noise and strokes are anchored to full-res layer coordinates, so they stay fixed to the layer and Half /
// Quarter previews see the same pattern. Colour: straight values; alpha is never changed. Transparent pixels
// never contribute colour (normalised filtering): the outside of the layer / of the comp acts like "repeat edge".
#pragma once
#include <functional>
#ifdef __APPLE__
#include <locale.h>
#undef LC_ALL
#endif

namespace sk {

enum View { V_RESULT = 0, V_LINES, V_FINE, V_MID, V_LARGE, V_RIM, V_FILL, V_STROKES, V_GLAZE, V_MAP, V_DEPTHW, V_COUNT };
enum MapTarget { MT_ALL = 0, MT_TEXTURE, MT_FILLBLUR, MT_WATERCOLOR, MT_LINES, MT_COUNT };
enum EdgeMode { EM_REPEAT = 0, EM_PAPER = 1 };
enum LineColor { LC_DARK = 0, LC_COLOR, LC_ALL };       // which lines are detected
enum LineStyle { LS_ASIS = 0, LS_PENCIL, LS_INK };       // how lines are drawn back
enum WcMode { WCM_FILTER = 0, WCM_PAINT };                 // watercolour: round-2 filter | round-6 region painter

// UI units (the plug-in passes slider values as they are; percentages are 0..100 etc.)
struct Params
{
    double amount = 100;  // %
    // texture
    double fine = 0, mid = 0, large = 0;  // % (-100 = remove, +100 = double, +200 = triple)
    double detailSize = 1.5;              // px, the fine / mid split (sigma)
    double halo = 70;                     // % halo control (big differences get less gain)
    double edgeAware = 60;                // % 0 = plain Gaussian bands
    double colorDetail = 30;              // % how much colour detail is boosted (removal always all)
    // lines
    bool protectLines = true;
    double lineSens = 50;   // %
    double lineWidth = 3;   // px, widest line detected
    int lineColor = LC_COLOR;
    bool blackWalls = false;      // black fills are walls for the blurs too
    bool bgDetailLines = false;   // dense thin dark texture (foliage, brush specks) counts as lines
    double fillBlur = 0;    // px
    double keepEdges = 30;  // % colour boundaries kept by the fill blur
    double lineThick = 0;   // px -3..3
    double lineDark = 0;    // % -100..100
    double lineSoft = 0;    // px
    int lineStyle = LS_ASIS;
    // watercolour
    double wc = 0;  // % overall
    int wcMode = WCM_FILTER;
    // the region painter (round 6)
    double paintScale = 16;  // px: size of the washes (the abstraction)
    double merge = 50;       // % how different colours may still become one wash
    double loose = 50;       // % registration broken: overshoot / unpainted slivers
    double liftDarks = 50;   // % darks lifted (watercolour never reaches black)
    double charge = 40;      // % pigments separate / colours drift inside a wash
    double margin = 0;       // % an irregular unpainted paper border
    double palette = 0;      // number of pigment mixtures (0 = every colour)
    double warp = 0;         // % round 9: hand-painted distortion of the whole painting (layer-anchored)
    int paintEngine = 0;     // 0.6.0.3: 0 = the current engine, 1 = the round-6 engine (the owner's confirmed look;
                             //   less stable in video; no depth matte / distortion / pigment set there)
    int pigSet = 0;          // round 9: pigment set of the limited palette: 0 watercolour, 1 storybook (soft),
                             //          2 storybook (bright), 3 source hues simplified
    double flatten = 60, flattenSize = 6;
    double strokes = 40, strokeLen = 18, strokeAngle = 35;  // brush strokes: amount %, length px, default angle deg
    double glaze = 50;                                      // % separation into a light wash + a shadow glaze
    double rim = 70, rimWidth = 4;
    double gravity = 30;                                    // % pigment runs down: stronger rims at the bottom
    double bleed = 25, bleedWidth = 8;
    double backrun = 0;
    double wobble = 2, wobbleScale = 40;
    double uneven = 30;
    double gran = 35, paper = 40, paperScale = 100;
    double keepWhite = 70, dilution = 30;
    double protectDetail = 70;  // % small features (eyes, mouths) keep their shape
    // common
    int mapTarget = MT_ALL;
    // depth matte (the map layer used as depth instead of a strength mask). Every influence is a signed % (-100..100):
    // its effect is (1 + k * far) where far = 0 at the reference depth and 1 at "range" away from it - a flat map at
    // the reference depth (or no map) leaves the picture exactly as it is
    bool depth = false;
    double mapBlur = 0;        // px, smooths a noisy depth pass
    double depthRef = 50;      // % the reference ("near") depth
    double depthRange = 30;    // % how far from the reference counts as fully "far"
    bool depthPick = false;    // take the reference from the map at a point
    double pickX = 0, pickY = 0;  // full-res layer coords of that point
    double dMaster = 100;      // % scales every influence
    double dAmount = 0, dScale = 0, dWet = 0, dLines = 0, dPale = 0, dTexture = 0, dBlur = 0;  // % -100..100
    int edgeMode = EM_REPEAT;
    int view = V_RESULT;
    int seed = 0;
    int paperFrame = 0;     // pattern index for "move paper" (0 = static)
    bool linear = false;    // linear working space: processed in gamma 2.2, written back linear
    double par = 1.0;       // pixel aspect ratio (noise / strokes stay round on screen)
};

// true when Render would return the input unchanged (the caller copies the input bit-exactly instead)
bool IsIdentity(const Params& p);

struct Rect
{
    int l = 0, t = 0, r = 0, b = 0;
};

struct Frame
{
    int w = 0, h = 0;
    int ox = 0, oy = 0;    // layer coords (at this downsample) of buffer pixel (0, 0)
    double ds = 1;         // buffer px per full-res px
    const float* rgba = nullptr;      // straight RGBA, w*h*4, finite, alpha 0..1 (caller sanitises)
    const float* map = nullptr;       // optional w*h strength 0..1
    const float* paperTex = nullptr;  // optional w*h paper height 0..1
    Rect layer;            // the layer's rect in buffer coords (edge mode "paper")
    float* out = nullptr;  // straight RGBA, w*h*4 (alpha = the input's; views: 1). MAY EQUAL rgba (in place).
    int threads = 1;
    std::function<bool()> abort;  // polled between stages on the calling thread
};

enum Status { ST_OK = 0, ST_ABORTED, ST_OOM, ST_ERROR };
Status Render(const Params& p, Frame& f);
// 0.6.0.6: the plug-in's global setdown - waits (bounded) for the GPU start-up self-test thread
void GpuShutdown();

extern int g_profile;  // tests: stage times on stderr

}  // namespace sk
