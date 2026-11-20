// Map Bilinear Blur - render engine (no After Effects dependency).
//
// Idea (the owner's): instead of a real blur, build a downsample pyramid of the layer once and, per
// output pixel, sample the pyramid level that matches that pixel's blur radius (from a depth / blur
// map), interpolating between levels. The cost per pixel is O(1) whatever the radius.
//
// v0.2: the look is set by independent choices:
//   interpolation  - how a pyramid level is resampled back to pixels (nearest ... Lanczos, hex mosaic)
//   downsample     - how levels are made (2x2 average = crisp cells, binomial 6-tap = smooth)
//   level blend    - between the two levels around the target blur: smooth / steps / dither
//   protect edges  - depth-aware layering (in-focus subject kept out of the background blur, front blur
//                    spread over what is behind it)
//
// Coordinates: "buffer px" = AE layer coordinates at the current downsample. The pyramid grid is
// anchored to the DOMAIN rect D, whose left / top the caller aligns to multiples of Align(s) in layer
// coords (right / bottom end at the content; the last cell of a level may be partial), so
// level L cells are 2^L buffer px squares on a grid fixed to the layer origin. At Half resolution the
// grid of level L-1 is the grid of full-res level L: Half / Quarter previews see the same cells.
// Every read outside D is clamped to D's extent at that level (a fixed rule relative to D, not to the
// tile), so any tile computes exactly the values the full frame would (tile / thread independent).
//
// Pixels are premultiplied RGBA float. q = "inside" weight per pixel (1, or 0 outside the layer for
// Clip to Layer): every level stores (q * rgba, q) and values are normalised by the q sum.
#pragma once
#include "mbbgpu.h"  // gpu\include: the GPU engine's frame description (plain C struct)
#include <functional>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace mbb {

// v0.3: a vector whose elements are not value-initialised on resize (no zero-fill of buffers that are
// fully written anyway: ~0.1 s per UHD frame in v0.2)
template <class T>
struct NoInitAlloc : std::allocator<T>
{
    template <class U>
    struct rebind
    {
        using other = NoInitAlloc<U>;
    };
    NoInitAlloc() = default;
    template <class U>
    NoInitAlloc(const NoInitAlloc<U>&) noexcept
    {
    }
    template <class U, class... A>
    void construct(U* p, A&&... a)
    {
        if constexpr (sizeof...(A) == 0)
            ::new ((void*)p) U;
        else
            ::new ((void*)p) U(std::forward<A>(a)...);
    }
};
template <class T, class U>
bool operator==(const NoInitAlloc<T>&, const NoInitAlloc<U>&) { return true; }
template <class T, class U>
bool operator!=(const NoInitAlloc<T>&, const NoInitAlloc<U>&) { return false; }
using FVec = std::vector<float, NoInitAlloc<float>>;

enum Interp
{
    IN_NEAREST = 0,
    IN_BILINEAR,
    IN_SMOOTHSTEP,
    IN_BSPLINE,
    IN_MITCHELL,
    IN_CATMULLROM,
    IN_LANCZOS3,
    IN_HEX,
    IN_COUNT
};
enum Blend { BL_SMOOTH = 0, BL_STEPS, BL_DITHER };
enum SharpType { SH_AE = 0, SH_UNSHARP = 1 };

struct Rect
{
    int l = 0, t = 0, r = 0, b = 0;  // half-open
    int w() const { return r - l; }
    int h() const { return b - t; }
    bool empty() const { return r <= l || b <= t; }
};

struct Settings
{
    int interp = IN_BSPLINE;
    bool downSmooth = true;  // binomial 6-tap levels (false: 2x2 average)
    int blend = BL_SMOOTH;
    bool antiRing = true;    // Mitchell / Catmull-Rom / Lanczos: clamp to the 2x2 nearest cells' range
    bool protect = true;     // depth-aware layering
    unsigned ditherSeed = 0;
    int ditherFrame = 0;     // part of the dither pattern (0 when not animated)
    double radius = 0;       // max blur radius, BUFFER px (full-res radius * ds)
    double ds = 1;           // buffer px per full-res px
    // sharpen (applied to the blurred result)
    int sharpType = SH_AE;
    double sharpAmount = 0;  // AE-style: the "a" of the 3x3 kernel (centre 1+8a, neighbours -a), already
                             // resolution-scaled by the caller; unsharp: gain (1 = 100 %)
    double sharpRadius = 0;  // unsharp only: radius, buffer px
    int threads = 1;
};

// sigma of the blur for radius r (buffer px): the visible spread ~ r
inline double SigmaForRadius(double r) { return 0.5 * r; }

// pyramid levels above level 0 needed for these settings (0 = only the base)
int PyramidLevels(const Settings& s);
// the domain must be aligned to this (in layer coords: left/top multiples of it)
inline int Align(const Settings& s) { return 1 << PyramidLevels(s); }
// how far the blurred result can reach beyond the content (Spread edge mode), buffer px
int SpreadPx(const Settings& s);
// AE-style sharpen needs one more pixel around the output
bool NeedsRing(const Settings& s);
// level-0 pixels (inside D) the engine reads to produce O
Rect InputRect(const Settings& s, const Rect& D, const Rect& O);

struct Job
{
    Settings s;
    Rect D;       // domain (aligned)
    Rect O;       // output rect, inside D
    Rect R0;      // = InputRect(s, D, O); the arrays below cover R0, row-major, dense
    const float* rgba = nullptr;    // premultiplied, 4 floats per px (unused when base5 is given)
    FVec* base5 = nullptr;  // optional instead of rgba / q: (q*r, q*g, q*b, q*a, q) per px over R0,
                                          // taken over (swapped out) by the engine: saves a copy
    const float* q = nullptr;       // inside weight, nullptr = all 1
    bool clip = false;              // some inside weights are 0 (Clip to Layer): normalise by them
    const float* amount = nullptr;  // signed blur amount -1..1 (negative = in front of focus), nullptr = +1
    float* out = nullptr;           // O, premultiplied RGBA
    std::function<bool()> abort;    // polled on the calling thread between stages; true = stop
};

enum Status { ST_OK = 0, ST_ABORTED, ST_OOM, ST_BAD };
Status Render(const Job& job);
// v0.3: the frame as the GPU engine needs it (ladder, level ranges, settings); false = not describable
bool DescribeGpu(const Job& job, mbbgpu_desc& d);
extern int g_profile;  // tests: print stage times to stderr

}  // namespace mbb
