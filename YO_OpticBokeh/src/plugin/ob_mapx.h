// Optic Bokeh v2.3: blur-map features done in the plug-in's map adapter, WITHOUT touching the engine
// (core\lenscore): Map Blur (pre-blur of the map), Focus Range (a fully sharp zone around Focus with a
// smooth shoulder), Front / Back Blur (separate scales on each side of focus) and Focus at Point (the
// focus value read from the map under a point). Shared by the plug-in (full and Kantan) and the tools.
//
// How it reaches the engine: lenscore computes amount = clamp(|m' - focus| * mapScale, 0, 1)^gamma and
// the side (front / behind) from the sign of m' - focus. When any feature is on, the adapter computes
// the final signed amount a in [-1, 1] itself and hands the engine m' = 0.5 + 0.5 a with focus 0.5,
// mapScale 2, gamma 1, no invert: the engine then gets |a| and sign(a) back (to float precision of the
// map buffer, ~6e-8). With every feature at its default the adapter is not used at all: the map and
// the params go to the engine exactly as before (bit-exact with v2.2 / v2.3.0 renders).
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace obmap {

struct Xform
{
    bool active = false;   // false: the classic path (raw map, the user's focus / scale / gamma)
    bool invert = false;
    double focus = 0;      // map units 0..1 (Focus %, or the value under the focus point)
    double scale = 1;      // Map Scale
    double gamma = 1;      // Map Gamma
    double half = 0;       // Focus Range / 2, map units
    double shoulder = 0;   // width of the smooth shoulder after the sharp zone
    double front = 1, back = 1;  // scales in front of / behind focus
    double sigma = 0;      // Map Blur, buffer px (Gaussian sigma)
    bool point = false;    // focus from the point
    double px = 0, py = 0; // point, effect-layer px (buffer coords at the current downsample)
};

// UI values -> Xform. blurPx: full-res px; sc: buffer px per full-res px (downsample).
inline Xform Make(bool invert, double focusPct, double mapScale, double mapGamma, double blurPx, double rangePct,
                  double frontPct, double backPct, bool pointOn, double px, double py, double sc)
{
    Xform x;
    x.invert = invert;
    x.focus = focusPct / 100.0;
    x.scale = mapScale;
    x.gamma = std::max(0.01, mapGamma);
    x.sigma = std::max(0.0, blurPx) * sc;
    x.half = std::max(0.0, rangePct) / 200.0;
    x.shoulder = 0.5 * x.half;
    x.front = std::max(0.0, frontPct) / 100.0;
    x.back = std::max(0.0, backPct) / 100.0;
    x.point = pointOn;
    x.px = px;
    x.py = py;
    x.active = blurPx > 0 || rangePct > 0 || frontPct != 100.0 || backPct != 100.0 || pointOn;
    return x;
}

// Focus Range: 0 inside the sharp zone |d| <= half, a quadratic shoulder of width 2*shoulder, then the
// distance minus (half + shoulder): continuous with a continuous slope (C1), = t when half = 0.
inline double Shoulder(double t, double half, double s)
{
    if (half <= 0)
        return t;
    if (t <= half)
        return 0;
    if (t >= half + 2 * s)
        return t - half - s;
    const double u = t - half;
    return u * u / (4 * s);
}

// raw map value -> the value the engine gets (see the header comment)
inline float Apply(const Xform& x, float m)
{
    double v = (m == m) ? (double)m : 0.0;
    if (x.invert)
        v = 1.0 - v;
    const double d = v - x.focus;
    double a = Shoulder(std::fabs(d), x.half, x.shoulder) * x.scale * (d < 0 ? x.front : x.back);
    a = std::min(1.0, std::max(0.0, a));
    if (x.gamma != 1.0 && a > 0)
        a = std::pow(a, x.gamma);
    return (float)(0.5 + 0.5 * (d < 0 ? -a : a));
}

// Map Blur: 3 extended box passes per axis (Gwosdek et al. 2011), = a Gaussian of `sigma` with a
// non-integer radius, so a Half / Quarter preview (sigma scaled) blurs like the full render. O(1) per
// pixel and pass; double running sums. Extent: a value depends on input within Extent(sigma) pixels.
struct Box
{
    int r = 0;
    double alpha = 0, norm = 1;
};
inline Box BoxFor(double sigma)
{
    Box b;
    const double v = sigma * sigma / 3.0;  // variance per pass
    int r = (int)std::floor((std::sqrt(12.0 * v + 1.0) - 1.0) / 2.0);
    while (r > 0 && r * (r + 1) / 3.0 > v) --r;
    while ((r + 1) * (r + 2) / 3.0 <= v) ++r;
    b.r = std::max(0, r);
    const double rr = b.r;
    const double den = 2.0 * ((rr + 1) * (rr + 1) - v);
    b.alpha = den > 1e-12 ? std::max(0.0, std::min(1.0, (2 * rr + 1) * (v - rr * (rr + 1) / 3.0) / den)) : 0.0;
    b.norm = 1.0 / (2 * rr + 1 + 2 * b.alpha);
    return b;
}
inline int Extent(double sigma)
{
    return sigma > 0 ? 3 * (BoxFor(sigma).r + 1) : 0;
}

// one pass along a line of n samples with stride st (edges clamped to the line)
inline void BoxLine(float* p, int n, int st, const Box& b, std::vector<double>& tmp)
{
    tmp.resize(n);
    for (int i = 0; i < n; ++i)
        tmp[i] = p[(size_t)i * st];
    auto at = [&](int i) { return tmp[std::min(n - 1, std::max(0, i))]; };
    double s = 0;
    for (int k = -b.r; k <= b.r; ++k)
        s += at(k);
    for (int i = 0; i < n; ++i)
    {
        p[(size_t)i * st] = (float)((s + b.alpha * (at(i - b.r - 1) + at(i + b.r + 1))) * b.norm);
        s += at(i + b.r + 1) - at(i - b.r);
    }
}

// in place, w x h, row-major. Values within Extent(sigma) of the border are approximate (clamped);
// callers sample Extent() more on every side and keep the inside.
inline void Blur(float* buf, int w, int h, double sigma)
{
    if (!(sigma > 0) || w <= 0 || h <= 0)
        return;
    const Box b = BoxFor(sigma);
    std::vector<double> tmp;
    for (int pass = 0; pass < 3; ++pass)
        for (int y = 0; y < h; ++y)
            BoxLine(buf + (size_t)y * w, w, 1, b, tmp);
    for (int pass = 0; pass < 3; ++pass)
        for (int x = 0; x < w; ++x)
            BoxLine(buf + x, h, w, b, tmp);
}

}  // namespace obmap
