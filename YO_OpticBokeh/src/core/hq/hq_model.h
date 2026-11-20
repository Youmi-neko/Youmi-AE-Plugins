// hq_model.h - internal to lenshq: continuous aperture model, texture, spectrum and the
// summed-area tables the scatter samples. Not part of the public API.
#pragma once

#include "lenshq.h"

#include <cstdint>
#include <vector>

namespace lenshq {
namespace detail {

const double kPi = 3.14159265358979323846;
inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline double smooth01(double t) { t = clampd(t, 0.0, 1.0); return t * t * (3 - 2 * t); }

// ------------------------------------------------------------------ aperture ---
// Convex unit aperture (vertex radius 1) = intersection of constraints, same construction as
// alpha (straight or circular-arc blades, rotation, vertex rounding). gauge(u,v) is its exact
// level-set scale: the shape scaled by r about the origin has (u,v) on its boundary.
struct Constraint {
    bool disk;
    double nx, ny, a;      // half-plane nx*x + ny*y <= a
    double cx, cy, rho2;   // disk |p - c|^2 <= rho2 (contains the origin)
    double k;              // disk: 1 / (rho2 - |c|^2)
};

struct Aperture {
    std::vector<Constraint> cs;
    bool circle = true;
    double xmin = -1, xmax = 1, ymin = -1, ymax = 1;
    void build(int blades, double curvature, double rotRad, double rounding);
    double gauge(double u, double v) const;
    double gaugeGrad(double u, double v, double& gx, double& gy) const;  // gauge and its gradient
};

// ------------------------------------------------------------- radial profile ---
// alpha's continuous intensity I(r) (r = gauge), evaluated exactly instead of in layers.
struct Profile {
    double e = 0, rw = 0.3, soft = 0, apod = 0, core = 0, obstr = 0, rimShape = 0;
    double band(double t) const;     // rim shape on t in [0,1] (0 = inner edge of the rim band)
    double base(double r) const;     // I(r) without the onion modulation, 0 for r > 1
    double edge(double d) const;     // only the edge-related factors (rim, soft edge) at gauge d
};

// ------------------------------------------------------------------- texture ---
struct Texture {
    double amount = 0, scale = 1;
    uint32_t seed = 1;
    double wGrain = 0, wDust = 0, wStreak = 0, wPolish = 0, wBusy = 0;
    double nGrain = 1, nStreak = 1, nPolish = 1, nBusy = 1;  // std normalisers
    struct Speck { double x, y, r, depth; };
    std::vector<Speck> specks;
    double streakAng[2] = {0, 0};
    void init(const HqOptions& q);
    bool active() const { return amount > 0; }
    double eval(double u, double v) const;   // multiplicative factor >= 0 (aperture coordinates)
    // raw components (zero-mean-ish)
    double grain(double u, double v) const;
    double dust(double u, double v) const;
    double streak(double u, double v) const;
    double polish(double u, double v) const;
    double busy(double u, double v) const;
};

// ------------------------------------------------------------------ spectrum ---
// Wavelength samples i: kernel scale per side, RGB weights (each channel's weights sum to 1, so a
// neutral source integrates to neutral), luminance weight, lateral-CA coefficient.
struct Spectrum {
    int N = 1;
    std::vector<double> s[2];   // [front, back]
    std::vector<double> w[3];   // channel weights
    std::vector<double> lum;    // luminance-weighted share (sums to 1)
    std::vector<double> lat;    // lateral CA coefficient (luminance-weighted mean 0)
    std::vector<double> ds;     // spread of the kernel scale inside the wavelength bin (>= 0)
    void buildSpectral(double fringe, double tint, int N);
    void buildThreeBand(double fringe, double tint);
    void buildNone();
    double smin(int side) const, smax(int side) const;
};

// --------------------------------------------------------------- SAT tables ---
// Summed-area table of a texel image (n x n texels over [-D, D]^2, texel = average of the
// continuous function over the texel). S at fractional texel coordinates is bilinear in the
// corner grid, i.e. the EXACT integral of the piecewise-constant texel image. Mip level l is the
// corner grid subsampled by 2^l (exact integrals of the 2^l-box-averaged image).
struct Sat {
    int n = 0;
    double D = 1;
    double total = 0;
    double bx0 = 0, bx1 = 0, by0 = 0, by1 = 0;   // bounding box of nonzero texels (unit coords)
    std::vector<std::vector<double>> lv;         // level corner grids, (n_l+1)^2
    std::vector<int> nl;
    void build(const std::vector<double>& texels, int n, double D);
    size_t bytes() const;
    // largest level with at least `tpp` texels per pixel when one pixel = 1/pxPerUnit units
    int levelFor(double pxPerUnit, double tpp) const;
};

// Clip-edge function C(d) of the cat-eye clip circle (d = distance / clip radius) and its first
// and second antiderivatives, for exact box/trapezoid filtering over a pixel footprint.
struct EdgeFn {
    double d0 = -1, d1 = 3, step = 1, inv = 1;
    int n = 0;
    std::vector<double> f, F1, F2;
    std::vector<int> runEnd;    // last index of the constant run starting at i (exact quick-out)
    double dIn = 0, dOut = 0;   // C == 1 for d <= dIn, C == 0 for d >= dOut
    // samples at d0 + i * step, i = 0..n-1 (step = (d1 - d0) / (n - 1)); C is piecewise linear
    void build(const std::vector<double>& samples, double d0, double d1);
    double val(double d) const;
    double I1(double d) const;
    double I2(double d) const;
    // mean of C over the sum of two uniform variables of widths a, b around d
    double filtered(double d, double a, double b) const;
    // the same mean restricted to [lo, hi] (the part of the footprint where the aperture exists);
    // falls back to filtered() when the footprint does not reach a limit or misses [lo, hi]
    double filteredIn(double d, double a, double b, double lo, double hi) const;
    // true (and the value) if C is exactly constant over [da, db]
    bool constOver(double da, double db, double& v) const;
};

} // namespace detail
} // namespace lenshq
