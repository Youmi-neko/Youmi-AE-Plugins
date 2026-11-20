// lenscore.cpp - scatter lens blur via span-delta accumulation.
//
// Model: every source pixel p (center at x+0.5, y+0.5) spreads its color over
// its own kernel K_p = sum_l beta_l * 1[S_l], where each S_l is a CONVEX shape
// (the aperture, scaled by s_l, optionally intersected with a shifted circle for
// cat-eye). The kernel is box-filtered by the output pixel grid:
//   weight(p -> q) = alpha * area(S ∩ pixel q),  alpha = 1 / sum_l beta_l*area(S_l)
// so every splat sums to exactly 1 (energy conserving).
//
// Per output row band [k-0.5, k+0.5] (k = row offset from the kernel center)
// the area of S inside the band, as a function of x, is approximated by a single
// span [xl, xr] of height h, where xl/xr are the band-AVERAGED span endpoints
// (from integral tables). This is exact for the total area in the band, so the
// discrete totals telescope to the analytic area. A span's per-pixel coverage
// is the exact box integral of [xl, xr], expressed with 4 deltas; a prefix sum
// over the row resolves all spans at once (O(pixels x radius x layers)).
//
// A weight channel accumulates the same coverage; output = color / weight,
// which removes border darkening.
//
// Radial profile (v2): the kernel intensity I(r) (r = level-set scale of the
// aperture shape) is a product of rim / soft edge / core / apodization / onion
// / obstruction factors. It is cut into constant-intensity bands whose
// boundaries are placed by equal total variation of I (so steep parts get many
// layers, flat parts none), subject to a minimum spacing in pixels. Because the
// useful layer count depends on the radius, there is one layer table per radius
// octave ("tier"); a splat near the top of an octave cross-fades into the next
// table, so the kernel changes continuously with radius. Layers may have
// negative weights (rings, hollow centre); I(r) >= 0 everywhere, and outputs are
// clamped at 0 against discretisation undershoot. With all v2 controls at their
// defaults the v1 layer construction is used verbatim (bit-exact v1 output).
//
// Highlights (v1.2, see computeHighlights): luminance-true expansion of bright
// pixels before the scatter, optionally selective to small details (white
// top-hat ratio, van Herk min/max filters, O(N)). Disc brightness stays
// monotonic in source luminance and size; amount 0 => no boost (identity).
// Optional output rolloff compresses only the added highlight energy.
//
// Glow: (1-g)*blur + g*wide(source), wide = 3x box blur on a downsampled grid,
// weight normalised at buffer edges. Vignetting: rgb *= 1 - v*rn^2.

#include "lenscore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <exception>
#include <cstdint>
#include <cstring>
#include <new>
#include <thread>
#include <vector>
#include <emmintrin.h>  // SSE2 (baseline of every x64 CPU)

// v2.3 profiling hooks (build\perf only: compile with /DLENSCORE_PERF_LOG). Off in every product build.
#ifdef LENSCORE_PERF_LOG
#include <cstdio>
#define LC_PERF_BEGIN() double lcT_[16]; const char* lcN_[16]; int lcK_ = 0; lcN_[0] = "start"; lcT_[lcK_++] = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()
#define LC_PERF_MARK(name) do { if (lcK_ < 16) { lcN_[lcK_] = #name; lcT_[lcK_++] = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); } } while (0)
#define LC_PERF_DUMP(stage) do { std::fprintf(stderr, "[%s]", stage); for (int i_ = 1; i_ < lcK_; ++i_) std::fprintf(stderr, " %s=%.1f", lcN_[i_], (lcT_[i_] - lcT_[i_ - 1]) * 1e3); std::fprintf(stderr, "\n"); } while (0)
#else
#define LC_PERF_BEGIN() do {} while (0)
#define LC_PERF_MARK(name) do {} while (0)
#define LC_PERF_DUMP(stage) do {} while (0)
#endif

namespace lenscore {
namespace {

const double kPi = 3.14159265358979323846;
inline double clampd(double v, double lo, double hi) { return std::min(hi, std::max(lo, v)); }
// v2.2: colour tints (linear RGB multipliers) are clamped to [0, 64]; NaN -> 0.
inline double tintC(double v) { return v >= 0 ? std::min(64.0, v) : 0.0; }
// v2.2: buffer pixel aspect (width / height), sane range, NaN -> 1
inline double aspectOf(const Params& p) {
    const double a = p.pixelAspect;
    return a > 0.05 && a < 20.0 ? a : 1.0;
}

// Input sanitising (v2.1f): a NaN / Inf pixel (32 bpc hosts, broken EXRs, divisions upstream) would
// otherwise poison the running span sums for the rest of its row. Non-finite -> 0, |v| <= kMaxValue.
const float kMaxValue = 1.0e6f;
inline float sanitize1(float v) {
    return std::isfinite(v) ? std::min(kMaxValue, std::max(-kMaxValue, v)) : (v > 0 ? kMaxValue : 0.f);
}
// Source pixels (v2.2): alpha is clamped to [0, 1] (NaN -> 0, +Inf -> 1). Colour: NaN / -Inf -> 0,
// +Inf -> 65504 (the half-float maximum: a broken pixel must not become the brightest light of the
// frame), finite values clamped to [0, kMaxValue]. Negative colours (legal in 32 bpc) are clamped
// to 0 for every profile: v2.1 kept them on the legacy profile and clamped the general one's output,
// so results depended on the lens shape. A scatter of negative light has no physical meaning.
const float kInfColour = 65504.f;
inline void sanitize4(const float* s, float* d) {
    for (int c = 0; c < 3; ++c) {
        const float v = s[c];
        d[c] = std::isfinite(v) ? std::min(kMaxValue, std::max(0.f, v)) : (v > 0 ? kInfColour : 0.f);
    }
    const float a = s[3];
    d[3] = std::isfinite(a) ? std::min(1.f, std::max(0.f, a)) : (a > 0 ? 1.f : 0.f);
}
const int kMaxLayers = 32;
const int kTiers = 8;          // radius octaves for the general profile
const double kTierBase = 2.0;  // tier t covers R in [2*2^t, 2*2^(t+1)) (tier 0 also R < 2)
const int kTierLayers[kTiers] = {3, 4, 6, 8, 11, 14, 18, 24};
const double kAdaptRadius = 0.08;  // highlight knee adaptation window radius, fraction of the half-diagonal
const double kAdaptGain = 1.3;     // effective knee >= 1.3 x local mean luminance
const double kLevelBase = 34.0;    // multi-resolution: level l takes coarse radii ~[34, 68) px
const int kMaxLevels = 7;
const double kHardEdgeFastBase = 20.0;  // alpha: level base for hard-edged kernels

// ---------------------------------------------------------------- shapes ----

struct Constraint {
    bool disk;
    double nx, ny, a;      // half-plane: nx*x + ny*y <= a
    double cx, cy, rho2;   // disk: |p - c|^2 <= rho2
};

bool spanAt(const std::vector<Constraint>& cs, double v, double& l, double& r) {
    l = -1e30; r = 1e30;
    for (const Constraint& c : cs) {
        if (c.disk) {
            const double dy = v - c.cy;
            const double q = c.rho2 - dy * dy;
            if (q <= 0) return false;
            const double hh = std::sqrt(q);
            l = std::max(l, c.cx - hh);
            r = std::min(r, c.cx + hh);
        } else {
            const double rhs = c.a - c.ny * v;
            if (std::fabs(c.nx) < 1e-12) {
                if (rhs < 0) return false;
            } else if (c.nx > 0) {
                r = std::min(r, rhs / c.nx);
            } else {
                l = std::max(l, rhs / c.nx);
            }
        }
    }
    return r > l;
}

// Integral tables of the left/right span endpoints of a unit convex shape:
// IL(u) = int_{ymin}^{u} xl(v) dv, IR likewise. Linear interpolation of the
// integral == exact integration of a stack of thin slabs (N per unit shape).
struct ShapeTable {
    double ymin = -1, ymax = 1, xmin = -1, xmax = 1, area = kPi;
    int N = 0;
    double invStep = 0;
    std::vector<double> I;  // interleaved IL, IR; size 2*(N+1)

    inline void lookup(double u, double& il, double& ir) const {
        double f = (u - ymin) * invStep;
        int i = (int)f;
        if (i < 0) i = 0;
        if (i > N - 1) i = N - 1;
        f -= i;
        const double* q = &I[2 * (size_t)i];
        il = q[0] + f * (q[2] - q[0]);
        ir = q[1] + f * (q[3] - q[1]);
    }

    void build(const std::vector<Constraint>& cs, int n) {
        N = n;
        const int M = 4001;
        int iFirst = -1, iLast = -1;
        double l, r;
        for (int i = 0; i < M; ++i) {
            const double v = -1.0 + 2.0 * i / (M - 1);
            if (spanAt(cs, v, l, r)) {
                if (iFirst < 0) iFirst = i;
                iLast = i;
            }
        }
        if (iFirst < 0) {  // cannot happen for valid params; fall back to circle
            std::vector<Constraint> c1(1);
            c1[0] = Constraint{true, 0, 0, 0, 0, 0, 1.0};
            build(c1, n);
            return;
        }
        auto vAt = [&](int i) { return -1.0 + 2.0 * i / (M - 1); };
        auto refine = [&](double empty, double full) {
            for (int it = 0; it < 60; ++it) {
                const double m = 0.5 * (empty + full);
                double a, b;
                if (spanAt(cs, m, a, b)) full = m; else empty = m;
            }
            return full;
        };
        ymin = iFirst > 0 ? refine(vAt(iFirst - 1), vAt(iFirst)) : -1.0;
        ymax = iLast < M - 1 ? refine(vAt(iLast + 1), vAt(iLast)) : 1.0;
        const double step = (ymax - ymin) / N;
        invStep = 1.0 / step;
        I.assign(2 * (size_t)(N + 1), 0.0);
        xmin = 1e30; xmax = -1e30;
        const int S = 16;
        double sl = 0, sr = 0;
        for (int i = 0; i < N; ++i) {
            for (int s = 0; s < S; ++s) {
                const double v = ymin + (i + (s + 0.5) / S) * step;
                if (spanAt(cs, v, l, r)) {
                    sl += l * step / S;
                    sr += r * step / S;
                    xmin = std::min(xmin, l);
                    xmax = std::max(xmax, r);
                }
            }
            I[2 * (size_t)(i + 1)] = sl;
            I[2 * (size_t)(i + 1) + 1] = sr;
        }
        if (xmin > xmax) { xmin = -1; xmax = 1; }
        area = sr - sl;
    }
};

std::vector<Constraint> makeAperture(int blades, double curvature, double rotRad, double rounding = 0) {
    std::vector<Constraint> cs;
    if (blades < 3 || curvature >= 0.9999) {
        cs.push_back(Constraint{true, 0, 0, 0, 0, 0, 1.0});
        return cs;
    }
    const int n = std::min(blades, 16);
    const double half = kPi / n;
    double apothem = std::cos(half);
    for (int k = 0; k < n; ++k) {
        // vertices at rot + k*2pi/n on the unit circle; edge normal halfway between.
        const double phi = rotRad + (k + 0.5) * 2.0 * kPi / n;
        const double ux = std::cos(phi), uy = std::sin(phi);
        Constraint c{};
        if (curvature < 1e-4) {
            c.disk = false; c.nx = ux; c.ny = uy; c.a = std::cos(half);
        } else {
            // blade edge = circular arc of radius rho through both adjacent vertices.
            const double rho = 1.0 / curvature;
            const double t = std::sqrt(rho * rho - std::sin(half) * std::sin(half)) - std::cos(half);
            c.disk = true; c.cx = -t * ux; c.cy = -t * uy; c.rho2 = rho * rho;
            apothem = rho - t;
        }
        cs.push_back(c);
    }
    if (rounding > 1e-4) {
        // clip the vertices with a centred circle between the vertex radius and the apothem
        const double rr = 1.0 - std::min(1.0, rounding) * (1.0 - apothem);
        cs.push_back(Constraint{true, 0, 0, 0, 0, 0, rr * rr});
    }
    return cs;
}

// ------------------------------------------------------------- profiles -----

struct Profile {
    int nl = 1;
    double ls[kMaxLayers] = {1};
    double lb[kMaxLayers] = {1};
    double s2 = 1;  // sum lb * ls^2
    bool share = false;          // alpha fast fringe: layers < shareFrom per channel, rest shared
    int shareFrom = 1;
    double s2out = 0, s2in = 0;  // sum lb*ls^2 over the per-channel layers / the shared layers
};

void finishProfile(Profile& P) {
    P.s2 = 0;
    for (int l = 0; l < P.nl; ++l) P.s2 += P.lb[l] * P.ls[l] * P.ls[l];
}

// v1 layer construction, verbatim.
void legacyProfile(Profile& P, double e, double rw) {
    const double K = 3.0;
    P.nl = 1; P.ls[0] = 1; P.lb[0] = 1;
    if (e > 1e-3) {
        P.nl = 2;
        P.ls[0] = 1; P.lb[0] = 1 + K * e;   // rim intensity
        P.ls[1] = 1 - rw; P.lb[1] = -K * e; // core back to 1
    } else if (e < -1e-3) {
        P.nl = 4;
        const double i0 = 1.0 / (1.0 + K * -e);  // outermost band intensity, core = 1
        double prev = 0;
        for (int l = 0; l < P.nl; ++l) {
            const double t = (double)l / (P.nl - 1);
            P.ls[l] = 1.0 - rw * t;
            const double inten = i0 + (1.0 - i0) * t;
            P.lb[l] = inten - prev;
            prev = inten;
        }
    }
    finishProfile(P);
}

struct ProfileSpec {
    double e, rw, soft, apod, core, obstr, onion, onionN;
};

double profileI(const ProfileSpec& s, double r, double onionAmp) {
    const double K = 3.0;
    double I = 1;
    if (s.e > 1e-3) {
        if (r >= 1 - s.rw) I += K * s.e;
    } else if (s.e < -1e-3) {
        const double i0 = 1.0 / (1.0 + K * -s.e);
        const double t = std::min(1.0, std::max(0.0, (1 - r) / s.rw));
        I *= i0 + (1 - i0) * t;
    }
    if (s.core > 0) I *= 1 + 2.0 * s.core * (1 - r * r);
    else if (s.core < 0) I *= 1 + 0.9 * s.core * (1 - r * r);
    if (s.apod > 0) I *= std::exp(-4.0 * s.apod * r * r);
    if (s.soft > 0) {
        const double t = std::min(1.0, std::max(0.0, (1 - r) / s.soft));
        I *= t * t * (3 - 2 * t);
    }
    if (onionAmp > 0) I *= 1 + 0.5 * onionAmp * std::cos(2 * kPi * s.onionN * r);
    if (r < s.obstr) I = 0;
    return std::max(0.0, I);
}

// General profile for one tier: Rref = smallest radius (px) the tier is used for.
void generalProfile(Profile& P, const ProfileSpec& s, double Rref, int nmax) {
    const int M = 4096;
    double onionAmp = 0;
    if (s.onion > 0) {
        const double periodPx = Rref / std::max(1.0, s.onionN);
        onionAmp = s.onion * std::min(1.0, std::max(0.0, (periodPx - 2.0) / 3.0));
    }
    std::vector<double> I(M), rr(M);
    double Imax = 0;
    for (int i = 0; i < M; ++i) {  // i = 0 is the outermost sample
        rr[i] = 1.0 - (i + 0.5) / M;
        I[i] = profileI(s, rr[i], onionAmp);
        Imax = std::max(Imax, I[i]);
    }
    if (Imax <= 0) {  // degenerate (cannot happen with clamped params): plain disc
        P = Profile();
        finishProfile(P);
        return;
    }
    double tv = 0;
    for (int i = 1; i < M; ++i) tv += std::fabs(I[i] - I[i - 1]);
    // boundaries (as sample indices where a new band starts), outermost band starts at 0
    std::vector<int> bnd(1, 0);
    const int extra = std::min(nmax - 1, (int)std::ceil(tv / (0.05 * Imax) - 1e-9));
    if (extra > 0) {
        const double step = tv / (extra + 1);
        const double minSpacing = std::max(1.0 / M, 1.0 / std::max(1.0, Rref));  // in r
        double acc = 0, next = step;
        for (int i = 1; i < M && (int)bnd.size() <= extra; ++i) {
            acc += std::fabs(I[i] - I[i - 1]);
            if (acc >= next) {
                if (rr[bnd.back()] + 0.5 / M - (rr[i] + 0.5 / M) >= minSpacing) bnd.push_back(i);
                while (next <= acc) next += step;
            }
        }
    }
    // band intensity = area-weighted mean of I over the band
    const int nb = (int)bnd.size();
    std::vector<double> bi(nb), bs(nb);
    for (int b = 0; b < nb; ++b) {
        const int i0 = bnd[b], i1 = b + 1 < nb ? bnd[b + 1] : M;
        double sw = 0, si = 0;
        for (int i = i0; i < i1; ++i) { sw += rr[i]; si += rr[i] * I[i]; }
        bi[b] = sw > 0 ? si / sw : 0;
        bs[b] = rr[i0] + 0.5 / M;  // outer edge of the band
    }
    P.nl = 0;
    double prev = 0;
    for (int b = 0; b < nb && P.nl < kMaxLayers; ++b) {
        const double d = bi[b] - prev;
        if (b > 0 && std::fabs(d) < 1e-6 * Imax) continue;
        P.ls[P.nl] = bs[b];
        P.lb[P.nl] = d;
        ++P.nl;
        prev = bi[b];
    }
    finishProfile(P);
    if (!(P.s2 > 1e-9)) { P = Profile(); finishProfile(P); }
}

// ---------------------------------------------------------------- context ---

struct Splat {
    float c[4];
    float R;     // blur radius in px; < 0 means "fits inside own pixel" (identity)
    float sgn;   // +1 behind focus, -1 in front
    float wb;    // cross-fade weight into tier + 1
    float eL;    // luminance of the highlight energy added to c (drives the output rolloff)
    float m;     // mass: share of the source(s) rendered by this splat on this level (1 = all)
    int x;
    int reach;   // max |row offset| this splat touches
    int tier;
};

struct CatInfo {
    float ox, oy;        // half of the aperture/clip-circle separation, unit aperture space
    float invD[2][3];    // exact weight / (sum beta * area) per [tier pass][channel kernel]
};

// v2.3: array of trivially-copyable T WITHOUT value-initialisation. The blur stage's per-pixel arrays
// (splats, cat-eye data, coarse-level copies and grids, ~200 MB at UHD) are completely written before
// they are read; std::vector::resize zero-filled them first, a measurable serial pass over fresh
// memory (page faults + the zeroing) before the parallel pass that writes them again. resize()
// discards the old contents.
template <class T>
class RawBuf {
public:
    void resize(size_t n) {
        if (n == n_) return;
        p_.reset();
        n_ = 0;
        if (n) p_.reset(new T[n]);  // default-init: no zeroing for trivial T
        n_ = n;
    }
    T* data() { return p_.get(); }
    const T* data() const { return p_.get(); }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return p_[i]; }
    const T& operator[](size_t i) const { return p_[i]; }
    void clear() { p_.reset(); n_ = 0; }
    void shrink_to_fit() {}
private:
    std::unique_ptr<T[]> p_;
    size_t n_ = 0;
};

struct Ctx {
    int W = 0, H = 0;
    ShapeTable ap, circ;
    double ax = 1, ay = 1;
    int nk = 1;
    double ks[2][3] = {{1, 1, 1}, {1, 1, 1}};  // [front/back][R,G,B] kernel scale
    int nTiers = 1;
    Profile prof[2][kTiers];                   // [front/back][tier]
    bool clampNeg = false;
    bool cat = false;
    double catK = 0;
    double xext = 1, yext = 1;
    int maxReach = 0;
    // frame geometry
    double fcx = 0, fcy = 0, halfDiag = 1;
    double pa = 1;  // buffer pixel aspect (width / height), v2.2
    RawBuf<Splat> splats;          // fine level
    RawBuf<CatInfo> cats;
    // multi-resolution levels l >= 1 (grid 2^l coarser): large-radius splats merged per block;
    // their raw accumulators are upsampled and added before the fine level normalises.
    struct Level {
        int f = 1, W = 0, H = 0, gox = 0, goy = 0, maxReach = 0;
        RawBuf<Splat> splats;
        RawBuf<CatInfo> cats;
        RawBuf<float> grid;        // W*H*NC prefix-summed accumulators after pass 2 (all written)
    };
    int nLevels = 1;
    double levelBase = 34.0;      // multi-resolution: level l takes coarse radii ~[base, 2 base)
    bool fringeShare = false;     // alpha: shared inner layers for the colour fringe
    std::vector<Level> levels;     // index 0 unused (the fine level is splats/cats above)
    struct Px { float c[4]; float eL, R, sgn; };
    RawBuf<Px> pix;                // fine-pixel data for building the coarse levels
    // general-profile cat eye: tabulated unit-shape area A(ox, oy) (every layer is a scaled
    // copy of the same clipped shape, so a splat's normaliser is Rx*Ry*s2*A).
    int catN = 0;
    double catLim = 1, catInv = 1;
    std::vector<double> catA;
    double catArea(double ox, double oy) const {
        double u = (ox + catLim) * catInv, v = (oy + catLim) * catInv;
        u = std::min((double)catN - 1.000001, std::max(0.0, u));
        v = std::min((double)catN - 1.000001, std::max(0.0, v));
        const int i = (int)u, j = (int)v;
        const double fu = u - i, fv = v - j;
        const double* r0 = &catA[(size_t)j * catN + i];
        const double* r1 = r0 + catN;
        const double a = r0[0] + (r0[1] - r0[0]) * fu, b = r1[0] + (r1[1] - r1[0]) * fu;
        return a + (b - a) * fv;
    }
};

// Band [a,b] (px, relative to kernel center) through the aperture of radii RxS,RyS.
inline bool apSpan(const ShapeTable& t, double RxS, double RyS, double a, double b,
                   double& h, double& xl, double& xr) {
    const double inv = 1.0 / RyS;
    double u0 = a * inv, u1 = b * inv;
    if (u0 < t.ymin) u0 = t.ymin;
    if (u1 > t.ymax) u1 = t.ymax;
    const double du = u1 - u0;
    if (du <= 1e-12) return false;
    double l0, r0, l1, r1;
    t.lookup(u0, l0, r0);
    t.lookup(u1, l1, r1);
    const double s = RxS / du;
    xl = (l1 - l0) * s;
    xr = (r1 - r0) * s;
    h = du * RyS;
    return xr > xl;
}

// Cat-eye: aperture centered at -(hx,hy) intersected with the unit clip circle
// centered at +(hx,hy) (unit space). Centering the pair on the source pixel keeps
// the kernel centroid in place; otherwise a position-dependent shift would act
// like a slight radial magnification of the whole image.
inline bool catSpan(const Ctx& C, double RxS, double RyS, double a, double b,
                    double hx, double hy, double& h, double& xl, double& xr) {
    const double inv = 1.0 / RyS;
    double u0 = a * inv, u1 = b * inv;
    u0 = std::max(u0, std::max(C.ap.ymin - hy, hy + C.circ.ymin));
    u1 = std::min(u1, std::min(C.ap.ymax - hy, hy + C.circ.ymax));
    const double du = u1 - u0;
    if (du <= 1e-12) return false;
    double l0, r0, l1, r1, cl0, cr0, cl1, cr1;
    C.ap.lookup(u0 + hy, l0, r0);
    C.ap.lookup(u1 + hy, l1, r1);
    C.circ.lookup(u0 - hy, cl0, cr0);
    C.circ.lookup(u1 - hy, cl1, cr1);
    const double idu = 1.0 / du;
    const double L = std::max((l1 - l0) * idu - hx, hx + (cl1 - cl0) * idu);
    const double R = std::min((r1 - r0) * idu - hx, hx + (cr1 - cr0) * idu);
    if (R <= L) return false;
    xl = L * RxS;
    xr = R * RxS;
    h = du * RyS;
    return true;
}

// ------------------------------------------------------------ threading -----

// Persistent worker pool, owned by one top-level call (prepare / blur / post / render) and published
// to parallelFor through a thread_local pointer for the duration of that call: no state shared
// between calls or threads, so concurrent renders (AE multi-frame rendering) stay independent.
// Workers are reused by every parallelFor of the call (thread creation per pass was measurable).
class Pool {
public:
    explicit Pool(int n) {
        for (int t = 1; t < n; ++t) {
            try {
                threads_.emplace_back([this, t] { loop(t); });
            } catch (...) {
                break;  // run with fewer threads
            }
        }
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    int workers() const { return (int)threads_.size(); }
    // runs (*job)(tid) once on every worker; returns immediately
    void start(const std::function<void(int)>* job) {
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = job;
            pending_ = (int)threads_.size();
            ++gen_;
        }
        cv_.notify_all();
    }
    // waits for the workers of the current job; polls `poll` every ~2 ms
    template <class P>
    void wait(P&& poll) {
        std::unique_lock<std::mutex> lk(m_);
        while (pending_ > 0) {
            if (doneCv_.wait_for(lk, std::chrono::milliseconds(2), [&] { return pending_ == 0; })) break;
            lk.unlock();
            poll();
            lk.lock();
        }
    }

private:
    void loop(int tid) {
        unsigned long long seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return gen_ != seen; });
            seen = gen_;
            if (quit_) return;
            const std::function<void(int)>* job = job_;
            lk.unlock();
            (*job)(tid);
            lk.lock();
            if (--pending_ == 0) doneCv_.notify_all();
        }
    }
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_, doneCv_;
    const std::function<void(int)>* job_ = nullptr;
    unsigned long long gen_ = 0;
    int pending_ = 0;
    bool quit_ = false;
};

thread_local Pool* tlPool = nullptr;

// Installs a pool for the current thread unless one of the same size is already active (nested
// stage calls inside render() reuse the outer pool).
struct PoolScope {
    std::unique_ptr<Pool> own;
    Pool* prev;
    explicit PoolScope(int nth) : prev(tlPool) {
        if (!tlPool || tlPool->workers() + 1 != nth) {
            own.reset(new Pool(nth));
            tlPool = own.get();
        }
    }
    ~PoolScope() { tlPool = prev; }
};

// v2.3: cancellation inside long work items. parallelFor publishes its stop flag (and, on the calling
// thread, its abort poll) through this thread_local for the duration of each item, so a long item
// (a doBlock of 8 output rows can take >100 ms on one thread at large radii) can give up early:
// workers see the stop flag, the calling thread also polls the host's abort callback (at most every
// ~2 ms; the callback is still only ever called on the calling thread). Per call, no shared state.
struct CancelCtx {
    const std::atomic<bool>* stop = nullptr;
    std::atomic<bool>* stopW = nullptr;               // caller only: set when the host aborts
    const std::function<bool()>* aborted = nullptr;   // caller only
    std::chrono::steady_clock::time_point next{};
};
thread_local CancelCtx* tlCancel = nullptr;
// true = the current parallelFor is stopping: the item may return early (its result is discarded)
inline bool cancelRequested() {
    CancelCtx* c = tlCancel;
    if (!c) return false;
    if (c->stop->load(std::memory_order_relaxed)) return true;
    if (c->aborted) {
        const auto t = std::chrono::steady_clock::now();
        if (t >= c->next) {
            c->next = t + std::chrono::milliseconds(2);
            if ((*c->aborted)()) { c->stopW->store(true); return true; }
        }
    }
    return false;
}
struct CancelScope {
    CancelCtx* prev;
    explicit CancelScope(CancelCtx* c) : prev(tlCancel) { tlCancel = c; }
    ~CancelScope() { tlCancel = prev; }
};

template <class F>
bool parallelFor(int n, int nthreads, int chunk, F&& f, AbortFn abortFn, void* user) {
    std::atomic<int> next{0};
    std::atomic<bool> stop{false};
    // v2.1f: an exception in a body (std::bad_alloc under memory pressure) must neither escape a worker
    // thread (std::terminate would take the host down) nor unwind this frame while workers still use
    // it: the first one is stored, everyone stops, and it is rethrown here after all workers are done.
    std::exception_ptr err;
    std::mutex errM;
    auto work = [&](int tid) {
        CancelCtx cw;
        cw.stop = &stop;
        CancelScope cs(&cw);
        try {
            for (;;) {
                if (stop.load(std::memory_order_relaxed)) break;
                const int i0 = next.fetch_add(chunk);
                if (i0 >= n) break;
                const int i1 = std::min(n, i0 + chunk);
                for (int i = i0; i < i1; ++i) f(i, tid);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(errM);
            if (!err) err = std::current_exception();
            stop = true;
        }
    };
    Pool* pool = tlPool;
    std::unique_ptr<Pool> local;
    if (!pool || pool->workers() + 1 > nthreads) {
        if (nthreads > 1) {
            local.reset(new Pool(nthreads));
            pool = local.get();
        } else {
            pool = nullptr;
        }
    }
    // v2.2: an exception thrown by the host's abort callback counts as an abort (it must not leave
    // this frame while workers still use it)
    const std::function<bool()> aborted = [&]() -> bool {
        if (!abortFn) return false;
        try { return abortFn(user) != 0; } catch (...) { return true; }
    };
    const std::function<void(int)> job = [&](int tid) { work(tid); };
    if (pool && pool->workers() > 0) pool->start(&job);
    // calling thread: works and is the only one that polls abort
    try {
        CancelCtx cc;
        cc.stop = &stop;
        cc.stopW = &stop;
        cc.aborted = abortFn ? &aborted : nullptr;
        cc.next = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
        CancelScope cs(&cc);
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) break;
            if (aborted()) { stop = true; break; }
            const int i0 = next.fetch_add(chunk);
            if (i0 >= n) break;
            const int i1 = std::min(n, i0 + chunk);
            for (int i = i0; i < i1; ++i) f(i, 0);
        }
    } catch (...) {
        std::lock_guard<std::mutex> lk(errM);
        if (!err) err = std::current_exception();
        stop = true;
    }
    if (pool && pool->workers() > 0)
        pool->wait([&] { if (!stop && aborted()) stop = true; });
    if (err) std::rethrow_exception(err);
    return !stop.load();
}

// v2.3: rows y = 0..H-1 in parallel, for element-wise passes that used to run on one thread (each
// pixel is computed exactly as before; nothing is accumulated across pixels)
template <class F>
bool parRows(int H, int nth, AbortFn ab, void* user, F&& f) {
    return parallelFor(H, nth, 8, [&](int y, int) { f(y); }, ab, user);
}

// ------------------------------------------------------------ amount --------

inline double amountOf(double m, const Params& p, double& sgn) {
    if (!(m == m)) m = 0;
    if (p.invertMap) m = 1.0 - m;
    const double d = m - p.focus;
    sgn = d < 0 ? -1.0 : 1.0;
    double amt = std::fabs(d) * p.mapScale;
    amt = std::min(1.0, std::max(0.0, amt));
    if (p.mapGamma != 1.f && amt > 0) amt = std::pow(amt, (double)p.mapGamma);
    return amt;
}

// ------------------------------------------------------------ highlights ----

// 1D running min/max over the window [i-r, i+r] clipped to [0, n-1] (van Herk / Gil-Werman).
template <bool MAX>
void runFilter(const float* in, float* out, int n, int r, float* g, float* hbuf) {
    const int k = 2 * r + 1;
    auto op = [](float a, float b) { return MAX ? std::max(a, b) : std::min(a, b); };
    for (int b0 = 0; b0 < n; b0 += k) {
        const int b1 = std::min(n, b0 + k);
        g[b0] = in[b0];
        for (int i = b0 + 1; i < b1; ++i) g[i] = op(g[i - 1], in[i]);
        hbuf[b1 - 1] = in[b1 - 1];
        for (int i = b1 - 2; i >= b0; --i) hbuf[i] = op(hbuf[i + 1], in[i]);
    }
    for (int i = 0; i < n; ++i) {
        const int a = std::max(0, i - r), b = std::min(n - 1, i + r);
        if (a / k == b / k) out[i] = (a % k == 0) ? g[b] : hbuf[a];
        else out[i] = op(hbuf[a], g[b]);
    }
}

// Per-thread scratch buffers reused by every filter pass of one prepare call (allocating and
// zero-filling them per pass cost more than the filtering itself).
struct Scratch {
    std::vector<std::vector<float>> buf;
    float* get(int tid, size_t n) {
        std::vector<float>& b = buf[tid];
        if (b.size() < n) b.resize(n);
        return b.data();
    }
    explicit Scratch(int nth) : buf(nth) {}
};

// Min (MAX=false) or max filter along straight lines of one direction, window radius r steps.
// dir: 0 = rows, 1 = columns, 2 = diagonal (+1,+1), 3 = anti-diagonal (+1,-1). In place.
template <bool MAX>
bool filterLines(std::vector<float>& img, int W, int H, int r, int dir, int nth, Scratch& sc, AbortFn ab, void* user) {
    if (r <= 0) return true;
    const int dx = dir == 1 ? 0 : 1, dy = dir == 0 ? 0 : (dir == 3 ? -1 : 1);
    std::vector<int> sx, sy;  // line start points: pixels whose predecessor is outside the image
    if (dir == 0) { for (int y = 0; y < H; ++y) { sx.push_back(0); sy.push_back(y); } }
    else if (dir == 1) { for (int x = 0; x < W; ++x) { sx.push_back(x); sy.push_back(0); } }
    else if (dir == 2) {
        for (int y = H - 1; y >= 0; --y) { sx.push_back(0); sy.push_back(y); }
        for (int x = 1; x < W; ++x) { sx.push_back(x); sy.push_back(0); }
    } else {
        for (int y = 0; y < H; ++y) { sx.push_back(0); sy.push_back(y); }
        for (int x = 1; x < W; ++x) { sx.push_back(x); sy.push_back(H - 1); }
    }
    const int n = std::max(W, H);
    return parallelFor((int)sx.size(), nth, 16, [&](int li, int tid) {
        float* a = sc.get(tid, 4 * (size_t)n);
        float* o = a + n;
        int x = sx[li], y = sy[li], len = 0;
        while (x >= 0 && x < W && y >= 0 && y < H) { a[len++] = img[(size_t)y * W + x]; x += dx; y += dy; }
        runFilter<MAX>(a, o, len, r, o + n, o + 2 * (size_t)n);
        x = sx[li]; y = sy[li];
        for (int i = 0; i < len; ++i) { img[(size_t)y * W + x] = o[i]; x += dx; y += dy; }
    }, ab, user);
}

// 2D separable min (MAX=false) or max filter with a (2r+1)^2 square, in place on `img`.
template <bool MAX>
bool filter2D(std::vector<float>& img, int W, int H, int r, int nth, Scratch& sc, AbortFn ab, void* user) {
    bool ok = parallelFor(H, nth, 8, [&](int y, int tid) {
        float* s = sc.get(tid, 4 * (size_t)std::max(W, H));
        float* row = img.data() + (size_t)y * W;
        std::copy(row, row + W, s);
        runFilter<MAX>(s, row, W, r, s + W, s + 2 * (size_t)W);
    }, ab, user);
    if (!ok) return false;
    // Column pass, van Herk on blocks of CB columns at once while sweeping rows: every access is
    // a contiguous run of CB floats (cache friendly, vectorisable). Same result as per column.
    // v2.3: CB 64 -> 16 (one cache line per row): the two CB x H scratch planes of a block now stay
    // in L2 at UHD heights (64 columns x 3300 rows x 2 planes = 1.7 MB per thread did not).
    const int CB = 16;
    const int nblk = (W + CB - 1) / CB;
    const int k = 2 * r + 1;
    auto op = [](float a, float b) { return MAX ? std::max(a, b) : std::min(a, b); };
    return parallelFor(nblk, nth, 1, [&](int bi, int tid) {
        const int x0 = bi * CB, nb = std::min(W, x0 + CB) - x0;
        float* g = sc.get(tid, (size_t)CB * H * 2);
        float* hb = g + (size_t)CB * H;
        for (int b0 = 0; b0 < H; b0 += k) {
            const int b1 = std::min(H, b0 + k);
            const float* in0 = img.data() + (size_t)b0 * W + x0;
            for (int c = 0; c < nb; ++c) g[(size_t)b0 * CB + c] = in0[c];
            for (int y = b0 + 1; y < b1; ++y) {
                const float* in = img.data() + (size_t)y * W + x0;
                float* gy = g + (size_t)y * CB;
                const float* gp = gy - CB;
                for (int c = 0; c < nb; ++c) gy[c] = op(gp[c], in[c]);
            }
            const float* inl = img.data() + (size_t)(b1 - 1) * W + x0;
            for (int c = 0; c < nb; ++c) hb[(size_t)(b1 - 1) * CB + c] = inl[c];
            for (int y = b1 - 2; y >= b0; --y) {
                const float* in = img.data() + (size_t)y * W + x0;
                float* hy = hb + (size_t)y * CB;
                const float* hn = hy + CB;
                for (int c = 0; c < nb; ++c) hy[c] = op(hn[c], in[c]);
            }
        }
        for (int y = 0; y < H; ++y) {
            const int a = std::max(0, y - r), b = std::min(H - 1, y + r);
            float* out = img.data() + (size_t)y * W + x0;
            const float* gb = g + (size_t)b * CB;
            const float* ha = hb + (size_t)a * CB;
            if (a / k == b / k) {
                const float* s = (a % k == 0) ? gb : ha;
                for (int c = 0; c < nb; ++c) out[c] = s[c];
            } else {
                for (int c = 0; c < nb; ++c) out[c] = op(ha[c], gb[c]);
            }
        }
    }, ab, user);
}

// Two box passes (radius r each, mean over the in-buffer part of the window) on a W x H image,
// in place on `a` (b = scratch). Used for the adaptive knee and on downsampled grids.
bool boxMean2(std::vector<float>& a, std::vector<float>& b, int W, int H, int r, int nth, AbortFn ab, void* user) {
    std::vector<std::vector<double>> pre(nth, std::vector<double>((size_t)std::max(W, H) + 1)), preB(nth);
    for (int pass = 0; pass < 2; ++pass) {
        bool ok = parallelFor(H, nth, 8, [&](int y, int tid) {
            double* ps = pre[tid].data();
            const float* in = a.data() + (size_t)y * W;
            ps[0] = 0;
            for (int x = 0; x < W; ++x) ps[x + 1] = ps[x] + in[x];
            float* o = b.data() + (size_t)y * W;
            for (int x = 0; x < W; ++x) {
                const int x0 = std::max(0, x - r), x1 = std::min(W, x + r + 1);
                o[x] = (float)((ps[x1] - ps[x0]) / (x1 - x0));
            }
        }, ab, user);
        if (!ok) return false;
        // v2.3: columns in blocks of CB, sweeping rows (contiguous reads instead of a W-float stride per
        // element; the strided version was ~1/3 of the prepare stage at UHD). Each column's prefix sum
        // and quotient are computed exactly as before.
        const int CB = 16;
        ok = parallelFor((W + CB - 1) / CB, nth, 1, [&](int bi, int tid) {
            const int x0 = bi * CB, nb = std::min(W, x0 + CB) - x0;
            std::vector<double>& pv = preB[tid];
            if (pv.size() < (size_t)(H + 1) * CB) pv.resize((size_t)(H + 1) * CB);
            double* ps = pv.data();
            for (int c = 0; c < nb; ++c) ps[c] = 0;
            for (int y = 0; y < H; ++y) {
                const float* row = b.data() + (size_t)y * W + x0;
                const double* prv = ps + (size_t)y * CB;
                double* cur = ps + (size_t)(y + 1) * CB;
                for (int c = 0; c < nb; ++c) cur[c] = prv[c] + row[c];
            }
            for (int y = 0; y < H; ++y) {
                const int y0 = std::max(0, y - r), y1 = std::min(H, y + r + 1);
                const double* p1 = ps + (size_t)y1 * CB;
                const double* p0 = ps + (size_t)y0 * CB;
                float* o = a.data() + (size_t)y * W + x0;
                for (int c = 0; c < nb; ++c) o[c] = (float)((p1[c] - p0[c]) / (y1 - y0));
            }
        }, ab, user);
        if (!ok) return false;
    }
    return true;
}

inline double smooth01(double t) {
    t = std::min(1.0, std::max(0.0, t));
    return t * t * (3 - 2 * t);
}

// grid phase so that coarse cells are aligned to the frame origin (tile-exact downsampling)
inline int gridPhase(const Params& p, bool x, int f) {
    if (!(p.frameWidth > 0 && p.frameHeight > 0) || f <= 1) return 0;
    const long long o = (long long)std::floor(-(x ? p.frameLeft : p.frameTop));
    return (int)(((o % f) + f) % f);
}

struct HlMaps {
    std::vector<float> E;  // expansion weight per pixel (empty = nothing to add)
    std::vector<float> D;  // detail weight per pixel (empty = selectivity 0)
};

const double kWhiteRadius = 0.10;   // White Protection neighbourhood radius, fraction of the diagonal
const int kWhiteLevels = 10;        // brightness levels, denser near white (paper vs light)
const double kWhiteTau[kWhiteLevels] = {0.30, 0.45, 0.60, 0.72, 0.82, 0.88, 0.925, 0.955, 0.98, 1.0};
const double kWhiteRel = 0.96;      // "as bright": >= 96% of the candidate's brightness

// Luminance-true highlight expansion (v1.2) + appearance controls (v2.0).
//   Y     = chroma-compensated brightness (a saturated clipped light counts like a white one).
//   E(Y)  = ((Y - k)/(1 - k))^curve above the knee k = max(knee, 1.3 x local mean); 1 + ln Y above 1.
//   D     = detail weight in [0,1]: mean over 5 scales (0.5..2 x detailSize) of the squared white
//           top-hat ratio ((Y - opening)/Y)^2; with detailMin > 0 the same measure at the minimum
//           size is subtracted (size window). Large flat areas, their edges and corners -> D ~ 0.
//   Dots  = (Y - max over 4 directions of a line opening)/Y: a stroke survives the line opening
//           along it (-> 0), a compact dot survives none (-> 1). D *= 1 - dotsOnly*(1 - dots^2).
//   White Protection: F = fraction of a wide neighbourhood (10% of the diagonal) that is at least
//           0.9 x as bright as the pixel (soft brightness levels, box means on a coarse grid);
//           E *= 1 - whiteProtection * smoothstep(1%, 10%, F). Common brightness = diffuse white.
//   Source: 1 = highlight layer only (E = layer, D = 1); 2 = automatic x layer.
// Every term is a continuous function of the pixel values (min/max/box filters), so the result is
// stable over time. Added energy (prepare) = colour * boost * E * [(1-sel)*amount + sel*D*S(R)],
// S(R) = pi * max(0, R^2 - 1) / 16.
bool computeHighlights(int W, int H, double halfDiag, const Params& p, const float* src, int ss,
                       const PrepareOptions& opt, HlMaps& M, int nth, AbortFn ab, void* user) {
    const size_t N = (size_t)W * H;
    Scratch sc(nth);
    const int srcMode = p.highlightSource == 1 ? 1 : (p.highlightSource == 2 ? 2 : 0);
    auto layerAt = [&](size_t i) -> float {
        if (!opt.highlightLayer) return 0.f;
        const int y = (int)(i / W), x = (int)(i % W);
        const float v = opt.highlightLayer[(size_t)y * opt.highlightLayerStride + x];
        return std::max(0.f, sanitize1(v));
    };
    if (srcMode == 1) {  // highlight layer only: the layer says where (and how strongly) lights are
        M.E.assign(N, 0.f);
        bool any = false;
        for (size_t i = 0; i < N; ++i) { M.E[i] = layerAt(i); any = any || M.E[i] > 0; }
        if (!any) { M.E.clear(); return true; }
        if (clampd(p.highlightSelectivity, 0.0, 1.0) > 0) M.D.assign(N, 1.f);
        return true;
    }
    LC_PERF_BEGIN();
    std::vector<float> L(N);
    if (!parallelFor(H, nth, 16, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            float s[4];
            sanitize4(src + (size_t)y * ss + (size_t)x * 4, s);
            // Chroma-compensated brightness: a saturated clipped light (red LED: L ~ 0.3, max = 1)
            // is as much a light source as a white one, so its brightness moves toward its max
            // channel by saturation^2. Neutral colours and pastels (skin) are nearly unchanged.
            const float l = 0.2126f * s[0] + 0.7152f * s[1] + 0.0722f * s[2];
            const float mx = std::max(s[0], std::max(s[1], s[2]));
            const float mn = std::min(s[0], std::min(s[1], s[2]));
            const float sat = mx > 1e-6f ? std::max(0.f, 1.f - mn / mx) : 0.f;
            const float br = l + std::max(0.f, mx - l) * sat * sat;
            L[(size_t)y * W + x] = std::isfinite(br) ? std::max(0.f, br) : 0.f;
        }
    }, ab, user)) return false;
    const double knee = clampd(p.highlightKnee, 0.0, 0.97);
    const double curve = clampd(p.highlightCurve, 1.0, 8.0);
    // Exposure-adaptive knee: a pixel is treated as a light only if it is clearly brighter than the
    // scene around it (local mean of min(L,1) over ~4% of the frame diagonal, two box passes).
    // Night scenes keep the user knee; high-key illustrations (skin ~0.85, walls ~0.7) get a knee
    // near their own brightness, so only true near-white highlights expand. Smooth in the image.
    std::vector<float> mean(N), tmp(N);
    if (!parallelFor(H, nth, 16, [&](int y, int) {  // v2.3: in parallel (was a serial pass)
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) tmp[i] = std::min(1.f, L[i]);
    }, ab, user)) return false;
    const int rm = std::max(2, (int)std::lround(kAdaptRadius * halfDiag));
    if (!boxMean2(tmp, mean, W, H, (rm + 1) / 2, nth, ab, user)) return false;
    LC_PERF_MARK(L_knee);
    M.E.assign(N, 0.f);
    std::vector<char> rowAny(H, 0);
    if (!parallelFor(H, nth, 16, [&](int y, int) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
            // SDR range: power curve from the knee to white (E(1) = 1). Above white (HDR input,
            // already scene-referred) expansion only grows logarithmically, so real HDR highlights
            // are not amplified a second time. Continuous and monotonic in L.
            if (L[i] > 1.f) { M.E[i] = (float)(1.0 + std::log((double)L[i])); rowAny[y] = 1; continue; }
            const double k = std::min(0.97, std::max(knee, kAdaptGain * (double)tmp[i]));
            const double t = (L[i] - k) / (1.0 - k);
            if (t > 0) { M.E[i] = (float)std::pow(t, curve); rowAny[y] = 1; }
        }
    }, ab, user)) return false;
    // v2.2 (renderVersion >= 2): energy linear in coverage. E(L) is non-linear, so applying it to
    // anti-aliased pixel values made a small light's total energy depend on how it straddles pixels
    // (2x flicker for a 3 px light under sub-pixel motion). Now the curve is evaluated once at the
    // light's peak (3x3 max) and each pixel gets its coverage share (L - bg) / (peak - bg), bg = 5x5
    // min: the total is ~ E(peak) x coverage, independent of the sub-pixel position. Interiors of
    // lights and flat areas (L = peak) are unchanged.
    if (p.renderVersion >= 2) {
        std::vector<float> pk = L, bgv = L;
        if (!filter2D<true>(pk, W, H, 1, nth, sc, ab, user)) return false;
        if (!filter2D<false>(bgv, W, H, 2, nth, sc, ab, user)) return false;
        std::fill(rowAny.begin(), rowAny.end(), (char)0);
        if (!parallelFor(H, nth, 16, [&](int y, int) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
                const double P = pk[i], B = std::min((double)bgv[i], (double)L[i]);
                double fp;
                if (P > 1.0) {
                    fp = 1.0 + std::log(P);
                } else {
                    const double k = std::min(0.97, std::max(knee, kAdaptGain * (double)tmp[i]));
                    const double t = (P - k) / (1.0 - k);
                    fp = t > 0 ? std::pow(t, curve) : 0.0;
                }
                if (fp <= 0) { M.E[i] = 0.f; continue; }
                const double den = P - B;
                const double w = den > 1e-4 * std::max(1.0, P) ? clampd((L[i] - B) / den, 0.0, 1.0) : 1.0;
                // prepareRow adds boost * E * colour, and the colour itself scales with the coverage:
                // E carries P / L so the added energy is fp * w * P * chromaticity (linear in w).
                // w <= L / P for B <= L <= P, so E <= fp; interiors (L = P) keep E = fp.
                const double lin = L[i] > 1e-12 ? std::min(1.0, w * P / (double)L[i]) : 0.0;
                M.E[i] = (float)(fp * lin);
                if (M.E[i] > 0) rowAny[y] = 1;
            }
        }, ab, user)) return false;
    }
    LC_PERF_MARK(E_peak);
    bool any = false;
    for (int y = 0; y < H; ++y) any = any || rowAny[y];
    if (srcMode == 2) {
        any = false;
        for (size_t i = 0; i < N; ++i) { M.E[i] *= layerAt(i); any = any || M.E[i] > 0; }
    }
    if (!any) { M.E.clear(); return true; }

    // ---- White Protection (rarity of the brightness in a wide neighbourhood)
    const double wpAmt = clampd(p.whiteProtection, 0.0, 1.0);
    if (wpAmt > 0) {
        const double rw = kWhiteRadius * 2.0 * halfDiag;
        const int f = std::max(1, (int)std::lround(rw / 16.0));
        const int gx = gridPhase(p, true, f), gy = gridPhase(p, false, f);
        const int gw = (W + gx + f - 1) / f, gh = (H + gy + f - 1) / f;
        const int rb = std::max(1, (int)std::lround(rw / f * 0.5));
        std::vector<std::vector<float>> lev(kWhiteLevels, std::vector<float>((size_t)gw * gh, 0.f));
        std::vector<float> cnt((size_t)gw * gh, 0.f);
        // v2.3: one task per coarse row (its f source rows): every cell still receives its pixels in
        // the original order (rows ascending, then x), so the float sums are unchanged
        if (!parallelFor(gh, nth, 1, [&](int gyr, int) {
            const int ya = std::max(0, gyr * f - gy), yb = std::min(H, (gyr + 1) * f - gy);
            for (int y = ya; y < yb; ++y)
                for (int x = 0; x < W; ++x) {
                    const size_t g = (size_t)((y + gy) / f) * gw + (x + gx) / f;
                    const double v = std::min(1.2f, L[(size_t)y * W + x]);
                    cnt[g] += 1.f;
                    for (int k = 0; k < kWhiteLevels; ++k) {
                        // soft step, width ~ the local level spacing (smooth in time)
                        const double tau = kWhiteTau[k];
                        const double wdt = 0.6 * (k > 0 ? tau - kWhiteTau[k - 1] : 0.15);
                        lev[k][g] += (float)smooth01((v - (tau - 0.5 * wdt)) / wdt);
                    }
                }
        }, ab, user)) return false;
        std::vector<float> scratch((size_t)gw * gh);
        for (int k = 0; k < kWhiteLevels; ++k) {
            for (size_t g = 0; g < cnt.size(); ++g) lev[k][g] = cnt[g] > 0 ? lev[k][g] / cnt[g] : 0.f;
            if (!boxMean2(lev[k], scratch, gw, gh, rb, nth, ab, user)) return false;
        }
        std::vector<char> rowAny2(H, 0);
        if (!parallelFor(H, nth, 8, [&](int y, int) {  // v2.3: rows in parallel (per-pixel, independent)
            bool any2 = false;
            const double v = (y + gy + 0.5) / f - 0.5;
            const double vc = std::min((double)gh - 1, std::max(0.0, v));
            const int y0 = (int)vc, y1 = std::min(y0 + 1, gh - 1);
            const double fy = vc - y0;
            for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                if (M.E[i] <= 0) continue;
                const double u = (x + gx + 0.5) / f - 0.5;
                const double uc = std::min((double)gw - 1, std::max(0.0, u));
                const int x0 = (int)uc, x1 = std::min(x0 + 1, gw - 1);
                const double fx = uc - x0;
                // fraction of the neighbourhood at >= 96% of this brightness (interpolated level)
                const double tv = std::min(1.0, kWhiteRel * L[i]);
                int k0 = 0;
                while (k0 + 1 < kWhiteLevels && kWhiteTau[k0 + 1] <= tv) ++k0;
                const int k1 = std::min(kWhiteLevels - 1, k0 + 1);
                const double ft = k1 == k0 ? 0.0 : std::min(1.0, std::max(0.0, (tv - kWhiteTau[k0]) / (kWhiteTau[k1] - kWhiteTau[k0])));
                auto bil = [&](int k) {
                    const std::vector<float>& G = lev[k];
                    const double a = G[(size_t)y0 * gw + x0] + (G[(size_t)y0 * gw + x1] - G[(size_t)y0 * gw + x0]) * fx;
                    const double b = G[(size_t)y1 * gw + x0] + (G[(size_t)y1 * gw + x1] - G[(size_t)y1 * gw + x0]) * fx;
                    return a + (b - a) * fy;
                };
                const double F = bil(k0) + (bil(k1) - bil(k0)) * ft;
                M.E[i] *= (float)(1.0 - wpAmt * smooth01((F - 0.01) / 0.09));
                any2 = any2 || M.E[i] > 0;
            }
            rowAny2[y] = any2 ? 1 : 0;
        }, ab, user)) return false;
        bool any2 = false;
        for (int y = 0; y < H; ++y) any2 = any2 || rowAny2[y];
        if (!any2) { M.E.clear(); return true; }
    }

    LC_PERF_MARK(white);
    const double sel = clampd(p.highlightSelectivity, 0.0, 1.0);
    if (sel <= 0) return true;
    // ---- detail weight (size window)
    const bool v2size = p.renderVersion >= 2;
    const double sqps = std::sqrt(clampd(p.pixelScale, 1e-4, 256.0));
    // v2.2: fractional structuring-element radii (q^2 interpolated between the neighbouring integer
    // radii, so Detail Size / Min Detail Size animate without pops), a tapered top of the size window
    // (the top scale spread over 1.8 / 2.2 / 2.6 x, so a growing light fades out instead of switching
    // off) and a radius rule r = max(1, (m D - (0.25 + 0.75 s)) / 2), s = sqrt(pixelScale) (v2.2.1, see
    // below; test\v22checks.cpp (3) and the AE Half / Quarter check).
    auto detailMeasureV2 = [&](double D, std::vector<float>& out) {
        // v2.1 used {0.5, 0.7, 1, 1.4, 2} x 0.2: a light larger than 2 x Detail Size lost its last
        // 20 % at once (7x pop). v2.2 spreads the top scale over 1.8 / 2.2 / 2.6 (0.12 / 0.06 / 0.02):
        // the same window (same total for small lights, the v2.1 energy up to 1.8 D) with a tapered top.
        const double mult[7] = {0.5, 0.7, 1.0, 1.4, 1.8, 2.2, 2.6};
        const double wgt[7] = {0.2, 0.2, 0.2, 0.2, 0.12, 0.06, 0.02};
        out.assign(N, 0.f);
        // opening per integer radius (index = radius). Sized once: pointers into it stay valid.
        const int rTop = (int)std::floor(std::max(0.0, (mult[6] * D - sqps) * 0.5)) + 2;
        std::vector<std::vector<float>> ops((size_t)rTop + 1);
        auto opening = [&](int r) -> const std::vector<float>* {
            if (r <= 0) return &L;  // radius 0: identity, q = 0
            if (r >= (int)ops.size()) return nullptr;
            if (ops[r].empty()) {
                ops[r] = L;
                if (!filter2D<false>(ops[r], W, H, r, nth, sc, ab, user)) return nullptr;
                if (!filter2D<true>(ops[r], W, H, r, nth, sc, ab, user)) return nullptr;
            }
            return &ops[r];
        };
        for (int s = 0; s < 7; ++s) {
            if (!(wgt[s] > 0)) continue;
            // v2.2.1: >= 1 px (as v2.1): an element below 3x3 cannot tell a light from its surroundings,
            // and at Quarter the small scales fell below it (AE Quarter: brightest 1 % at 0.81 x Full).
            // Offset 0.25 + 0.75 s (s = sqrt(pixelScale); 1 at Full): measured on AE's point-sampled
            // Half / Quarter input (sample1: 0.95 / 0.90 x Full) and on box-averaged isolated lights.
            const double rf = std::max(1.0, (mult[s] * D - (0.25 + 0.75 * sqps)) * 0.5);
            const int r0 = (int)std::floor(rf);
            const float t = (float)(rf - r0);
            // radii only grow with the scale: free the openings no later scale needs (memory ~2 N)
            for (int k = 1; k < r0 && k < (int)ops.size(); ++k) std::vector<float>().swap(ops[k]);
            const std::vector<float>* o0 = opening(r0);
            const std::vector<float>* o1 = t > 0 ? opening(r0 + 1) : o0;
            if (!o0 || !o1) return false;
            const float w = (float)wgt[s];
            if (!parallelFor(H, nth, 16, [&](int y, int) {
                for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i)
                    if (M.E[i] > 0) {
                        const float il = 1.f / std::max(L[i], 1e-6f);
                        const float q0 = std::max(0.f, L[i] - (*o0)[i]) * il, q1 = std::max(0.f, L[i] - (*o1)[i]) * il;
                        out[i] += w * (q0 * q0 + t * (q1 * q1 - q0 * q0));
                    }
            }, ab, user)) return false;
        }
        return true;
    };
    auto detailMeasure = [&](double D, std::vector<float>& out) {
        if (v2size) return detailMeasureV2(D, out);
        const double mult[5] = {0.5, 0.7, 1.0, 1.4, 2.0};
        std::vector<float>& op = mean;  // reuse (adaptive-knee scratch, no longer needed)
        out.assign(N, 0.f);
        int prevR = -1;
        for (double m : mult) {
            const int r = std::max(1, (int)std::lround((m * D - 1.0) * 0.5));
            if (r != prevR) {
                prevR = r;
                op = L;
                if (!filter2D<false>(op, W, H, r, nth, sc, ab, user)) return false;
                if (!filter2D<true>(op, W, H, r, nth, sc, ab, user)) return false;
            }
            if (!parallelFor(H, nth, 16, [&](int y, int) {
                for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i)
                    if (M.E[i] > 0) {
                        // contrast ratio squared: clearly-brighter-than-surroundings details count
                        // fully, mild texture (line-art gaps, cloth shading) only a little
                        const float q = std::max(0.f, L[i] - op[i]) / std::max(L[i], 1e-6f);
                        out[i] += 0.2f * q * q;
                    }
            }, ab, user)) return false;
        }
        return true;
    };
    if (!detailMeasure(clampd(p.detailSize, 1.0, 400.0), M.D)) return false;
    const double dmin = clampd(p.detailMin, 0.0, 400.0);
    // the fade-in is defined in full-resolution px (v2.1f), so a half-res preview matches
    const double dminFull = dmin / std::sqrt(clampd(p.pixelScale, 1e-4, 256.0));
    if (dmin > 1e-3 && dminFull > 0.5) {
        std::vector<float> Dm;
        if (!detailMeasure(std::min(std::max(dmin, 0.5), clampd(p.detailSize, 1.0, 400.0)), Dm)) return false;
        // fade the minimum in over 0.5..3 px: the smallest structuring element (3x3) would otherwise
        // switch it on with a jump
        const float ramp = (float)smooth01((dminFull - 0.5) / 2.5);
        if (!parallelFor(H, nth, 16, [&](int y, int) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) M.D[i] = std::max(0.f, M.D[i] - ramp * Dm[i]);
        }, ab, user)) return false;
    }
    LC_PERF_MARK(detail);
    // ---- dots vs lines (oriented line openings, length ~2 x detailSize)
    const double dots = clampd(p.dotsOnly, 0.0, 1.0);
    if (dots > 0) {
        const double len = std::max(3.0, 2.0 * clampd(p.detailSize, 1.0, 400.0));
        std::vector<float> best(N, 0.f), op(N);
        for (int dir = 0; dir < 4; ++dir) {
            const int r = std::max(1, (int)std::lround(0.5 * len / (dir >= 2 ? 1.41421356 : 1.0)));
            std::copy(L.begin(), L.end(), op.begin());
            if (!filterLines<false>(op, W, H, r, dir, nth, sc, ab, user)) return false;
            if (!filterLines<true>(op, W, H, r, dir, nth, sc, ab, user)) return false;
            if (!parallelFor(H, nth, 16, [&](int y, int) {
                for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) best[i] = std::max(best[i], op[i]);
            }, ab, user)) return false;
        }
        if (!parallelFor(H, nth, 16, [&](int y, int) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
                if (M.D[i] <= 0) continue;
                const float q = std::max(0.f, L[i] - best[i]) / std::max(L[i], 1e-6f);
                M.D[i] *= (float)(1.0 - dots * (1.0 - (double)q * q));
            }
        }, ab, user)) return false;
    }
    LC_PERF_MARK(dots);
    LC_PERF_DUMP("highlights");
    return true;
}

// ------------------------------------------------------------ pass 1 --------

// Multi-resolution: level l (>= 1) works on a grid 2^l times coarser and takes splats whose radius
// is ~[34, 68) coarse px; the fine level takes R < ~55..68 px. A splat's share moves to the next
// level over the top 30% of each octave (smooth in R), so the result is continuous in the radius.
inline void levelOf(const Ctx& C, double R, int& l0, double& w1) {
    l0 = 0;
    w1 = 0;
    if (C.nLevels <= 1 || R <= C.levelBase) return;
    const double lam = std::log2(R / C.levelBase);
    const int l = (int)lam;
    if (l >= C.nLevels - 1) { l0 = C.nLevels - 1; return; }
    double w = std::min(1.0, std::max(0.0, (lam - l - 0.7) / 0.3));
    l0 = l;
    w1 = l + 1 < C.nLevels ? w * w * (3 - 2 * w) : 0.0;
}

// Radius / tier / reach / cat-eye data of one splat. R is in the splat's own level px; (px, py)
// is its centre in fine buffer px (cat-eye is a function of the frame position).
void finishSplat(const Ctx& C, Splat& sp, CatInfo* ci, double R, double sgn, double px, double py, int maxReach) {
    const double* ks = C.ks[sgn > 0 ? 1 : 0];
    double smax = ks[0];
    for (int ch = 1; ch < C.nk; ++ch) smax = std::max(smax, ks[ch]);
    sp.tier = 0;
    sp.wb = 0.f;
    const double Rxe = R * C.ax * smax * C.xext, Rye = R * C.ay * smax * C.yext;
    if (Rxe <= 0.5 && Rye <= 0.5) {
        sp.R = -1.f;
        sp.reach = 0;
    } else {
        sp.R = (float)R;
        sp.reach = std::min(maxReach, std::max(0, (int)std::ceil(Rye - 0.5)));
        if (C.nTiers > 1) {
            const double tau = std::log2(std::max(R, kTierBase) / kTierBase);
            int t = (int)tau;
            double f = tau - t;
            if (t >= C.nTiers - 1) { t = C.nTiers - 1; f = 0; }
            double wb = std::min(1.0, std::max(0.0, (f - 0.7) / 0.3));
            wb = wb * wb * (3 - 2 * wb);
            sp.tier = t;
            sp.wb = (float)wb;
        }
    }
    if (!ci) return;
    *ci = CatInfo{};
    const double dxp = px - C.fcx, dyp = py - C.fcy;
    const double len = std::sqrt(dxp * dxp + dyp * dyp);
    const double lenSq = C.pa == 1.0 ? len : std::sqrt(dxp * dxp * C.pa * C.pa + dyp * dyp);  // square px
    if (len > 1e-9) {
        // radial direction mapped into unit (pre-anamorphic) space
        double dx = dxp / len / C.ax, dy = dyp / len / C.ay;
        const double dl = std::sqrt(dx * dx + dy * dy);
        const double m = 0.5 * C.catK * std::min(1.5, lenSq / C.halfDiag);  // half offset
        ci->ox = (float)(m * dx / dl);
        ci->oy = (float)(m * dy / dl);
    }
    if (sp.R > 0 && C.catN > 0) {
        const double A = C.catArea(ci->ox, ci->oy);
        bool any = false;
        for (int pass = 0; pass < 2; ++pass) {
            const double wgt = pass ? sp.wb : 1.0 - sp.wb;
            if (pass && wgt <= 0) break;
            const Profile& P = C.prof[sgn > 0 ? 1 : 0][sp.tier + pass];
            for (int ch = 0; ch < C.nk; ++ch) {
                const double sum = P.share ? A * (sp.R * C.ax) * (sp.R * C.ay) * (P.s2out * ks[ch] * ks[ch] + P.s2in)
                                           : A * P.s2 * (sp.R * C.ax * ks[ch]) * (sp.R * C.ay * ks[ch]);
                ci->invD[pass][ch] = sum > 1e-20 ? (float)(wgt / sum) : 0.f;
                any = any || sum > 1e-20;
            }
        }
        if (!any) { sp.R = -1.f; sp.reach = 0; }
    } else if (sp.R > 0) {
        bool any = false;
        for (int pass = 0; pass < 2; ++pass) {
            const double wgt = pass ? sp.wb : 1.0 - sp.wb;
            if (pass && wgt <= 0) break;
            const Profile& P = C.prof[sgn > 0 ? 1 : 0][sp.tier + pass];
            for (int ch = 0; ch < C.nk; ++ch) {
                const double Rx = sp.R * C.ax * ks[ch], Ry = sp.R * C.ay * ks[ch];
                double sum = 0;
                for (int k = -sp.reach; k <= sp.reach; ++k) {
                    for (int l = 0; l < P.nl; ++l) {
                        double h, xl, xr;
                        if (catSpan(C, Rx * P.ls[l], Ry * P.ls[l], k - 0.5, k + 0.5, ci->ox, ci->oy, h, xl, xr))
                            sum += P.lb[l] * h * (xr - xl);
                    }
                }
                ci->invD[pass][ch] = sum > 1e-20 ? (float)(wgt / sum) : 0.f;
                any = any || sum > 1e-20;
            }
        }
        if (!any) { sp.R = -1.f; sp.reach = 0; }
    }
}

// Stable counting sort of one row by reach, descending (pass 2 stops early per source row).
// Skipped splats (reach -1, all of their energy went to coarser levels) go last.
void sortRow(int W, const std::vector<Splat>& tmpS, const std::vector<CatInfo>& tmpC, bool cat, int maxR,
             std::vector<int>& cnt, Splat* outS, CatInfo* outC) {
    cnt.assign(maxR + 3, 0);
    for (int x = 0; x < W; ++x) cnt[maxR - tmpS[x].reach + 1]++;
    for (int i = 1; i <= maxR + 2; ++i) cnt[i] += cnt[i - 1];
    for (int x = 0; x < W; ++x) {
        const int pos = cnt[maxR - tmpS[x].reach]++;
        outS[pos] = tmpS[x];
        if (cat) outC[pos] = tmpC[x];
    }
}

// Stage 1, one row: working colour + radius + side + added-energy luminance.
void prepareRow(const Params& p, const HlMaps& M, const float* src, int ss, const float* map, int ms, int sy,
                int W, bool keepExtra, Prepared& P) {
    const double gain = p.highlightGain, thr = p.highlightThreshold;
    const bool hl = !M.E.empty();
    const bool hasD = !M.D.empty();
    const double boost = p.highlightBoost;
    const double sel = clampd(p.highlightSelectivity, 0.0, 1.0);
    const double hSat = 1.0 + 2.0 * clampd(p.highlightSaturation, 0.0, 1.0);
    const double ps = clampd(p.pixelScale, 1e-4, 256.0);
    for (int x = 0; x < W; ++x) {
        const size_t i = (size_t)sy * W + x;
        float sv[4];  // sanitised source pixel (non-finite -> 0, magnitudes clamped)
        sanitize4(src + (size_t)sy * ss + (size_t)x * 4, sv);
        const float* s = sv;
        double r = s[0], g = s[1], b = s[2];
        double amt = 1.0, sgn = 1.0;
        if (map) amt = amountOf(sanitize1(map[(size_t)sy * ms + x]), p, sgn);
        double eL = 0;  // luminance of the energy added by highlight expansion (for the rolloff)
        // Highlight boost fades in with the blur amount, so in-focus pixels are
        // left untouched and only out-of-focus highlights bloom into bokeh.
        if (gain > 0 && amt > 0) {
            const double L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            if (L > thr && L > 0) {
                const double f = 1.0 + gain * amt * (L - thr) / L;
                eL += (f - 1.0) * L;
                r *= f; g *= f; b *= f;
            }
        }
        const double R = amt * p.maxRadius;
        float* xv = keepExtra ? P.extra.data() + i * 3 : nullptr;
        if (xv) xv[0] = xv[1] = xv[2] = 0.f;
        if (hl && amt > 0) {
            const double E = M.E[i];
            if (E > 0) {
                // disc area in FULL-RESOLUTION pixels (a downsampled buffer has 1/pixelScale fewer
                // source pixels per detail; this keeps the disc brightness resolution-independent)
                const double area = kPi * std::max(0.0, R * R / ps - 1.0) / 16.0;
                const double dw = hasD ? (double)M.D[i] : 0.0;
                const double k = std::min(1e6, boost * E * ((1.0 - sel) * amt + (hasD ? sel * dw * area : 0.0)));
                if (k > 0) {
                    double er = k * s[0], eg = k * s[1], eb = k * s[2];
                    const double Le = 0.2126 * er + 0.7152 * eg + 0.0722 * eb;
                    if (hSat != 1.0) {
                        er = std::max(0.0, Le + (er - Le) * hSat);
                        eg = std::max(0.0, Le + (eg - Le) * hSat);
                        eb = std::max(0.0, Le + (eb - Le) * hSat);
                        const double Ln = 0.2126 * er + 0.7152 * eg + 0.0722 * eb;
                        if (Ln > 1e-12) { const double q = Le / Ln; er *= q; eg *= q; eb *= q; }
                    }
                    r += er; g += eg; b += eb;
                    eL += Le;
                    if (xv) {
                        // view units: the same energy with the disc-area factor replaced by 1
                        // (~ brightness of the disc this source makes per 4x4 px of it)
                        const double kv = boost * E * ((1.0 - sel) * amt + (hasD ? sel * dw * (R * R > ps ? 1.0 : 0.0) : 0.0));
                        const double q = 4.0 * kv / k;  // x4: faint sources stay visible in the view
                        xv[0] = (float)(er * q); xv[1] = (float)(eg * q); xv[2] = (float)(eb * q);
                    }
                }
            }
        }
        float* o = P.rgba.data() + i * 4;
        o[0] = (float)r; o[1] = (float)g; o[2] = (float)b; o[3] = s[3];
        P.radius[i] = R;
        P.side[i] = sgn > 0 ? 1 : -1;
        P.eL[i] = (float)eL;
    }
}

// Stage 2, one row: splats from the prepared data.
void prepRow(Ctx& C, const Prepared& P, int sy, std::vector<Splat>& tmpS, std::vector<CatInfo>& tmpC, std::vector<int>& cnt) {
    const int W = C.W;
    tmpS.resize(W);
    if (C.cat) tmpC.resize(W);
    int maxR = 0;
    for (int x = 0; x < W; ++x) {
        const size_t i = (size_t)sy * W + x;
        const float* c = P.rgba.data() + i * 4;
        const double R = P.radius[i];
        const double sgn = P.side[i] > 0 ? 1.0 : -1.0;
        Splat sp;
        sp.c[0] = c[0]; sp.c[1] = c[1]; sp.c[2] = c[2]; sp.c[3] = c[3];
        sp.eL = P.eL[i];
        sp.x = x;
        sp.sgn = (float)sgn;
        sp.m = 1.f;
        double fineShare = 1.0;
        if (C.nLevels > 1) {
            Ctx::Px& q = C.pix[i];
            q.c[0] = sp.c[0]; q.c[1] = sp.c[1]; q.c[2] = sp.c[2]; q.c[3] = sp.c[3];
            q.eL = sp.eL; q.R = (float)R; q.sgn = (float)sgn;
            int l0; double w1;
            levelOf(C, R, l0, w1);
            fineShare = l0 == 0 ? 1.0 - w1 : 0.0;
        }
        CatInfo* ci = C.cat ? &tmpC[x] : nullptr;
        if (fineShare <= 0) {  // everything goes to coarser levels
            sp.R = 0.f; sp.reach = -1; sp.tier = 0; sp.wb = 0.f; sp.m = 0.f;
            if (ci) *ci = CatInfo{};
        } else {
            sp.m = (float)fineShare;
            finishSplat(C, sp, ci, R, sgn, x + 0.5, sy + 0.5, C.maxReach);
        }
        tmpS[x] = sp;
        maxR = std::max(maxR, sp.reach);
    }
    sortRow(W, tmpS, tmpC, C.cat, maxR, cnt, C.splats.data() + (size_t)sy * W,
            C.cat ? C.cats.data() + (size_t)sy * W : nullptr);
}

// Coarse level l: merge each f x f block's share for this level into one splat (energy-weighted
// colour, radius and side), placed at the block centre. Blocks are aligned to the frame origin.
void prepLevelRow(Ctx& C, int l, int by, std::vector<Splat>& tmpS, std::vector<CatInfo>& tmpC, std::vector<int>& cnt) {
    Ctx::Level& Lv = C.levels[l];
    const int f = Lv.f, W = Lv.W;
    tmpS.resize(W);
    if (C.cat) tmpC.resize(W);
    int maxR = 0;
    const int y0 = std::max(0, by * f - Lv.goy), y1 = std::min(C.H, (by + 1) * f - Lv.goy);
    // v2.3: radii that certainly have no share on this level skip levelOf's log2 (the weight below
    // is exactly 0 for them anyway). Level l takes R in (base 2^(l - 0.3), base 2^(l + 1)) (the top
    // level everything above); the 1 % margins keep every borderline radius on the exact path.
    const double rLo = 0.99 * C.levelBase * std::pow(2.0, l - 0.3);
    const double rHi = l < C.nLevels - 1 ? 1.01 * C.levelBase * std::pow(2.0, l + 1) : 1e300;
    for (int bx = 0; bx < W; ++bx) {
        const int x0 = std::max(0, bx * f - Lv.gox), x1 = std::min(C.W, (bx + 1) * f - Lv.gox);
        double m = 0, c[4] = {0, 0, 0, 0}, eL = 0, Rs = 0, sg = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const Ctx::Px& q = C.pix[(size_t)y * C.W + x];
                if (!(q.R > rLo && q.R < rHi)) continue;
                int l0; double w1;
                levelOf(C, q.R, l0, w1);
                const double w = (l0 == l ? 1.0 - w1 : 0.0) + (l0 + 1 == l ? w1 : 0.0);
                if (w <= 0) continue;
                m += w;
                for (int k = 0; k < 4; ++k) c[k] += w * q.c[k];
                eL += w * q.eL; Rs += w * q.R; sg += w * q.sgn;
            }
        Splat sp;
        sp.x = bx;
        CatInfo* ci = C.cat ? &tmpC[bx] : nullptr;
        if (m < 1e-9) {
            sp.c[0] = sp.c[1] = sp.c[2] = sp.c[3] = 0.f; sp.eL = 0.f; sp.sgn = 1.f;
            sp.R = 0.f; sp.reach = -1; sp.tier = 0; sp.wb = 0.f; sp.m = 0.f;
            if (ci) *ci = CatInfo{};
        } else {
            const double inv = 1.0 / m;
            for (int k = 0; k < 4; ++k) sp.c[k] = (float)(c[k] * inv);
            sp.eL = (float)(eL * inv);
            sp.sgn = sg >= 0 ? 1.f : -1.f;
            sp.m = (float)m;
            const double cx = (bx + 0.5) * f - Lv.gox, cy = (by + 0.5) * f - Lv.goy;
            finishSplat(C, sp, ci, Rs * inv / f, sp.sgn, cx, cy, Lv.maxReach);
        }
        tmpS[bx] = sp;
        maxR = std::max(maxR, sp.reach);
    }
    sortRow(W, tmpS, tmpC, C.cat, maxR, cnt, Lv.splats.data() + (size_t)by * W,
            C.cat ? Lv.cats.data() + (size_t)by * W : nullptr);
}

// ------------------------------------------------------------ post stage ----

struct PostCtx {
    int W = 0, H = 0;
    double fcx = 0, fcy = 0, halfDiag = 1;
    double pa = 1;  // buffer pixel aspect (width / height), v2.2
    // glow
    double glowAmt = 0;
    int gf = 1, gw = 0, gh = 0;
    int gox = 0, goy = 0;          // grid phase: cells are aligned to the frame origin (tile-exact)
    std::vector<float> glowGrid;   // gw*gh*6 (rgba, eL, weight), blurred
    double vig = 0;
    bool roll = false;
    double rollKnee = 1.0;
    int rollV2 = 0;   // v2.2: rolloff compresses only the added energy on HDR sources
};

const int kGC = 6;  // glow grid channels: r g b a eL weight

bool buildGlow(PostCtx& C, const Prepared& P, const Params& p, int nth, AbortFn ab, void* user) {
    const int W = C.W, H = C.H;
    const double sigma = std::min(0.25, std::max(0.005, (double)p.glowSize)) * 2.0 * C.halfDiag;
    const int f = std::max(1, (int)(sigma / 3.0));
    C.gox = gridPhase(p, true, f);
    C.goy = gridPhase(p, false, f);
    const int gw = (W + C.gox + f - 1) / f, gh = (H + C.goy + f - 1) / f;
    C.gf = f; C.gw = gw; C.gh = gh;
    std::vector<float>& G = C.glowGrid;
    G.assign((size_t)gw * gh * kGC, 0.f);
    for (int y = 0; y < H; ++y) {
        float* g = G.data() + (size_t)((y + C.goy) / f) * gw * kGC;
        for (int x = 0; x < W; ++x) {
            float* q = g + (size_t)((x + C.gox) / f) * kGC;
            const size_t i = (size_t)y * W + x;
            const float* sx = P.rgba.data() + i * 4;
            q[0] += sx[0]; q[1] += sx[1]; q[2] += sx[2]; q[3] += sx[3]; q[4] += P.eL[i]; q[5] += 1.f;
        }
    }
    const double sd = sigma / f;
    const double wbox = std::sqrt(4.0 * sd * sd + 1.0);
    const int rb = std::max(1, (int)std::lround((wbox - 1) * 0.5));
    // 3 box passes per axis, zero padding (the weight channel normalises edges)
    auto boxLine = [rb](float* d, int n, int stride, std::vector<double>& tmp) {
        tmp.assign((size_t)(n + 1) * kGC, 0.0);
        for (int pass = 0; pass < 3; ++pass) {
            for (int c = 0; c < kGC; ++c) tmp[c] = 0;
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < kGC; ++c)
                    tmp[(size_t)(i + 1) * kGC + c] = tmp[(size_t)i * kGC + c] + d[(size_t)i * stride + c];
            for (int i = 0; i < n; ++i) {
                const int a = std::max(0, i - rb), b = std::min(n, i + rb + 1);
                for (int c = 0; c < kGC; ++c)
                    d[(size_t)i * stride + c] = (float)((tmp[(size_t)b * kGC + c] - tmp[(size_t)a * kGC + c]) / (2 * rb + 1));
            }
        }
    };
    std::vector<std::vector<double>> tmps(nth);
    bool ok = parallelFor(gh, nth, 4, [&](int y, int tid) {
        boxLine(G.data() + (size_t)y * gw * kGC, gw, kGC, tmps[tid]);
    }, ab, user);
    if (!ok) return false;
    return parallelFor(gw, nth, 4, [&](int x, int tid) {
        boxLine(G.data() + (size_t)x * kGC, gh, gw * kGC, tmps[tid]);
    }, ab, user);
}

// out: r g b a eL (weight-normalised)
inline void sampleGlow(const PostCtx& C, double x, double y, double out[5]) {
    double u = (x + C.gox + 0.5) / C.gf - 0.5, v = (y + C.goy + 0.5) / C.gf - 0.5;
    u = std::min((double)C.gw - 1, std::max(0.0, u));
    v = std::min((double)C.gh - 1, std::max(0.0, v));
    const int x0 = (int)u, y0 = (int)v;
    const int x1 = std::min(x0 + 1, C.gw - 1), y1 = std::min(y0 + 1, C.gh - 1);
    const double fx = u - x0, fy = v - y0;
    const float* a = C.glowGrid.data() + ((size_t)y0 * C.gw + x0) * kGC;
    const float* b = C.glowGrid.data() + ((size_t)y0 * C.gw + x1) * kGC;
    const float* c = C.glowGrid.data() + ((size_t)y1 * C.gw + x0) * kGC;
    const float* d = C.glowGrid.data() + ((size_t)y1 * C.gw + x1) * kGC;
    double vv[kGC];
    for (int k = 0; k < kGC; ++k) {
        const double t = a[k] + (b[k] - a[k]) * fx;
        const double s = c[k] + (d[k] - c[k]) * fx;
        vv[k] = t + (s - t) * fy;
    }
    const double inv = vv[5] > 1e-9 ? 1.0 / vv[5] : 0.0;
    for (int k = 0; k < 5; ++k) out[k] = vv[k] * inv;
}

// Output highlight shoulder for SDR delivery (see v1.2 notes): the whole pixel goes through a
// hue-preserving shoulder (knee*a .. a), blended in by how much of it is added highlight energy,
// then desaturated toward white in proportion to the compression. Identity where eL == 0.
inline void rolloffPixelV1(float* o, double eL, double knee);
// v2.2 (renderVersion >= 2): an SDR source (estimated source peak (1 - phi) M <= alpha) keeps the
// v2.1 shoulder exactly. On an HDR source (sky, window, emitters in a 32 bpc comp) only the ADDED
// part phi*o is compressed: the shoulder starts at max(knee*a, source peak) with the same span, so
// the source's own HDR values pass through. Blended over source peak a .. 1.5a (no step).
inline void rolloffPixel(float* o, double eL, double knee, bool v2) {
    const double a = o[3];
    if (!(eL > 0) || !(a > 1e-6)) return;
    const double M = std::max((double)o[0], std::max((double)o[1], (double)o[2]));
    const double Lt = 0.2126 * o[0] + 0.7152 * o[1] + 0.0722 * o[2];
    if (!v2 || !(Lt > 1e-12) || !(M > 0)) { rolloffPixelV1(o, eL, knee); return; }
    const double phi = std::min(1.0, eL / Lt);
    const double Ms = (1.0 - phi) * M;   // source peak estimate
    const double t = smooth01((Ms - a) / (0.5 * a));
    if (t <= 0) { rolloffPixelV1(o, eL, knee); return; }
    float v1[3] = {o[0], o[1], o[2]};
    if (t < 1) { float q[4] = {o[0], o[1], o[2], o[3]}; rolloffPixelV1(q, eL, knee); v1[0] = q[0]; v1[1] = q[1]; v1[2] = q[2]; }
    const double span = a * (1.0 - knee);
    const double kA = std::max(knee * a, Ms);
    double r = 1.0;   // scale of the added part
    if (M > kA && span > 0 && M - Ms > 1e-12) {
        const double Mt = kA + span * (1.0 - std::exp(-(M - kA) / span));
        r = std::max(0.0, (Mt - Ms) / (M - Ms));
    }
    for (int c = 0; c < 3; ++c) {
        const double v2c = o[c] - (1.0 - r) * phi * o[c];
        o[c] = (float)(t >= 1 ? v2c : v1[c] + t * (v2c - v1[c]));
    }
}
inline void rolloffPixelV1(float* o, double eL, double knee) {
    const double a = o[3];
    if (!(eL > 0) || !(a > 1e-6)) return;
    const double M = std::max((double)o[0], std::max((double)o[1], (double)o[2]));
    const double kA = knee * a;
    if (M <= kA) return;
    const double span = a - kA;
    const double Mt = kA + span * (1.0 - std::exp(-(M - kA) / span));
    const double Lt = 0.2126 * o[0] + 0.7152 * o[1] + 0.0722 * o[2];
    if (!(Lt > 1e-12)) return;
    // phi = fraction of the pixel that is added highlight energy. The whole pixel goes through the
    // shoulder, blended in by g(phi): robust to the approximate phi (eL rides on the G kernel only)
    // and still the identity where no energy was added.
    const double phi = std::min(1.0, eL / Lt);
    double g = std::min(1.0, phi / 0.2);
    g = g * g * (3.0 - 2.0 * g);
    if (g <= 0) return;
    const double f = 1.0 - g + g * (Mt / M);
    const double Mf = M * f;
    const double comp = std::max(0.0, 1.0 - Mt / M);
    const double dsat = 0.85 * g * comp * comp;
    for (int c = 0; c < 3; ++c) {
        const double v = o[c] * f;
        o[c] = (float)(v + dsat * (Mf - v));
    }
}

// glow / vignetting on one finished output row (le: added-energy luminance, may be null)
void postRowA(const PostCtx& C, int y, float* out, float* le) {
    const int W = C.W;
    if (C.glowAmt > 0) {
        const double g = C.glowAmt;
        double v[5];
        for (int x = 0; x < W; ++x) {
            sampleGlow(C, x, y, v);
            float* o = out + (size_t)x * 4;
            for (int k = 0; k < 4; ++k) o[k] = (float)((1 - g) * o[k] + g * v[k]);
            if (C.roll) le[x] = (float)((1 - g) * le[x] + g * v[4]);
        }
    }
    if (C.vig > 0) {
        const double dy = (y + 0.5 - C.fcy) / C.halfDiag;
        for (int x = 0; x < W; ++x) {
            const double dx = (x + 0.5 - C.fcx) * C.pa / C.halfDiag;
            const double rn2 = std::min(1.0, dx * dx + dy * dy);
            const float f = (float)(1.0 - C.vig * rn2);
            float* o = out + (size_t)x * 4;
            o[0] *= f; o[1] *= f; o[2] *= f;
            if (C.roll) le[x] *= f;
        }
    }
}

// Gaussian-like blur of a scalar field (3 box passes). sigma <= 6 px: full resolution; larger:
// downsampled grid aligned to the frame origin (tile-exact), bilinear upsampling. Weight-normalised
// at the buffer edges (zero padding + coverage channel).
bool gaussField(const std::vector<float>& in, int W, int H, double sigma, std::vector<float>& out,
                int nth, AbortFn ab, void* user);
bool blurFieldBox(const std::vector<float>& in, int W, int H, double sigma, const Params& p,
                  std::vector<float>& out, int nth, AbortFn ab, void* user, int rbForce);

// v2.2 (renderVersion >= 2): no staircase when a width is animated. sigma <= 6: the exact Gaussian;
// above: the 3-box blur interpolated between the two neighbouring box radii by the variance.
bool blurField(const std::vector<float>& in, int W, int H, double sigma, const Params& p,
               std::vector<float>& out, int nth, AbortFn ab, void* user) {
    if (p.renderVersion < 2) return blurFieldBox(in, W, H, sigma, p, out, nth, ab, user, -1);
    if (sigma <= 6.0) return gaussField(in, W, H, std::max(0.3, sigma), out, nth, ab, user);
    const int f = std::max(1, (int)(sigma / 3.0));
    const double sd = sigma / f;
    const int r0 = std::max(1, (int)std::floor((std::sqrt(4.0 * sd * sd + 1.0) - 1) * 0.5));
    const double v0 = r0 * (r0 + 1.0), v1 = (r0 + 1.0) * (r0 + 2.0);
    const double t = clampd((sd * sd - v0) / (v1 - v0), 0.0, 1.0);
    if (!blurFieldBox(in, W, H, sigma, p, out, nth, ab, user, r0)) return false;
    if (t <= 1e-6) return true;
    std::vector<float> o1;
    if (!blurFieldBox(in, W, H, sigma, p, o1, nth, ab, user, r0 + 1)) return false;
    return parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) out[i] += (float)t * (o1[i] - out[i]);
    });
}

bool blurFieldBox(const std::vector<float>& in, int W, int H, double sigma, const Params& p,
                  std::vector<float>& out, int nth, AbortFn ab, void* user, int rbForce) {
    const int f = sigma > 6.0 ? std::max(1, (int)(sigma / 3.0)) : 1;
    const int gx = gridPhase(p, true, f), gy = gridPhase(p, false, f);
    const int gw = (W + gx + f - 1) / f, gh = (H + gy + f - 1) / f;
    std::vector<float> G((size_t)gw * gh * 2, 0.f);
    // v2.3: one task per coarse row; each cell still sums its pixels in the original order (y, then x)
    if (!parRows(gh, nth, ab, user, [&](int gyr) {
        const int ya = std::max(0, gyr * f - gy), yb = std::min(H, (gyr + 1) * f - gy);
        for (int y = ya; y < yb; ++y)
            for (int x = 0; x < W; ++x) {
                float* q = G.data() + ((size_t)((y + gy) / f) * gw + (x + gx) / f) * 2;
                q[0] += in[(size_t)y * W + x];
                q[1] += 1.f;
            }
    })) return false;
    const double sd = sigma / f;
    const int rb = rbForce > 0 ? rbForce : std::max(1, (int)std::lround((std::sqrt(4.0 * sd * sd + 1.0) - 1) * 0.5));
    auto boxLine = [rb](float* d, int n, int stride, std::vector<double>& tmp) {
        tmp.assign((size_t)(n + 1) * 2, 0.0);
        for (int pass = 0; pass < 3; ++pass) {
            tmp[0] = tmp[1] = 0;
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 2; ++c) tmp[(size_t)(i + 1) * 2 + c] = tmp[(size_t)i * 2 + c] + d[(size_t)i * stride + c];
            for (int i = 0; i < n; ++i) {
                const int a = std::max(0, i - rb), b = std::min(n, i + rb + 1);
                for (int c = 0; c < 2; ++c)
                    d[(size_t)i * stride + c] = (float)((tmp[(size_t)b * 2 + c] - tmp[(size_t)a * 2 + c]) / (2 * rb + 1));
            }
        }
    };
    std::vector<std::vector<double>> tmps(nth);
    bool ok = parallelFor(gh, nth, 4, [&](int y, int tid) { boxLine(G.data() + (size_t)y * gw * 2, gw, 2, tmps[tid]); }, ab, user);
    if (!ok) return false;
    ok = parallelFor(gw, nth, 4, [&](int x, int tid) { boxLine(G.data() + (size_t)x * 2, gh, gw * 2, tmps[tid]); }, ab, user);
    if (!ok) return false;
    out.assign((size_t)W * H, 0.f);
    return parallelFor(H, nth, 8, [&](int y, int) {
        double v = (y + gy + 0.5) / f - 0.5;
        v = std::min((double)gh - 1, std::max(0.0, v));
        const int y0 = (int)v, y1 = std::min(y0 + 1, gh - 1);
        const double fy = v - y0;
        for (int x = 0; x < W; ++x) {
            double u = (x + gx + 0.5) / f - 0.5;
            u = std::min((double)gw - 1, std::max(0.0, u));
            const int x0 = (int)u, x1 = std::min(x0 + 1, gw - 1);
            const double fx = u - x0;
            double s[2];
            for (int c = 0; c < 2; ++c) {
                const double a = G[((size_t)y0 * gw + x0) * 2 + c] + (G[((size_t)y0 * gw + x1) * 2 + c] - G[((size_t)y0 * gw + x0) * 2 + c]) * fx;
                const double b = G[((size_t)y1 * gw + x0) * 2 + c] + (G[((size_t)y1 * gw + x1) * 2 + c] - G[((size_t)y1 * gw + x0) * 2 + c]) * fx;
                s[c] = a + (b - a) * fy;
            }
            out[(size_t)y * W + x] = s[1] > 1e-9 ? (float)(s[0] / s[1]) : 0.f;
        }
    }, ab, user);
}

inline double lumOf(const float* o) { return 0.2126 * o[0] + 0.7152 * o[1] + 0.0722 * o[2]; }

// Film halation: light that passed the emulsion reflects off the film base and re-exposes the
// red(-sensitive) layer around bright areas: a tight reddish rim right at the highlight edge plus a
// wider soft orange glow. Source = the part of the image above a soft luminance threshold (HDR
// values give proportionally more halation). Added, mostly outside the source itself.
// v2.1: the source threshold is relative to the local scene level (wide mean of the luminance,
// sigma kHalScene of the diagonal), floored by halationThreshold: a white shirt in a high-key frame
// is not a light source (no overall cast), a street light at night is. The source is capped at 16x
// its threshold (energy bound for extreme HDR values).
const double kHalScene = 0.05;
bool applyHalation(const PostCtx& C, const Params& p, float* rgba, int stride, int nth, AbortFn ab, void* user) {
    const int W = C.W, H = C.H;
    const double amt = clampd(p.halation, 0.0, 4.0);
    const double floorT = std::max(1e-3, (double)p.halationThreshold);
    const double relK = std::pow(2.0, clampd(p.halationRelative, -4.0, 8.0));
    const double soft = clampd(p.halationSoftness, 0.0, 1.0);
    std::vector<float> h((size_t)W * H), m((size_t)W * H), sc;
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (int x = 0; x < W; ++x)
            h[(size_t)y * W + x] = (float)std::max(0.0, lumOf(rgba + (size_t)y * stride + (size_t)x * 4));
    })) return false;
    if (!blurField(h, W, H, std::max(1.0, kHalScene * 2.0 * C.halfDiag), p, sc, nth, ab, user)) return false;
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
            const double Y = h[i];
            const double thr = std::max(floorT, relK * sc[i]);
            const double t0 = thr * (1.0 - 0.6 * soft), t1 = thr * (1.0 + 0.6 * soft) + 1e-4;
            const double mk = smooth01((Y - t0) / (t1 - t0));
            m[i] = (float)mk;
            h[i] = (float)(std::min(Y, 16.0 * thr) * mk);
        }
    })) return false;
    const double sw = std::max(1.0, clampd(p.halationSize, 0.001, 0.2) * 2.0 * C.halfDiag);
    const double st = std::max(0.8, 0.12 * sw);
    const double rim = clampd(p.halationRim, 0.0, 1.0);
    std::vector<float> bt, bw;
    if (rim > 0 && !blurField(h, W, H, st, p, bt, nth, ab, user)) return false;
    if (rim < 1 && !blurField(h, W, H, sw, p, bw, nth, ab, user)) return false;
    const double tr = tintC(p.halationTint[0]), tg = tintC(p.halationTint[1]), tb = tintC(p.halationTint[2]);
    return parallelFor(H, nth, 8, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            float* o = rgba + (size_t)y * stride + (size_t)x * 4;
            double v = 0;
            if (rim > 0) v += rim * 1.6 * bt[i];   // the tight rim is visibly brighter, as on film
            if (rim < 1) v += (1.0 - rim) * bw[i];
            v *= amt * (1.0 - 0.9 * m[i]) * std::max(0.f, o[3]);  // shows on the darker side
            o[0] += (float)(tr * v); o[1] += (float)(tg * v); o[2] += (float)(tb * v);
        }
    }, ab, user);
}

// "Luminance aberration" (the user's term): at bright/dark boundaries warm light bleeds across the
// edge; the larger the luminance difference (in stops), the stronger the bleed and the deeper its
// hue (yellow-orange -> laTint). Mode 0 = bleed into the dark side (additive, energy ~ the
// brightness difference), 1 = tint the bright side (hue shift toward warm at constant luminance),
// 2 = both. Flat areas and small (noise) differences give exactly nothing (threshold in stops).
// Exact separable Gaussian (radius ceil(3 sigma)) of a scalar field for small sigmas, normalised by
// the in-buffer weights. Used where blurField's 3-box approximation is too coarse (sub-2px rims).
bool gaussField(const std::vector<float>& in, int W, int H, double sigma, std::vector<float>& out,
                int nth, AbortFn ab, void* user) {
    const int r = std::max(1, (int)std::ceil(3.0 * sigma));
    std::vector<double> k((size_t)2 * r + 1);
    for (int i = -r; i <= r; ++i) k[(size_t)(i + r)] = std::exp(-0.5 * i * i / (sigma * sigma));
    std::vector<float> t((size_t)W * H);
    out.assign((size_t)W * H, 0.f);
    bool ok = parallelFor(H, nth, 8, [&](int y, int) {
        const float* a = in.data() + (size_t)y * W;
        for (int x = 0; x < W; ++x) {
            double s = 0, n = 0;
            for (int i = std::max(-r, -x); i <= std::min(r, W - 1 - x); ++i) { s += k[(size_t)(i + r)] * a[x + i]; n += k[(size_t)(i + r)]; }
            t[(size_t)y * W + x] = (float)(s / n);
        }
    }, ab, user);
    if (!ok) return false;
    return parallelFor(H, nth, 8, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            double s = 0, n = 0;
            for (int i = std::max(-r, -y); i <= std::min(r, H - 1 - y); ++i) {
                s += k[(size_t)(i + r)] * t[(size_t)(y + i) * W + x];
                n += k[(size_t)(i + r)];
            }
            out[(size_t)y * W + x] = (float)(s / n);
        }
    }, ab, user);
}

// Geometry of the contrast rim, shared by the effect and postMargin (px).
struct RimGeom { int rl, rc, rn; double sReg, sd, sb, sf; };
inline RimGeom rimGeom(const Params& p, double halfDiag) {
    const double diag = 2.0 * halfDiag;
    RimGeom g;
    g.rl = (int)std::lround(clampd(p.laLineProtect, 0.0, 0.02) * diag);
    g.rc = std::max(1, (int)std::lround(clampd(p.laContrastScale, 0.0005, 0.05) * diag));
    g.sReg = std::max(0.7, 0.5 * g.rc);
    const double sharp = clampd(p.laSharpness, 0.0, 1.0);
    g.sd = std::max(0.35, clampd(p.laWidth, 0.0001, 0.1) * diag);
    g.sb = g.sd * (1.0 - sharp) * 0.8;
    g.rn = (int)std::ceil(2.5 * g.sd) + 1;
    // fringe mode (laBlur > 0): red-channel defocus sigma
    g.sf = std::max(0.5, (clampd(p.laWidth, 0.0001, 0.1) + clampd(p.laBlur, 0.0, 0.05)) * diag);
    return g;
}

// laMode 3, "contrast rim" (the user's reference look): a crisp, saturated line exactly on the
// boundary between a bright region and a much darker one, replacing the line art there.
//  1. Line protection: grey closing (max then min, square radius rl) of the luminance. Dark lines
//     narrower than ~2 rl with bright on BOTH sides (eyelashes, mouth) are filled and so are not
//     a dark region; a line between bright and dark keeps the dark level (it becomes rim).
//  2. Region levels: max (B) and min (D) of the closed luminance over radius rc, smoothed; the
//     contrast in stops c = log2(B/D) drives strength (threshold/curve), colour and brightness.
//  3. Membership b = closed luminance above the geometric mid level sqrt(B*D) (soft, +-0.5 stop).
//  4. Rim weight: dark-side pixels near bright ones (Gaussian sd = rim width, plateau then a
//     falloff whose steepness is set by Sharpness) + optionally bright-side pixels (sb).
//  5. The pixel is blended toward the rim colour (laTint for high contrast, red-brown for lower;
//     optionally a warm-shifted bright-side colour) at a luminance tied to the bright level.
// Gaussian of a field: exact for small sigmas, the frame-aligned 3-box approximation above that.
bool gaussOrBox(const std::vector<float>& in, int W, int H, double sigma, const Params& p, std::vector<float>& out,
                int nth, AbortFn ab, void* user) {
    return sigma <= 6.0 ? gaussField(in, W, H, sigma, out, nth, ab, user)
                        : blurField(in, W, H, sigma, p, out, nth, ab, user);
}

// Contrast Rim with Rim Blur > 0: an optical fringe instead of a drawn line. The light of the
// qualifying bright regions (gated by the same contrast / genuine-region tests as the rim) is
// defocused per channel - red spreads furthest (sigma), green ~0.55 sigma, blue ~0.25 sigma - and
// the difference to its own (barely blurred) image is what spills across the boundary: positive on
// the dark side (an orange -> red -> dark gradient, peak at the boundary, Gaussian-like falloff,
// added on top of what is there), negative just inside the bright side (red moved away = a faint
// cyan, used only as much as laComplement). The spilled light is tinted toward laTint (red-brown
// for weaker differences), or keeps the bright side's own colour (laBrightColor).
// The source colour is the bright region's local colour without line-art holes, so dark lines
// inside a bright area (protected by the closing) receive nothing.
bool contrastFringe(const PostCtx& C, const Params& p, float* rgba, int stride, const RimGeom& G,
                    const std::vector<float>& Y, const std::vector<float>& Yc, const std::vector<float>& Bs,
                    const std::vector<float>& Ds, const std::vector<float>& Bn, const std::vector<float>& b,
                    int nth, AbortFn ab, void* user) {
    const int W = C.W, H = C.H;
    const size_t N = (size_t)W * H;
    const double eps = 1.0 / 256.0;
    const double amt = clampd(p.lumAberration, 0.0, 4.0);
    const double thr = clampd(p.laThreshold, 0.0, 10.0), span = 2.0;
    const double curve = clampd(p.laCurve, 0.25, 4.0);
    // bright-region colour: members whose own luminance is the closed level (no line-art holes)
    std::vector<float> m(N), f(N), cbs[3], wm;
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
            const double r = std::min(1.0, (double)Y[i] / (Yc[i] + 1e-6));
            m[i] = (float)(b[i] * r * r);
        }
    })) return false;
    for (int ch = 0; ch < 3; ++ch) {
        if (!parRows(H, nth, ab, user, [&](int y) {
            for (int x = 0; x < W; ++x) {
                const size_t i = (size_t)y * W + x;
                f[i] = std::max(0.f, rgba[(size_t)y * stride + (size_t)x * 4 + ch]) * m[i];
            }
        })) return false;
        if (!blurField(f, W, H, G.sReg, p, cbs[ch], nth, ab, user)) return false;
    }
    if (!blurField(m, W, H, G.sReg, p, wm, nth, ab, user)) return false;
    std::vector<float>().swap(m);
    // gate (contrast response x genuine bright region) and the per-pixel contrast term t
    std::vector<float> gate(N), tt(N);
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
            const double c = std::log2((Bs[i] + eps) / (Ds[i] + eps));
            const double t = std::min(1.0, std::max(0.0, (c - thr) / span));
            tt[i] = (float)t;
            if (t <= 0) { gate[i] = 0.f; continue; }
            const double q = smooth01(1.0 - (std::log2(Bs[i] + eps) - std::log2(Bn[i] + eps))) *
                             smooth01((std::log2(Bs[i] + eps) + 3.5) / 1.5);
            gate[i] = (float)(std::pow(t, curve) * q);
        }
    })) return false;
    const double sR = G.sf, sig[3] = {sR, 0.55 * sR, 0.25 * sR}, s0 = 0.12 * sR;
    std::vector<float> D[3];
    for (int ch = 0; ch < 3; ++ch) {
        if (!parRows(H, nth, ab, user, [&](int y) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i)
                f[i] = wm[i] > 1e-6f ? gate[i] * b[i] * std::max(0.f, cbs[ch][i] / wm[i]) : 0.f;
        })) return false;
        std::vector<float> spread, base;
        if (sig[ch] >= 0.3) {
            if (!gaussOrBox(f, W, H, sig[ch], p, spread, nth, ab, user)) return false;
        } else {
            spread = f;
        }
        if (s0 >= 0.3) {
            if (!gaussOrBox(f, W, H, s0, p, base, nth, ab, user)) return false;
        } else {
            base = f;
        }
        if (!parRows(H, nth, ab, user, [&](int y) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) spread[i] -= base[i];
        })) return false;
        D[ch].swap(spread);
    }
    const double hs = clampd(p.laHueShift, 0.0, 1.0);
    const double mixB = clampd(p.laBrightColor, 0.0, 1.0);
    const double comp = clampd(p.laComplement, 0.0, 1.0);
    const double lo[3] = {0.55, 0.12, 0.05};  // red-brown for smaller differences
    const double hi[3] = {tintC(p.laTint[0]), tintC(p.laTint[1]), tintC(p.laTint[2])};
    const double K = 2.0 * amt;  // the defocused spill at the boundary is ~half the source: x2
    return parallelFor(H, nth, 8, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            const double d[3] = {D[0][i], D[1][i], D[2][i]};
            if (d[0] == 0 && d[1] == 0 && d[2] == 0) continue;
            float* o = rgba + (size_t)y * stride + (size_t)x * 4;
            const double al = std::max(0.f, o[3]);
            const double pos[3] = {std::max(0.0, d[0]), std::max(0.0, d[1]), std::max(0.0, d[2])};
            const double lp = 0.2126 * pos[0] + 0.7152 * pos[1] + 0.0722 * pos[2];
            double col[3];
            const double u = (1.0 - hs) + hs * tt[i];
            for (int k = 0; k < 3; ++k) col[k] = lo[k] + (hi[k] - lo[k]) * u;
            const double lc = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2];
            for (int k = 0; k < 3; ++k) {
                const double tinted = lc > 1e-9 ? col[k] / lc * lp : 0.0;
                double add = (1.0 - mixB) * tinted + mixB * pos[k];
                add += comp * std::min(0.0, d[k]);
                o[k] = (float)std::max(0.0, o[k] + K * add * al);
            }
        }
    }, ab, user);
}

bool applyContrastRim(const PostCtx& C, const Params& p, float* rgba, int stride, int nth, AbortFn ab, void* user) {
    const int W = C.W, H = C.H;
    const size_t N = (size_t)W * H;
    const double amt = clampd(p.lumAberration, 0.0, 4.0);
    const RimGeom G = rimGeom(p, C.halfDiag);
    const double eps = 1.0 / 256.0;
    std::vector<float> Y(N);
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (int x = 0; x < W; ++x)
            Y[(size_t)y * W + x] = (float)std::max(0.0, lumOf(rgba + (size_t)y * stride + (size_t)x * 4));
    })) return false;
    Scratch sc(nth);
    // octagonal morphology (row/column segments + diagonal segments): no square corners
    const int s1 = std::max(1, (int)std::lround(0.6 * G.rl)), s2 = (int)std::lround(0.42 * G.rl);
    auto morph = [&](std::vector<float>& img, bool maxFirst) {
        for (int pass = 0; pass < 2; ++pass)
            for (int dir = 0; dir < 4; ++dir) {
                const int r = dir < 2 ? s1 : s2;
                const bool mx = (pass == 0) == maxFirst;
                const bool ok = mx ? filterLines<true>(img, W, H, r, dir, nth, sc, ab, user)
                                   : filterLines<false>(img, W, H, r, dir, nth, sc, ab, user);
                if (!ok) return false;
            }
        return true;
    };
    std::vector<float> Yc = Y;
    if (G.rl > 0 && !morph(Yc, true)) return false;  // closing: thin dark lines inside bright areas
    // opening of that: small bright specks (catchlights, highlights on hair) are not a "bright
    // region" either, so they neither raise the region level nor get a rim around them
    std::vector<float> Yo = Yc;
    if (G.rl > 0 && !morph(Yo, false)) return false;
    // brightest region level right next to each pixel: the rim belongs to the boundary of the
    // genuinely bright region (within ~1 stop of the region level), not to lighter strands
    // inside a dark area that merely sit above the mid level
    std::vector<float> Bn = Yo;
    if (!filter2D<true>(Bn, W, H, G.rn, nth, sc, ab, user)) return false;
    std::vector<float> B = Yo, D = Yc, Bs, Ds;
    if (!filter2D<true>(B, W, H, G.rc, nth, sc, ab, user)) return false;
    if (!filter2D<false>(D, W, H, G.rc, nth, sc, ab, user)) return false;
    if (!blurField(B, W, H, G.sReg, p, Bs, nth, ab, user)) return false;
    if (!blurField(D, W, H, G.sReg, p, Ds, nth, ab, user)) return false;
    std::vector<float>().swap(B); std::vector<float>().swap(D);
    std::vector<float> b(N), nb(N);
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
            const double mid = 0.5 * (std::log2(Bs[i] + eps) + std::log2(Ds[i] + eps));
            const double v = smooth01((std::log2(Yc[i] + eps) - mid) + 0.5);
            b[i] = (float)v;
            nb[i] = (float)(1.0 - v);
        }
    })) return false;
    if (p.laBlur > 1e-6) return contrastFringe(C, p, rgba, stride, G, Y, Yc, Bs, Ds, Bn, b, nth, ab, user);
    std::vector<float> Fb, Fd;
    if (!gaussField(b, W, H, G.sd, Fb, nth, ab, user)) return false;
    const bool brightSide = G.sb >= 0.3;
    if (brightSide && !gaussField(nb, W, H, G.sb, Fd, nth, ab, user)) return false;
    // bright-side colour (mean colour of the bright members around the boundary)
    const double mixB = clampd(p.laBrightColor, 0.0, 1.0);
    std::vector<float> cb[3], wb;
    if (mixB > 1e-4) {
        std::vector<float> f(N);
        for (int ch = 0; ch < 3; ++ch) {
            if (!parRows(H, nth, ab, user, [&](int y) {
                for (int x = 0; x < W; ++x) {
                    const size_t i = (size_t)y * W + x;
                    f[i] = rgba[(size_t)y * stride + (size_t)x * 4 + ch] * b[i];
                }
            })) return false;
            if (!blurField(f, W, H, G.sReg, p, cb[ch], nth, ab, user)) return false;
        }
        if (!blurField(b, W, H, G.sReg, p, wb, nth, ab, user)) return false;
    }
    const double thr = clampd(p.laThreshold, 0.0, 10.0), span = 2.0;
    const double curve = clampd(p.laCurve, 0.25, 4.0);
    const double hs = clampd(p.laHueShift, 0.0, 1.0);
    const double sharp = clampd(p.laSharpness, 0.0, 1.0);
    const double ex = 1.0 + 3.0 * sharp;
    const double F0 = 0.16;  // ~ Gaussian tail at 1 sigma: full weight within ~1 rim width
    const double lo[3] = {0.55, 0.10, 0.035};  // red-brown for smaller differences
    const double hi[3] = {tintC(p.laTint[0]), tintC(p.laTint[1]), tintC(p.laTint[2])};
    auto lumN = [](double c[3]) {
        const double l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        if (l > 1e-9) for (int k = 0; k < 3; ++k) c[k] /= l;
        return l > 1e-9;
    };
    return parallelFor(H, nth, 8, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            const double c = std::log2((Bs[i] + eps) / (Ds[i] + eps));
            if (c <= thr) continue;
            const double t = std::min(1.0, (c - thr) / span);
            const double s = std::pow(t, curve);
            // the bright side must be bright in absolute terms too (lit skin, not a lighter strand of
            // dark hair): 0 below ~-3.5 stops (linear 0.09), full from -2 stops (0.25)
            const double q = smooth01(1.0 - (std::log2(Bs[i] + eps) - std::log2(Bn[i] + eps))) *
                             smooth01((std::log2(Bs[i] + eps) + 3.5) / 1.5);
            if (q <= 0) continue;
            double w = (1.0 - b[i]) * std::pow(std::min(1.0, Fb[i] / F0), ex);
            if (brightSide) w += b[i] * std::pow(std::min(1.0, Fd[i] / F0), ex);
            w *= q;
            const double a = std::min(1.0, amt * s * std::min(1.0, w));
            if (a <= 0) continue;
            float* o = rgba + (size_t)y * stride + (size_t)x * 4;
            double col[3];
            const double u = (1.0 - hs) + hs * t;
            for (int k = 0; k < 3; ++k) col[k] = lo[k] + (hi[k] - lo[k]) * u;
            lumN(col);
            if (mixB > 1e-4 && wb[i] > 1e-6) {
                double wc[3];
                for (int k = 0; k < 3; ++k) wc[k] = std::max(0.0, (double)cb[k][i] / wb[i]) * hi[k];
                if (lumN(wc))
                    for (int k = 0; k < 3; ++k) col[k] += (wc[k] - col[k]) * mixB;
            }
            const double L = 0.55 * Bs[i] * (0.35 + 0.65 * t);
            const double al = std::max(0.f, o[3]);
            for (int k = 0; k < 3; ++k) o[k] = (float)(o[k] + a * (col[k] * L * al - o[k]));
        }
    }, ab, user);
}

// ------------------------------------------------------------ film grain ----
// Gaussian-filtered white noise on the FULL-RESOLUTION layer pixel grid (no lattice cells, isotropic):
// every full-res pixel gets an approximately Gaussian random value hashed from (its absolute layer
// coordinates, layer, octave, seed, frame); the field is blurred with an exact separable Gaussian
// (sigma from Grain Size / Softness) and normalised to unit variance analytically. Two octaves: fine
// grain (sigma_f) plus coarser clumps (1.8 sigma_f, weight 0.6) - a continuous mixture of sizes with a
// Gaussian histogram (sum of many independent values). The noise is generated for the buffer's own
// full-res footprint plus the kernel support, so no input margin is needed and any tile / thread / render
// order gives the same values. Each buffer pixel averages the full-res pixels of its footprint, so a
// half-resolution render shows the same grain, area-averaged. Colour grain uses independent fields per
// channel with larger, softer clumps (R/G 1.25x, B 1.5x sigma).
// Applied per channel in the gamma-encoded (display) domain, scaled by a tone curve: sqrt(L) near black
// (black stays black, shadows still show grain), fading out in the highlights by Tone Response; HDR
// values above 1 get the highlight weight and are never clipped. Straight colour; alpha is untouched.
inline uint32_t grainHash(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t h = a * 0x8da6b343u ^ b * 0xd8163841u ^ c * 0xcb1ab31fu ^ d * 0x165667b1u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h;
}
// unit-variance, approximately Gaussian white noise (Irwin-Hall of 3 uniforms in [-1, 1])
inline float grainWhite(int u, int v, uint32_t key) {
    const uint32_t h1 = grainHash((uint32_t)u, (uint32_t)v, key, 0x9e3779b9u);
    const uint32_t h2 = grainHash(h1, key, 0x2545f491u, 0x68e31da4u);
    const float a = (h1 & 0xffff) * (2.f / 65535.f) - 1.f, b = (h1 >> 16) * (2.f / 65535.f) - 1.f;
    const float c = (h2 & 0xffff) * (2.f / 65535.f) - 1.f;
    return a + b + c;  // variance 3 * 1/3 = 1
}

bool applyGrain(const PostCtx& C, const Params& p, float* rgba, int stride, int nth, AbortFn ab, void* user) {
    const int W = C.W, H = C.H;
    const size_t N = (size_t)W * H;
    const double amt = clampd(p.grain, 0.0, 4.0) * 0.07;   // encoded-domain std at full weight
    const double size = clampd(p.grainSize, 0.3, 8.0);
    const double soft = clampd(p.grainSoftness, 0.0, 1.0);
    const double col = clampd(p.grainColor, 0.0, 1.0);
    const double tone = clampd(p.grainTone, 0.0, 1.0);
    const double sx = clampd(p.grainScaleX, 0.01, 16.0), sy = clampd(p.grainScaleY, 0.01, 16.0);
    const double ox = p.frameWidth > 0 ? p.frameLeft : 0.0, oy = p.frameWidth > 0 ? p.frameTop : 0.0;
    const uint32_t seed = (uint32_t)p.grainSeed, frame = (uint32_t)p.grainFrame;
    // fine-grain sigma (full-res px): Softness widens it (crisp fine grain -> soft clumps)
    const double sf = 0.5 * size * (0.55 + 0.9 * soft);
    const double layerScale[4] = {1.0, 1.25, 1.25, 1.5};
    const int nLayers = col > 1e-4 ? 4 : 1;
    double smax = 0;
    for (int l = 0; l < nLayers; ++l) smax = std::max(smax, 1.8 * sf * layerScale[l]);
    const int R = (int)std::ceil(3.0 * smax) + 2;
    const double cn = 1.0 / std::sqrt((1.0 - col) * (1.0 - col) + col * col);
    std::vector<float> g[4];
    auto applyField = [&]() {
        return parallelFor(H, nth, 8, [&](int y, int) {
            for (int x = 0; x < W; ++x) {
                float* o = rgba + (size_t)y * stride + (size_t)x * 4;
                const double a = o[3];
                if (!(a > 1e-6)) continue;
                const size_t i = (size_t)y * W + x;
                double e[3];
                for (int c = 0; c < 3; ++c) {
                    const double s = o[c] / a;
                    e[c] = s >= 0 ? std::pow(s, 1.0 / 2.2) : -std::pow(-s, 1.0 / 2.2);
                }
                const double L = std::max(0.0, 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]);
                const double Lc = std::min(1.0, L);
                const double w = std::sqrt(Lc) * (1.0 - tone * smooth01((Lc - 0.45) / 0.5));
                for (int c = 0; c < 3; ++c) {
                    const double n = nLayers > 1 ? ((1.0 - col) * g[0][i] + col * g[c + 1][i]) * cn : g[0][i];
                    double ec = e[c] + amt * w * n;
                    if (e[c] >= 0 && ec < 0) ec = 0;
                    const double s = ec >= 0 ? std::pow(ec, 2.2) : -std::pow(-ec, 2.2);
                    o[c] = (float)(s * a);
                }
            }
        }, ab, user);
    };
    // v2.2: downsampled renders (Half / Quarter previews of big comps) no longer build the grain at
    // full resolution. The same full-resolution white noise is box-summed into each buffer pixel's
    // footprint (so the pattern stays the one of the final render, area-averaged), then blurred on the
    // BUFFER grid with the equivalent sigma and rescaled to the variance the full-resolution path
    // would give. Cost and memory scale with the buffer; nothing is skipped.
    // (v2.2.1) only for square buffer pixels: the buffer-grid blur is isotropic. Non-square pixels
    // (pixel aspect, downsample x != y) take the full-resolution path, which is exact per axis
    // (AE 2026: a 2:1 layer's grain came out round in BUFFER px, i.e. stretched 2x on screen).
    const bool squareBuf = std::fabs(sx / sy - 1.0) < 0.02;
    if (squareBuf && (1.0 / sx >= 1.5 || 1.0 / sy >= 1.5)) {
        const double sb = std::sqrt(sx * sy);
        const int Rb = (int)std::ceil(3.0 * smax * sb) + 2;
        const int BW = W + 2 * Rb, BH = H + 2 * Rb;
        std::vector<int> fx0(BW), fx1(BW), fy0(BH), fy1(BH);
        auto footAbs = [](double a, double b, int& i0, int& i1) {
            i0 = (int)std::ceil(a - 0.5); i1 = (int)std::ceil(b - 0.5) - 1;
            if (i1 < i0) i0 = i1 = (int)std::floor(0.5 * (a + b));
        };
        for (int x = 0; x < BW; ++x) footAbs((x - Rb - ox) / sx, (x - Rb + 1 - ox) / sx, fx0[x], fx1[x]);
        for (int y = 0; y < BH; ++y) footAbs((y - Rb - oy) / sy, (y - Rb + 1 - oy) / sy, fy0[y], fy1[y]);
        const int stepX = std::max(1, (int)std::ceil((1.0 / sx) / 16.0)), stepY = std::max(1, (int)std::ceil((1.0 / sy) / 16.0));
        const double kx = 1.0 / sx, ky = 1.0 / sy;
        // variance of the area average (k x k) of a unit-variance field with Gaussian correlation
        // exp(-d^2 / (4 sigma^2)) (white noise blurred with sigma), per axis
        auto vAxis = [](double k, double sig) {
            const int n = std::max(1, (int)std::lround(k));
            double s = 0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) s += std::exp(-(double)(i - j) * (i - j) / (4.0 * sig * sig));
            return s / ((double)n * n);
        };
        std::vector<float> B((size_t)BW * BH), blurred;
        for (int l = 0; l < nLayers; ++l) {
            std::vector<float> acc((size_t)BW * BH, 0.f);
            for (int o = 0; o < 2; ++o) {
                const double sig = sf * layerScale[l] * (o ? 1.8 : 1.0);
                const double wgt = o ? 0.6 : 1.0;
                const uint32_t key = grainHash(seed, frame, (uint32_t)(l * 2 + o), 0x51ed270bu);
                bool ok = parallelFor(BH, nth, 8, [&](int y, int) {
                    for (int x = 0; x < BW; ++x) {
                        double s = 0; int n = 0;
                        for (int v = fy0[y]; v <= fy1[y]; v += stepY)
                            for (int u = fx0[x]; u <= fx1[x]; u += stepX) { s += grainWhite(u, v, key); ++n; }
                        B[(size_t)y * BW + x] = (float)(s / std::sqrt((double)std::max(1, n)));  // unit variance
                    }
                }, ab, user);
                if (!ok) return false;
                const double sigb = std::max(0.3, sig * sb);
                const int r = std::max(1, (int)std::ceil(3.0 * sigb));
                double s1 = 0, s2 = 0;
                for (int i = -r; i <= r; ++i) { const double k = std::exp(-0.5 * i * i / (sigb * sigb)); s1 += k; s2 += k * k; }
                const double vNew = (s2 / (s1 * s1)) * (s2 / (s1 * s1));
                const double vOld = vAxis(kx, sig) * vAxis(ky, sig);
                const double nrm = wgt * std::sqrt(vOld / std::max(1e-30, vNew));
                if (!gaussField(B, BW, BH, sigb, blurred, nth, ab, user)) return false;
                if (!parallelFor(BH, nth, 16, [&](int y, int) {  // v2.3: element-wise, in parallel
                    for (size_t i = (size_t)y * BW; i < (size_t)(y + 1) * BW; ++i) acc[i] += (float)(nrm * blurred[i]);
                }, ab, user)) return false;
            }
            const float an = (float)(1.0 / std::sqrt(1.0 + 0.36));
            g[l].assign(N, 0.f);
            if (!parallelFor(H, nth, 16, [&](int y, int) {
                for (int x = 0; x < W; ++x) g[l][(size_t)y * W + x] = an * acc[(size_t)(y + Rb) * BW + x + Rb];
            }, ab, user)) return false;
        }
        return applyField();
    }
    // full-res pixels covered by the buffer (+ kernel support)
    const int u0 = (int)std::floor((0 - ox) / sx) - R, u1 = (int)std::ceil((W - ox) / sx) + R;
    const int v0 = (int)std::floor((0 - oy) / sy) - R, v1 = (int)std::ceil((H - oy) / sy) + R;
    const int GW = u1 - u0, GH = v1 - v0;
    if (GW <= 0 || GH <= 0) return true;
    // footprint of each buffer column / row in full-res pixel indices (centres inside the footprint)
    std::vector<int> cx0(W), cx1(W), cy0(H), cy1(H);
    auto foot = [](double a, double b, int g0, int& i0, int& i1) {
        i0 = (int)std::ceil(a - 0.5); i1 = (int)std::ceil(b - 0.5) - 1;
        if (i1 < i0) i0 = i1 = (int)std::floor(0.5 * (a + b));
        i0 -= g0; i1 -= g0;
    };
    for (int x = 0; x < W; ++x) foot((x - ox) / sx, (x + 1 - ox) / sx, u0, cx0[x], cx1[x]);
    for (int y = 0; y < H; ++y) foot((y - oy) / sy, (y + 1 - oy) / sy, v0, cy0[y], cy1[y]);
    std::vector<float> field((size_t)GW * GH), blurred;
    for (int l = 0; l < nLayers; ++l) {
        std::vector<float> acc((size_t)GW * GH, 0.f);
        for (int o = 0; o < 2; ++o) {
            const double sig = sf * layerScale[l] * (o ? 1.8 : 1.0);
            const double wgt = o ? 0.6 : 1.0;
            const uint32_t key = grainHash(seed, frame, (uint32_t)(l * 2 + o), 0x51ed270bu);
            bool ok = parallelFor(GH, nth, 16, [&](int j, int) {
                float* f = field.data() + (size_t)j * GW;
                for (int i = 0; i < GW; ++i) f[i] = grainWhite(u0 + i, v0 + j, key);
            }, ab, user);
            if (!ok) return false;
            // unit variance after the blur: std = sum(k^2) of the normalised 1D kernel (separable)
            const int r = std::max(1, (int)std::ceil(3.0 * sig));
            double s1 = 0, s2 = 0;
            for (int i = -r; i <= r; ++i) { const double k = std::exp(-0.5 * i * i / (sig * sig)); s1 += k; s2 += k * k; }
            const double nrm = wgt / (s2 / (s1 * s1));
            if (!gaussField(field, GW, GH, sig, blurred, nth, ab, user)) return false;
            if (!parallelFor(GH, nth, 16, [&](int j, int) {  // v2.3: element-wise, in parallel
                for (size_t i = (size_t)j * GW; i < (size_t)(j + 1) * GW; ++i) acc[i] += (float)(nrm * blurred[i]);
            }, ab, user)) return false;
        }
        const float an = (float)(1.0 / std::sqrt(1.0 + 0.36));
        // area-average into buffer pixels
        g[l].assign(N, 0.f);
        bool ok = parallelFor(H, nth, 8, [&](int y, int) {
            for (int x = 0; x < W; ++x) {
                double s = 0;
                for (int j = cy0[y]; j <= cy1[y]; ++j)
                    for (int i = cx0[x]; i <= cx1[x]; ++i) s += acc[(size_t)j * GW + i];
                g[l][(size_t)y * W + x] = (float)(an * s / ((cy1[y] - cy0[y] + 1) * (cx1[x] - cx0[x] + 1)));
            }
        }, ab, user);
        if (!ok) return false;
    }
    std::vector<float>().swap(field); std::vector<float>().swap(blurred);
    return applyField();
}

// Reach (px, per axis) of the octagonal closing used by Line Protection: two passes (max then
// min), each of a row/column segment of radius s1 plus two diagonal segments of radius s2 (each
// diagonal moves one px in x and y per step): 2 x (s1 + 2 s2) ~ 2.88 rl. (v2.1 budgeted 2 rl.)
inline double morphReach(int rl) {
    if (rl <= 0) return 0.0;
    const int s1 = std::max(1, (int)std::lround(0.6 * rl)), s2 = (int)std::lround(0.42 * rl);
    return 2.0 * (s1 + 2.0 * s2);
}

// laMode 4 geometry (px), shared with postMargin
struct DriftGeom { int rl; double sm, len, kmax; int rb; int spread; double dirSigma, lenCap; };
inline DriftGeom driftGeom(const Params& p, double halfDiag) {
    const double diag = 2.0 * halfDiag;
    DriftGeom g;
    g.rl = (int)std::lround(clampd(p.laLineProtect, 0.0, 0.02) * diag);
    g.sm = std::max(0.7, clampd(p.laMapSoftness, 0.0, 0.05) * diag);
    g.len = clampd(p.laWidth, 0.0, 0.05) * diag * clampd(p.lumAberration, 0.0, 4.0);
    g.kmax = 0;
    for (int c = 0; c < 3; ++c) g.kmax = std::max(g.kmax, std::fabs(clampd(p.laShift[c], -2.0, 2.0)));
    g.rb = (int)std::ceil(g.len * g.kmax) + 2;
    // v2.2 band spreading (renderVersion >= 2): dilation radius = the smear length, direction from a
    // blurred field, fetch length capped at 3 % of the diagonal (no colour from far-away content)
    g.spread = 0; g.dirSigma = 0; g.lenCap = 1e30;
    if (p.renderVersion >= 2) {
        g.lenCap = 0.03 * diag;
        g.spread = (int)std::ceil(std::min(g.len, g.lenCap) * 0.5);
        g.dirSigma = std::max(g.sm, 0.5 * g.spread + 0.5);
        g.rb = (int)std::ceil(std::min(g.len, g.lenCap) * g.kmax) + 2 + 2 * g.spread + (int)std::ceil(3.5 * g.dirSigma);
    }
    return g;
}

// laMode 4, "channel drift" - the AE recipe behind the reference look (Channel Combiner + CC Vector
// Blur on green/blue only, added back over red), done properly:
//  1. luminance map (optionally line-protected by a closing), softened (Map Softness);
//  2. edge strength in stops from the gradient of its log2 (an ideal step of c stops softened by
//     sigma has a peak gradient of c / (sigma sqrt(2 pi))), response = threshold/curve as for the
//     other modes; Preserve Detail additionally requires the bright side to be bright in absolute
//     terms (lit skin, not mid-tone line art), so faces and soft shading do not distort;
//  3. every channel is smeared (directional box, like a vector blur) from each pixel along the
//     gradient: toward the dark side (retreat: that channel goes missing on the bright edge) or toward
//     the bright side (advance), length = laShift[c] x laWidth x diag x amount x response.
// The default G+B retreat leaves red on the bright edge: a soft orange-red fringe whose colour comes
// from the picture itself. Premultiplied colour is resampled; alpha stays untouched. HDR-safe
// (linear interpolation only). Nothing changes where the response is 0.
bool applyChannelDrift(const PostCtx& C, const Params& p, float* rgba, int stride, int nth, AbortFn ab, void* user) {
    const int W = C.W, H = C.H;
    const size_t N = (size_t)W * H;
    const DriftGeom G = driftGeom(p, C.halfDiag);
    if (G.len * G.kmax < 1e-3) return true;
    const double eps = 1.0 / 256.0;
    // straight (unpremultiplied) luminance: a layer's own transparent surroundings are not "dark"
    std::vector<float> Y(N), A(N), Ls;
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float* o = rgba + (size_t)y * stride + (size_t)x * 4;
            const double a = std::min(1.0, std::max(0.0, (double)o[3]));
            A[(size_t)y * W + x] = (float)a;
            Y[(size_t)y * W + x] = a > 1e-4 ? (float)std::max(0.0, lumOf(o) / a) : 0.f;
        }
    })) return false;
    Scratch sc(nth);
    if (G.rl > 0) {  // octagonal closing: thin dark lines inside bright areas do not make edges
        const int s1 = std::max(1, (int)std::lround(0.6 * G.rl)), s2 = (int)std::lround(0.42 * G.rl);
        for (int pass = 0; pass < 2; ++pass)
            for (int dir = 0; dir < 4; ++dir) {
                const int r = dir < 2 ? s1 : s2;
                const bool ok = pass == 0 ? filterLines<true>(Y, W, H, r, dir, nth, sc, ab, user)
                                          : filterLines<false>(Y, W, H, r, dir, nth, sc, ab, user);
                if (!ok) return false;
            }
    }
    // softened map by normalised convolution over the covered pixels (alpha-weighted)
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) Y[i] *= A[i];
    })) return false;
    std::vector<float> As;
    if (!gaussOrBox(Y, W, H, G.sm, p, Ls, nth, ab, user)) return false;
    if (!gaussOrBox(A, W, H, G.sm, p, As, nth, ab, user)) return false;
    std::vector<float> lg(N);
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
            Ls[i] = As[i] > 1e-3f ? Ls[i] / As[i] : 0.f;
            lg[i] = (float)std::log2(Ls[i] + eps);
        }
    })) return false;
    std::vector<float>().swap(Y); std::vector<float>().swap(As);
    const double thr = clampd(p.laThreshold, 0.0, 10.0), span = 2.0;
    const double curve = clampd(p.laCurve, 0.25, 4.0);
    const double pres = clampd(p.laPreserve, 0.0, 1.0);
    const double kstop = G.sm * 2.5066;  // gradient (stops/px) -> edge step in stops
    // displacement field (px, toward the bright side), 0 where the edge does not qualify
    std::vector<float> vx(N, 0.f), vy(N, 0.f);
    bool ok = parallelFor(H, nth, 8, [&](int y, int) {
        const int y0 = std::max(0, y - 1), y1 = std::min(H - 1, y + 1);
        for (int x = 0; x < W; ++x) {
            const int x0 = std::max(0, x - 1), x1 = std::min(W - 1, x + 1);
            const double gx = (lg[(size_t)y * W + x1] - lg[(size_t)y * W + x0]) / std::max(1, x1 - x0);
            const double gy = (lg[(size_t)y1 * W + x] - lg[(size_t)y0 * W + x]) / std::max(1, y1 - y0);
            const double gm = std::sqrt(gx * gx + gy * gy);
            const double c = gm * kstop;
            if (c <= thr || gm <= 0) continue;
            double s = std::pow(std::min(1.0, (c - thr) / span), curve);
            if (pres > 0) {  // bright side (one map-softness step up the gradient) must be really bright
                const double ux = gx / gm, uy = gy / gm;
                const int bx = std::min(W - 1, std::max(0, (int)std::lround(x + ux * 2.0 * G.sm)));
                const int by = std::min(H - 1, std::max(0, (int)std::lround(y + uy * 2.0 * G.sm)));
                const double q = smooth01((lg[(size_t)by * W + bx] + 3.5) / 1.5);
                s *= (1.0 - pres) + pres * q;
            }
            const double L = G.len * s / gm;
            vx[(size_t)y * W + x] = (float)(gx * L);
            vy[(size_t)y * W + x] = (float)(gy * L);
        }
    }, ab, user);
    if (!ok) return false;
    std::vector<float>().swap(lg); std::vector<float>().swap(Ls);
    const bool v2 = p.renderVersion >= 2;
    if (v2 && G.spread > 0) {
        // v2.2: Width widens the fringe. The qualifying band is dilated by the smear length (octagonal
        // max filter of |v|) and the direction is taken from the blurred displacement field, so the
        // fringe width grows proportionally with Width instead of staying at the map-softness width.
        std::vector<float> mag(N), bx, by;
        if (!parRows(H, nth, ab, user, [&](int y) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) mag[i] = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
        })) return false;
        const int s1 = std::max(1, (int)std::lround(0.6 * G.spread)), s2 = (int)std::lround(0.42 * G.spread);
        for (int dir = 0; dir < 4; ++dir)
            if (!filterLines<true>(mag, W, H, dir < 2 ? s1 : s2, dir, nth, sc, ab, user)) return false;
        if (!gaussOrBox(vx, W, H, G.dirSigma, p, bx, nth, ab, user)) return false;
        if (!gaussOrBox(vy, W, H, G.dirSigma, p, by, nth, ab, user)) return false;
        if (!parRows(H, nth, ab, user, [&](int y) {
            for (size_t i = (size_t)y * W; i < (size_t)(y + 1) * W; ++i) {
                const double bm = std::sqrt((double)bx[i] * bx[i] + (double)by[i] * by[i]);
                if (mag[i] <= 0.f || bm < 1e-9) { vx[i] = vy[i] = 0.f; continue; }
                const double l = std::min((double)mag[i], G.lenCap) / bm;
                vx[i] = (float)(bx[i] * l);
                vy[i] = (float)(by[i] * l);
            }
        })) return false;
    }
    const double sgn = p.laShiftDir == 1 ? 1.0 : -1.0;  // retreat = sample toward the dark side
    double k[3];
    for (int c = 0; c < 3; ++c) k[c] = sgn * clampd(p.laShift[c], -2.0, 2.0);
    // premultiplied colour + alpha of the input
    std::vector<float> src((size_t)N * 4);
    if (!parRows(H, nth, ab, user, [&](int y) {
        std::copy(rgba + (size_t)y * stride, rgba + (size_t)y * stride + (size_t)W * 4, src.data() + (size_t)y * W * 4);
    })) return false;
    auto sample = [&](double fx, double fy, int c) {
        fx = std::min((double)W - 1, std::max(0.0, fx));
        fy = std::min((double)H - 1, std::max(0.0, fy));
        const int x0 = (int)fx, y0 = (int)fy, x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
        const double ax = fx - x0, ay = fy - y0;
        const float* s0 = src.data() + ((size_t)y0 * W) * 4;
        const float* s1 = src.data() + ((size_t)y1 * W) * 4;
        const double a = s0[(size_t)x0 * 4 + c] + (s0[(size_t)x1 * 4 + c] - s0[(size_t)x0 * 4 + c]) * ax;
        const double b = s1[(size_t)x0 * 4 + c] + (s1[(size_t)x1 * 4 + c] - s1[(size_t)x0 * 4 + c]) * ax;
        return a + (b - a) * ay;
    };
    return parallelFor(H, nth, 8, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            if (vx[i] == 0.f && vy[i] == 0.f) continue;
            float* o = rgba + (size_t)y * stride + (size_t)x * 4;
            const double ad = o[3];
            if (!(ad > 1e-6)) continue;  // no colour where there is no coverage (robust to rounding noise)
            for (int c = 0; c < 3; ++c) {
                if (k[c] == 0) continue;
                const double dx = k[c] * vx[i], dy = k[c] * vy[i];
                const double dl = std::sqrt(dx * dx + dy * dy);
                const int n = v2 ? std::min(128, std::max(1, (int)std::ceil(dl)))
                                 : std::min(16, std::max(1, (int)std::ceil(dl * 2.0)));
                // directional box smear over [0, d] (vector blur) of the STRAIGHT colour, alpha-weighted,
                // re-premultiplied by this pixel's alpha: colour never exceeds alpha x the straight
                // colours it was taken from (v2.2 fix: premultiplied colour was resampled before)
                double accC = 0, accA = 0;
                for (int j = 0; j < n; ++j) {
                    const double t = (j + 0.5) / n;
                    accC += sample(x + dx * t, y + dy * t, c);
                    accA += sample(x + dx * t, y + dy * t, 3);
                }
                if (accA > 1e-9 * n) o[c] = (float)std::max(0.0, accC / accA * ad);
            }
        }
    }, ab, user);
}

bool applyLumAberration(const PostCtx& C, const Params& p, float* rgba, int stride, int nth, AbortFn ab, void* user) {
    if (p.laMode == 3) return applyContrastRim(C, p, rgba, stride, nth, ab, user);
    if (p.laMode == 4) return applyChannelDrift(C, p, rgba, stride, nth, ab, user);
    const int W = C.W, H = C.H;
    const double amt = clampd(p.lumAberration, 0.0, 4.0);
    std::vector<float> Y((size_t)W * H), G;
    if (!parRows(H, nth, ab, user, [&](int y) {
        for (int x = 0; x < W; ++x)
            Y[(size_t)y * W + x] = (float)std::max(0.0, lumOf(rgba + (size_t)y * stride + (size_t)x * 4));
    })) return false;
    const double sigma = std::max(0.6, clampd(p.laWidth, 0.0005, 0.1) * 2.0 * C.halfDiag);
    if (!blurField(Y, W, H, sigma, p, G, nth, ab, user)) return false;
    const double thr = clampd(p.laThreshold, 0.0, 10.0), span = 3.0;
    const double curve = clampd(p.laCurve, 0.25, 4.0);
    const double hs = clampd(p.laHueShift, 0.0, 1.0);
    const double lo[3] = {1.0, 0.62, 0.16};  // yellow-orange for small differences
    const double hi[3] = {tintC(p.laTint[0]), tintC(p.laTint[1]), tintC(p.laTint[2])};
    const int mode = p.laMode == 1 ? 1 : (p.laMode == 2 ? 2 : 0);
    const double eps = 1.0 / 256.0;
    return parallelFor(H, nth, 8, [&](int y, int) {
        for (int x = 0; x < W; ++x) {
            const size_t i = (size_t)y * W + x;
            float* o = rgba + (size_t)y * stride + (size_t)x * 4;
            const double yv = Y[i], g = G[i];
            const double d = std::log2((g + eps) / (yv + eps));  // > 0: neighbourhood brighter (dark side)
            auto resp = [&](double dd) {
                const double t = std::min(1.0, std::max(0.0, (dd - thr) / span));
                return std::pow(t, curve);
            };
            auto colourAt = [&](double dd, double c[3]) {
                const double t = hs * std::min(1.0, std::max(0.0, (dd - thr) / span));
                for (int k = 0; k < 3; ++k) c[k] = lo[k] + (hi[k] - lo[k]) * (1.0 - hs + t);
            };
            if (mode != 1 && d > thr) {  // warm light spilling into the darker side
                const double s = resp(d);
                double c[3];
                colourAt(d, c);
                const double e = amt * s * std::max(0.0, g - yv) * std::max(0.f, o[3]);
                o[0] += (float)(c[0] * e); o[1] += (float)(c[1] * e); o[2] += (float)(c[2] * e);
            }
            if (mode != 0 && -d > thr) {  // bright side of the edge turns warm
                const double s = resp(-d);
                double c[3];
                colourAt(-d, c);
                const double lc = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                const double k = std::min(1.0, 0.6 * amt * s);
                if (lc > 1e-9 && k > 0)
                    for (int ch = 0; ch < 3; ++ch) o[ch] = (float)(o[ch] + k * (c[ch] / lc * yv - o[ch]));
            }
        }
    }, ab, user);
}

// ------------------------------------------------------------ pass 2 --------

struct LevelView {
    int W, H, maxReach;
    const Splat* splats;
    const CatInfo* cats;
};

// Adds the coarse levels' accumulators (bilinear, as densities: / f^2) at fine pixel (x, y).
template <int NC>
inline void addCoarse(const Ctx& C, int x, int y, double* v) {
    for (int l = 1; l < C.nLevels; ++l) {
        const Ctx::Level& Lv = C.levels[l];
        const double f = Lv.f;
        double u = (x + 0.5 + Lv.gox) / f - 0.5, w = (y + 0.5 + Lv.goy) / f - 0.5;
        u = std::min((double)Lv.W - 1, std::max(0.0, u));
        w = std::min((double)Lv.H - 1, std::max(0.0, w));
        const int x0 = (int)u, y0 = (int)w;
        const int x1 = std::min(x0 + 1, Lv.W - 1), y1 = std::min(y0 + 1, Lv.H - 1);
        const double fx = u - x0, fy = w - y0, s = 1.0 / (f * f);
        const double w00 = (1 - fx) * (1 - fy) * s, w10 = fx * (1 - fy) * s, w01 = (1 - fx) * fy * s, w11 = fx * fy * s;
        const float* a = Lv.grid.data() + ((size_t)y0 * Lv.W + x0) * NC;
        const float* b = Lv.grid.data() + ((size_t)y0 * Lv.W + x1) * NC;
        const float* c = Lv.grid.data() + ((size_t)y1 * Lv.W + x0) * NC;
        const float* d = Lv.grid.data() + ((size_t)y1 * Lv.W + x1) * NC;
        for (int k = 0; k < NC; ++k) v[k] += w00 * a[k] + w10 * b[k] + w01 * c[k] + w11 * d[k];
    }
}

// ---- v2.3: blocked form of v2.2's doRow (same arithmetic, fewer redundant evaluations) ----
// (v2.2's doRow rendered one output row per call; its source is kept in build\perf\base\core.)
// doBlock renders kBlockRows consecutive output rows at once. Every splat of a source row is visited
// once per block instead of once per output row, its per-(pass, channel, layer) constants (1/Ry,
// normaliser, lb * invD) are computed once per block, and the shape-table lookup at a band boundary
// is shared by the two bands that meet there (band k's upper edge (k + 0.5) / Ry is band k+1's lower
// edge). For each output row the sequence of additions into each accumulator is exactly the one
// doRow performs (source rows ascending, splats in sorted order, then pass, channel, layer), and every
// double operation is the same IEEE operation on the same operands (SSE2 packed lanes are plain
// IEEE double ops; no FMA, no reassociation), so the result is bit-identical to doRow. The per-pixel
// accumulator channels are stored in a different ORDER than doRow's (pairs that are updated
// together are adjacent, see AccLayout); each channel's own summation order is unchanged.
// Verified bit-identical: build\perf (tools\perf_run.py verify), renderVersion 1 and 2.
#ifndef LC_BLOCK_ROWS
#define LC_BLOCK_ROWS 8
#endif
const int kBlockRows = LC_BLOCK_ROWS;

// accumulator channel indices. FR (three kernels): R WR | B WB | G A | WG eL ; else R G | B A | W eL
template <bool FR>
struct AccLayout {
    static constexpr int R = 0, G = FR ? 4 : 1, B = 2, A = FR ? 5 : 3;
    static constexpr int WR = FR ? 1 : 4, WG = FR ? 6 : 4, WB = FR ? 3 : 4, EL = FR ? 7 : 5;
};

// Resolve one accumulated output row (prefix sum, coarse levels, normalisation): doRow's tail.
template <bool FR, bool ROLL, bool RAW>
inline void resolveRow(const Ctx& C, int W, int y, double* buf, float* out, float* le) {
    using A = AccLayout<FR>;
    constexpr int NB = FR ? 7 : 5;
    constexpr int NC = NB + (ROLL ? 1 : 0);
    double acc[NC] = {};
    const double eps = 1e-9;
    const bool coarse = !RAW && C.nLevels > 1;
    for (int i = 0; i < W; ++i) {
        double* q = buf + (size_t)i * NC;
        for (int c = 0; c < NC; ++c) { acc[c] += q[c]; q[c] = 0; }
        if constexpr (RAW) {
            float* o = out + (size_t)i * NC;
            for (int c = 0; c < NC; ++c) o[c] = (float)acc[c];
            continue;
        }
        double v[NC];
        const double* s = acc;
        if (coarse) {
            for (int c = 0; c < NC; ++c) v[c] = acc[c];
            addCoarse<NC>(C, i, y, v);
            s = v;
        }
        float* o = out + (size_t)i * 4;
        if (FR) {
            o[0] = s[A::WR] > eps ? (float)(s[A::R] / s[A::WR]) : 0.f;
            o[1] = s[A::WG] > eps ? (float)(s[A::G] / s[A::WG]) : 0.f;
            o[2] = s[A::WB] > eps ? (float)(s[A::B] / s[A::WB]) : 0.f;
            o[3] = s[A::WG] > eps ? (float)(s[A::A] / s[A::WG]) : 0.f;
        } else if (s[A::WG] > eps) {
            const double inv = 1.0 / s[A::WG];
            o[0] = (float)(s[A::R] * inv); o[1] = (float)(s[A::G] * inv);
            o[2] = (float)(s[A::B] * inv); o[3] = (float)(s[A::A] * inv);
        } else {
            o[0] = o[1] = o[2] = o[3] = 0.f;
        }
        if constexpr (ROLL) le[i] = s[A::WG] > eps ? (float)std::max(0.0, s[A::EL] / s[A::WG]) : 0.f;
    }
    std::fill(buf + (size_t)W * NC, buf + (size_t)(W + 2) * NC, 0.0);
}

// ShapeTable::lookup as one packed pair (il, ir): lane-for-lane the same operations.
inline __m128d lookup2(const ShapeTable& T, double u) {
    double f = (u - T.ymin) * T.invStep;
    int i = (int)f;
    if (i < 0) i = 0;
    if (i > T.N - 1) i = T.N - 1;
    f -= i;
    const double* q = &T.I[2 * (size_t)i];
    const __m128d a = _mm_loadu_pd(q), b = _mm_loadu_pd(q + 2);
    return _mm_add_pd(a, _mm_mul_pd(_mm_set1_pd(f), _mm_sub_pd(b, a)));
}

// Band geometry of one (splat, pass, channel, layer) for the bands t = 0..nk-1 (row offsets ka + t):
// h, xl, xr and ok per band, identical to apSpan / catSpan of each band.
struct BandOut { double h[kBlockRows], xl[kBlockRows], xr[kBlockRows]; bool ok[kBlockRows]; };

template <bool CAT>
inline void bandSpans(const Ctx& C, double RxS, double RyS, int ka, int nk, double hx, double hy, BandOut& o) {
    const double inv = 1.0 / RyS;
    double U[kBlockRows + 1];
    for (int t = 0; t <= nk; ++t) U[t] = ((double)(ka + t) - 0.5) * inv;  // == (k - 0.5) * inv, (k + 0.5) * inv
    double du[kBlockRows + 1];
    if constexpr (!CAT) {
        const ShapeTable& T = C.ap;
        __m128d lr[kBlockRows + 1];  // (IL, IR) at each boundary
        for (int t = 0; t <= nk; ++t) {
            // the clamped value a successful band uses at this boundary (both one-sided clamps of
            // apSpan agree whenever du > 1e-12)
            double u = U[t];
            if (u < T.ymin) u = T.ymin;
            if (u > T.ymax) u = T.ymax;
            lr[t] = lookup2(T, u);
        }
        for (int t = 0; t < nk; ++t) {
            double u0 = U[t], u1 = U[t + 1];
            if (u0 < T.ymin) u0 = T.ymin;
            if (u1 > T.ymax) u1 = T.ymax;
            du[t] = u1 - u0;
            o.ok[t] = !(du[t] <= 1e-12);
        }
        const __m128d rx = _mm_set1_pd(RxS);
        for (int t = 0; t < nk; t += 2) {  // two bands per packed division: s = RxS / du
            const bool two = t + 1 < nk;
            const __m128d d2 = _mm_set_pd(two && o.ok[t + 1] ? du[t + 1] : 1.0, o.ok[t] ? du[t] : 1.0);
            const __m128d s2 = _mm_div_pd(rx, d2);
            for (int j = 0; j < 2 && t + j < nk; ++j) {
                const int b = t + j;
                if (!o.ok[b]) continue;
                const double s = j ? _mm_cvtsd_f64(_mm_unpackhi_pd(s2, s2)) : _mm_cvtsd_f64(s2);
                const __m128d x = _mm_mul_pd(_mm_sub_pd(lr[b + 1], lr[b]), _mm_set1_pd(s));
                o.xl[b] = _mm_cvtsd_f64(x);
                o.xr[b] = _mm_cvtsd_f64(_mm_unpackhi_pd(x, x));
                o.h[b] = du[b] * RyS;
                o.ok[b] = o.xr[b] > o.xl[b];
            }
        }
    } else {
        const double lo = std::max(C.ap.ymin - hy, hy + C.circ.ymin);
        const double hi = std::min(C.ap.ymax - hy, hy + C.circ.ymax);
        __m128d a2[kBlockRows + 1], c2[kBlockRows + 1];  // aperture / clip-circle (IL, IR)
        for (int t = 0; t <= nk; ++t) {
            double u = std::max(U[t], lo);
            u = std::min(u, hi);
            a2[t] = lookup2(C.ap, u + hy);
            c2[t] = lookup2(C.circ, u - hy);
        }
        for (int t = 0; t < nk; ++t) {
            const double u0 = std::max(U[t], lo), u1 = std::min(U[t + 1], hi);
            du[t] = u1 - u0;
            o.ok[t] = !(du[t] <= 1e-12);
        }
        const __m128d one = _mm_set1_pd(1.0), hx2 = _mm_set1_pd(hx);
        for (int t = 0; t < nk; t += 2) {  // two bands per packed division: idu = 1 / du
            const bool two = t + 1 < nk;
            const __m128d d2 = _mm_set_pd(two && o.ok[t + 1] ? du[t + 1] : 1.0, o.ok[t] ? du[t] : 1.0);
            const __m128d i2 = _mm_div_pd(one, d2);
            for (int j = 0; j < 2 && t + j < nk; ++j) {
                const int b = t + j;
                if (!o.ok[b]) continue;
                const __m128d idu = j ? _mm_unpackhi_pd(i2, i2) : _mm_unpacklo_pd(i2, i2);
                // (l, r) lanes: aperture span - hx, hx + clip-circle span
                const __m128d ap = _mm_sub_pd(_mm_mul_pd(_mm_sub_pd(a2[b + 1], a2[b]), idu), hx2);
                const __m128d ci = _mm_add_pd(hx2, _mm_mul_pd(_mm_sub_pd(c2[b + 1], c2[b]), idu));
                const double L = std::max(_mm_cvtsd_f64(ap), _mm_cvtsd_f64(ci));
                const double R = std::min(_mm_cvtsd_f64(_mm_unpackhi_pd(ap, ap)), _mm_cvtsd_f64(_mm_unpackhi_pd(ci, ci)));
                if (R <= L) { o.ok[b] = false; continue; }
                o.xl[b] = L * RxS;
                o.xr[b] = R * RxS;
                o.h[b] = du[b] * RyS;
            }
        }
    }
}

// q[i..i+1] += w * (a, b)
inline void add2(double* q, __m128d w2, __m128d ab) { _mm_storeu_pd(q, _mm_add_pd(_mm_loadu_pd(q), _mm_mul_pd(w2, ab))); }

// nb (<= kBlockRows) output rows y0..y0+nb-1 of one level; bufs: nb accumulator rows of (W+2)*NC.
// out / le: row 0 pointers, strides in floats (RAW: NC floats per pixel).
template <bool FR, bool CAT, bool ROLL, bool RAW>
void doBlock(const Ctx& C, const LevelView& V, int y0, int nb, double* bufs, float* out, size_t outStride,
             float* le, size_t leStride, bool clampNeg) {
    using A = AccLayout<FR>;
    constexpr int NB = FR ? 7 : 5;
    constexpr int NC = NB + (ROLL ? 1 : 0);
    const int W = V.W;
    const size_t rowLen = (size_t)(W + 2) * NC;
    const int y1 = y0 + nb - 1;
    const int syA = std::max(0, y0 - V.maxReach), syB = std::min(V.H - 1, y1 + V.maxReach);
    BandOut bo;
    for (int sy = syA; sy <= syB; ++sy) {
        if (cancelRequested()) return;  // v2.3: abort latency (the whole stage result is discarded)
        const int kmin = y0 - sy, kmax = y1 - sy;  // row offsets of the block's rows from this source row
        const int dmin = (kmin <= 0 && kmax >= 0) ? 0 : std::min(std::abs(kmin), std::abs(kmax));
        const Splat* const base = V.splats + (size_t)sy * W;
        const Splat* const end = base + W;
        const CatInfo* const cbase = CAT ? V.cats + (size_t)sy * W : nullptr;
        for (const Splat* sp = base; sp < end && sp->reach >= dmin; ++sp) {
            const double c0 = sp->c[0], c1 = sp->c[1], c2 = sp->c[2], c3 = sp->c[3];
            const double ce = ROLL ? (double)sp->eL : 0.0;
            if (sp->R < 0.f) {  // identity splat: reach 0, only the row with k == 0 (inside: dmin == 0)
                double* q = bufs + (size_t)(0 - kmin) * rowLen + (size_t)sp->x * NC;
                double* q1 = q + NC;
                q[A::R] += c0; q[A::G] += c1; q[A::B] += c2; q[A::A] += c3;
                q1[A::R] -= c0; q1[A::G] -= c1; q1[A::B] -= c2; q1[A::A] -= c3;
                q[A::WR] += 1; q1[A::WR] -= 1;
                if (FR) { q[A::WG] += 1; q[A::WB] += 1; q1[A::WG] -= 1; q1[A::WB] -= 1; }
                if constexpr (ROLL) { q[A::EL] += ce; q1[A::EL] -= ce; }
                continue;
            }
            // packed colour pairs in accumulator order
            const __m128d pR = _mm_set_pd(FR ? 1.0 : c1, c0);     // FR: (R, WR)       else (R, G)
            const __m128d pB = _mm_set_pd(FR ? 1.0 : c3, c2);     // FR: (B, WB)       else (B, A)
            const __m128d pG = _mm_set_pd(c3, c1);                 // FR: (G, A)
            const __m128d pW = _mm_set_pd(ce, 1.0);                // (W or WG, eL)
            const int ka = std::max(kmin, -sp->reach), kb = std::min(kmax, sp->reach);
            const int nk = kb - ka + 1;
            double* const rb0 = bufs + (size_t)(ka - kmin) * rowLen;  // buffer row of band t = 0
            const CatInfo* cc = CAT ? cbase + (sp - base) : nullptr;
            const double hx = CAT ? (double)cc->ox : 0.0, hy = CAT ? (double)cc->oy : 0.0;
            const double R = sp->R;
            const double cx = sp->x + 0.5;
            const double mass = sp->m;
            const int side = sp->sgn > 0.f ? 1 : 0;
            const double* ks = C.ks[side];
            bool alive[kBlockRows];
            for (int pass = 0; pass < 2; ++pass) {
                const double wgt = pass ? (double)sp->wb : 1.0 - (double)sp->wb;
                if (pass && wgt <= 0) break;
                const Profile& P = C.prof[side][sp->tier + pass];
                const bool share = FR && P.share;
                double invDs[3] = {0, 0, 0};
                for (int ch = 0; ch < (FR ? 3 : 1); ++ch) {
                    const double Rx = R * C.ax * ks[ch], Ry = R * C.ay * ks[ch];
                    const double invD = CAT ? (double)cc->invD[pass][ch]
                                      : share ? wgt / (C.ap.area * (R * C.ax) * (R * C.ay) * (P.s2out * ks[ch] * ks[ch] + P.s2in))
                                              : wgt / (C.ap.area * P.s2 * Rx * Ry);
                    invDs[ch] = invD;
                    const int nlCh = share ? P.shareFrom : P.nl;
                    int nAlive = nk;
                    for (int t = 0; t < nk; ++t) alive[t] = true;
                    for (int l = 0; l < nlCh && nAlive > 0; ++l) {
                        const double RxS = Rx * P.ls[l], RyS = Ry * P.ls[l];
                        bandSpans<CAT>(C, RxS, RyS, ka, nk, hx, hy, bo);
                        const double lbInvD = P.lb[l] * invD;
                        for (int t = 0; t < nk; ++t) {
                            if constexpr (!CAT) { if (!alive[t]) continue; }
                            bool ok = bo.ok[t];
                            double L = 0, Rr = 0;
                            if (ok) {
                                L = cx + bo.xl[t]; Rr = cx + bo.xr[t];
                                if (L < 0) L = 0;
                                if (Rr > W) Rr = W;
                                ok = Rr > L;
                            }
                            if (!ok) { if (!CAT) { alive[t] = false; --nAlive; } continue; }  // inner layers are subsets
                            const double w = lbInvD * bo.h[t] * mass;
                            const int il = (int)L, ir = (int)Rr;
                            const double fl = L - il, fr = Rr - ir;
                            const double ws[4] = {w - w * fl, w * fl, w * fr - w, -w * fr};
                            double* const buf = rb0 + (size_t)t * rowLen;
                            double* const qs[4] = {buf + (size_t)il * NC, buf + (size_t)il * NC + NC,
                                                   buf + (size_t)ir * NC, buf + (size_t)ir * NC + NC};
                            for (int kk = 0; kk < 4; ++kk) {  // in doRow's order q0, q1, q2, q3 (they may coincide)
                                double* q = qs[kk];
                                const __m128d w2 = _mm_set1_pd(ws[kk]);
                                if (!FR) {
                                    add2(q + A::R, w2, pR);
                                    add2(q + A::B, w2, pB);
                                    if constexpr (ROLL) add2(q + A::WG, w2, pW); else q[A::WG] += ws[kk];
                                } else if (ch == 0) {
                                    add2(q + A::R, w2, pR);
                                } else if (ch == 1) {  // G kernel also carries alpha (and eL)
                                    add2(q + A::G, w2, pG);
                                    if constexpr (ROLL) add2(q + A::WG, w2, pW); else q[A::WG] += ws[kk];
                                } else {
                                    add2(q + A::B, w2, pB);
                                }
                            }
                        }
                    }
                }
                if constexpr (FR) {
                    if (share) {
                        // inner layers once, neutral geometry, each channel's own normaliser (doRow)
                        const double Rx = R * C.ax, Ry = R * C.ay;
                        int nAlive = nk;
                        for (int t = 0; t < nk; ++t) alive[t] = true;
                        for (int l = P.shareFrom; l < P.nl && nAlive > 0; ++l) {
                            const double RxS = Rx * P.ls[l], RyS = Ry * P.ls[l];
                            bandSpans<CAT>(C, RxS, RyS, ka, nk, hx, hy, bo);
                            for (int t = 0; t < nk; ++t) {
                                if constexpr (!CAT) { if (!alive[t]) continue; }
                                bool ok = bo.ok[t];
                                double L = 0, Rr = 0;
                                if (ok) {
                                    L = cx + bo.xl[t]; Rr = cx + bo.xr[t];
                                    if (L < 0) L = 0;
                                    if (Rr > W) Rr = W;
                                    ok = Rr > L;
                                }
                                if (!ok) { if (!CAT) { alive[t] = false; --nAlive; } continue; }
                                const double wl = P.lb[l] * bo.h[t] * mass;
                                const int il = (int)L, ir = (int)Rr;
                                const double fl = L - il, fr = Rr - ir;
                                const double gs[4] = {1.0 - fl, fl, fr - 1.0, -fr};
                                const double wr = wl * invDs[0], wg = wl * invDs[1], wbl = wl * invDs[2];
                                double* const buf = rb0 + (size_t)t * rowLen;
                                double* const qs[4] = {buf + (size_t)il * NC, buf + (size_t)il * NC + NC,
                                                       buf + (size_t)ir * NC, buf + (size_t)ir * NC + NC};
                                for (int kk = 0; kk < 4; ++kk) {
                                    double* q = qs[kk];
                                    const __m128d r2 = _mm_set1_pd(wr * gs[kk]), g2 = _mm_set1_pd(wg * gs[kk]),
                                                  b2 = _mm_set1_pd(wbl * gs[kk]);
                                    add2(q + A::R, r2, pR);
                                    add2(q + A::G, g2, pG);
                                    if constexpr (ROLL) add2(q + A::WG, g2, pW); else q[A::WG] += wg * gs[kk];
                                    add2(q + A::B, b2, pB);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int r = 0; r < nb; ++r) {
        float* o = out + (size_t)r * outStride;
        resolveRow<FR, ROLL, RAW>(C, W, y0 + r, bufs + (size_t)r * rowLen, o, ROLL && le ? le + (size_t)r * leStride : nullptr);
        if constexpr (!RAW) {
            if (clampNeg)
                for (int i = 0; i < 4 * W; ++i) o[i] = std::max(0.f, o[i]);
        }
    }
}

template <bool RAW>
void dispatchBlock(const Ctx& C, const LevelView& V, int y0, int nb, double* b, float* o, size_t os, float* le, size_t ls,
                   bool clampNeg, bool FR, bool CAT, bool RL) {
    if (FR) {
        if (CAT) { if (RL) doBlock<true, true, true, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); else doBlock<true, true, false, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); }
        else     { if (RL) doBlock<true, false, true, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); else doBlock<true, false, false, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); }
    } else {
        if (CAT) { if (RL) doBlock<false, true, true, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); else doBlock<false, true, false, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); }
        else     { if (RL) doBlock<false, false, true, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); else doBlock<false, false, false, RAW>(C, V, y0, nb, b, o, os, le, ls, clampNeg); }
    }
}

// Coarse radius (px) from which a splat may be rendered on a coarser grid. Exact mode: 34 (v1.2).
// Alpha (fastBlur): as low as 20, but a narrow radial feature of the kernel (soft edge, rim, onion
// ring) must stay >= ~1.2 coarse px wide, and hard edges keep 34 (bilinear upsampling would soften
// them by ~1 coarse px).
double levelBaseFor(const Params& p) {
    if (!p.fastBlur) return kLevelBase;
    double base = 12.0;
    const double soft = clampd(p.edgeSoftness, 0.0, 1.0);
    if (soft < 0.03 || p.obstruction > 1e-3) base = std::max(base, kHardEdgeFastBase);  // hard edge
    else base = std::max(base, 0.55 / soft);                                            // soft roll-off
    const double rim = std::max(std::fabs(p.edgeRing + p.sphericalAberration), std::fabs(p.edgeRing - p.sphericalAberration));
    if (rim > 0.05)  // rim width, weighted by how visible the rim is (a faint rim may soften a little)
        base = std::max(base, 1.2 / clampd(p.ringWidth, 0.02, 0.98) * std::min(1.0, rim / 0.3));
    if (p.onionRings > 0.05) base = std::max(base, 2.4 * clampd(p.onionCount, 1.0, 10.0));  // ring period
    return std::min(base, kLevelBase);
}

// number of resolution levels for a maximum radius (1 = fine only)
int levelCount(const Params& p) {
    if (p.fullResolution) return 1;
    const double base = levelBaseFor(p);
    const double R = std::max(0.0, (double)p.maxRadius);
    if (R <= base * std::pow(2.0, 0.7)) return 1;
    const double lam = std::log2(R / base);
    const int lmax = (int)lam;
    return std::min(kMaxLevels, lmax + 1 + (lam - lmax > 0.7 ? 1 : 0));
}

bool overlaps(const void* a0, size_t an, const void* b0, size_t bn) {
    const char* a = (const char*)a0;
    const char* b = (const char*)b0;
    return a < b + bn && b < a + an;
}

bool finite(float v) { return std::isfinite(v); }

void frameOf(const Params& p, int w, int h, double& cx, double& cy, double& halfDiag) {
    double fl = 0, ft = 0, fw = w, fh = h;
    if (p.frameWidth > 0 && p.frameHeight > 0) { fl = p.frameLeft; ft = p.frameTop; fw = p.frameWidth; fh = p.frameHeight; }
    cx = fl + 0.5 * fw;
    cy = ft + 0.5 * fh;
    // v2.2: in square pixels (units: buffer px vertically); pixelAspect 1 = the v2.1 formula
    const double fws = fw * aspectOf(p);
    halfDiag = std::max(1.0, 0.5 * std::sqrt(fws * fws + fh * fh));
}

bool paramsValid(const Params& p) {
    if (!finite(p.maxRadius) || p.maxRadius < 0 || p.maxRadius > 4000) return false;
    const float fs[] = {p.focus, p.mapScale, p.mapGamma, p.bladeCurvature, p.rotationDeg, p.anamorphic,
                        p.catEye, p.edgeRing, p.ringWidth, p.chromaFringe, p.highlightThreshold, p.highlightGain,
                        p.sphericalAberration, p.cornerRounding, p.edgeSoftness, p.apodization, p.coreWeight,
                        p.obstruction, p.onionRings, p.onionCount, p.vignetting, p.glow, p.glowSize,
                        p.highlightBoost, p.highlightKnee, p.highlightCurve, p.highlightSelectivity,
                        p.detailSize, p.highlightSaturation, p.highlightRolloff, p.fringeTint,
                        p.whiteProtection, p.dotsOnly, p.detailMin,
                        p.halation, p.halationSize, p.halationThreshold, p.halationSoftness, p.halationRim,
                        p.halationRelative, p.laContrastScale, p.laLineProtect, p.laSharpness, p.laBrightColor,
                        p.laBlur, p.laComplement, p.laMapSoftness, p.laShift[0], p.laShift[1], p.laShift[2],
                        p.laPreserve, p.grain, p.grainSize, p.grainSoftness, p.grainColor, p.grainTone,
                        p.grainScaleX, p.grainScaleY,
                        p.halationTint[0], p.halationTint[1], p.halationTint[2],
                        p.lumAberration, p.laWidth, p.laThreshold, p.laCurve, p.laHueShift,
                        p.laTint[0], p.laTint[1], p.laTint[2],
                        p.frameLeft, p.frameTop, p.frameWidth, p.frameHeight};
    for (float v : fs) if (!finite(v)) return false;
    return p.mapGamma > 0;
}

int threadCount(const Params& p, int h) {
    int nth = p.threads > 0 ? p.threads : (int)std::thread::hardware_concurrency();
    return std::max(1, std::min(nth, std::min(h, 256)));
}

} // namespace

float blurAmount(float mapValue, const Params& p) {
    double sgn;
    return (float)amountOf(mapValue, p, sgn);
}

int prepareMargin(const Params& p) {
    if (!(p.highlightBoost > 0)) return 0;
    double cx, cy, hd;
    frameOf(p, 1, 1, cx, cy, hd);
    // every highlight term at a pixel depends on the brightness within the largest of these windows
    double m = std::ceil(kAdaptRadius * hd) + 2;
    const double D = clampd(p.detailSize, 1.0, 400.0);
    if (p.highlightSelectivity > 0) {
        m = std::max(m, 2.0 * std::ceil(D) + 3);
        if (p.dotsOnly > 0) m = std::max(m, 2.0 * std::ceil(D) + 3);
        // v2.2: openings up to radius ceil((3 D) / 2) + 1 (tapered size window)
        if (p.renderVersion >= 2) m = std::max(m, 2.0 * (std::ceil(1.5 * D) + 1) + 3);
    }
    if (p.renderVersion >= 2) m += 3;  // peak (3x3) / background (5x5) of the coverage-linear energy
    if (p.whiteProtection > 0) {
        const double rw = kWhiteRadius * 2.0 * hd;
        const double f = std::max(1.0, std::round(rw / 16.0));
        m = std::max(m, 2.0 * std::ceil(0.5 * rw) + 3.0 * f + 2);
    }
    return (int)std::min(100000.0, m + 2);
}

int blurMargin(const Params& p) {
    const double a = clampd(p.anamorphic, 0.1, 10.0);
    const double axis = std::max(std::sqrt(a), 1.0 / std::sqrt(a));
    const double chroma = 1.0 + 0.2 * std::fabs(clampd(p.chromaFringe, -1.0, 1.0));
    double m = std::max(0.0, (double)p.maxRadius) * axis * chroma + 2;
    const int nl = levelCount(p);  // coarse blocks + bilinear upsampling reach a little further
    if (nl > 1) m += 2.0 * (1 << (nl - 1)) + 2;
    return (int)std::min(100000.0, std::ceil(m));
}

int postMargin(const Params& p) {
    double cx, cy, hd;
    frameOf(p, 1, 1, cx, cy, hd);
    double m = 0;
    auto field = [](double sigma) {
        const double f = sigma > 6.0 ? std::max(1.0, std::floor(sigma / 3.0)) : 1.0;
        return 3.5 * sigma + 2.0 * f + 2;
    };
    if (p.glow > 0) {
        const double sigma = clampd(p.glowSize, 0.005, 0.25) * 2.0 * hd;
        m += 3.5 * sigma + 2.0 * std::max(1.0, sigma / 3.0) + 2;
    }
    if (p.halation > 0)  // scene level, then the source blur
        m += field(std::max(1.0, kHalScene * 2.0 * hd)) + field(std::max(1.0, clampd(p.halationSize, 0.001, 0.2) * 2.0 * hd));
    if (p.lumAberration > 0) {
        if (p.laMode == 4) {  // closing, softened map, gradient, preserve lookup, smear
            const DriftGeom g = driftGeom(p, hd);
            m += morphReach(g.rl) + 4 + field(g.sm) + 2 + std::ceil(2.0 * g.sm) + 1 + g.rb;
        } else if (p.laMode == 3) {  // closing, region max/min + smoothing, rim spread, bright-side colour
            const RimGeom g = rimGeom(p, hd);
            m += 2.0 * morphReach(g.rl) + 8 + g.rc + g.rn + field(g.sReg) + std::ceil(3.0 * std::max(g.sd, g.sb)) + 1 + field(g.sReg);
            if (p.laBlur > 1e-6)  // fringe: bright-colour field + per-channel defocus
                m += field(g.sReg) + field(g.sf);
        } else {
            m += field(std::max(0.6, clampd(p.laWidth, 0.0005, 0.1) * 2.0 * hd));
        }
    }
    return (int)std::min(100000.0, std::ceil(m));
}

int inputMargin(const Params& p) {
    return (int)std::min(100000, prepareMargin(p) + blurMargin(p) + postMargin(p));
}

// v2.2: overloads for hosts that do not set the frame rect: the w x h buffer is the frame.
static Params withFrame(const Params& p, int w, int h) {
    Params q = p;
    if (!(q.frameWidth > 0) || !(q.frameHeight > 0)) {
        q.frameLeft = 0; q.frameTop = 0;
        q.frameWidth = (float)std::max(1, w); q.frameHeight = (float)std::max(1, h);
    }
    return q;
}
int prepareMargin(const Params& p, int w, int h) { return prepareMargin(withFrame(p, w, h)); }
int blurMargin(const Params& p, int w, int h) { return blurMargin(withFrame(p, w, h)); }
int postMargin(const Params& p, int w, int h) { return postMargin(withFrame(p, w, h)); }
int inputMargin(const Params& p, int w, int h) { return inputMargin(withFrame(p, w, h)); }
bool postNeedsEL(const Params& p, const Prepared& prep) {
    return p.highlightRolloff > 1e-4 && prep.rollRelevant;
}

Status prepareHighlights(const float* src, int w, int h, int srcStride, const float* map, int mapStride,
                         const Params& p, const PrepareOptions& opt, Prepared& P, AbortFn abortFn, void* user) {
    if (!src || w <= 0 || h <= 0 || (long long)w * 4 > srcStride) return Status::BadArgs;
    if (map && mapStride < w) return Status::BadArgs;
    if (opt.highlightLayer && opt.highlightLayerStride < w) return Status::BadArgs;
    if (!paramsValid(p)) return Status::BadArgs;
    try {
        double fcx, fcy, hd;
        frameOf(p, w, h, fcx, fcy, hd);
        const int nth = threadCount(p, h);
        PoolScope pool(nth);
        LC_PERF_BEGIN();
        HlMaps M;
        if (p.highlightBoost > 0 && p.maxRadius > 0) {
            if (!computeHighlights(w, h, hd, p, src, srcStride, opt, M, nth, abortFn, user)) return Status::Aborted;
        }
        LC_PERF_MARK(maps);
        const size_t N = (size_t)w * h;
        P.w = w; P.h = h;
        P.rgba.resize(N * 4);
        P.radius.resize(N);
        P.side.resize(N);
        P.eL.resize(N);
        P.extra.assign(opt.keepExtra ? N * 3 : 0, 0.f);
        P.rollRelevant = p.highlightGain > 0 || !M.E.empty();
        LC_PERF_MARK(alloc);
        const bool ok = parallelFor(h, nth, 4, [&](int sy, int) {
            prepareRow(p, M, src, srcStride, map, mapStride, sy, w, opt.keepExtra, P);
        }, abortFn, user);
        LC_PERF_MARK(rows);
        LC_PERF_DUMP("prep");
        return ok ? Status::Ok : Status::Aborted;
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;   // v2.2: nothing escapes a stage (system_error, length_error, ...)
    }
}

Prepared cropPrepared(const Prepared& prep, int x0, int y0, int w, int h) {
    Prepared q;
    const int X0 = std::max(0, x0), Y0 = std::max(0, y0);
    const int X1 = std::min(prep.w, x0 + std::max(0, w)), Y1 = std::min(prep.h, y0 + std::max(0, h));
    if (X1 <= X0 || Y1 <= Y0) return q;
    const size_t N = (size_t)prep.w * prep.h;
    if (prep.rgba.size() != N * 4 || prep.radius.size() != N || prep.side.size() != N || prep.eL.size() != N) return q;
    q.w = X1 - X0; q.h = Y1 - Y0;
    q.rollRelevant = prep.rollRelevant;
    const size_t M = (size_t)q.w * q.h;
    q.rgba.resize(M * 4); q.radius.resize(M); q.side.resize(M); q.eL.resize(M);
    const bool ex = prep.extra.size() == N * 3;
    if (ex) q.extra.resize(M * 3);
    for (int y = 0; y < q.h; ++y) {
        const size_t s = (size_t)(Y0 + y) * prep.w + X0, d = (size_t)y * q.w;
        std::copy(prep.rgba.begin() + s * 4, prep.rgba.begin() + (s + q.w) * 4, q.rgba.begin() + d * 4);
        std::copy(prep.radius.begin() + s, prep.radius.begin() + s + q.w, q.radius.begin() + d);
        std::copy(prep.side.begin() + s, prep.side.begin() + s + q.w, q.side.begin() + d);
        std::copy(prep.eL.begin() + s, prep.eL.begin() + s + q.w, q.eL.begin() + d);
        if (ex) std::copy(prep.extra.begin() + s * 3, prep.extra.begin() + (s + q.w) * 3, q.extra.begin() + d * 3);
    }
    return q;
}

Prepared highlightsOnly(const Prepared& prep) {
    Prepared q;
    q.w = prep.w; q.h = prep.h;
    const size_t N = (size_t)prep.w * prep.h;
    q.rgba.assign(N * 4, 0.f);
    if (prep.extra.size() == N * 3)
        for (size_t i = 0; i < N; ++i) {
            q.rgba[i * 4] = prep.extra[i * 3]; q.rgba[i * 4 + 1] = prep.extra[i * 3 + 1];
            q.rgba[i * 4 + 2] = prep.extra[i * 3 + 2]; q.rgba[i * 4 + 3] = 1.f;
        }
    q.radius = prep.radius;
    q.side = prep.side;
    q.eL.assign(N, 0.f);
    return q;
}

// v2.3 (GPU integration): the blur stage's setup (shape tables, layer profiles, cat-eye table, levels),
// shared by blurAlpha and describeBlur. Moved here verbatim from blurAlpha: same statements, same
// order, so the CPU result is bit-identical. Returns Ok or Aborted (cat-eye table pass).
static Status buildCtx(const Params& p, int w, int h, Ctx& C, double& smaxG, AbortFn abortFn, void* user) {
    C.W = w; C.H = h;
    frameOf(p, w, h, C.fcx, C.fcy, C.halfDiag);
    C.pa = aspectOf(p);
    const double an = clampd(p.anamorphic, 0.1, 10.0);
    C.ax = 1.0 / std::sqrt(an);
    C.ay = std::sqrt(an);
    const double curv = clampd(p.bladeCurvature, 0.0, 1.0);
    C.ap.build(makeAperture(p.blades, curv, p.rotationDeg * kPi / 180.0, clampd(p.cornerRounding, 0.0, 1.0)), 4096);
    C.circ.build(makeAperture(0, 1, 0), 4096);
    C.xext = std::max(-C.ap.xmin, C.ap.xmax);
    C.yext = std::max(-C.ap.ymin, C.ap.ymax);

    // v2: negative values swap the fringe colours (blue outside behind focus).
    // v1.2: fringeTint 0 = red/blue (v1), 1 = green behind / magenta in front (classic LoCA).
    const double fr = clampd(p.chromaFringe, -1.0, 1.0);
    if (std::fabs(fr) > 5e-4) {
        const double kf = 0.2 * fr;
        const double tt = clampd(p.fringeTint, 0.0, 1.0);
        C.nk = 3;
        if (tt <= 0) {
            C.ks[1][0] = 1 + kf; C.ks[1][1] = 1; C.ks[1][2] = 1 - kf;  // behind focus: red outside
            C.ks[0][0] = 1 - kf; C.ks[0][1] = 1; C.ks[0][2] = 1 + kf;  // in front: blue outside
        } else {
            const double d[3] = {1.0 - 1.5 * tt, tt, -1.0 + 0.5 * tt};
            for (int ch = 0; ch < 3; ++ch) { C.ks[1][ch] = 1 + kf * d[ch]; C.ks[0][ch] = 1 - kf * d[ch]; }
        }
    }
    // radial intensity profile as nested scaled copies of the shape
    const double e = clampd(p.edgeRing, -1.0, 1.0);
    const double sa = clampd(p.sphericalAberration, -1.0, 1.0);
    const double rw = clampd(p.ringWidth, 0.02, 0.98);
    ProfileSpec spec;
    spec.rw = rw;
    spec.soft = clampd(p.edgeSoftness, 0.0, 1.0);
    spec.apod = clampd(p.apodization, 0.0, 1.0);
    spec.core = clampd(p.coreWeight, -1.0, 1.0);
    spec.obstr = clampd(p.obstruction, 0.0, 0.9);
    spec.onion = clampd(p.onionRings, 0.0, 1.0);
    spec.onionN = clampd(p.onionCount, 1.0, 10.0);
    const bool general = spec.soft > 1e-4 || spec.apod > 1e-4 || std::fabs(spec.core) > 1e-4 ||
                         spec.obstr > 1e-4 || spec.onion > 1e-4;
    for (int side = 0; side < 2; ++side) {
        spec.e = clampd(side ? e + sa : e - sa, -1.0, 1.0);  // side 1 = behind focus
        if (!general) {
            legacyProfile(C.prof[side][0], spec.e, rw);
        } else {
            for (int t = 0; t < kTiers; ++t)
                generalProfile(C.prof[side][t], spec, t == 0 ? 1.0 : kTierBase * std::pow(2.0, t), kTierLayers[t]);
        }
    }
    C.nTiers = general ? kTiers : 1;
    C.clampNeg = general;
    C.levelBase = levelBaseFor(p);
    // alpha: shared inner layers for the colour fringe where every channel's outer edge stays
    // outside the first inner layer (kernel stays non-negative, normalisation exact per channel)
    C.fringeShare = p.fastBlur && C.nk == 3;
    if (C.fringeShare) {
        double kmin = 1.0;
        for (int s = 0; s < 2; ++s) for (int ch = 0; ch < 3; ++ch) kmin = std::min(kmin, C.ks[s][ch]);
        for (int s = 0; s < 2; ++s)
            for (int tt = 0; tt < kTiers; ++tt) {
                Profile& pf = C.prof[s][tt];
                // first layer whose gap to the previous one is wider than the fringe scale:
                // everything outside it is drawn per channel (the fringe zone), the rest once
                int from = pf.nl;
                for (int l = 1; l < pf.nl; ++l)
                    if (pf.ls[l] < kmin * pf.ls[l - 1] - 1e-6) { from = l; break; }
                pf.share = from < pf.nl;
                pf.shareFrom = from;
                pf.s2out = 0;
                for (int l = 0; l < from; ++l) pf.s2out += pf.lb[l] * pf.ls[l] * pf.ls[l];
                pf.s2in = pf.s2 - pf.s2out;
            }
    }
    const double ce = clampd(p.catEye, 0.0, 1.0);
    C.cat = ce > 1e-3;
    C.catK = 1.2 * ce;
    // the recentered intersection stays inside the unit circle, but may poke
    // outside the (shifted) aperture's own extents
    if (C.cat) { C.xext = std::max(C.xext, 1.0); C.yext = std::max(C.yext, 1.0); }
    if (C.cat && (general || p.fastBlur)) {  // alpha: tabulated normaliser for every profile
        C.catN = 65;
        C.catLim = 0.75 * C.catK + 1e-3;
        C.catInv = (C.catN - 1) / (2 * C.catLim);
        C.catA.assign((size_t)C.catN * C.catN, 0.0);
        const int nb = 1024;
        // v2.3: cells in parallel (4.3 M span evaluations, ~35 ms on one thread); each cell's sum is
        // computed exactly as before, so the table is identical
        const int nthA = threadCount(p, h);
        PoolScope poolA(nthA);
        if (!parallelFor(C.catN * C.catN, nthA, 16, [&](int cell, int) {
            const int j = cell / C.catN, i = cell % C.catN;
            const double ox = -C.catLim + i / C.catInv, oy = -C.catLim + j / C.catInv;
            double sum = 0;
            for (int k = 0; k < nb; ++k) {
                double hh, xl, xr;
                const double a = -1.1 + 2.2 * k / nb, b = -1.1 + 2.2 * (k + 1) / nb;
                if (catSpan(C, 1.0, 1.0, a, b, ox, oy, hh, xl, xr)) sum += hh * (xr - xl);
            }
            C.catA[(size_t)j * C.catN + i] = sum;
        }, abortFn, user)) return Status::Aborted;
    }
    smaxG = 1;
    for (int i = 0; i < 2; ++i) for (int ch = 0; ch < C.nk; ++ch) smaxG = std::max(smaxG, C.ks[i][ch]);
    C.maxReach = std::max(0, (int)std::ceil(p.maxRadius * C.ay * smaxG * C.yext - 0.5));
    C.nLevels = levelCount(p);
    if (C.nLevels > 1) {
        C.levels.resize(C.nLevels);
        for (int l = 1; l < C.nLevels; ++l) {
            Ctx::Level& Lv = C.levels[l];
            Lv.f = 1 << l;
            Lv.gox = gridPhase(p, true, Lv.f);
            Lv.goy = gridPhase(p, false, Lv.f);
            Lv.W = (w + Lv.gox + Lv.f - 1) / Lv.f;
            Lv.H = (h + Lv.goy + Lv.f - 1) / Lv.f;
            Lv.maxReach = std::max(0, (int)std::ceil(p.maxRadius / Lv.f * C.ay * smaxG * C.yext - 0.5));
        }
    }
    return Status::Ok;
}

// output rows [rb, re) the blur stage computes (Params.outRowBegin/End grown by the post margin)
static void blurRowRange(const Params& p, int w, int h, int& rb, int& re) {
    rb = 0;
    re = h;
    if (p.outRowEnd > p.outRowBegin) {
        const long long pm = postMargin(p, w, h);
        rb = (int)std::min<long long>(h, std::max<long long>(0, (long long)p.outRowBegin - pm));
        re = (int)std::min<long long>(h, std::max<long long>(rb, (long long)p.outRowEnd + pm));
    }
}

// v2.3 (GPU integration): the blur stage's setup for an external blur implementation (gpu\obgpu).
// The same buildCtx / blurRowRange code as blurAlpha, so the description cannot drift from the CPU path.
Status describeBlur(const Params& p, int w, int h, BlurSetup& o, AbortFn abortFn, void* user) {
    if (w <= 0 || h <= 0 || !paramsValid(p)) return Status::BadArgs;
    try {
        Ctx C;
        double smaxG = 1;
        const Status st = buildCtx(p, w, h, C, smaxG, abortFn, user);
        if (st != Status::Ok) return st;
        o = BlurSetup();
        o.w = w; o.h = h;
        o.nk = C.nk; o.nTiers = C.nTiers; o.nLevels = C.nLevels;
        o.cat = C.cat; o.catN = C.catN; o.clampNeg = C.clampNeg; o.fringeShare = C.fringeShare;
        o.maxReach = C.maxReach;
        blurRowRange(p, w, h, o.rowBegin, o.rowEnd);
        o.ax = C.ax; o.ay = C.ay; o.xext = C.xext; o.yext = C.yext;
        o.catK = C.catK; o.catLim = C.catLim; o.catInv = C.catInv; o.levelBase = C.levelBase;
        o.fcx = C.fcx; o.fcy = C.fcy; o.halfDiag = C.halfDiag; o.pa = C.pa;
        o.maxRadius = std::max(0.0, (double)p.maxRadius); o.smaxG = smaxG;
        for (int s = 0; s < 2; ++s)
            for (int ch = 0; ch < 3; ++ch) o.ks[s][ch] = C.ks[s][ch];
        static_assert(BlurSetup::kMaxLayers == kMaxLayers && BlurSetup::kTiers == kTiers, "BlurSetup layout");
        for (int s = 0; s < 2; ++s)
            for (int tt = 0; tt < kTiers; ++tt) {
                const Profile& P = C.prof[s][tt];
                BlurSetup::Layers& L = o.prof[s][tt];
                L.nl = P.nl; L.share = P.share; L.shareFrom = P.shareFrom;
                L.s2 = P.s2; L.s2out = P.s2out; L.s2in = P.s2in;
                for (int l = 0; l < kMaxLayers; ++l) { L.ls[l] = P.ls[l]; L.lb[l] = P.lb[l]; }
            }
        auto shape = [](const ShapeTable& T, BlurSetup::Shape& S) {
            S.n = T.N; S.ymin = T.ymin; S.ymax = T.ymax; S.invStep = T.invStep; S.area = T.area; S.table = T.I;
        };
        shape(C.ap, o.ap);
        shape(C.circ, o.circ);
        o.catA = C.catA;
        o.levels.assign((size_t)std::max(1, C.nLevels), BlurSetup::Level());
        for (int l = 1; l < C.nLevels; ++l) {
            const Ctx::Level& Lv = C.levels[l];
            BlurSetup::Level& q = o.levels[l];
            q.f = Lv.f; q.w = Lv.W; q.h = Lv.H; q.gox = Lv.gox; q.goy = Lv.goy; q.maxReach = Lv.maxReach;
            // coarse rows the fine rows [rowBegin, rowEnd) interpolate from (blurAlpha's coarseRow rule)
            auto coarseRow = [&](int y, bool upper) {
                double wv = (y + 0.5 + Lv.goy) / (double)Lv.f - 0.5;
                wv = std::min((double)Lv.H - 1, std::max(0.0, wv));
                const int y0c = (int)wv;
                return upper ? std::min(y0c + 1, Lv.H - 1) : y0c;
            };
            if (o.rowEnd > o.rowBegin) { q.rowLo = coarseRow(o.rowBegin, false); q.rowHi = coarseRow(o.rowEnd - 1, true) + 1; }
        }
        return Status::Ok;
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;
    }
}
Status blurAlpha(const Prepared& P, const Params& p, float* dst, int dstStride, float* eLOut,
                 AbortFn abortFn, void* user) {
    const int w = P.w, h = P.h;
    if (!dst || w <= 0 || h <= 0 || (long long)w * 4 > dstStride) return Status::BadArgs;
    if (P.rgba.size() != (size_t)w * h * 4 || P.radius.size() != (size_t)w * h ||
        P.side.size() != (size_t)w * h || P.eL.size() != (size_t)w * h) return Status::BadArgs;  // v2.2: all sizes
    if (!paramsValid(p)) return Status::BadArgs;
    const bool needEL = postNeedsEL(p, P);
    if (needEL && !eLOut) return Status::BadArgs;
    try {
        LC_PERF_BEGIN();
        Ctx C;
        double smaxG = 1;
        {
            const Status sst = buildCtx(p, w, h, C, smaxG, abortFn, user);
            if (sst != Status::Ok) return sst;
        }
        const int nth = threadCount(p, h);
        PoolScope pool(nth);
        LC_PERF_MARK(setup);

        C.splats.resize((size_t)w * h);
        if (C.cat) C.cats.resize((size_t)w * h);
        if (C.nLevels > 1) C.pix.resize((size_t)w * h);
        LC_PERF_MARK(alloc);

        struct Tmp { std::vector<Splat> s; std::vector<CatInfo> c; std::vector<int> cnt; };
        std::vector<Tmp> tmp(nth);
        bool ok = parallelFor(h, nth, 4, [&](int sy, int tid) {
            prepRow(C, P, sy, tmp[tid].s, tmp[tid].c, tmp[tid].cnt);
        }, abortFn, user);
        if (!ok) return Status::Aborted;
        LC_PERF_MARK(splats);
        for (int l = 1; l < C.nLevels; ++l) {
            Ctx::Level& Lv = C.levels[l];
            Lv.splats.resize((size_t)Lv.W * Lv.H);
            if (C.cat) Lv.cats.resize((size_t)Lv.W * Lv.H);
            ok = parallelFor(Lv.H, nth, 2, [&](int by, int tid) {
                prepLevelRow(C, l, by, tmp[tid].s, tmp[tid].c, tmp[tid].cnt);
            }, abortFn, user);
            if (!ok) return Status::Aborted;
        }
        C.pix.clear(); C.pix.shrink_to_fit();
        tmp.clear();
        LC_PERF_MARK(levels);

        const int NC = (C.nk == 3 ? 7 : 5) + (needEL ? 1 : 0);
        // v2.3: blocks of kBlockRows output rows per task (doBlock; bit-identical to doRow per row)
        const int BR = kBlockRows;
        std::vector<std::vector<double>> bufs(nth, std::vector<double>((size_t)BR * (w + 2) * NC, 0.0));
        const bool FRm = C.nk == 3, CATm = C.cat, RL = needEL;
        // v2.3: output rows [rb, re) (Params.outRowBegin/End grown by the post margin; default all)
        int rb = 0, re = h;
        blurRowRange(p, w, h, rb, re);
        // coarse levels first: raw accumulators, upsampled by the fine pass
        for (int l = 1; l < C.nLevels && re > rb; ++l) {
            Ctx::Level& Lv = C.levels[l];
            Lv.grid.resize((size_t)Lv.W * Lv.H * NC);  // every cell read below is written by its row's resolve
            const LevelView V{Lv.W, Lv.H, Lv.maxReach, Lv.splats.data(), C.cat ? Lv.cats.data() : nullptr};
            // coarse rows the fine rows [rb, re) interpolate from (addCoarse's own index rule; monotonic)
            auto coarseRow = [&](int y, bool upper) {
                double wv = (y + 0.5 + Lv.goy) / (double)Lv.f - 0.5;
                wv = std::min((double)Lv.H - 1, std::max(0.0, wv));
                const int y0c = (int)wv;
                return upper ? std::min(y0c + 1, Lv.H - 1) : y0c;
            };
            const int cb = coarseRow(rb, false), cn = coarseRow(re - 1, true) + 1 - cb;
            ok = parallelFor((cn + BR - 1) / BR, nth, 1, [&](int bi, int tid) {
                const int y0 = cb + bi * BR, nb = std::min(BR, cb + cn - y0);
                dispatchBlock<true>(C, V, y0, nb, bufs[tid].data(), Lv.grid.data() + (size_t)y0 * Lv.W * NC, (size_t)Lv.W * NC,
                                    nullptr, 0, false, FRm, CATm, RL);
            }, abortFn, user);
            if (!ok) return Status::Aborted;
            Lv.splats.clear(); Lv.splats.shrink_to_fit();
            Lv.cats.clear(); Lv.cats.shrink_to_fit();
        }
        LC_PERF_MARK(coarse);
        for (int y = 0; y < h; ++y) {  // rows nobody reads: defined zeros
            if (y >= rb && y < re) continue;
            std::fill(dst + (size_t)y * dstStride, dst + (size_t)y * dstStride + 4 * (size_t)w, 0.f);
            if (RL) std::fill(eLOut + (size_t)y * w, eLOut + (size_t)(y + 1) * w, 0.f);
        }
        const LevelView V0{w, h, C.maxReach, C.splats.data(), C.cat ? C.cats.data() : nullptr};
        ok = parallelFor((re - rb + BR - 1) / BR, nth, 1, [&](int bi, int tid) {
            const int y0 = rb + bi * BR, nb = std::min(BR, re - y0);
            dispatchBlock<false>(C, V0, y0, nb, bufs[tid].data(), dst + (size_t)y0 * dstStride, (size_t)dstStride,
                                 RL ? eLOut + (size_t)y0 * w : nullptr, (size_t)w, C.clampNeg, FRm, CATm, RL);
        }, abortFn, user);
        LC_PERF_MARK(fine);
        LC_PERF_DUMP("blur");
        return ok ? Status::Ok : Status::Aborted;
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;   // v2.2: nothing escapes a stage (system_error, length_error, ...)
    }
}

Status postProcess(const Prepared& P, const Params& p, float* rgba, int stride, const float* eL,
                   AbortFn abortFn, void* user) {
    const int w = P.w, h = P.h;
    if (!rgba || w <= 0 || h <= 0 || (long long)w * 4 > stride) return Status::BadArgs;
    if (!paramsValid(p)) return Status::BadArgs;
    try {
        PostCtx C;
        C.W = w; C.H = h;
        frameOf(p, w, h, C.fcx, C.fcy, C.halfDiag);
        C.pa = aspectOf(p);
        C.glowAmt = clampd(p.glow, 0.0, 1.0);
        if (C.glowAmt < 1e-4) C.glowAmt = 0;
        C.vig = clampd(p.vignetting, 0.0, 1.0);
        if (C.vig < 1e-4) C.vig = 0;
        C.roll = postNeedsEL(p, P);
        if (C.roll && !eL) return Status::BadArgs;
        C.rollKnee = 1.0 - 0.5 * clampd(p.highlightRolloff, 0.0, 1.0);
        C.rollV2 = p.renderVersion >= 2 ? 1 : 0;
        const bool hal = p.halation > 1e-5, la = p.lumAberration > 1e-5, gr = p.grain > 1e-5;
        if (!(C.glowAmt > 0 || C.vig > 0 || C.roll || hal || la || gr)) return Status::Ok;
        const int nth = threadCount(p, h);
        PoolScope pool(nth);
        if (C.glowAmt > 0 && (P.rgba.size() != (size_t)w * h * 4 || !buildGlow(C, P, p, nth, abortFn, user)))
            return P.rgba.size() != (size_t)w * h * 4 ? Status::BadArgs : Status::Aborted;
        std::vector<float> le;
        if (C.roll) le.assign(eL, eL + (size_t)w * h);
        bool ok = true;
        if (C.glowAmt > 0 || C.vig > 0)
            ok = parallelFor(h, nth, 2, [&](int y, int) {
                postRowA(C, y, rgba + (size_t)y * stride, C.roll ? le.data() + (size_t)y * w : nullptr);
            }, abortFn, user);
        if (ok && hal) ok = applyHalation(C, p, rgba, stride, nth, abortFn, user);
        if (ok && la) ok = applyLumAberration(C, p, rgba, stride, nth, abortFn, user);
        if (ok && C.roll)
            ok = parallelFor(h, nth, 2, [&](int y, int) {
                float* o = rgba + (size_t)y * stride;
                const float* l = le.data() + (size_t)y * w;
                for (int x = 0; x < w; ++x) rolloffPixel(o + (size_t)x * 4, l[x], C.rollKnee, C.rollV2 != 0);
            }, abortFn, user);
        if (ok && gr) ok = applyGrain(C, p, rgba, stride, nth, abortFn, user);  // on the recorded image
        return ok ? Status::Ok : Status::Aborted;
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;   // v2.2: nothing escapes a stage (system_error, length_error, ...)
    }
}

Status render(const float* src, int w, int h, int srcStride, const float* map, int mapStride,
              const float* hlLayer, int hlStride, float* dst, int dstStride, const Params& p,
              AbortFn abortFn, void* user) {
    if (!src || !dst || w <= 0 || h <= 0) return Status::BadArgs;
    if ((long long)w * 4 > srcStride || (long long)w * 4 > dstStride) return Status::BadArgs;
    if (map && mapStride < w) return Status::BadArgs;
    if (!paramsValid(p)) return Status::BadArgs;
    const size_t srcBytes = ((size_t)(h - 1) * srcStride + (size_t)w * 4) * sizeof(float);
    const size_t dstBytes = ((size_t)(h - 1) * dstStride + (size_t)w * 4) * sizeof(float);
    if (overlaps(src, srcBytes, dst, dstBytes)) return Status::BadArgs;
    if (map && overlaps(map, ((size_t)(h - 1) * mapStride + w) * sizeof(float), dst, dstBytes))
        return Status::BadArgs;
    try {
        PoolScope pool(threadCount(p, h));
        Prepared P;
        PrepareOptions opt;
        opt.highlightLayer = hlLayer;
        opt.highlightLayerStride = hlStride;
        Status s = prepareHighlights(src, w, h, srcStride, map, mapStride, p, opt, P, abortFn, user);
        if (s != Status::Ok) return s;
        std::vector<float> eL(postNeedsEL(p, P) ? (size_t)w * h : 0);
        s = blurAlpha(P, p, dst, dstStride, eL.empty() ? nullptr : eL.data(), abortFn, user);
        if (s != Status::Ok) return s;
        return postProcess(P, p, dst, dstStride, eL.empty() ? nullptr : eL.data(), abortFn, user);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;   // v2.2: nothing escapes a stage (system_error, length_error, ...)
    }
}

Status render(const float* src, int w, int h, int srcStride, const float* map, int mapStride,
              float* dst, int dstStride, const Params& p, AbortFn abortFn, void* user) {
    return render(src, w, h, srcStride, map, mapStride, nullptr, 0, dst, dstStride, p, abortFn, user);
}


} // namespace lenscore
