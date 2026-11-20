// hq_model.cpp - continuous aperture / profile / texture / spectrum model and summed-area tables.
#include "hq_model.h"

#include <algorithm>
#include <cmath>

namespace lenshq {
namespace detail {

// ================================================================== aperture ===

void Aperture::build(int blades, double curvature, double rotRad, double rounding) {
    cs.clear();
    circle = blades < 3 || curvature >= 0.9999;
    if (!circle) {
        const int n = std::min(blades, 16);
        const double half = kPi / n;
        double apothem = std::cos(half);
        for (int k = 0; k < n; ++k) {
            const double phi = rotRad + (k + 0.5) * 2.0 * kPi / n;
            const double ux = std::cos(phi), uy = std::sin(phi);
            Constraint c{};
            if (curvature < 1e-4) {
                c.disk = false; c.nx = ux; c.ny = uy; c.a = std::cos(half);
            } else {
                const double rho = 1.0 / curvature;
                const double t = std::sqrt(rho * rho - std::sin(half) * std::sin(half)) - std::cos(half);
                c.disk = true; c.cx = -t * ux; c.cy = -t * uy; c.rho2 = rho * rho;
                c.k = 1.0 / (c.rho2 - t * t);
                apothem = rho - t;
            }
            cs.push_back(c);
        }
        if (rounding > 1e-4) {
            const double rr = 1.0 - std::min(1.0, rounding) * (1.0 - apothem);
            Constraint c{};
            c.disk = true; c.cx = 0; c.cy = 0; c.rho2 = rr * rr; c.k = 1.0 / c.rho2;
            cs.push_back(c);
        }
    }
    // bounding box from the boundary radius along many directions
    xmin = ymin = 1e30; xmax = ymax = -1e30;
    const int M = 8192;
    for (int i = 0; i < M; ++i) {
        const double a = 2 * kPi * i / M, cx = std::cos(a), sy = std::sin(a);
        const double g = gauge(cx, sy);
        const double r = g > 1e-12 ? 1.0 / g : 1.0;
        xmin = std::min(xmin, r * cx); xmax = std::max(xmax, r * cx);
        ymin = std::min(ymin, r * sy); ymax = std::max(ymax, r * sy);
    }
    const double pad = 2e-3;
    xmin -= pad; ymin -= pad; xmax += pad; ymax += pad;
}

double Aperture::gauge(double u, double v) const {
    if (circle) return std::sqrt(u * u + v * v);
    double g = 0;
    for (const Constraint& c : cs) {
        double r;
        if (!c.disk) {
            r = (c.nx * u + c.ny * v) / c.a;
        } else {
            const double uc = u * c.cx + v * c.cy, uu = u * u + v * v;
            const double c2 = c.cx * c.cx + c.cy * c.cy;
            r = (-uc + std::sqrt(std::max(0.0, uc * uc + (c.rho2 - c2) * uu))) * c.k;
        }
        g = std::max(g, r);
    }
    return g;
}

double Aperture::gaugeGrad(double u, double v, double& gx, double& gy) const {
    if (circle) {
        const double r = std::sqrt(u * u + v * v);
        if (r > 1e-15) { gx = u / r; gy = v / r; } else { gx = 1; gy = 0; }
        return r;
    }
    double g = -1e30;
    gx = gy = 0;
    for (const Constraint& c : cs) {
        double r, ax, ay;
        if (!c.disk) {
            r = (c.nx * u + c.ny * v) / c.a;
            ax = c.nx / c.a; ay = c.ny / c.a;
        } else {
            const double uc = u * c.cx + v * c.cy, uu = u * u + v * v;
            const double c2 = c.cx * c.cx + c.cy * c.cy, q = c.rho2 - c2;
            const double sq = std::sqrt(std::max(1e-300, uc * uc + q * uu));
            r = (-uc + sq) * c.k;
            ax = c.k * (-c.cx + (uc * c.cx + q * u) / sq);
            ay = c.k * (-c.cy + (uc * c.cy + q * v) / sq);
        }
        if (r > g) { g = r; gx = ax; gy = ay; }
    }
    return std::max(0.0, g);
}

// =================================================================== profile ===

double Profile::band(double t) const {
    // flat band (alpha) blended with a caustic shape 0.2 + 2.4 t^2 (same integral over t in [0,1])
    t = clampd(t, 0.0, 1.0);
    return (1.0 - rimShape) + rimShape * (0.2 + 2.4 * t * t);
}

double Profile::base(double r) const {
    if (r > 1.0) return 0.0;
    const double K = 3.0;
    double I = 1;
    if (e > 1e-3) {
        if (r >= 1 - rw) I += K * e * band((r - (1 - rw)) / rw);
    } else if (e < -1e-3) {
        const double i0 = 1.0 / (1.0 + K * -e);
        const double t = clampd((1 - r) / rw, 0.0, 1.0);
        I *= i0 + (1 - i0) * t;
    }
    if (core > 0) I *= 1 + 2.0 * core * (1 - r * r);
    else if (core < 0) I *= 1 + 0.9 * core * (1 - r * r);
    if (apod > 0) I *= std::exp(-4.0 * apod * r * r);
    if (soft > 0) {
        const double t = clampd((1 - r) / soft, 0.0, 1.0);
        I *= t * t * (3 - 2 * t);
    }
    if (r < obstr) I = 0;
    return std::max(0.0, I);
}

double Profile::edge(double d) const {
    // no hard cut here: the clip mask supplies the edge (rim continues to d = 1 and beyond)
    const double K = 3.0;
    double I = 1;
    if (e > 1e-3) {
        if (d >= 1 - rw) I += K * e * band((d - (1 - rw)) / rw);
    } else if (e < -1e-3) {
        const double i0 = 1.0 / (1.0 + K * -e);
        const double t = clampd((1 - d) / rw, 0.0, 1.0);
        I *= i0 + (1 - i0) * t;
    }
    if (soft > 0) {
        const double t = clampd((1 - d) / soft, 0.0, 1.0);
        I *= t * t * (3 - 2 * t);
    }
    return std::max(0.0, I);
}

// =================================================================== texture ===

namespace {

inline uint32_t hash3(int x, int y, uint32_t s) {
    uint32_t h = (uint32_t)x * 0x8da6b343u ^ (uint32_t)y * 0xd8163841u ^ (s + 0x9e3779b9u) * 0xcb1ab31fu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

const double kGx[16] = {1, 0.92388, 0.70711, 0.38268, 0, -0.38268, -0.70711, -0.92388,
                        -1, -0.92388, -0.70711, -0.38268, 0, 0.38268, 0.70711, 0.92388};
const double kGy[16] = {0, 0.38268, 0.70711, 0.92388, 1, 0.92388, 0.70711, 0.38268,
                        0, -0.38268, -0.70711, -0.92388, -1, -0.92388, -0.70711, -0.38268};

inline double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }

// 2D gradient noise, roughly [-1, 1]
double gnoise(double x, double y, uint32_t s) {
    const double fxl = std::floor(x), fyl = std::floor(y);
    const int xi = (int)fxl, yi = (int)fyl;
    const double fx = x - fxl, fy = y - fyl;
    auto g = [&](int ix, int iy, double dx, double dy) {
        const uint32_t h = hash3(ix, iy, s) & 15u;
        return kGx[h] * dx + kGy[h] * dy;
    };
    const double n00 = g(xi, yi, fx, fy), n10 = g(xi + 1, yi, fx - 1, fy);
    const double n01 = g(xi, yi + 1, fx, fy - 1), n11 = g(xi + 1, yi + 1, fx - 1, fy - 1);
    const double u = fade(fx), v = fade(fy);
    const double a = n00 + u * (n10 - n00), b = n01 + u * (n11 - n01);
    return 1.4 * (a + v * (b - a));
}

// 1D gradient noise with integer period P (P <= 0: not periodic)
double gnoise1(double x, uint32_t s, int P = 0) {
    const double fl = std::floor(x);
    int i = (int)fl;
    const double f = x - fl;
    auto g = [&](int k) {
        if (P > 0) k = ((k % P) + P) % P;
        const uint32_t h = hash3(k, 0x51ed27, s);
        return ((h & 0xffff) / 32767.5 - 1.0);
    };
    const double a = g(i) * f, b = g(i + 1) * (f - 1);
    return 2.0 * (a + fade(f) * (b - a));
}

inline double urand(uint32_t& st) {
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (st & 0xffffff) / 16777216.0;
}

} // namespace

double Texture::grain(double u, double v) const {
    const double f = 30.0 / scale;
    double n = gnoise(u * f, v * f, seed + 11);
    n += 0.55 * gnoise(u * f * 2.03 + 3.1, v * f * 2.03 - 1.7, seed + 12);
    n += 0.3 * gnoise(u * f * 4.1 - 5.3, v * f * 4.1 + 2.2, seed + 13);
    // slight "orange peel": a ring-oriented low-amplitude component
    const double r = std::sqrt(u * u + v * v);
    n += 0.35 * gnoise1(r * f * 1.3, seed + 14);
    return n;
}

double Texture::dust(double u, double v) const {
    double t = 0;
    for (const Speck& s : specks) {
        const double dx = u - s.x, dy = v - s.y;
        const double d2 = dx * dx + dy * dy;
        const double lim = 4.5 * s.r;
        if (d2 > lim * lim) continue;
        const double rho = std::sqrt(d2) / s.r;
        // opaque particle: dark shadow, bright Fresnel fringe just outside, decaying ripples,
        // and a faint Poisson-Arago spot in the centre
        double val = -s.depth * (1.0 - smooth01((rho - 0.82) / 0.36));
        val += 0.6 * s.depth * std::exp(-(rho - 1.12) * (rho - 1.12) / 0.03);
        if (rho > 1.0) val += 0.3 * s.depth * std::cos(2 * kPi * 1.25 * (rho * rho - 1.0)) *
                               std::exp(-(rho - 1.0) / 1.1) * smooth01((4.5 - rho) / 1.0);
        val += 0.25 * s.depth * std::exp(-rho * rho / 0.012);
        t += val;
    }
    return t;
}

double Texture::streak(double u, double v) const {
    double t = 0;
    for (int k = 0; k < 2; ++k) {
        const double ca = std::cos(streakAng[k]), sa = std::sin(streakAng[k]);
        const double p = u * ca + v * sa, q = -u * sa + v * ca;
        const double fHi = 46.0 / scale, fLo = 1.3 / scale;
        double n = gnoise(p * fHi, q * fLo + 17.0 * k, seed + 21 + k);
        n += 0.5 * gnoise(p * fHi * 2.1 + 1.3, q * fLo * 1.7 - 4.0, seed + 23 + k);
        // streaks occur in patches (where the element was wiped)
        const double env = smooth01((gnoise(u * 1.6 + 7 * k, v * 1.6 - 3 * k, seed + 25 + k) + 0.15) / 0.6);
        t += (k == 0 ? 1.0 : 0.6) * n * env;
    }
    return t;
}

double Texture::polish(double u, double v) const {
    const double r = std::sqrt(u * u + v * v);
    const double f = 38.0 / scale;
    // concentric marks with azimuthal variation of their strength
    double c = gnoise1(r * f, seed + 31) + 0.5 * gnoise1(r * f * 2.3 + 0.7, seed + 32);
    c *= 0.55 + 0.45 * gnoise(u * 2.2, v * 2.2, seed + 33);
    c *= smooth01((r - 0.05) / 0.3);  // no bull's-eye at the polishing centre
    // fine radial marks (periodic in the angle), fading toward the centre
    const double a = std::atan2(v, u) / (2 * kPi);
    const int P = std::max(8, (int)std::lround(70.0 / scale));
    const double rad = gnoise1(a * P, seed + 34, P) * smooth01(r / 0.6);
    return c + 0.6 * rad;
}

double Texture::busy(double u, double v) const {
    const double r = std::sqrt(u * u + v * v);
    double t = 0.6 * gnoise(u * 3.1 / scale + 1.1, v * 3.1 / scale - 0.4, seed + 41);
    t += 0.5 * gnoise(u * 7.5 / scale, v * 7.5 / scale, seed + 42);
    // uneven rim (soap-bubble lenses): an angular modulation that only acts near the edge
    const double a = std::atan2(v, u) / (2 * kPi);
    const int P = 7;
    const double ang = gnoise1(a * P, seed + 43, P) + 0.5 * gnoise1(a * 2 * P + 0.3, seed + 44, 2 * P);
    t += 1.6 * ang * smooth01((r - 0.72) / 0.28);
    return t;
}

void Texture::init(const HqOptions& q) {
    amount = clampd(q.textureAmount, 0.0, 2.0);
    scale = clampd(q.textureScale, 0.25, 4.0);
    seed = (uint32_t)q.textureSeed * 2654435761u + 12345u;
    wGrain = wDust = wStreak = wPolish = wBusy = 0;
    int nSpecks = 0;
    switch (q.textureType) {
    case TexGrain: wGrain = 1; break;
    case TexDust: wDust = 1; wGrain = 0.25; nSpecks = 14; break;
    case TexStreaks: wStreak = 1; wGrain = 0.25; break;
    case TexPolish: wPolish = 1; wGrain = 0.25; break;
    case TexBusy: wBusy = 1; wGrain = 0.3; break;
    default: wGrain = 0.6; wDust = 0.7; wStreak = 0.3; wPolish = 0.3; wBusy = 0.2; nSpecks = 6; break;
    }
    uint32_t st = seed | 1u;
    for (int i = 0; i < 7; ++i) urand(st);
    streakAng[0] = urand(st) * kPi;
    streakAng[1] = streakAng[0] + (0.35 + 0.6 * urand(st)) * (urand(st) < 0.5 ? -1 : 1);
    specks.clear();
    for (int i = 0; i < nSpecks; ++i) {
        const double rr = 0.9 * std::sqrt(urand(st)), aa = 2 * kPi * urand(st);
        Speck s;
        s.x = rr * std::cos(aa); s.y = rr * std::sin(aa);
        s.r = (0.012 + 0.03 * urand(st) * urand(st)) * scale;
        s.depth = 0.35 + 0.5 * urand(st);
        specks.push_back(s);
    }
    // normalise the continuous components to unit standard deviation over the disc
    nGrain = nStreak = nPolish = nBusy = 1;
    if (amount <= 0) return;
    double sg = 0, ss = 0, sp = 0, sb = 0, mg = 0, ms = 0, mp = 0, mb = 0;
    int cnt = 0;
    const int G = 160;
    for (int j = 0; j < G; ++j)
        for (int i = 0; i < G; ++i) {
            const double u = -1 + (i + 0.5) * 2.0 / G, v = -1 + (j + 0.5) * 2.0 / G;
            if (u * u + v * v > 1) continue;
            const double a = wGrain > 0 ? grain(u, v) : 0, b = wStreak > 0 ? streak(u, v) : 0;
            const double c = wPolish > 0 ? polish(u, v) : 0, d = wBusy > 0 ? busy(u, v) : 0;
            mg += a; ms += b; mp += c; mb += d;
            sg += a * a; ss += b * b; sp += c * c; sb += d * d;
            ++cnt;
        }
    auto nrm = [cnt](double m, double s2) {
        const double var = s2 / cnt - (m / cnt) * (m / cnt);
        return var > 1e-12 ? 1.0 / std::sqrt(var) : 1.0;
    };
    nGrain = nrm(mg, sg); nStreak = nrm(ms, ss); nPolish = nrm(mp, sp); nBusy = nrm(mb, sb);
}

double Texture::eval(double u, double v) const {
    if (amount <= 0) return 1.0;
    double t = 0;
    if (wGrain > 0) t += wGrain * 0.17 * nGrain * grain(u, v);
    if (wDust > 0) t += wDust * 0.75 * dust(u, v);
    if (wStreak > 0) t += wStreak * 0.30 * nStreak * streak(u, v);
    if (wPolish > 0) t += wPolish * 0.22 * nPolish * polish(u, v);
    if (wBusy > 0) t += wBusy * 0.2 * nBusy * busy(u, v);
    return std::max(0.03, 1.0 + amount * t);
}

// ================================================================== spectrum ===

namespace {
// Non-negative, smooth camera-like channel sensitivities (so every wavelength adds energy, no
// negative lobes). Hue sweeps violet-blue -> cyan -> green -> yellow -> orange -> red.
inline double resp(int c, double lam) {
    static const double mu[3] = {607, 537, 455}, sg[3] = {34, 35, 28};
    const double z = (lam - mu[c]) / sg[c];
    return std::exp(-0.5 * z * z);
}
// solve 3x3 (row-major) A x = b
bool solve3(double A[3][3], const double b[3], double x[3]) {
    double M[3][4];
    for (int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) M[i][j] = A[i][j]; M[i][3] = b[i]; }
    for (int c = 0; c < 3; ++c) {
        int p = c;
        for (int r = c + 1; r < 3; ++r) if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
        if (std::fabs(M[p][c]) < 1e-14) return false;
        for (int j = 0; j < 4; ++j) std::swap(M[c][j], M[p][j]);
        for (int r = 0; r < 3; ++r) {
            if (r == c) continue;
            const double f = M[r][c] / M[c][c];
            for (int j = 0; j < 4; ++j) M[r][j] -= f * M[c][j];
        }
    }
    for (int i = 0; i < 3; ++i) x[i] = M[i][3] / M[i][i];
    return true;
}
const double kYw[3] = {0.2126, 0.7152, 0.0722};
} // namespace

void Spectrum::buildNone() {
    N = 1;
    for (int k = 0; k < 2; ++k) s[k].assign(1, 1.0);
    for (int c = 0; c < 3; ++c) w[c].assign(1, 1.0);
    lum.assign(1, 1.0);
    lat.assign(1, 0.0);
    ds.assign(1, 0.0);
}

void Spectrum::buildThreeBand(double fringe, double tint) {
    N = 3;
    const double kf = 0.2 * clampd(fringe, -1.0, 1.0);
    const double t = clampd(tint, 0.0, 1.0);
    double d[3];
    if (t <= 0) { d[0] = 1; d[1] = 0; d[2] = -1; }
    else { d[0] = 1.0 - 1.5 * t; d[1] = t; d[2] = -1.0 + 0.5 * t; }
    for (int k = 0; k < 2; ++k) s[k].resize(3);
    for (int i = 0; i < 3; ++i) { s[1][i] = 1 + kf * d[i]; s[0][i] = 1 - kf * d[i]; }
    for (int c = 0; c < 3; ++c) { w[c].assign(3, 0.0); w[c][c] = 1.0; }
    lum.assign(kYw, kYw + 3);
    lat = {1.0, 0.0, -1.0};
    ds.assign(3, 0.0);
    double m = 0;
    for (int i = 0; i < 3; ++i) m += lum[i] * lat[i];
    for (int i = 0; i < 3; ++i) lat[i] -= m;
}

// Longitudinal CA: kernel scale s(lambda) = 1 +- kf * D(lambda) (+ behind focus, - in front),
// kf = 0.2 * chromaFringe as in alpha; D(lambda) see below. Wavelength bins are equal slices of
// 400..700 nm; each bin's channel weight is its share of the channel's response.
void Spectrum::buildSpectral(double fringe, double tint, int n) {
    N = std::max(1, n);
    const double kf = 0.2 * clampd(fringe, -1.0, 1.0);
    const double t = clampd(tint, 0.0, 1.0);
    const int L = 3000;  // 0.1 nm integration steps
    double tot[3] = {};
    for (int k = 0; k < L; ++k) {
        const double lam = 400 + (k + 0.5) * 300.0 / L;
        for (int c = 0; c < 3; ++c) tot[c] += resp(c, lam);
    }
    // D(x), x = (lambda - 550) / 100, is a polynomial a + b x + c x^p, log-compressed in the tails
    // (D = sign * log(1 + |poly|): violet / deep red neither spread far inside the disc nor pile
    // up into a separate ring), fitted so the response-weighted mean of D over each channel equals
    // alpha's per-channel scale offsets exactly:
    //   D0: p = 2, targets (R, G, B) = (+1, 0, -1)   primary spectrum (red/blue rims)
    //   D1: p = 4, targets (-0.5, +1, -0.5)          secondary spectrum: a flat-topped curve, so
    //       the green wavelengths focus together into a crisp green (behind) / magenta (in front)
    //       outline while red and blue spread inward.
    // D = (1 - tint) D0 + tint D1 keeps the means exact (linear).
    auto fit = [&](int p, const double tgt[3], double q[3]) {
        auto poly = [p](const double* qq, double x) { return qq[0] + qq[1] * x + qq[2] * std::pow(std::fabs(x), p); };
        auto Dc = [poly](const double* qq, double lam) {
            const double d = poly(qq, (lam - 550) / 100);
            return (d < 0 ? -1.0 : 1.0) * std::log1p(std::fabs(d));
        };
        double A[3][3] = {};
        for (int k = 0; k < L; ++k) {
            const double lam = 400 + (k + 0.5) * 300.0 / L, x = (lam - 550) / 100;
            for (int c = 0; c < 3; ++c) {
                const double r = resp(c, lam) / tot[c];
                A[c][0] += r; A[c][1] += r * x; A[c][2] += r * std::pow(std::fabs(x), p);
            }
        }
        q[0] = q[1] = q[2] = 0;
        solve3(A, tgt, q);
        for (int it = 0; it < 40; ++it) {
            double m[3] = {0, 0, 0};
            for (int k = 0; k < L; ++k) {
                const double lam = 400 + (k + 0.5) * 300.0 / L;
                const double d = Dc(q, lam);
                for (int c = 0; c < 3; ++c) m[c] += resp(c, lam) * d / tot[c];
            }
            double r[3] = {tgt[0] - m[0], tgt[1] - m[1], tgt[2] - m[2]}, dq[3] = {0, 0, 0};
            if (std::fabs(r[0]) + std::fabs(r[1]) + std::fabs(r[2]) < 1e-10) break;
            solve3(A, r, dq);
            for (int k = 0; k < 3; ++k) q[k] += dq[k];
        }
        return Dc;
    };
    const double t0[3] = {1, 0, -1}, t1[3] = {-0.5, 1, -0.5};
    double q0[3], q1[3];
    auto D0 = fit(2, t0, q0);
    auto D1 = fit(4, t1, q1);
    auto D = [&](double lam) { return (1 - t) * D0(q0, lam) + t * D1(q1, lam); };
    for (int k = 0; k < 2; ++k) s[k].assign(N, 1.0);
    for (int c = 0; c < 3; ++c) w[c].assign(N, 0.0);
    lum.assign(N, 0.0);
    lat.assign(N, 0.0);
    std::vector<double> dsum(N, 0.0), dw(N, 0.0), xs(N, 0.0), dlo(N, 1e9), dhi(N, -1e9);
    for (int k = 0; k < L; ++k) {
        const double lam = 400 + (k + 0.5) * 300.0 / L;
        const int i = std::min(N - 1, (int)((lam - 400) / 300.0 * N));
        double rs = 0;
        for (int c = 0; c < 3; ++c) { const double r = resp(c, lam); w[c][i] += r / tot[c]; rs += r; }
        const double Dl = D(lam);
        dsum[i] += Dl * rs; dw[i] += rs; xs[i] += (lam - 550) / 100 * rs;
        dlo[i] = std::min(dlo[i], Dl); dhi[i] = std::max(dhi[i], Dl);
    }
    ds.assign(N, 0.0);
    for (int i = 0; i < N && N > 1; ++i) ds[i] = std::fabs(kf) * (dhi[i] - dlo[i]);
    for (int i = 0; i < N; ++i) {
        const double Di = N == 1 ? 0.0 : dsum[i] / std::max(1e-30, dw[i]);
        s[1][i] = std::max(0.2, 1 + kf * Di);
        s[0][i] = std::max(0.2, 1 - kf * Di);
        lum[i] = kYw[0] * w[0][i] + kYw[1] * w[1][i] + kYw[2] * w[2][i];
        lat[i] = N == 1 ? 0.0 : xs[i] / std::max(1e-30, dw[i]);
    }
    double m = 0;
    for (int i = 0; i < N; ++i) m += lum[i] * lat[i];
    for (int i = 0; i < N; ++i) lat[i] -= m;
}

double Spectrum::smin(int side) const { return *std::min_element(s[side].begin(), s[side].end()); }
double Spectrum::smax(int side) const { return *std::max_element(s[side].begin(), s[side].end()); }

// ======================================================================= SAT ===

void Sat::build(const std::vector<double>& tex, int n_, double D_) {
    n = n_; D = D_;
    const double ta = (2 * D / n) * (2 * D / n);
    const int s0 = n + 1;
    lv.clear();
    lv.reserve(16);  // S below is a reference into lv[0]: no reallocation while mips are added
    lv.push_back(std::vector<double>((size_t)s0 * s0, 0.0));
    nl.assign(1, n);
    std::vector<double>& S = lv[0];
    int ix0 = n, ix1 = -1, iy0 = n, iy1 = -1;
    for (int j = 0; j < n; ++j) {
        double row = 0;
        const double* t = tex.data() + (size_t)j * n;
        double* sp = S.data() + (size_t)j * s0;
        double* sn = sp + s0;
        sn[0] = 0;
        for (int i = 0; i < n; ++i) {
            if (t[i] != 0) { ix0 = std::min(ix0, i); ix1 = std::max(ix1, i); iy0 = std::min(iy0, j); iy1 = std::max(iy1, j); }
            row += t[i] * ta;
            sn[i + 1] = sp[i + 1] + row;
        }
    }
    total = S[(size_t)n * s0 + n];
    const double tx = 2 * D / n;
    if (ix1 < 0) { bx0 = bx1 = by0 = by1 = 0; }
    else {
        bx0 = -D + ix0 * tx; bx1 = -D + (ix1 + 1) * tx;
        by0 = -D + iy0 * tx; by1 = -D + (iy1 + 1) * tx;
    }
    for (int l = 1; ; ++l) {
        const int f = 1 << l;
        if (n % f || n / f < 16) break;
        const int m = n / f, sm = m + 1;
        std::vector<double> L((size_t)sm * sm);
        for (int j = 0; j <= m; ++j)
            for (int i = 0; i <= m; ++i) L[(size_t)j * sm + i] = S[(size_t)(j * f) * s0 + (size_t)i * f];
        lv.push_back(std::move(L));
        nl.push_back(m);
    }
}

size_t Sat::bytes() const {
    size_t b = 0;
    for (const auto& v : lv) b += v.size() * sizeof(double);
    return b;
}

int Sat::levelFor(double pxPerUnit, double tpp) const {
    int best = 0;
    for (int l = 0; l < (int)nl.size(); ++l) {
        const double texPerUnit = nl[l] / (2 * D);
        if (texPerUnit >= tpp * pxPerUnit) best = l; else break;
    }
    return best;
}

// =================================================================== EdgeFn ===

void EdgeFn::build(const std::vector<double>& smp, double d0_, double d1_) {
    n = (int)smp.size();
    d0 = d0_; d1 = d1_;
    step = (d1 - d0) / (n - 1);
    inv = 1.0 / step;
    f = smp;
    F1.assign(n, 0.0);
    F2.assign(n, 0.0);
    for (int i = 0; i + 1 < n; ++i) {
        F1[i + 1] = F1[i] + step * (f[i] + f[i + 1]) * 0.5;
        F2[i + 1] = F2[i] + step * F1[i] + step * step * (2 * f[i] + f[i + 1]) / 6.0;
    }
    runEnd.assign(n, n - 1);
    for (int i = n - 2; i >= 0; --i) runEnd[i] = f[i] == f[i + 1] ? runEnd[i + 1] : i;
    // C == 1 up to dIn, == 0 beyond dOut
    dIn = d0;
    for (int i = 0; i < n; ++i) { if (std::fabs(f[i] - 1.0) > 1e-12) break; dIn = d0 + i * step; }
    dOut = d1;
    for (int i = n - 1; i >= 0; --i) { if (f[i] != 0.0) break; dOut = d0 + i * step; }
}

double EdgeFn::val(double d) const {
    double x = (d - d0) * inv;
    if (x <= 0) return f[0];
    if (x >= n - 1) return f[n - 1];
    const int i = (int)x;
    const double t = x - i;
    return f[i] + t * (f[i + 1] - f[i]);
}

double EdgeFn::I1(double d) const {
    double x = (d - d0) * inv;
    if (x <= 0) return f[0] * (d - d0);
    if (x >= n - 1) return F1[n - 1] + f[n - 1] * (d - d1);
    const int i = (int)x;
    const double h = (x - i) * step;
    return F1[i] + f[i] * h + (f[i + 1] - f[i]) * h * h * 0.5 * inv;
}

double EdgeFn::I2(double d) const {
    double x = (d - d0) * inv;
    if (x <= 0) { const double h = d - d0; return f[0] * h * h * 0.5; }
    if (x >= n - 1) { const double h = d - d1; return F2[n - 1] + F1[n - 1] * h + f[n - 1] * h * h * 0.5; }
    const int i = (int)x;
    const double h = (x - i) * step;
    return F2[i] + F1[i] * h + f[i] * h * h * 0.5 + (f[i + 1] - f[i]) * h * h * h * inv / 6.0;
}

bool EdgeFn::constOver(double da, double db, double& v) const {
    const double x0 = (da - d0) * inv, x1 = (db - d0) * inv;
    if (x1 <= 0) { v = f[0]; return true; }
    if (x0 >= n - 1) { v = f[n - 1]; return true; }
    const int i0 = x0 <= 0 ? 0 : (int)x0, i1 = x1 >= n - 1 ? n - 1 : (int)std::ceil(x1);
    if (runEnd[i0] >= i1) { v = f[i0]; return true; }
    return false;
}

double EdgeFn::filtered(double d, double a, double b) const {
    if (a < b) std::swap(a, b);
    {   // exact quick-out: f constant over the whole footprint [d - (a+b)/2, d + (a+b)/2]
        const double hw = 0.5 * (a + b);
        const double x0 = (d - hw - d0) * inv, x1 = (d + hw - d0) * inv;
        if (x1 <= 0) return f[0];
        if (x0 >= n - 1) return f[n - 1];
        const int i0 = x0 <= 0 ? 0 : (int)x0, i1 = x1 >= n - 1 ? n - 1 : (int)std::ceil(x1);
        if (runEnd[i0] >= i1 && (x0 > 0 || runEnd[0] >= i1)) return f[i0];
    }
    if (a < 1e-9) return val(d);
    if (b < 1e-4 * a) return (I1(d + 0.5 * a) - I1(d - 0.5 * a)) / a;
    const double p = 0.5 * (a + b), m = 0.5 * (a - b);
    return (I2(d + p) - I2(d + m) - I2(d - m) + I2(d - p)) / (a * b);
}

double EdgeFn::filteredIn(double d, double a, double b, double lo, double hi) const {
    if (a < b) std::swap(a, b);
    const double P = 0.5 * (a + b), Mh = 0.5 * (a - b);
    if (lo <= d - P && hi >= d + P) return filtered(d, a, b);   // limits outside the footprint
    const double u0 = std::max(d - P, lo), u1 = std::min(d + P, hi);
    if (!(u1 > u0) || a < 1e-9) return filtered(d, a, b);
    // kernel (sum of uniforms of widths a >= b): ramps on [d-P, d-Mh] and [d+Mh, d+P], flat between;
    // on each piece k(x) = al + be x, so int f k = al [I1] + be [x I1 - I2]
    auto J = [&](double x) { return x * I1(x) - I2(x); };
    const double top = 1.0 / a;
    double sfk = 0, sk = 0;
    auto piece = [&](double x0, double x1, double al, double be) {
        x0 = std::max(x0, u0); x1 = std::min(x1, u1);
        if (!(x1 > x0)) return;
        sfk += al * (I1(x1) - I1(x0)) + be * (J(x1) - J(x0));
        sk += al * (x1 - x0) + be * 0.5 * (x1 * x1 - x0 * x0);
    };
    if (b < 1e-9) piece(d - P, d + P, top, 0.0);
    else {
        const double s = top / b;   // ramp slope
        piece(d - P, d - Mh, -s * (d - P), s);
        piece(d - Mh, d + Mh, top, 0.0);
        piece(d + Mh, d + P, s * (d + P), -s);
    }
    if (!(sk > 1e-12)) return filtered(d, a, b);
    return sfk / sk;
}

} // namespace detail
} // namespace lenshq
