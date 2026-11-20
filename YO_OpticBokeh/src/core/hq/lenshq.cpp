// lenshq.cpp - quality-first ("beta", Quality = Best) scatter lens blur.   v2.2
//
// MODEL. Every source pixel p spreads its colour over its OWN kernel
//     K_p(x) = sum_i w_i(ch) * P_i(x) / Z,          i = wavelength sub-scale
//     P_i(x) = A(p_i) * C(p_i) * tilt,   p_i = sg * (x - c - delta_i) / (R * s_i * (ax, ay))
// A = aperture intensity in pupil coordinates: exact level-set profile I(gauge(u)) of the bladed
//     aperture (rim / soft edge / core / apodisation / onion / obstruction, alpha's continuous
//     definition without layer quantisation) times the in-disc texture T(u) (hq_model.cpp).
// C = cat-eye clip circle (centre 2h from the aperture centre; the pair is centred on the source
//     pixel like alpha) with a soft mechanical edge and the edge character of whichever boundary is
//     outermost (clipFactor). tilt = optional cos^4-like falloff.
// s_i = longitudinal CA scale (sign flips front/back), delta_i = optional lateral CA shift,
// c = aperture centre (chief ray), sg = -1 for front-of-focus discs if invertFrontBokeh.
// The value landing on an output pixel is the EXACT integral of the kernel over the pixel square
// (alpha's box-filter semantics), from summed-area tables (SAT) of 4x4-supersampled texel averages,
// so sums over all pixels telescope to the analytic total.
//
// TABLES (immutable, cached process-wide). A[side][part] (pupil, onion delta part) and the per-
// channel spectral tables K_ch = sum_i w_i(ch) A(u / s_i) / s_i^2 built from the SAME wavelength
// sub-scale list that the per-wavelength code uses (no pop between the two). Resolution from the
// maxRadius slider (data independent -> tile exact), bounded by tableBudgetMB.
//
// STAMPS. path A (kernel depends on R and side only): 3 SAT lookups per pixel. path B (cat eye /
// lateral CA / disc falloff): table value for pixels where the clip is 1 for every wavelength,
// complement form K - sum_i w_i box_i (1 - C_i) where a clip edge crosses, every pixel per
// wavelength with lateral CA. Each plane is normalised by its own discrete sum (exact energy).
//
// LATTICE (speed). The kernel is a smooth function of (R, g), g = frame position / half diagonal.
// Exact stamps are computed on a lattice (sqrt(R) step 0.07*tol, g step such that the clip edge /
// lateral shift moves <= 0.07*sqrt(R)*tol px between nodes) and every source blends its 2..8
// neighbouring node stamps (weights sum to 1: energy stays exact). Lattice nodes depend only on
// parameters, so the result is still independent of threads and tiling. R < 8 px: exact stamps.
// R lattice is anchored at sqrt(maxRadius), so a uniform blur uses exact-radius stamps.
//
// ACCUMULATION. Source tiles (64x64) are scattered into a private float buffer (tile + reach) and
// merged into a global two-word fixed-point accumulator: hi at 2^a (a chosen per render from the
// largest |value| so nothing can overflow) + lo 2^40 finer for the remainder. Integer addition is
// associative: bit-identical for any thread count. Output = colour / weight, clamped >= 0.

#include "hq_model.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lenshq {
using namespace detail;
namespace {

const int kBand = 16;          // rows per accumulator lock
const double kWeightCap = 65536.0;  // bound on a pixel's summed kernel weight (pile-up), for the scale
const double kLatMinR = 8.0;   // below this radius every source gets its exact stamp
const double kStampPinBudget = 24.0 * 1048576.0;  // per thread: stamps pinned by one tile

typedef std::chrono::steady_clock Clock;

// ------------------------------------------------------------ abort / poll -----
// parallelFor publishes a poll context to every participating thread; long loops call cancelled()
// (cheap). On the calling thread it also calls the host abort function at most every 10 ms.
struct PollCtx {
    std::atomic<bool>* stop = nullptr;
    AbortFn fn = nullptr;
    void* user = nullptr;
    Clock::time_point last;
};
thread_local PollCtx* tlPoll = nullptr;
thread_local bool tlMain = false;

inline bool cancelled() {
    PollCtx* p = tlPoll;
    if (!p) return false;
    if (p->stop->load(std::memory_order_relaxed)) return true;
    if (tlMain && p->fn) {
        const Clock::time_point now = Clock::now();
        if (now - p->last >= std::chrono::milliseconds(10)) {
            p->last = now;
            if (p->fn(p->user)) { p->stop->store(true); return true; }
        }
    }
    return false;
}

template <class F>
bool parallelFor(int n, int nthreads, int chunk, F&& f, AbortFn abortFn, void* user) {
    std::atomic<int> next{0};
    std::atomic<bool> stop{false};
    std::atomic<int> done{0};
    PollCtx ctx;
    ctx.stop = &stop; ctx.fn = abortFn; ctx.user = user; ctx.last = Clock::now();
    // an exception in a body (std::bad_alloc under memory pressure) must neither escape a worker
    // thread (std::terminate) nor unwind this frame while workers run: store the first one, stop
    // everybody, rethrow on the calling thread after the join.
    std::exception_ptr err;
    std::mutex errM;
    auto work = [&](int tid) {
        PollCtx* prev = tlPoll;
        const bool prevMain = tlMain;
        tlPoll = &ctx; tlMain = tid == 0;
        try {
            for (;;) {
                if (stop.load(std::memory_order_relaxed)) break;
                if (tid == 0 && cancelled()) break;
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
        tlPoll = prev; tlMain = prevMain;
    };
    if (abortFn && abortFn(user)) return false;
    std::vector<std::thread> th;
    for (int t = 1; t < nthreads; ++t) {
        try {
            th.emplace_back([&, t] { work(t); done.fetch_add(1); });
        } catch (...) {
            break;
        }
    }
    const int spawned = (int)th.size();
    work(0);
    while (done.load() < spawned) {
        if (!stop && abortFn && abortFn(user)) stop = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (auto& t : th) t.join();
    if (err) std::rethrow_exception(err);
    return !stop.load();
}

double nowSec() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

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

void frameOf(const Params& p, int w, int h, double& cx, double& cy, double& halfDiag) {
    double fl = 0, ft = 0, fw = w, fh = h;
    if (p.frameWidth > 0 && p.frameHeight > 0) { fl = p.frameLeft; ft = p.frameTop; fw = p.frameWidth; fh = p.frameHeight; }
    cx = fl + 0.5 * fw;
    cy = ft + 0.5 * fh;
    halfDiag = std::max(1.0, 0.5 * std::sqrt(fw * fw + fh * fh));
}

// bilinear sample of a corner grid (stride n+1) at texel coordinates (clamped)
inline double satAt(const double* S, int n, double tx, double ty) {
    tx = tx < 0 ? 0 : (tx > n ? n : tx);
    ty = ty < 0 ? 0 : (ty > n ? n : ty);
    int ix = (int)tx, iy = (int)ty;
    if (ix > n - 1) ix = n - 1;
    if (iy > n - 1) iy = n - 1;
    const double fx = tx - ix, fy = ty - iy;
    const double* r0 = S + (size_t)iy * (n + 1) + ix;
    const double* r1 = r0 + (n + 1);
    const double a = r0[0] + fx * (r0[1] - r0[0]);
    const double b = r1[0] + fx * (r1[1] - r1[0]);
    return a + fy * (b - a);
}

// ------------------------------------------------------------------ geometry ---

struct SubScale { double s, w; int k; };  // kernel scale, weight inside its bin, bin index

struct Geo {  // everything derived from the parameters without tables (cheap)
    double ax = 1, ay = 1;
    Aperture ap;
    Spectrum spec;
    std::vector<SubScale> sub[2];  // per side
    bool ca = false, lat = false, cat = false, pathB = false;
    double catK = 0, catCurve = 1, rho = 1, csoft = 0.02, clipRim = 1, loss = 0;
    double latK = 0, discG = 0, diff = 0;
    double maxR = 0;
    double extX = 1, extY = 1;   // unit-space half extents of any kernel incl. spectral scale
    double latMax = 0;           // max |lat_i| * |latK| (px per px of radius per unit |g|)
    double obstr = 0;
    double catMaxM = 0.9;        // cap of the clip half offset: aperture and clip always intersect
};

void buildGeo(Geo& G, const Params& p, const HqOptions& q) {
    const double an = clampd(p.anamorphic, 0.1, 10.0);
    G.ax = 1.0 / std::sqrt(an);
    G.ay = std::sqrt(an);
    G.ap.build(p.blades, clampd(p.bladeCurvature, 0.0, 1.0), p.rotationDeg * kPi / 180.0,
               clampd(p.cornerRounding, 0.0, 1.0));
    G.maxR = clampd(p.maxRadius, 0.0, 4000.0);
    G.obstr = clampd(p.obstruction, 0.0, 0.9);
    const double fr = clampd(p.chromaFringe, -1.0, 1.0);
    G.ca = std::fabs(fr) > 5e-4;
    G.latK = 0.08 * clampd(q.lateralCA, -1.0, 1.0);
    G.lat = std::fabs(G.latK) > 1e-6;
    const double ce = clampd(p.catEye, 0.0, 1.0);
    G.cat = ce > 1e-3;
    G.catK = 1.2 * ce;
    G.catCurve = clampd(q.catEyeCurve, 0.5, 3.0);
    // the clip circle must stay larger than the central obstruction (else whole discs vanish)
    G.rho = clampd(q.catEyeClipRadius, std::max(0.7, std::min(1.5, G.obstr + 0.15)), 1.5);
    // a clip circle smaller than the aperture cuts every disc (even at the frame centre) as soon as
    // the cat eye is on: fade it in over catEye 0..0.2 so keyframing Cat Eye from 0 has no pop
    if (G.rho < 1.0) {
        const double t = clampd(ce / 0.2, 0.0, 1.0);
        G.rho = 1.0 - (1.0 - G.rho) * t * t * (3.0 - 2.0 * t);
    }
    G.csoft = clampd(q.catEyeSoftness, 0.0, 0.3);
    // alpha reaches at most 0.9 (catEye 1 at rf 1.5); steeper curves / smaller clips must not push
    // the clip circle off the aperture (the disc would vanish or detach from its source pixel)
    G.catMaxM = std::min(0.9, 0.5 * (1.0 + G.rho) - 0.1);
    G.clipRim = clampd(q.catEyeClipRim, 0.0, 1.0);
    G.loss = G.cat ? clampd(q.catEyeLightLoss, 0.0, 1.0) : 0.0;
    G.discG = clampd(q.discFalloff, 0.0, 1.0);
    G.diff = clampd(q.diffraction, 0.0, 1.0);
    G.pathB = G.cat || G.lat || G.discG > 1e-4;
    const double amax = std::max(G.ax, G.ay);
    if (!G.ca && !G.lat) {
        G.spec.buildNone();
    } else if (q.fringeModel == FringeThreeBand) {
        G.spec.buildThreeBand(fr, p.fringeTint);
    } else {
        Spectrum probe;
        probe.buildSpectral(G.ca ? fr : 0.0, p.fringeTint, 64);
        double spread = 0;
        for (int s = 0; s < 2; ++s) spread = std::max(spread, probe.smax(s) - probe.smin(s));
        double lmin = 0, lmax = 0;
        for (double l : probe.lat) { lmin = std::min(lmin, l); lmax = std::max(lmax, l); }
        const double spreadPx = G.maxR * amax * spread + G.maxR * std::fabs(G.latK) * 1.5 * (lmax - lmin);
        int nb = q.spectralSamples > 0 ? std::min(48, std::max(3, q.spectralSamples))
                                       : (int)clampd(std::ceil(spreadPx / 1.5) + 1, 5, 24);
        G.spec.buildSpectral(G.ca ? fr : 0.0, p.fringeTint, nb);
    }
    // sub-scales: every bin integrated over its own scale spread, <= ~0.5 px apart at maxRadius,
    // at most 96 per side (beyond that neighbouring copies of a rim are > 0.5 px apart but still
    // overlap through the rim width and the pixel footprint)
    const Spectrum& sp = G.spec;
    const double rmax = std::max(1.0, G.maxR * amax);
    std::vector<int> m(sp.N, 1);
    int tot = 0;
    for (int k = 0; k < sp.N; ++k) { m[k] = std::max(1, std::min(64, (int)std::ceil(sp.ds[k] * rmax / 0.5))); tot += m[k]; }
    if (tot > 96)
        for (int k = 0; k < sp.N; ++k) m[k] = std::max(1, (int)std::floor(m[k] * 96.0 / tot));
    for (int side = 0; side < 2; ++side) {
        G.sub[side].clear();
        for (int k = 0; k < sp.N; ++k)
            for (int u = 0; u < m[k]; ++u)
                G.sub[side].push_back({sp.s[side][k] + ((u + 0.5) / m[k] - 0.5) * sp.ds[k], 1.0 / m[k], k});
    }
    double smax = 0, smin = 1e9;
    for (int s = 0; s < 2; ++s)
        for (const SubScale& ss : G.sub[s]) { smax = std::max(smax, ss.s); smin = std::min(smin, ss.s); }
    double ex = std::max(-G.ap.xmin, G.ap.xmax) * smax, ey = std::max(-G.ap.ymin, G.ap.ymax) * smax;
    if (G.cat) {
        // aperture (inside the unit circle) at -h, clip of radius rho' at +h: |u|^2 + |h|^2 <=
        // (1 + rho'^2) / 2 at scale 1; wavelength s scales about the aperture centre (chief ray),
        // which moves points by at most |s - 1| (pupil radius 1)
        const double ro = G.rho * (1 + 0.5 * G.csoft / G.rho);
        const double b = std::sqrt(0.5 * (1 + ro * ro)) + std::max(smax - 1.0, 1.0 - smin);
        ex = std::max(ex, b); ey = std::max(ey, b);
    }
    G.extX = ex + 4e-3;
    G.extY = ey + 4e-3;
    double lm = 0;
    for (double l : G.spec.lat) lm = std::max(lm, std::fabs(l));
    G.latMax = G.lat ? std::fabs(G.latK) * lm : 0.0;
}

inline double diffSigma(const Geo& G, double R) { return G.diff > 0 ? 0.15 * G.diff * std::sqrt(R) : 0.0; }
inline int diffRad(double sig) { return sig > 0.03 ? (int)std::ceil(3.0 * sig) : 0; }

// ------------------------------------------------------------------ tables ----

struct Tables {
    int n = 512;
    Profile prof[2];
    bool onion = false;
    double onionN = 4;
    Sat A[2][2];          // [table][part]
    int aSide[2] = {0, 0};
    int nch = 1;
    Sat K[2][3][2];       // [table][channel][part]
    int kSide[2] = {0, 0};
    bool kFromA = false;  // K == A (single wavelength)
    double kbx0 = 0, kbx1 = 0, kby0 = 0, kby1 = 0;  // union bbox of the K tables (unit)
    EdgeFn cfM, cfE[2], cfME[2];
    double eMax[2] = {1, 1};
    int lossN = 0;
    double lossLim = 1;
    std::vector<double> lossF[2];  // [side]
    double buildSec = 0;
    size_t bytes() const {
        size_t b = 0;
        for (int t = 0; t < 2; ++t) for (int p = 0; p < 2; ++p) b += A[t][p].bytes();
        for (int t = 0; t < 2; ++t) for (int c = 0; c < 3; ++c) for (int p = 0; p < 2; ++p) b += K[t][c][p].bytes();
        b += (cfM.f.size() + cfE[0].f.size() * 2 + cfME[0].f.size() * 2) * 3 * sizeof(double);
        b += (lossF[0].size() + lossF[1].size()) * sizeof(double);
        return b;
    }
    const Sat& kTab(int side, int c, int part) const {
        return kFromA ? A[aSide[side]][part] : K[kSide[side]][c][part];
    }
};

struct Model : Geo {
    int W = 0, H = 0;
    double fcx = 0, fcy = 0, halfDiag = 1;
    double onionStr = 0, onionN = 4;
    bool onion = false;
    double tpp = 6;
    bool flipFront = false;
    double latTol = 1;
    double latEps0 = 0.07;  // lattice step coefficient (smaller for hard edges / bright rims)
    int tile = 64;          // source tile size (32 for large radii: fewer stamps pinned per tile)
    std::shared_ptr<const Tables> T;

    double onionAmp(double R) const {
        if (!onion) return 0.0;
        return onionStr * clampd((R / onionN - 2.0) / 3.0, 0.0, 1.0);
    }
    double lossAt(int side, double hx, double hy) const {
        const Tables& t = *T;
        if (t.lossN == 0) return 1.0;
        double u = (hx + t.lossLim) / (2 * t.lossLim) * (t.lossN - 1), v = (hy + t.lossLim) / (2 * t.lossLim) * (t.lossN - 1);
        u = clampd(u, 0.0, t.lossN - 1.000001); v = clampd(v, 0.0, t.lossN - 1.000001);
        const int i = (int)u, j = (int)v;
        const double fu = u - i, fv = v - j;
        const double* r0 = &t.lossF[side][(size_t)j * t.lossN + i];
        const double* r1 = r0 + t.lossN;
        const double a = r0[0] + fu * (r0[1] - r0[0]), b = r1[0] + fu * (r1[1] - r1[0]);
        return a + fv * (b - a);
    }
};

// Cat-eye clip factor at pupil point p (aperture coordinates), w = p - 2h (clip-centred), averaged
// over a footprint of (fx, fy) pupil units. The aperture table A already carries the edge character
// E of the APERTURE boundary (rim, soft edge at gauge d_ap). Where the clip boundary is the outer
// one (d_cl = |w| / rho > d_ap) the edge character must follow the clip instead, exactly once:
//     C = mech(d_cl) * [ (1 - clipRim) + clipRim * E(max(d_ap, d_cl)) / E(d_ap) ]
// (alpha's equivalent: the profile of the intersection's gauge max(g_ap, g_clip)). At the frame
// centre the clip coincides with / contains the aperture and C == 1 (identical to no cat eye).
// rs: relative scale spread of the wavelength bin (ds / s): the bin's sub-scales move p radially by
// up to p * rs, which widens the footprint of both gauges (clip: |p.w^| rs / rho, aperture: dap rs).
//
// C is the APERTURE-WEIGHTED mean of the clip function over the footprint: the stamp multiplies C
// with the exact pixel integral of A, and A == 0 where the aperture gauge exceeds 1, so that part of
// the footprint must not count. Along w^ the aperture gauge changes by gproj * rho per unit of clip
// gauge, i.e. the aperture boundary lies at clip gauge dcl + (1 - dap) / (gproj * rho); the filter is
// restricted to the inside. Without this, a clip boundary that (nearly) coincides with the aperture
// boundary (catEyeClipRadius 1, any tiny cat-eye offset) would be averaged against the empty outside
// and darken the rim: a visible pop when catEye leaves 0 (v2.2 review follow-up).
double clipFactor(const Model& M, int side, double wx, double wy, double px, double py, double fx, double fy,
                  double dapIn = -1.0, double rs = 0.0) {
    (void)dapIn;
    const Tables& T = *M.T;
    const double rho = M.rho;
    const double dw = std::sqrt(wx * wx + wy * wy);
    const double ux = dw > 1e-12 ? std::fabs(wx) / dw : 1.0, uy = dw > 1e-12 ? std::fabs(wy) / dw : 0.0;
    const double dcl = dw / rho;
    double gax = 0, gay = 0;
    const double dap = M.ap.gaugeGrad(px, py, gax, gay);
    const double dp = std::sqrt(px * px + py * py);
    const double pdw = dw > 1e-12 ? (px * wx + py * wy) / dw : dp;
    const double smCl = std::fabs(pdw) * rs / rho, smAp = dap * rs;
    double aw = ux * fx / rho, bw = uy * fy / rho;
    if (aw >= bw) aw += smCl; else bw += smCl;
    const double gproj = dw > 1e-12 ? (gax * wx + gay * wy) / dw : 0.0;
    double lo = -1e30, hi = 1e30;
    if (gproj > 1e-9) hi = dcl + (1.0 - dap) / (gproj * rho);
    else if (gproj < -1e-9) lo = dcl + (1.0 - dap) / (gproj * rho);
    const double mech = T.cfM.filteredIn(dcl, aw, bw, lo, hi);
    if (M.clipRim <= 0) return mech;
    if (dcl <= dap) return mech;
    const double vx = dp > 1e-12 ? std::fabs(px) / dp : 1.0, vy = dp > 1e-12 ? std::fabs(py) / dp : 0.0;
    double ax = vx * fx, by = vy * fy;
    if (ax >= by) ax += smAp; else by += smAp;
    const double den = T.cfE[side].filteredIn(dap, ax, by, -1e30, 1.0);
    const double num = T.cfME[side].filteredIn(dcl, aw, bw, lo, hi);
    // Where the aperture's own soft edge makes E(d_ap) -> 0, A is ~0 too; the ratio must not
    // amplify the (antialiased) residue: clamp to the physically possible range.
    double r = num / std::max(den, 1e-3);
    r = std::min(r, T.eMax[side]);
    if (den < 0.25) r = std::min(r, 1.0);
    return (1.0 - M.clipRim) * mech + M.clipRim * r;
}

bool sameProfile(const Profile& a, const Profile& b) {
    return a.e == b.e && a.rw == b.rw && a.soft == b.soft && a.apod == b.apod && a.core == b.core &&
           a.obstr == b.obstr && a.rimShape == b.rimShape;
}

// texel images of A (base and onion delta) for one profile
bool buildA(const Model& M, Tables& T, const Profile& P, Sat& A0, Sat& A1, const Texture& tex, int nth, AbortFn ab, void* user) {
    const int n = T.n;
    std::vector<double> t0((size_t)n * n), t1(T.onion ? (size_t)n * n : 0);
    const double tx = 2.0 / n;
    const int SS = 4;
    const bool ok = parallelFor(n, nth, 4, [&](int j, int) {
        if (cancelled()) return;
        for (int i = 0; i < n; ++i) {
            const double u0 = -1 + i * tx, v0 = -1 + j * tx;
            const double uc = u0 + 0.5 * tx, vc = v0 + 0.5 * tx;
            const double nx = std::max(std::fabs(uc) - 0.5 * tx, 0.0), ny = std::max(std::fabs(vc) - 0.5 * tx, 0.0);
            if (nx * nx + ny * ny > 1.0) { t0[(size_t)j * n + i] = 0; if (T.onion) t1[(size_t)j * n + i] = 0; continue; }
            double s0 = 0, s1 = 0;
            for (int b = 0; b < SS; ++b)
                for (int a = 0; a < SS; ++a) {
                    const double u = u0 + (a + 0.5) * tx / SS, v = v0 + (b + 0.5) * tx / SS;
                    const double r = M.ap.gauge(u, v);
                    if (r > 1.0) continue;
                    const double I = P.base(r);
                    s0 += I;
                    if (T.onion) s1 += I * 0.5 * std::cos(2 * kPi * T.onionN * r);
                }
            const double Tv = tex.eval(uc, vc);
            t0[(size_t)j * n + i] = s0 * Tv / (SS * SS);
            if (T.onion) t1[(size_t)j * n + i] = s1 * Tv / (SS * SS);
        }
    }, ab, user);
    if (!ok) return false;
    A0.build(t0, n, 1.0);
    if (T.onion) A1.build(t1, n, 1.0);
    return true;
}

// per-channel spectral tables K_ch(u) = sum_i w_i(ch) A(u / s_i) / s_i^2 (texel averages, exact
// from A's SAT over the scaled texel box), over the shared sub-scale list. Row bands share corner
// rows, so a texel costs one SAT sample per sub-scale.
bool buildK(const Model& M, Tables& T, int side, int nth, AbortFn ab, void* user) {
    const Spectrum& sp = M.spec;
    const std::vector<SubScale>& sub = M.sub[side];
    const Sat& A0 = T.A[T.aSide[side]][0];
    const Sat* A1 = T.onion ? &T.A[T.aSide[side]][1] : nullptr;
    double smax = 0;
    for (const SubScale& s : sub) smax = std::max(smax, s.s);
    const int nk = std::max(T.n, (int)std::ceil(T.n * smax / 64.0) * 64);
    const double Dk = (double)nk / T.n;
    const double tx = 2.0 * Dk / nk;  // == 2 / n
    const int parts = T.onion ? 2 : 1;
    std::vector<double> img[3][2];
    for (int c = 0; c < T.nch; ++c) for (int pp = 0; pp < parts; ++pp) img[c][pp].assign((size_t)nk * nk, 0.0);
    const double* S[2] = {A0.lv[0].data(), A1 ? A1->lv[0].data() : nullptr};
    const int na = A0.n;
    const double sc = na / 2.0;
    const int NS = (int)sub.size();
    const int band = 16;
    const int nb = (nk + band - 1) / band;
    const double inv = 1.0 / (tx * tx);
    const bool ok = parallelFor(nb, nth, 1, [&](int b, int) {
        const int j0 = b * band, j1 = std::min(nk, j0 + band);
        std::vector<double> prev(nk + 1), cur(nk + 1), cx(nk + 1);
        for (int ss = 0; ss < NS; ++ss) {
            if (cancelled()) return;
            const double s = sub[ss].s;
            double wc[3];
            for (int c = 0; c < T.nch; ++c) wc[c] = sp.w[c][sub[ss].k] * sub[ss].w * inv;
            for (int i = 0; i <= nk; ++i) cx[i] = ((-Dk + i * tx) / s + 1) * sc;
            for (int pp = 0; pp < parts; ++pp) {
                const double* SS = S[pp];
                for (int j = j0; j <= j1; ++j) {
                    const double ty = ((-Dk + j * tx) / s + 1) * sc;
                    for (int i = 0; i <= nk; ++i) cur[i] = satAt(SS, na, cx[i], ty);
                    if (j > j0) {
                        const size_t ro = (size_t)(j - 1) * nk;
                        for (int i = 0; i < nk; ++i) {
                            const double v = (cur[i + 1] - cur[i]) - (prev[i + 1] - prev[i]);
                            if (v == 0) continue;
                            for (int c = 0; c < T.nch; ++c) img[c][pp][ro + i] += wc[c] * v;
                        }
                    }
                    std::swap(prev, cur);
                }
            }
        }
    }, ab, user);
    if (!ok) return false;
    for (int c = 0; c < T.nch; ++c)
        for (int pp = 0; pp < parts; ++pp) { T.K[side][c][pp].build(img[c][pp], nk, Dk); img[c][pp].clear(); img[c][pp].shrink_to_fit(); }
    return true;
}

// estimated bytes of a table set at resolution n (corner grids + mips, doubles)
double estimateTableBytes(const Model& M, int n, bool sameSides) {
    double smax = 1;
    for (int s = 0; s < 2; ++s) for (const SubScale& ss : M.sub[s]) smax = std::max(smax, ss.s);
    const double a = 1.34 * (n + 1.0) * (n + 1.0) * 8.0;
    const int parts = M.onion ? 2 : 1;
    const int aT = sameSides ? 1 : 2;
    double b = 0;
    const bool mono = M.spec.N == 1;
    if (!mono && !M.lat) b += 2.0 * 3 * parts * a * smax * smax;           // K tables
    if (M.pathB || mono) b += aT * parts * a;                               // A tables kept
    else b += aT * parts * a;                                               // transient during buildK
    return b;
}

std::string tableKey(const Model& M, const Params& p, const HqOptions& q, int n) {
    char buf[1024];
    std::snprintf(buf, sizeof buf,
        "v22|n%d|bl%d|%.9g|%.9g|%.9g|e%.9g|sa%.9g|rw%.9g|so%.9g|ap%.9g|co%.9g|ob%.9g|rs%.9g|on%.9g|%.9g|"
        "tx%.9g|%.9g|%d|%d|fr%.9g|ti%.9g|fm%d|ns%d|nsub%d,%d|pb%d|lat%d|cat%d|%.9g|%.9g|%.9g|%.9g|loss%d|%.9g",
        n, p.blades, (double)clampd(p.bladeCurvature, 0, 1), (double)p.rotationDeg, (double)clampd(p.cornerRounding, 0, 1),
        (double)p.edgeRing, (double)p.sphericalAberration, (double)p.ringWidth, (double)p.edgeSoftness,
        (double)p.apodization, (double)p.coreWeight, M.obstr, (double)clampd(q.rimShape, 0, 1),
        (double)clampd(p.onionRings, 0, 1), (double)clampd(p.onionCount, 1, 10),
        (double)q.textureAmount, (double)q.textureScale, q.textureSeed, q.textureType,
        (double)p.chromaFringe, (double)p.fringeTint, q.fringeModel, M.spec.N,
        (int)M.sub[0].size(), (int)M.sub[1].size(), (int)M.pathB, (int)M.lat, (int)M.cat,
        M.rho, M.csoft, M.clipRim, M.catK * 1000 + M.catCurve, (int)(M.loss > 0), M.catK);
    return buf;
}

// ---- process-wide cache of immutable table sets (the only shared state; see lenshq.h)
struct TableCache {
    std::mutex m;
    std::list<std::pair<std::string, std::shared_ptr<const Tables>>> lru;  // front = most recent
    size_t bytes = 0;
    size_t budget() const { return (size_t)1536 * 1048576ull; }  // see lenshq.h
    std::shared_ptr<const Tables> get(const std::string& k) {
        std::lock_guard<std::mutex> lk(m);
        for (auto it = lru.begin(); it != lru.end(); ++it)
            if (it->first == k) { lru.splice(lru.begin(), lru, it); return lru.front().second; }
        return nullptr;
    }
    void put(const std::string& k, const std::shared_ptr<const Tables>& t) {
        std::lock_guard<std::mutex> lk(m);
        for (auto& e : lru) if (e.first == k) return;
        const size_t b = t->bytes(), cap = budget();
        if (b > cap) return;
        lru.emplace_front(k, t);
        bytes += b;
        while (bytes > cap || lru.size() > 8) {
            bytes -= lru.back().second->bytes();
            lru.pop_back();
        }
    }
    void clear() { std::lock_guard<std::mutex> lk(m); lru.clear(); bytes = 0; }
};
TableCache& tableCache() { static TableCache c; return c; }

Status buildTables(Model& M, const Params& p, const HqOptions& q, int nth, AbortFn ab, void* user, bool& hit) {
    hit = false;
    const double t0 = nowSec();
    Profile prof[2];
    const double e = clampd(p.edgeRing, -1.0, 1.0), sa = clampd(p.sphericalAberration, -1.0, 1.0);
    for (int side = 0; side < 2; ++side) {
        Profile& P = prof[side];
        P.e = clampd(side ? e + sa : e - sa, -1.0, 1.0);
        P.rw = clampd(p.ringWidth, 0.02, 0.98);
        P.soft = clampd(p.edgeSoftness, 0.0, 1.0);
        P.apod = clampd(p.apodization, 0.0, 1.0);
        P.core = clampd(p.coreWeight, -1.0, 1.0);
        P.obstr = M.obstr;
        P.rimShape = clampd(q.rimShape, 0.0, 1.0);
        if (P.soft < 1e-4) P.soft = 0;
        if (P.apod < 1e-4) P.apod = 0;
        if (std::fabs(P.core) < 1e-4) P.core = 0;
        if (P.obstr < 1e-4) P.obstr = 0;
    }
    const bool same = sameProfile(prof[0], prof[1]);
    // resolution: from the maxRadius slider (data independent -> tile exact), multiple of 64,
    // reduced until the estimated size fits the budget
    int n;
    if (q.tableResolution > 0) n = (int)clampd(std::ceil(q.tableResolution / 64.0) * 64, 256, 4096);
    else {
        double smax = 1;
        for (int s = 0; s < 2; ++s) for (const SubScale& ss : M.sub[s]) smax = std::max(smax, ss.s);
        const double need = 2.0 * M.tpp * M.maxR * std::max(M.ax, M.ay) * smax;
        n = (int)clampd(std::ceil(need / 64.0) * 64, 256, 4096);
    }
    const double budget = clampd(q.tableBudgetMB, 64, 16384) * 1048576.0;
    while (n > 256 && estimateTableBytes(M, n, same) > budget) n -= 64;
    const std::string key = tableKey(M, p, q, n);
    if (std::shared_ptr<const Tables> c = tableCache().get(key)) { M.T = c; hit = true; return Status::Ok; }

    std::shared_ptr<Tables> T = std::make_shared<Tables>();
    T->n = n;
    T->prof[0] = prof[0]; T->prof[1] = prof[1];
    T->onion = M.onion; T->onionN = M.onionN;
    Texture tex;
    tex.init(q);
    T->aSide[0] = 0; T->aSide[1] = same ? 0 : 1;
    for (int t = 0; t < (same ? 1 : 2); ++t)
        if (!buildA(M, *T, prof[t], T->A[t][0], T->A[t][1], tex, nth, ab, user)) return Status::Aborted;
    const bool mono = M.spec.N == 1;
    T->nch = mono ? 1 : 3;
    T->kFromA = mono;
    for (int t = 0; t < 2; ++t) T->kSide[t] = mono ? T->aSide[t] : t;
    if (!mono && !M.lat) {
        for (int side = 0; side < 2; ++side)
            if (!buildK(M, *T, side, nth, ab, user)) return Status::Aborted;
        if (!M.pathB) for (int t = 0; t < 2; ++t) for (int pp = 0; pp < 2; ++pp) T->A[t][pp] = Sat();
    }
    T->kbx0 = T->kby0 = 1e30; T->kbx1 = T->kby1 = -1e30;
    for (int t = 0; t < 2; ++t)
        for (int c = 0; c < T->nch; ++c) {
            if (M.lat && !mono) continue;
            const Sat& S = T->kTab(t, c, 0);
            if (S.lv.empty()) continue;
            T->kbx0 = std::min(T->kbx0, S.bx0); T->kbx1 = std::max(T->kbx1, S.bx1);
            T->kby0 = std::min(T->kby0, S.by0); T->kby1 = std::max(T->kby1, S.by1);
        }
    if (M.pathB) {
        // clip-edge functions: mech(d) = mechanical cut (1 up to the clip radius, soft roll-off of
        // width catEyeSoftness beyond it), E(d) = the aperture's edge character (rim / soft edge) as a
        // function of a gauge, ME = mech * E. See clipFactor().
        const int NS = 26001;
        const double d0 = -1.0, d1 = 1.6;
        const double sig = M.csoft / M.rho;
        {
            std::vector<double> smp(NS);
            for (int i = 0; i < NS; ++i) {
                const double d = d0 + (d1 - d0) * i / (NS - 1);
                smp[i] = sig < 1e-5 ? (d <= 1.0 ? 1.0 : 0.0) : 1.0 - smooth01((d - 1.0) / sig);
            }
            T->cfM.build(smp, d0, d1);
        }
        for (int side = 0; side < 2; ++side) {
            std::vector<double> se(NS), sme(NS);
            for (int i = 0; i < NS; ++i) {
                const double d = d0 + (d1 - d0) * i / (NS - 1);
                const double ed = d < 0 ? 1.0 : prof[side].edge(d);
                se[i] = ed;
                sme[i] = T->cfM.f[i] * ed;
            }
            T->cfE[side].build(se, d0, d1);
            T->cfME[side].build(sme, d0, d1);
            T->eMax[side] = *std::max_element(se.begin(), se.end());
        }
        M.T = T;  // clipFactor needs the edge functions below
        // light-loss table F(h) = transmitted fraction of the pupil, per side
        if (M.loss > 0) {
            T->lossN = 41;
            T->lossLim = std::min(M.catMaxM, 0.5 * M.catK * std::pow(1.5, M.catCurve)) + 1e-3;
            for (int side = 0; side < 2; ++side) {
                const Sat& A0 = T->A[T->aSide[side]][0];
                int lv = 0;
                while (lv + 1 < (int)A0.nl.size() && A0.nl[lv] > 256) ++lv;
                const int nn = A0.nl[lv];
                const double* S = A0.lv[lv].data();
                std::vector<double> ta((size_t)nn * nn);
                for (int j = 0; j < nn; ++j)
                    for (int i = 0; i < nn; ++i) {
                        const double* r0 = S + (size_t)j * (nn + 1) + i;
                        const double* r1 = r0 + nn + 1;
                        ta[(size_t)j * nn + i] = r1[1] - r1[0] - r0[1] + r0[0];
                    }
                T->lossF[side].assign((size_t)T->lossN * T->lossN, 1.0);
                const double txl = 2.0 / nn;
                const double tot = A0.total;
                std::vector<double>& LF = T->lossF[side];
                const bool okL = parallelFor(T->lossN, nth, 1, [&](int jj, int) {
                    for (int ii = 0; ii < T->lossN; ++ii) {
                        if (cancelled()) return;
                        const double hx = -T->lossLim + 2 * T->lossLim * ii / (T->lossN - 1);
                        const double hy = -T->lossLim + 2 * T->lossLim * jj / (T->lossN - 1);
                        double s = 0;
                        for (int j = 0; j < nn; ++j) {
                            const double py = -1 + (j + 0.5) * txl;
                            for (int i = 0; i < nn; ++i) {
                                const double a = ta[(size_t)j * nn + i];
                                if (a == 0) continue;
                                const double px = -1 + (i + 0.5) * txl;
                                s += a * clipFactor(M, side, px - 2 * hx, py - 2 * hy, px, py, txl, txl);
                            }
                        }
                        LF[(size_t)jj * T->lossN + ii] = tot > 0 ? std::min(1.0, s / tot) : 1.0;
                    }
                }, ab, user);
                if (!okL) return Status::Aborted;
            }
        }
    }
    T->buildSec = nowSec() - t0;
    M.T = T;
    tableCache().put(key, T);
    return Status::Ok;
}

Status buildModel(Model& M, const Params& p, const HqOptions& q, int w, int h, int nth, AbortFn ab, void* user, bool& hit) {
    buildGeo(M, p, q);
    M.W = w; M.H = h;
    frameOf(p, w, h, M.fcx, M.fcy, M.halfDiag);
    M.flipFront = q.invertFrontBokeh;
    M.tpp = clampd(q.minTexelsPerPixel, 1.0, 64.0);
    M.latTol = std::isfinite(q.latticeTolerance) ? clampd(q.latticeTolerance, 0.0, 8.0) : 1.0;
    M.tile = M.maxR * std::max(M.ax, M.ay) >= 64 ? 32 : 64;  // from the slider: data independent
    M.onionStr = clampd(p.onionRings, 0.0, 1.0);
    M.onionN = clampd(p.onionCount, 1.0, 10.0);
    M.onion = M.onionStr > 1e-4;
    {   // blending stamps whose sharp edges are eps apart costs ~eps^2 x (edge contrast): hard outer
        // edges and bright rim steps get a finer lattice (measured against exact stamps, harness)
        const double e = clampd(p.edgeRing, -1.0, 1.0), sa = clampd(p.sphericalAberration, -1.0, 1.0);
        const double rim = 3.0 * std::max(0.0, std::max(clampd(e + sa, -1.0, 1.0), clampd(e - sa, -1.0, 1.0)));
        const bool hard = clampd(p.edgeSoftness, 0.0, 1.0) < 0.04 && clampd(p.apodization, 0.0, 1.0) < 0.3;
        // fitted to measured lattice-vs-exact L1 at eps = 0.07 (plain hard disc 5e-4, premium85 6e-4,
        // sparkle 3e-3, outlined 5e-3, chromatic / swirly 1.4e-3): target ~7e-4
        double f = (hard ? 1.25 : 1.0) + 0.8 * rim;
        if (M.onion) f += 1.5 * M.onionStr * M.onionN / 6.0;
        if (clampd(q.textureAmount, 0.0, 2.0) > 0) f *= 1.0 + 0.3 * clampd(q.textureAmount, 0.0, 2.0);
        M.latEps0 = 0.055 / std::max(1.0, f);
    }
    return buildTables(M, p, q, nth, ab, user, hit);
}

// ------------------------------------------------------------ per-source ----

struct Src {
    float c[4];
    float aux;
    float R;
    int8_t side;   // 0 front, 1 back
    bool ident;
    float gx, gy;  // frame position / half diagonal (|g| <= 1.5)
    float hx, hy;  // cat-eye half offset (unit space)
    float dirx, diry, rf;
};

// position-dependent kernel parameters from g
inline void paramsFromG(const Model& M, double gx, double gy, Src& s) {
    double len = std::sqrt(gx * gx + gy * gy);
    if (len > 1.5) { gx *= 1.5 / len; gy *= 1.5 / len; len = 1.5; }
    s.gx = (float)gx; s.gy = (float)gy;
    s.rf = (float)len;
    s.hx = s.hy = 0.f; s.dirx = 1.f; s.diry = 0.f;
    if (len > 1e-12) {
        s.dirx = (float)(gx / len); s.diry = (float)(gy / len);
        if (M.cat) {
            double dx = gx / len / M.ax, dy = gy / len / M.ay;
            const double dl = std::sqrt(dx * dx + dy * dy);
            const double m = std::min(M.catMaxM, 0.5 * M.catK * std::pow(len, M.catCurve));
            s.hx = (float)(m * dx / dl);
            s.hy = (float)(m * dy / dl);
        }
    }
}

struct Stamp {
    int i0 = 0, j0 = 0, w = 0, h = 0, np = 1;
    std::vector<float> pl;
    std::vector<int> rlo, rhi;  // per row: nonzero span [rlo, rhi) over all planes
    long long lastUse = 0;
    void spans() {
        const size_t pn = (size_t)w * h;
        rlo.assign(h, 0); rhi.assign(h, 0);
        for (int j = 0; j < h; ++j) {
            int lo = w, hi = 0;
            for (int c = 0; c < np; ++c) {
                const float* r = pl.data() + (size_t)c * pn + (size_t)j * w;
                int a = 0, b = w;
                while (a < b && r[a] == 0.f) ++a;
                while (b > a && r[b - 1] == 0.f) --b;
                if (b > a) { lo = std::min(lo, a); hi = std::max(hi, b); }
            }
            rlo[j] = lo < hi ? lo : 0; rhi[j] = lo < hi ? hi : 0;
        }
    }
};

struct StampKey {
    int64_t a, b, c;
    bool operator==(const StampKey& o) const { return a == o.a && b == o.b && c == o.c; }
};
struct StampKH { size_t operator()(const StampKey& k) const { return (size_t)(k.a * 1000003 ^ k.b * 9176 ^ k.c * 7919); } };

// lattice stamps of one render, shared by all threads: each node stamp is computed once (a thread
// that needs a stamp another thread is computing waits for it, polling abort every 10 ms). Tiles pin
// the stamps they use (shared_ptr); the LRU only evicts stamps no tile pins, down to the budget.
// Values do not depend on which thread computed a stamp, so results stay thread-count independent.
struct SharedStamps {
    struct E { std::shared_ptr<Stamp> st; bool ready = false; std::list<StampKey>::iterator it; size_t bytes = 0; };
    std::mutex m;
    std::condition_variable cv;
    std::unordered_map<StampKey, E, StampKH> map;
    std::list<StampKey> lru;  // ready entries, most recent first
    size_t bytes = 0, budget = 0, peak = 0;
    void evictLocked() {
        auto it = lru.end();
        while (bytes > budget && it != lru.begin()) {
            --it;
            auto f = map.find(*it);
            if (f != map.end() && f->second.st.use_count() == 1) {  // pinned by no tile
                bytes -= f->second.bytes;
                map.erase(f);
                it = lru.erase(it);
            }
        }
    }
};

// per thread: the stamps the current tile pins
struct StampCache {
    using Key = StampKey;
    std::unordered_map<StampKey, std::shared_ptr<Stamp>, StampKH> map;
    void release() { map.clear(); }
};

// a lattice source of the current tile: its node stamps (pinned in the cache for the tile)
struct LatSrc {
    int x, y, n;
    const Stamp* st[8];
    float w[8];
    int i0, j0, w_, h_;   // union bbox of the node stamps (offsets from the source pixel)
    int srcIdx;
};

struct Work {
    std::vector<float> buf;
    std::vector<LatSrc> lat;
    std::vector<float> rowbuf;
    std::vector<Src> srcs;
    Stamp st;            // exact stamp (and the previous one for reuse)
    bool stValid = false;
    double kR = -1; int kSide = -1; float kgx = 0, kgy = 0;
    Stamp blend;
    StampCache cache;
    SharedStamps* shared = nullptr;
    std::vector<float> tmp, tmp2;
    std::vector<int> ix;
    std::vector<double> fx, fx2, rowA, rowB;
    std::vector<uint8_t> cls;
    long long nStamps = 0, nStampPx = 0, nLamPx = 0, nLat = 0;
};

inline void reachOf(const Model& M, double R, int& rx, int& ry) {
    const double sig = diffSigma(M, R);
    const double extra = M.latMax * 1.5 * R + 3.0 * sig + 1.0;
    rx = (int)std::ceil(R * M.ax * M.extX + extra);
    ry = (int)std::ceil(R * M.ay * M.extY + extra);
}

inline bool isIdentity(const Model& M, double R) {
    if (R <= 0) return true;
    const double sig = diffSigma(M, R);
    const double extra = M.latMax * 1.5 * R + (sig > 0.03 ? 3.0 * sig : 0.0);
    return R * M.ax * M.extX + extra <= 0.5 && R * M.ay * M.extY + extra <= 0.5;
}

inline void setIdentity(Stamp& st, int np) {
    st.i0 = st.j0 = 0; st.w = st.h = 1; st.np = np;
    st.pl.assign((size_t)np, 1.f);
    st.rlo.assign(1, 0); st.rhi.assign(1, 1);
}

// separable Gaussian blur of the stamp planes (diffraction), grows the stamp by rr each side
void diffuseStamp(Stamp& st, double sig, Work& W) {
    const int rr = diffRad(sig);
    if (rr <= 0) return;
    std::vector<double> g(2 * rr + 1);
    double gs = 0;
    for (int k = -rr; k <= rr; ++k) { g[k + rr] = std::exp(-0.5 * k * k / (sig * sig)); gs += g[k + rr]; }
    for (double& v : g) v /= gs;
    const int w2 = st.w + 2 * rr, h2 = st.h + 2 * rr;
    W.tmp2.assign((size_t)st.np * w2 * h2, 0.f);
    std::vector<float> row((size_t)w2 * st.h);
    for (int c = 0; c < st.np; ++c) {
        const float* src = st.pl.data() + (size_t)c * st.w * st.h;
        std::fill(row.begin(), row.end(), 0.f);
        for (int j = 0; j < st.h; ++j)
            for (int i = 0; i < st.w; ++i) {
                const float v = src[(size_t)j * st.w + i];
                if (v == 0.f) continue;
                float* o = row.data() + (size_t)j * w2 + i;
                for (int k = 0; k <= 2 * rr; ++k) o[k] += (float)(v * g[k]);
            }
        float* dst = W.tmp2.data() + (size_t)c * w2 * h2;
        for (int j = 0; j < st.h; ++j)
            for (int i = 0; i < w2; ++i) {
                const float v = row[(size_t)j * w2 + i];
                if (v == 0.f) continue;
                for (int k = 0; k <= 2 * rr; ++k) dst[(size_t)(j + k) * w2 + i] += (float)(v * g[k]);
            }
    }
    st.pl.swap(W.tmp2);
    st.i0 -= rr; st.j0 -= rr; st.w = w2; st.h = h2;
}

// normalise each plane by its discrete sum; a plane without energy -> identity (no black hole)
void normalizeStamp(Stamp& st) {
    const size_t pn = (size_t)st.w * st.h;
    double z[3] = {0, 0, 0};
    bool bad = false;
    for (int c = 0; c < st.np; ++c) {
        const float* pl = st.pl.data() + (size_t)c * pn;
        for (size_t i = 0; i < pn; ++i) z[c] += pl[i];
        if (!(z[c] > 1e-30) || !std::isfinite(z[c])) bad = true;
    }
    if (bad) { setIdentity(st, st.np); return; }
    for (int c = 0; c < st.np; ++c) {
        float* pl = st.pl.data() + (size_t)c * pn;
        const float iz = (float)(1.0 / z[c]);
        for (size_t i = 0; i < pn; ++i) pl[i] *= iz;
    }
    st.spans();
}

// ---- path A stamp: kernel depends on (R, side) only
void stampA(const Model& M, double R, int side, Stamp& st, Work& W) {
    const Tables& T = *M.T;
    const double amp = M.onionAmp(R);
    const double sig = (side == 0 && M.flipFront) ? -1.0 : 1.0;
    const double Rx = R * M.ax, Ry = R * M.ay;
    const double bx = std::max(-T.kbx0, T.kbx1), by = std::max(-T.kby0, T.kby1);
    st.i0 = (int)std::floor(-bx * Rx - 0.5); const int i1 = (int)std::ceil(bx * Rx + 0.5);
    st.j0 = (int)std::floor(-by * Ry - 0.5); const int j1 = (int)std::ceil(by * Ry + 0.5);
    st.w = i1 - st.i0 + 1; st.h = j1 - st.j0 + 1;
    st.np = T.nch;
    const size_t pn = (size_t)st.w * st.h;
    st.pl.assign((size_t)st.np * pn, 0.f);
    const Sat& ref = T.kTab(side, 0, 0);
    const int lv = ref.levelFor(R * std::max(M.ax, M.ay), M.tpp);
    const int n = ref.nl[lv];
    const double D = ref.D, scl = n / (2 * D);
    W.ix.resize(st.w + 1); W.fx.resize(st.w + 1);
    for (int k = 0; k <= st.w; ++k) {
        double t = (sig * (st.i0 - 0.5 + k) / Rx + D) * scl;
        t = t < 0 ? 0 : (t > n ? n : t);
        int ii = (int)t;
        if (ii > n - 1) ii = n - 1;
        W.ix[k] = ii; W.fx[k] = t - ii;
    }
    W.rowA.resize(st.w + 1); W.rowB.resize(st.w + 1);
    for (int c = 0; c < st.np; ++c) {
        const Sat& K0 = T.kTab(side, c, 0);
        const Sat* K1 = T.onion ? &T.kTab(side, c, 1) : nullptr;
        const double* S0 = K0.lv[lv].data();
        const double* S1 = K1 ? K1->lv[lv].data() : nullptr;
        float* out = st.pl.data() + (size_t)c * pn;
        double* prev = W.rowA.data();
        double* cur = W.rowB.data();
        for (int l = 0; l <= st.h; ++l) {
            if ((l & 15) == 0 && cancelled()) return;
            double t = (sig * (st.j0 - 0.5 + l) / Ry + D) * scl;
            t = t < 0 ? 0 : (t > n ? n : t);
            int iy = (int)t;
            if (iy > n - 1) iy = n - 1;
            const double fy = t - iy;
            const double* r0 = S0 + (size_t)iy * (n + 1);
            const double* r1 = r0 + (n + 1);
            for (int k = 0; k <= st.w; ++k) {
                const int ii = W.ix[k];
                const double f = W.fx[k];
                const double a = r0[ii] + f * (r0[ii + 1] - r0[ii]);
                const double b = r1[ii] + f * (r1[ii + 1] - r1[ii]);
                cur[k] = a + fy * (b - a);
            }
            if (S1 && amp > 0) {
                const double* q0 = S1 + (size_t)iy * (n + 1);
                const double* q1 = q0 + (n + 1);
                for (int k = 0; k <= st.w; ++k) {
                    const int ii = W.ix[k];
                    const double f = W.fx[k];
                    const double a = q0[ii] + f * (q0[ii + 1] - q0[ii]);
                    const double b = q1[ii] + f * (q1[ii + 1] - q1[ii]);
                    cur[k] += amp * (a + fy * (b - a));
                }
            }
            if (l > 0) {
                // with sig = -1 both axes are reversed: inclusion-exclusion keeps its sign
                float* o = out + (size_t)(l - 1) * st.w;
                for (int k = 0; k < st.w; ++k) {
                    const double v = (cur[k + 1] - cur[k]) - (prev[k + 1] - prev[k]);
                    o[k] = (float)(v > 0 ? v : 0.0);
                }
            }
            std::swap(prev, cur);
        }
    }
}

#include "lenshq_stampb.inl"

// ------------------------------------------------------------ scatter -------

void computeStamp(const Model& M, const Src& s, Stamp& st, Work& W) {
    if (M.pathB) stampB(M, s, st, W);
    else stampA(M, s.R, s.side, st, W);
    diffuseStamp(st, diffSigma(M, s.R), W);
    normalizeStamp(st);
    ++W.nStamps;
}

// two-word fixed-point accumulator (see the file header)
struct Accum {
    int W = 0, H = 0, NC = 5, NV = 4;  // NV = value channels (r g b a [aux]); weights after them
    bool tri = false, aux = false;
    int a = 40;                         // hi word scale 2^a for values
    double sHi = 0, sLo = 0, iHi = 0, iLo = 0, sW = 0, iW = 0;
    std::vector<int64_t> hi, lo, wt;     // values hi/lo (NV per pixel), weights (NW per pixel)
    int NW = 1;
    std::unique_ptr<std::mutex[]> locks;
    int nLocks = 0;
};

inline int64_t toI64(double v) {
    const double lim = 4.0e18;
    if (!(v == v)) return 0;
    v = v < -lim ? -lim : (v > lim ? lim : v);
    return (int64_t)(v >= 0 ? v + 0.5 : v - 0.5);
}

void mergeTile(Accum& A, const float* buf, int bx0, int by0, int bw, int bh) {
    const int NC = A.NC, NV = A.NV, NW = A.NW;
    int curLock = -1;
    for (int j = 0; j < bh; ++j) {
        const int y = by0 + j;
        const int lk = y / kBand;
        if (lk != curLock) {
            if (curLock >= 0) A.locks[curLock].unlock();
            A.locks[lk].lock();
            curLock = lk;
        }
        const float* b = buf + (size_t)j * bw * NC;
        const size_t base = (size_t)y * A.W + bx0;
        int64_t* hi = A.hi.data() + base * NV;
        int64_t* lo = A.lo.data() + base * NV;
        int64_t* wt = A.wt.data() + base * NW;
        for (int i = 0; i < bw; ++i, b += NC, hi += NV, lo += NV, wt += NW) {
            for (int c = 0; c < NV; ++c) {
                const double v = b[c];
                if (v == 0) continue;
                const double h = std::floor(v * A.sHi + 0.5);
                hi[c] += toI64(h);
                lo[c] += toI64((v - h * A.iHi) * A.sLo);
            }
            for (int c = 0; c < NW; ++c) {
                const double v = b[NV + c];
                if (v != 0) wt[c] += toI64(v * A.sW);
            }
        }
    }
    if (curLock >= 0) A.locks[curLock].unlock();
}

// channel layout in the tile buffer: values r g b a [aux], then weights (1 or 3)
inline void addIdentity(float* b, const Accum& A, const Src& s) {
    b[0] += s.c[0]; b[1] += s.c[1]; b[2] += s.c[2]; b[3] += s.c[3];
    if (A.aux) b[4] += s.aux;
    for (int c = 0; c < A.NW; ++c) b[A.NV + c] += 1.f;
}

void accumulate(const Stamp& st, const Src& s, int sx, int sy, float* buf, int bx0, int by0, int bw, int bh, const Accum& A) {
    const int NC = A.NC, NV = A.NV;
    const size_t pn = (size_t)st.w * st.h;
    const int iA = std::max(0, bx0 - sx - st.i0), iB = std::min(st.w, bx0 + bw - sx - st.i0);
    const int jA = std::max(0, by0 - sy - st.j0), jB = std::min(st.h, by0 + bh - sy - st.j0);
    const float cr = s.c[0], cg = s.c[1], cb = s.c[2], ca = s.c[3], cx = s.aux;
    const bool aux = A.aux;
    const bool tri = st.np == 3;
    for (int j = jA; j < jB; ++j) {
        float* row = buf + ((size_t)(sy + st.j0 + j - by0) * bw + (sx + st.i0 - bx0)) * NC;
        const float* p0 = st.pl.data() + (size_t)j * st.w;
        if (!tri) {
            for (int i = iA; i < iB; ++i) {
                const float k = p0[i];
                if (k == 0) continue;
                float* q = row + (size_t)i * NC;
                q[0] += k * cr; q[1] += k * cg; q[2] += k * cb; q[3] += k * ca;
                if (aux) q[4] += k * cx;
                for (int c = 0; c < A.NW; ++c) q[NV + c] += k;
            }
        } else {
            const float* p1 = p0 + pn;
            const float* p2 = p1 + pn;
            for (int i = iA; i < iB; ++i) {
                const float kr = p0[i], kg = p1[i], kb = p2[i];
                if (kr == 0 && kg == 0 && kb == 0) continue;
                float* q = row + (size_t)i * NC;
                q[0] += kr * cr; q[1] += kg * cg; q[2] += kb * cb; q[3] += kg * ca;
                if (aux) q[4] += kg * cx;
                q[NV] += kr; q[NV + 1] += kg; q[NV + 2] += kb;
            }
        }
    }
}

// fill source info for pixel (x, y)
inline void prepSrc(const Model& M, const Params& p, const HqOptions& q, const float* src, int ss,
                    const float* map, int ms, int x, int y, Src& s) {
    const float* c = src + (size_t)y * ss + (size_t)x * 4;
    for (int k = 0; k < 4; ++k) s.c[k] = std::isfinite(c[k]) ? c[k] : 0.f;
    s.aux = q.auxSrc ? q.auxSrc[(size_t)y * q.auxSrcStride + x] : 0.f;
    if (!std::isfinite(s.aux)) s.aux = 0.f;
    double amt = 1.0, sgn = 1.0, R;
    if (q.radiusIn) {
        const size_t i = (size_t)y * M.W + x;
        R = q.radiusIn[i];
        if (!(R > 0)) R = 0;  // NaN / negative -> in focus
        R = std::min(R, M.maxR);
        sgn = q.sideIn && q.sideIn[i] < 0 ? -1.0 : 1.0;
    } else {
        if (map) amt = amountOf(map[(size_t)y * ms + x], p, sgn);
        R = amt * M.maxR;
    }
    s.R = (float)R;
    s.side = (int8_t)(sgn > 0 ? 1 : 0);
    paramsFromG(M, (x + 0.5 - M.fcx) / M.halfDiag, (y + 0.5 - M.fcy) / M.halfDiag, s);
    if (M.loss > 0) {
        const double f = M.lossAt(s.side, s.hx, s.hy);
        const float l = (float)(1.0 - M.loss * (1.0 - f));
        s.c[0] *= l; s.c[1] *= l; s.c[2] *= l; s.aux *= l;
    }
    s.ident = isIdentity(M, R);
}

// ---- lattice -------------------------------------------------------------------------------
struct LatNode { double R; int side; double gx, gy; StampCache::Key key; double w; };

inline double latEps(const Model& M) { return M.latEps0 * M.latTol; }  // * sqrt(R) px

// g step at radius R: kernel features move <= eps*sqrt(R) px between nodes
double gStep(const Model& M, double R) {
    if (!M.pathB) return 0;
    const double amax = std::max(M.ax, M.ay);
    double rate = 0;  // px per unit g
    if (M.cat) {
        const double c = M.catCurve;
        const double rfMax = 1.5;
        const double slope = c >= 1 ? 0.5 * M.catK * c * std::pow(rfMax, c - 1) : 0.5 * M.catK * c * std::pow(0.2, c - 1);
        rate += 2.0 * R * amax * std::max(slope, 0.5 * M.catK * std::pow(rfMax, c - 1));
    }
    if (M.lat) rate += M.latMax * R;
    // the clip / lateral shift moves only part of the outline: 1.4 x the radius step
    double dg = rate > 0 ? 1.4 * latEps(M) * std::sqrt(R) / rate : 1.0;
    if (M.discG > 0) dg = std::min(dg, 0.004 * M.latTol / M.discG);
    // QUALITY CAP (time / memory): a lattice cell is never smaller than 4.5 frame px (at most one
    // exact stamp per ~20 sources), nor so small that the stamps a tile pins exceed the per-thread
    // budget. Where this engages the blend error grows beyond the ~1e-3 L1 target (documented).
    const double sideL = 2.0 * (R * std::max(M.ax, M.ay) * std::max(M.extX, M.extY) + 1.5 * M.latMax * R) + 3.0;
    const double stampBytes = sideL * sideL * 3.0 * 4.0;
    const double perAxis = std::sqrt(std::max(1.0, kStampPinBudget / stampBytes));
    double cellMin = 4.5;
    if (perAxis > 2.5) cellMin = std::max(cellMin, M.tile / (perAxis - 2.0));
    else cellMin = std::max(cellMin, (double)M.tile * 2.0);
    dg = std::max(dg, cellMin / M.halfDiag);
    return clampd(dg, 1e-4, 1.0);
}

// up to 8 nodes (2 radii x 4 positions) with weights for a source
int latticeNodes(const Model& M, const Src& s, LatNode* nd) {
    const double dt = latEps(M) / 2.0;  // sqrt(R) step: dR = 2 sqrt(R) dt = eps sqrt(R)
    const double t = std::sqrt((double)s.R), tm = std::sqrt(M.maxR);
    double kf = (tm - t) / dt;
    long long k = (long long)std::floor(kf);
    double fr = kf - (double)k;
    double Rn[2], wr[2];
    int nr;
    if (fr < 1e-9) { nr = 1; Rn[0] = (tm - k * dt) * (tm - k * dt); wr[0] = 1; }
    else {
        nr = 2;
        Rn[0] = (tm - k * dt) * (tm - k * dt); wr[0] = 1 - fr;
        Rn[1] = (tm - (k + 1) * dt) * (tm - (k + 1) * dt); wr[1] = fr;
    }
    int n = 0;
    for (int r = 0; r < nr; ++r) {
        const long long kk = k + r;
        if (!M.pathB) {
            nd[n].R = Rn[r]; nd[n].side = s.side; nd[n].gx = nd[n].gy = 0; nd[n].w = wr[r];
            nd[n].key = {kk * 2 + s.side, 0, 0};
            ++n;
            continue;
        }
        const double dg = gStep(M, Rn[r]);
        const double ux = s.gx / dg, uy = s.gy / dg;
        const long long ix = (long long)std::floor(ux), iy = (long long)std::floor(uy);
        const double fx = ux - ix, fy = uy - iy;
        for (int b = 0; b < 2; ++b)
            for (int a = 0; a < 2; ++a) {
                const double w = wr[r] * (a ? fx : 1 - fx) * (b ? fy : 1 - fy);
                if (w < 1e-7) continue;
                nd[n].R = Rn[r]; nd[n].side = s.side;
                nd[n].gx = (ix + a) * dg; nd[n].gy = (iy + b) * dg;
                nd[n].w = w;
                nd[n].key = {kk * 2 + s.side, ix + a, iy + b};
                ++n;
            }
    }
    double ws = 0;
    for (int i = 0; i < n; ++i) ws += nd[i].w;
    for (int i = 0; i < n; ++i) nd[i].w /= ws;
    return n;
}

// the node stamp, pinned for the current tile; nullptr when the render was cancelled
const Stamp* nodeStamp(const Model& M, const LatNode& nd, Work& W) {
    StampCache& C = W.cache;
    auto it = C.map.find(nd.key);
    if (it != C.map.end()) return it->second.get();
    SharedStamps& S = *W.shared;
    std::shared_ptr<Stamp> sp;
    {
        std::unique_lock<std::mutex> lk(S.m);
        for (;;) {
            auto f = S.map.find(nd.key);
            if (f == S.map.end()) { S.map[nd.key];  break; }  // placeholder: this thread computes it
            if (f->second.ready) {
                S.lru.splice(S.lru.begin(), S.lru, f->second.it);
                sp = f->second.st;
                break;
            }
            S.cv.wait_for(lk, std::chrono::milliseconds(10));
            lk.unlock();
            const bool c = cancelled();
            lk.lock();
            if (c) return nullptr;
        }
    }
    if (!sp) {
        auto unpublish = [&]() {
            { std::lock_guard<std::mutex> lk(S.m); S.map.erase(nd.key); }
            S.cv.notify_all();
        };
        std::shared_ptr<Stamp> st;
        try {
            st = std::make_shared<Stamp>();
            Src s{};
            s.R = (float)nd.R; s.side = (int8_t)nd.side;
            paramsFromG(M, nd.gx, nd.gy, s);
            if (isIdentity(M, nd.R)) setIdentity(*st, M.T->nch);
            else computeStamp(M, s, *st, W);
        } catch (...) { unpublish(); throw; }
        if (cancelled()) { unpublish(); return nullptr; }  // possibly partial: never shared
        {
            std::lock_guard<std::mutex> lk(S.m);
            SharedStamps::E& e = S.map[nd.key];
            e.st = st; e.ready = true;
            e.bytes = st->pl.capacity() * sizeof(float) + (st->rlo.capacity() + st->rhi.capacity()) * sizeof(int);
            S.lru.push_front(nd.key);
            e.it = S.lru.begin();
            S.bytes += e.bytes;
            S.peak = std::max(S.peak, S.bytes);
            S.evictLocked();
        }
        S.cv.notify_all();
        sp = std::move(st);
    }
    const Stamp* p = sp.get();
    C.map.emplace(nd.key, std::move(sp));
    return p;
}

// one output row of a lattice source: blend its node stamps' rows (weights sum to 1 -> energy
// exact) in an L1-sized scratch row and accumulate it with the source colour
void latRow(const LatSrc& L, const Src& s, int y, float* buf, int bx0, int by0, int bw, const Accum& A, Work& W) {
    const int np = L.st[0]->np;
    const int wU = L.w_;
    W.rowbuf.assign((size_t)np * wU, 0.f);
    float* rb = W.rowbuf.data();
    int lo = wU, hi = 0;
    for (int k = 0; k < L.n; ++k) {
        const Stamp& st = *L.st[k];
        const int jr = y - L.y - st.j0;
        if (jr < 0 || jr >= st.h) continue;
        const int a = st.rlo[jr], b = st.rhi[jr];
        if (b <= a) continue;
        const int off = st.i0 - L.i0;
        lo = std::min(lo, off + a); hi = std::max(hi, off + b);
        const float w = L.w[k];
        const size_t pn = (size_t)st.w * st.h;
        for (int c = 0; c < np; ++c) {
            const float* src = st.pl.data() + (st.np == 1 ? 0 : (size_t)c * pn) + (size_t)jr * st.w;
            float* o = rb + (size_t)c * wU + off;
            for (int i = a; i < b; ++i) o[i] += w * src[i];
        }
    }
    if (hi <= lo) return;
    const int NC = A.NC, NV = A.NV;
    const int xa = std::max(L.x + L.i0 + lo, bx0), xb = std::min(L.x + L.i0 + hi, bx0 + bw);
    if (xb <= xa) return;
    float* row = buf + ((size_t)(y - by0) * bw) * NC;
    const float cr = s.c[0], cg = s.c[1], cb = s.c[2], ca = s.c[3], cx = s.aux;
    const int base = L.x + L.i0;
    if (np == 1) {
        for (int x = xa; x < xb; ++x) {
            const float k = rb[x - base];
            if (k == 0) continue;
            float* q = row + (size_t)(x - bx0) * NC;
            q[0] += k * cr; q[1] += k * cg; q[2] += k * cb; q[3] += k * ca;
            if (A.aux) q[4] += k * cx;
            q[NV] += k;
        }
    } else {
        const float* r1 = rb + wU;
        const float* r2 = r1 + wU;
        for (int x = xa; x < xb; ++x) {
            const int i = x - base;
            const float kr = rb[i], kg = r1[i], kb = r2[i];
            if (kr == 0 && kg == 0 && kb == 0) continue;
            float* q = row + (size_t)(x - bx0) * NC;
            q[0] += kr * cr; q[1] += kg * cg; q[2] += kb * cb; q[3] += kg * ca;
            if (A.aux) q[4] += kg * cx;
            q[NV] += kr; q[NV + 1] += kg; q[NV + 2] += kb;
        }
    }
}

bool validate(const float* src, int w, int h, int srcStride, const float* map, int mapStride,
              const float* dst, int dstStride, const Params& p) {
    if (!src || !dst || w <= 0 || h <= 0) return false;
    if ((long long)w * 4 > srcStride || (long long)w * 4 > dstStride) return false;
    if (map && mapStride < w) return false;
    if (!std::isfinite(p.maxRadius) || p.maxRadius < 0 || p.maxRadius > 4000) return false;
    const float fs[] = {p.focus, p.mapScale, p.mapGamma, p.bladeCurvature, p.rotationDeg, p.anamorphic,
                        p.catEye, p.edgeRing, p.ringWidth, p.chromaFringe, p.sphericalAberration,
                        p.cornerRounding, p.edgeSoftness, p.apodization, p.coreWeight, p.obstruction,
                        p.onionRings, p.onionCount, p.vignetting, p.fringeTint,
                        p.frameLeft, p.frameTop, p.frameWidth, p.frameHeight};
    for (float v : fs) if (!std::isfinite(v)) return false;
    if (p.mapGamma <= 0) return false;
    auto overlaps = [](const void* a0, size_t an, const void* b0, size_t bn) {
        const char* a = (const char*)a0;
        const char* b = (const char*)b0;
        return a < b + bn && b < a + an;
    };
    const size_t srcBytes = ((size_t)(h - 1) * srcStride + (size_t)w * 4) * sizeof(float);
    const size_t dstBytes = ((size_t)(h - 1) * dstStride + (size_t)w * 4) * sizeof(float);
    if (overlaps(src, srcBytes, dst, dstBytes)) return false;
    if (map && overlaps(map, ((size_t)(h - 1) * mapStride + w) * sizeof(float), dst, dstBytes)) return false;
    return true;
}

Status renderImpl(const float* src, int w, int h, int srcStride, const float* map, int mapStride,
                  float* dst, int dstStride, const Params& p, const HqOptions& q, AbortFn abortFn, void* user) {
    if (!validate(src, w, h, srcStride, map, mapStride, dst, dstStride, p)) return Status::BadArgs;
    if ((q.auxSrc && (!q.auxDst || q.auxSrcStride < w || q.auxDstStride < w)) || (!q.auxSrc && q.auxDst))
        return Status::BadArgs;
    HqStats stats;
    const double t0 = nowSec();
    int nth = p.threads > 0 ? p.threads : (int)std::thread::hardware_concurrency();
    nth = std::max(1, std::min(nth, 256));
    if (abortFn && abortFn(user)) return Status::Aborted;
    Model M;
    bool hit = false;
    Status st = buildModel(M, p, q, w, h, nth, abortFn, user, hit);
    if (st != Status::Ok) return st;
    const double t1 = nowSec();
    stats.secModel = t1 - t0;
    stats.tableMB = M.T->bytes() / 1048576.0;
    stats.tableRes = M.T->n;
    stats.spectralSamplesA = stats.spectralSamplesB = M.spec.N;
    stats.pathB = M.pathB;
    stats.tableCacheHit = hit;

    Accum A;
    A.W = w; A.H = h;
    A.tri = M.T->nch == 3;
    A.aux = q.auxSrc != nullptr;
    A.NV = 4 + (A.aux ? 1 : 0);
    A.NW = A.tri ? 3 : 1;
    A.NC = A.NV + A.NW;
    // fixed-point scale from the largest |value| (after light loss <= input) so that
    // value * kWeightCap * 2^a < 2^62
    double vmax = 1e-30;
    for (int y = 0; y < h; ++y) {
        const float* r = src + (size_t)y * srcStride;
        for (int i = 0; i < 4 * w; ++i) if (std::isfinite(r[i])) vmax = std::max(vmax, (double)std::fabs(r[i]));
        if (q.auxSrc) {
            const float* ax = q.auxSrc + (size_t)y * q.auxSrcStride;
            for (int x = 0; x < w; ++x) if (std::isfinite(ax[x])) vmax = std::max(vmax, (double)std::fabs(ax[x]));
        }
    }
    A.a = (int)std::floor(std::log2(std::ldexp(1.0, 62) / (vmax * kWeightCap)));
    A.a = std::max(-200, std::min(40, A.a));
    A.sHi = std::ldexp(1.0, A.a); A.iHi = std::ldexp(1.0, -A.a);
    A.sLo = std::ldexp(1.0, A.a + 40); A.iLo = std::ldexp(1.0, -(A.a + 40));
    A.sW = std::ldexp(1.0, 40); A.iW = std::ldexp(1.0, -40);
    stats.accumScaleLog2 = A.a;
    A.hi.assign((size_t)w * h * A.NV, 0);
    A.lo.assign((size_t)w * h * A.NV, 0);
    A.wt.assign((size_t)w * h * A.NW, 0);
    A.nLocks = (h + kBand - 1) / kBand;
    A.locks.reset(new std::mutex[A.nLocks]);
    stats.accumMB = (A.hi.size() + A.lo.size() + A.wt.size()) * sizeof(int64_t) / 1048576.0;

    const bool useLat = M.latTol > 0;
    const int kT = M.tile;
    const int tw = (w + kT - 1) / kT, th = (h + kT - 1) / kT;
    // lattice stamps: one store shared by the threads (see SharedStamps). One tile pins at most
    // ~kStampPinBudget (see gStep); the unpinned LRU keeps 1.25x that per thread in total.
    SharedStamps shared;
    shared.budget = (size_t)(1.25 * kStampPinBudget * nth);
    std::vector<Work> works(nth);
    for (Work& W : works) W.shared = &shared;
    // tiles in vertical strips of kStrip tile columns (rows within a strip): the tiles running at
    // the same time are neighbours and use the same lattice nodes
    const int kStrip = 4;
    std::atomic<long long> nSplats{0};
    const bool ok = parallelFor(tw * th, nth, 1, [&](int t, int tid) {
        Work& W = works[tid];
        const int sIdx = t / (kStrip * th), sr = t % (kStrip * th);
        const int sCols = std::min(kStrip, tw - sIdx * kStrip);
        const int tx0 = (sIdx * kStrip + sr % sCols) * kT, ty0 = (sr / sCols) * kT;
        const int tx1 = std::min(w, tx0 + kT), ty1 = std::min(h, ty0 + kT);
        W.srcs.resize((size_t)(tx1 - tx0) * (ty1 - ty0));
        int rx = 0, ry = 0;
        for (int y = ty0; y < ty1; ++y)
            for (int x = tx0; x < tx1; ++x) {
                Src& s = W.srcs[(size_t)(y - ty0) * (tx1 - tx0) + (x - tx0)];
                prepSrc(M, p, q, src, srcStride, map, mapStride, x, y, s);
                if (!s.ident) { int a, b; reachOf(M, s.R, a, b); rx = std::max(rx, a); ry = std::max(ry, b); }
            }
        const int bx0 = std::max(0, tx0 - rx), bx1 = std::min(w, tx1 + rx);
        const int by0 = std::max(0, ty0 - ry), by1 = std::min(h, ty1 + ry);
        const int bw = bx1 - bx0, bh = by1 - by0;
        W.buf.assign((size_t)bw * bh * A.NC, 0.f);
        long long cnt = 0;
        LatNode nd[8];
        // phase 1: identity / exact-stamp sources directly; lattice sources gather their node stamps
        // (pinned in the cache for this tile)
        W.lat.clear();
        W.cache.release();
        for (int y = ty0; y < ty1; ++y) {
            if (cancelled()) return;
            for (int x = tx0; x < tx1; ++x) {
                const int si = (y - ty0) * (tx1 - tx0) + (x - tx0);
                const Src& s = W.srcs[(size_t)si];
                if (s.ident) {
                    addIdentity(W.buf.data() + ((size_t)(y - by0) * bw + (x - bx0)) * A.NC, A, s);
                    continue;
                }
                if (s.c[0] == 0 && s.c[1] == 0 && s.c[2] == 0 && s.c[3] == 0 && s.aux == 0 && !q.normalizeByWeight)
                    continue;  // raw mode: a fully transparent black source adds nothing
                ++cnt;
                if (useLat && s.R >= kLatMinR) {
                    const int n = latticeNodes(M, s, nd);
                    LatSrc L;
                    L.x = x; L.y = y; L.n = n; L.srcIdx = si;
                    int i0 = 1 << 30, j0 = 1 << 30, i1 = -(1 << 30), j1 = -(1 << 30);
                    for (int k = 0; k < n; ++k) {
                        L.st[k] = nodeStamp(M, nd[k], W);
                        if (!L.st[k]) return;  // cancelled
                        L.w[k] = (float)nd[k].w;
                        i0 = std::min(i0, L.st[k]->i0); j0 = std::min(j0, L.st[k]->j0);
                        i1 = std::max(i1, L.st[k]->i0 + L.st[k]->w); j1 = std::max(j1, L.st[k]->j0 + L.st[k]->h);
                    }
                    if (cancelled()) return;
                    L.i0 = i0; L.j0 = j0; L.w_ = i1 - i0; L.h_ = j1 - j0;
                    W.lat.push_back(L);
                    ++W.nLat;
                    continue;
                }
                const bool same = W.stValid && W.kR == s.R && W.kSide == s.side &&
                                  (!M.pathB || (W.kgx == s.gx && W.kgy == s.gy));
                if (!same) {
                    computeStamp(M, s, W.st, W);
                    W.stValid = true; W.kR = s.R; W.kSide = s.side; W.kgx = s.gx; W.kgy = s.gy;
                }
                if (cancelled()) return;
                accumulate(W.st, s, x, y, W.buf.data(), bx0, by0, bw, bh, A);
            }
        }
        // phase 2: lattice sources band by band (16 output rows): the touched part of the tile buffer
        // stays in cache instead of streaming the whole kernel footprint per source
        if (!W.lat.empty()) {
            const int BAND = 16;
            for (int Y0 = by0; Y0 < by1; Y0 += BAND) {
                if (cancelled()) return;
                const int Y1 = std::min(by1, Y0 + BAND);
                for (const LatSrc& L : W.lat) {
                    const int ya = std::max(Y0, L.y + L.j0), yb = std::min(Y1, L.y + L.j0 + L.h_);
                    for (int y = ya; y < yb; ++y)
                        latRow(L, W.srcs[(size_t)L.srcIdx], y, W.buf.data(), bx0, by0, bw, A, W);
                }
            }
        }
        W.cache.release();
        nSplats += cnt;
        mergeTile(A, W.buf.data(), bx0, by0, bw, bh);
    }, abortFn, user);
    if (!ok) return Status::Aborted;
    const double t2 = nowSec();
    stats.secScatter = t2 - t1;
    size_t workBytes = shared.peak;
    for (const Work& W : works) {
        workBytes += W.buf.capacity() * sizeof(float) + W.st.pl.capacity() * sizeof(float) +
                     W.blend.pl.capacity() * sizeof(float) + (W.tmp.capacity() + W.tmp2.capacity()) * sizeof(float);
        stats.stampsComputed += W.nStamps;
        stats.stampPixels += W.nStampPx;
        stats.wavelengthPixels += W.nLamPx;
        stats.latticeSplats += W.nLat;
    }
    stats.workMB = workBytes / 1048576.0;
    stats.splats = nSplats.load();
    works.clear();

    // resolve: output >= 0 (stage contract)
    const bool norm = q.normalizeByWeight;
    const bool okR = parallelFor(h, nth, 4, [&](int y, int) {
        float* o = dst + (size_t)y * dstStride;
        float* axo = q.auxDst ? q.auxDst + (size_t)y * q.auxDstStride : nullptr;
        const int64_t* hi = A.hi.data() + (size_t)y * w * A.NV;
        const int64_t* lo = A.lo.data() + (size_t)y * w * A.NV;
        const int64_t* wt = A.wt.data() + (size_t)y * w * A.NW;
        for (int x = 0; x < w; ++x, hi += A.NV, lo += A.NV, wt += A.NW) {
            double v[5];
            for (int c = 0; c < A.NV; ++c) v[c] = (double)hi[c] * A.iHi + (double)lo[c] * A.iLo;
            const double wr = (double)wt[0] * A.iW;
            const double wg = A.tri ? (double)wt[1] * A.iW : wr, wb = A.tri ? (double)wt[2] * A.iW : wr;
            float* q4 = o + 4 * (size_t)x;
            double r[5];
            if (!norm) { for (int c = 0; c < A.NV; ++c) r[c] = v[c]; }
            else if (!(wr > 1e-9) && !(wg > 1e-9) && !(wb > 1e-9)) {
                // no kernel landed here (only with extreme cat-eye offsets next to the buffer edge):
                // keep the input pixel instead of a black hole
                const float* sp = src + (size_t)y * srcStride + 4 * (size_t)x;
                for (int c = 0; c < 4; ++c) r[c] = std::isfinite(sp[c]) ? sp[c] : 0.0;
                if (A.aux) r[4] = std::isfinite(q.auxSrc[(size_t)y * q.auxSrcStride + x]) ? q.auxSrc[(size_t)y * q.auxSrcStride + x] : 0.0;
            } else {
                const double eps = 1e-9;
                r[0] = wr > eps ? v[0] / wr : 0.0;
                r[1] = wg > eps ? v[1] / wg : 0.0;
                r[2] = wb > eps ? v[2] / wb : 0.0;
                r[3] = wg > eps ? v[3] / wg : 0.0;
                if (A.aux) r[4] = wg > eps ? v[4] / wg : 0.0;
            }
            for (int c = 0; c < 4; ++c) q4[c] = r[c] > 0 && std::isfinite(r[c]) ? (float)std::min(r[c], 3.0e38) : 0.f;
            if (axo) axo[x] = A.aux && r[4] > 0 && std::isfinite(r[4]) ? (float)std::min(r[4], 3.0e38) : 0.f;
        }
    }, abortFn, user);
    if (!okR) return Status::Aborted;
    if (q.applyVignetting) applyVignetting(dst, w, h, dstStride, p);
    stats.secResolve = nowSec() - t2;
    if (q.stats) *q.stats = stats;
    return Status::Ok;
}

} // namespace

// ================================================================== public ===

float blurAmount(float mapValue, const Params& p, int* sign) {
    double sgn;
    const double a = amountOf(mapValue, p, sgn);
    if (sign) *sign = sgn > 0 ? 1 : -1;
    return (float)a;
}

int inputMargin(const Params& p, const HqOptions& q) {
    Geo G;
    buildGeo(G, p, q);
    const double R = G.maxR;
    const double amax = std::max(G.ax, G.ay);
    const double ext = std::max(G.extX, G.extY);
    const double sig = diffSigma(G, R);
    // kernel reach bound (tight, see harness "inputMargin covers the kernels and is tight") + 1 px
    const double m = R * amax * ext + G.latMax * 1.5 * R + 3.0 * sig + 1.0;
    return (int)std::min(100000.0, std::ceil(m));
}

void purgeTableCache() { tableCache().clear(); }

void applyVignetting(float* rgba, int w, int h, int stride, const Params& p) {
    const double v = clampd(p.vignetting, 0.0, 1.0);
    if (v < 1e-4) return;
    double cx, cy, hd;
    frameOf(p, w, h, cx, cy, hd);
    for (int y = 0; y < h; ++y) {
        const double dy = (y + 0.5 - cy) / hd;
        float* o = rgba + (size_t)y * stride;
        for (int x = 0; x < w; ++x) {
            const double dx = (x + 0.5 - cx) / hd;
            const double rn2 = std::min(1.0, dx * dx + dy * dy);
            const float f = (float)(1.0 - v * rn2);
            o[4 * x] *= f; o[4 * x + 1] *= f; o[4 * x + 2] *= f;
        }
    }
}

Status kernelStamp(const Params& p, const HqOptions& q, int w, int h, double R, int side,
                   int px, int py, int& x0, int& y0, int& kw, int& kh, std::vector<float>& planes) {
    if (w <= 0 || h <= 0 || !(R >= 0) || R > 4000 || !std::isfinite(p.maxRadius)) return Status::BadArgs;
    try {
        Model M;
        const int nth = std::max(1, (int)std::thread::hardware_concurrency());
        Params pp = p;
        pp.maxRadius = (float)std::max((double)pp.maxRadius, R);
        if (!(pp.maxRadius <= 4000)) return Status::BadArgs;
        bool hit;
        Status st = buildModel(M, pp, q, w, h, nth, nullptr, nullptr, hit);
        if (st != Status::Ok) return st;
        Src s{};
        s.R = (float)R;
        s.side = (int8_t)(side > 0 ? 1 : 0);
        paramsFromG(M, (px + 0.5 - M.fcx) / M.halfDiag, (py + 0.5 - M.fcy) / M.halfDiag, s);
        if (isIdentity(M, R)) {
            x0 = 0; y0 = 0; kw = 1; kh = 1;
            planes.assign(3, 1.f);
            return Status::Ok;
        }
        Work W;
        computeStamp(M, s, W.st, W);
        const Stamp& stp = W.st;
        x0 = stp.i0; y0 = stp.j0; kw = stp.w; kh = stp.h;
        const size_t pn = (size_t)kw * kh;
        planes.assign(3 * pn, 0.f);
        for (int c = 0; c < 3; ++c)
            std::copy(stp.pl.begin() + (stp.np == 1 ? 0 : c) * pn, stp.pl.begin() + ((stp.np == 1 ? 0 : c) + 1) * pn,
                      planes.begin() + c * pn);
        return Status::Ok;
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;
    }
}

Status render(const float* src, int w, int h, int srcStride,
              const float* map, int mapStride,
              float* dst, int dstStride,
              const Params& p, const HqOptions& q, AbortFn abortFn, void* user) {
    // nothing may escape into the host: bad_alloc -> OutOfMemory, anything else -> Error
    try {
        return renderImpl(src, w, h, srcStride, map, mapStride, dst, dstStride, p, q, abortFn, user);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory;
    } catch (...) {
        return Status::Error;
    }
}

} // namespace lenshq
