// Map Bilinear Blur - render engine (v0.2). See mbb_core.h for the model.
//
// Pyramid: level L = level L-1 downsampled 2:1 with either a 2x2 average (crisp cells) or the binomial
// 6-tap [1 5 10 10 5 1]/32 (a proper low-pass). Each output pixel samples the level(s) that match its blur
// through the chosen interpolation kernel:
//   Nearest      - the cell itself: mosaic blocks whose size follows the map
//   Bilinear     - tent, the literal idea (blocky / diamond shading on 2x2-average levels)
//   Smoothstep   - bilinear with smoothstep weights: rounded, pillowy blocks
//   B-spline     - cubic B-spline: the softest, no ringing (Gaussian-like; the former "Smooth")
//   Mitchell     - cubic B = C = 1/3: a little crisper than B-spline, faint ringing
//   Catmull-Rom  - cubic B = 0, C = 1/2: crisp, visible ringing at strong edges
//   Lanczos 3    - 6x6 taps, the crispest, strongest ringing
//   Hex mosaic   - nearest cell of a hexagonal lattice (area = one cell), its colour sampled bilinearly
// Anti-ringing clamps the negative-lobe kernels to the range of the 2x2 nearest cells (per channel, on
// premultiplied values, so no negative alpha and no colour beyond the local min / max).
//
// Ladder: entry 0 = the pixel itself, entry k >= 1 = a pyramid level through the kernel, with its exact
// variance (downsample cascade + kernel second moment averaged over the level's pixel phases). A pixel
// whose target variance lies between two entries blends them with weights linear in variance (Smooth),
// takes the nearer one (Steps) or picks one at random with that probability (Dither, deterministic hash
// of the full-res pixel, frame and seed).
//
// Protect Edges (depth-aware layering):
//   far / in-focus layer = per-entry weighted ("normalised convolution") pyramids: building entry k,
//     pixels much sharper than that blur (in-focus subject, nearer things) get weight 0, so a blurred
//     background never smears the sharp subject's colours (the hole behind the subject is filled from the
//     background around it, pull-push style when nothing is near).
//   near layer (in front of focus) = the front pixels split into blur bands (a soft partition of their own
//     entry), each band blurred at ITS entry and summed: out = near + (1 - near.alpha) * far.
#include "mbb_core.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <memory>
#include <new>
#include <condition_variable>
#include <functional>
#include <thread>
#include "platform/plat.h"  // mac_port: MBB_SIMD_SSE2 / MBB_SIMD_NEON / MBB_SIMD_SCALAR
#if MBB_SIMD_SSE2
#include <emmintrin.h>
#elif MBB_SIMD_NEON
#include <arm_neon.h>
#endif

namespace mbb {
int g_profile = 0;
namespace {

const int kMaxIdx = 28;
const int CH = 5;  // q*r, q*g, q*b, q*a, q (premultiplied colour, inside weight)
const double kPi = 3.14159265358979323846;

// ----------------------------------------------------------------- kernels ---
struct KernInfo
{
    int lo, hi;  // tap offsets relative to floor(u), inclusive
    bool neg;    // has negative lobes (anti-ringing applies)
};
KernInfo KI(int interp)
{
    switch (interp)
    {
    case IN_NEAREST:
    case IN_BILINEAR:
    case IN_SMOOTHSTEP: return {0, 1, false};
    case IN_BSPLINE: return {-1, 2, false};
    case IN_MITCHELL:
    case IN_CATMULLROM: return {-1, 2, true};
    case IN_LANCZOS3: return {-2, 3, true};
    default: return {-2, 3, false};  // hex: the bilinear taps around a hex centre within ~0.62 cells
    }
}

inline double CubicBC(double x, double B, double C)
{
    x = std::fabs(x);
    if (x < 1)
        return ((12 - 9 * B - 6 * C) * x * x * x + (-18 + 12 * B + 6 * C) * x * x + (6 - 2 * B)) / 6.0;
    if (x < 2)
        return ((-B - 6 * C) * x * x * x + (6 * B + 30 * C) * x * x + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6.0;
    return 0;
}
inline double Sinc(double x) { return std::fabs(x) < 1e-9 ? 1.0 : std::sin(kPi * x) / (kPi * x); }

// weights for the taps lo..hi of a separable kernel at fractional position f in [0, 1)
inline void Weights(int interp, float f, float* w)
{
    switch (interp)
    {
    case IN_NEAREST:
        w[0] = f < 0.5f ? 1.f : 0.f;
        w[1] = 1.f - w[0];
        return;
    case IN_BILINEAR:
        w[0] = 1.f - f;
        w[1] = f;
        return;
    case IN_SMOOTHSTEP:
    {
        const float s = f * f * (3.f - 2.f * f);
        w[0] = 1.f - s;
        w[1] = s;
        return;
    }
    case IN_BSPLINE:
    {
        const float x2 = f * f, x3 = x2 * f, a = 1.f - f;
        w[0] = a * a * a * (1.f / 6.f);
        w[1] = (3.f * x3 - 6.f * x2 + 4.f) * (1.f / 6.f);
        w[2] = (-3.f * x3 + 3.f * x2 + 3.f * f + 1.f) * (1.f / 6.f);
        w[3] = x3 * (1.f / 6.f);
        return;
    }
    case IN_MITCHELL:
    case IN_CATMULLROM:
    {
        const double B = interp == IN_MITCHELL ? 1.0 / 3.0 : 0.0, C = interp == IN_MITCHELL ? 1.0 / 3.0 : 0.5;
        for (int i = 0; i < 4; ++i)
            w[i] = (float)CubicBC((i - 1) - (double)f, B, C);
        return;
    }
    case IN_LANCZOS3:
    {
        double s = 0, t[6];
        for (int i = 0; i < 6; ++i)
        {
            const double x = (i - 2) - (double)f;
            t[i] = std::fabs(x) < 3 ? Sinc(x) * Sinc(x / 3.0) : 0.0;
            s += t[i];
        }
        for (int i = 0; i < 6; ++i)
            w[i] = (float)(t[i] / s);
        return;
    }
    default:
        return;
    }
}

// second moment (px^2 of the cell size) of the kernel about the sample point, at phase f
double Moment2(int interp, double f)
{
    const KernInfo k = KI(interp);
    float w[6];
    Weights(interp, (float)f, w);
    double m = 0;
    for (int i = k.lo; i <= k.hi; ++i)
        m += w[i - k.lo] * (i - f) * (i - f);
    return m;
}

// ------------------------------------------------------------------ ladder ---
struct Ladder
{
    int top = 0;  // highest entry
    int lev[kMaxIdx + 2] = {};    // pyramid level of entry k (entry 0: identity, lev -1)
    double var[kMaxIdx + 2] = {};
    int level(int k) const { return lev[k]; }
};

double SigmaMax(const Settings& s)
{
    double sg = SigmaForRadius(std::max(0.0, s.radius));
    if (s.sharpType == SH_UNSHARP && s.sharpAmount > 0)
    {
        const double ss = SigmaForRadius(std::max(0.0, s.sharpRadius));
        sg = std::sqrt(sg * sg + ss * ss);
    }
    return sg;
}

// variance (per axis, buffer px^2) of sampling level L with the settings' kernel
double LevelVar(const Settings& s, int L)
{
    const double f4 = std::ldexp(1.0, 2 * L);  // 4^L
    const double step = s.downSmooth ? 1.25 : 0.25;
    const double vd = step * (f4 - 1) / 3.0;
    double m2;
    if (s.interp == IN_HEX)
        m2 = 5.0 * 2.0 / (24.0 * 3.0 * std::sqrt(3.0)) + 1.0 / 6.0;  // hexagon of area 1 (5 a^2 / 24) + bilinear
    else if (L == 0)
        m2 = Moment2(s.interp, 0.0);
    else
    {
        const int sP = 1 << std::min(L, 6);  // the level's pixel phases (64 = uniform enough)
        m2 = 0;
        for (int p = 0; p < sP; ++p)
        {
            double f = (p + 0.5) / sP - 0.5;
            f -= std::floor(f);
            m2 += Moment2(s.interp, f);
        }
        m2 /= sP;
    }
    return vd + f4 * m2;
}

Ladder MakeLadder(const Settings& s)
{
    Ladder l;
    l.lev[0] = -1;
    l.var[0] = 0;
    const double vmax = SigmaMax(s) * SigmaMax(s);
    int top = 0;
    if (vmax > 0)
    {
        // level 0 only for the non-interpolating kernels (the others reproduce the pixel there)
        const bool l0 = s.interp == IN_BSPLINE || s.interp == IN_MITCHELL;
        for (int L = l0 ? 0 : 1; L <= 24 && top < kMaxIdx; ++L)
        {
            const double v = LevelVar(s, L);
            if (!(v > l.var[top] + 1e-9))
                continue;
            ++top;
            l.lev[top] = L;
            l.var[top] = v;
            if (v >= vmax)
                break;
        }
    }
    l.top = top;
    l.lev[top + 1] = l.lev[top];
    l.var[top + 1] = l.var[top];
    return l;
}

// target variance -> (entry, blend t towards entry+1)
inline void Locate(const Ladder& l, double v, int& k, float& t)
{
    if (!(v > 0) || l.top == 0)
    {
        k = 0;
        t = 0;
        return;
    }
    if (v >= l.var[l.top])
    {
        k = l.top;
        t = 0;
        return;
    }
    int i = 0;
    while (i + 1 < l.top && l.var[i + 1] <= v)
        ++i;
    k = i;
    t = (float)((v - l.var[i]) / (l.var[i + 1] - l.var[i]));
}

struct Range
{
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;  // cell indices, half-open, relative to D
};

struct Lvl
{
    int L = 0;
    int x0 = 0, y0 = 0, w = 0, h = 0;
    int extW = 1, extH = 1;  // the level's extent over D
    FVec d;  // v0.3: not zero-filled (every cell is written by its producer)
    void alloc(int lev, const Range& r, int ew, int eh)
    {
        L = lev;
        x0 = r.x0;
        y0 = r.y0;
        w = r.x1 - r.x0;
        h = r.y1 - r.y0;
        extW = ew;
        extH = eh;
        d.clear();
        d.resize((size_t)std::max(0, w) * std::max(0, h) * CH);
    }
    // clamped to the level extent (the rule that makes tiles exact), then to the stored range (no-op
    // when the ranges are right; the tile tests would show it otherwise)
    inline const float* at(int cx, int cy) const
    {
        cx = std::min(extW - 1, std::max(0, cx));
        cy = std::min(extH - 1, std::max(0, cy));
        cx = std::min(x0 + w - 1, std::max(x0, cx));
        cy = std::min(y0 + h - 1, std::max(y0, cy));
        return &d[((size_t)(cy - y0) * w + (cx - x0)) * CH];
    }
};

// ---------------------------------------------------------------- threads ---
// v0.3: one set of worker threads per render, reused by every parallel loop of the frame (v0.2 created
// threads for each loop: dozens per frame). Same chunking rule; each index is processed by exactly one
// thread with the same arithmetic, so results do not depend on the pool (bit-identical to v0.2).
class Pool
{
public:
    explicit Pool(int threads) : T_(std::max(1, threads))
    {
        for (int i = 0; i < T_ - 1; ++i)
            th_.emplace_back([this] { Worker(); });
    }
    ~Pool()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_)
            t.join();
    }
    int Threads() const { return T_; }
    template <class Fn>
    void For(int n, const Fn& fn)  // fn(begin, end)
    {
        if (n <= 0)
            return;
        const int T = std::max(1, std::min(T_, (n + 7) / 8));
        if (T == 1)
        {
            fn(0, n);
            return;
        }
        std::function<void(int, int)> f = fn;
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &f;
            n_ = n;
            chunk_ = std::max(1, n / (T * 8));
            next_.store(0);
            ex_ = nullptr;
            want_ = T - 1;  // helpers taking part (the caller is the T-th)
            joined_ = 0;
            done_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        Run();
        std::unique_lock<std::mutex> lk(m_);
        doneCv_.wait(lk, [&] { return done_ == joined_; });
        job_ = nullptr;
        want_ = 0;
        if (ex_)
            std::rethrow_exception(ex_);
    }

private:
    void Run()
    {
        try
        {
            for (;;)
            {
                const int b = next_.fetch_add(chunk_);
                if (b >= n_)
                    break;
                (*job_)(b, std::min(n_, b + chunk_));
            }
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!ex_)
                ex_ = std::current_exception();
            next_.store(n_);
        }
    }
    void Worker()
    {
        unsigned long long seen = 0;
        for (;;)
        {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || (gen_ != seen && joined_ < want_); });
                if (stop_)
                    return;
                seen = gen_;
                ++joined_;
            }
            Run();
            {
                std::lock_guard<std::mutex> lk(m_);
                ++done_;
            }
            doneCv_.notify_all();
        }
    }
    const int T_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, doneCv_;
    bool stop_ = false;
    unsigned long long gen_ = 0;
    std::function<void(int, int)>* job_ = nullptr;
    int n_ = 0, chunk_ = 1, want_ = 0, joined_ = 0, done_ = 0;
    std::atomic<int> next_{0};
    std::exception_ptr ex_;
};

// the v0.2 one-shot version, still used where no pool exists
template <class Fn>
void ParallelFor(int n, int threads, const Fn& fn)  // fn(begin, end)
{
    if (n <= 0)
        return;
    const int T = std::max(1, std::min(threads, (n + 7) / 8));
    if (T == 1)
    {
        fn(0, n);
        return;
    }
    std::exception_ptr ex;
    std::mutex exm;
    std::atomic<int> next{0};
    const int chunk = std::max(1, n / (T * 8));
    auto work = [&] {
        try
        {
            for (;;)
            {
                const int b = next.fetch_add(chunk);
                if (b >= n)
                    break;
                fn(b, std::min(n, b + chunk));
            }
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lk(exm);
            if (!ex)
                ex = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(T - 1);
    for (int i = 0; i < T - 1; ++i)
        pool.emplace_back(work);
    work();
    for (auto& th : pool)
        th.join();
    if (ex)
        std::rethrow_exception(ex);
}

// ---------------------------------------------------------------- ranges ---
inline int FloorDiv(double v) { return (int)std::floor(v); }
// cells of level L over n px (the last cell may be partial: the domain is aligned on its left / top only)
inline int Ext(int n, int L) { return std::max(1, (n + (1 << L) - 1) >> L); }

// cells of level L read when sampling pixels [a, b) (relative to D) with the kernel
void NeedCells(const KernInfo& k, int L, int a, int b, int& lo, int& hi)
{
    const double s = std::ldexp(1.0, L);
    const int ia = FloorDiv((a + 0.5) / s - 0.5), ib = FloorDiv((b - 1 + 0.5) / s - 0.5);
    lo = ia + k.lo;
    hi = ib + k.hi + 1;
}

void Hull(int& lo, int& hi, int a, int b)
{
    if (hi <= lo)
    {
        lo = a;
        hi = b;
        return;
    }
    lo = std::min(lo, a);
    hi = std::max(hi, b);
}

// downsample support: cell i of level L reads level L-1 cells 2i + dLo .. 2i + dHi
inline int DownLo(bool smooth) { return smooth ? -2 : 0; }
inline int DownHi(bool smooth) { return smooth ? 3 : 1; }

// ranges[L] for L = 0..Lp, for output pixels O1 (absolute) in domain D
std::vector<Range> MakeRanges(const Settings& s, const Ladder& lad, int Lp, const Rect& D, const Rect& O1)
{
    KernInfo ki = KI(s.interp);
    if (s.protect)  // FarLam samples lamPyr with the cubic B-spline whatever the interpolation
    {
        ki.lo = std::min(ki.lo, -1);
        ki.hi = std::max(ki.hi, 2);
    }
    std::vector<char> sampled((size_t)Lp + 1, 0);
    for (int k = 1; k <= lad.top; ++k)
        if (lad.level(k) >= 0 && lad.level(k) <= Lp)
            sampled[(size_t)lad.level(k)] = 1;
    std::vector<Range> rg((size_t)Lp + 1);
    const int ax = O1.l - D.l, bx = O1.r - D.l, ay = O1.t - D.t, by = O1.b - D.t;
    const int dl = DownLo(s.downSmooth), dh = DownHi(s.downSmooth);
    for (int L = Lp; L >= 0; --L)
    {
        Range r;
        int xl = 0, xh = 0, yl = 0, yh = 0;
        if (sampled[(size_t)L])
        {
            NeedCells(ki, L, ax, bx, xl, xh);
            NeedCells(ki, L, ay, by, yl, yh);
        }
        if (L == 0)  // identity
        {
            Hull(xl, xh, ax, bx);
            Hull(yl, yh, ay, by);
        }
        if (L < Lp)  // dependencies of the level above
        {
            const Range& u = rg[(size_t)L + 1];
            Hull(xl, xh, 2 * u.x0 + dl, 2 * (u.x1 - 1) + dh + 1);
            Hull(yl, yh, 2 * u.y0 + dl, 2 * (u.y1 - 1) + dh + 1);
        }
        const int ew = Ext(D.w(), L), eh = Ext(D.h(), L);
        r.x0 = std::max(0, std::min(ew - 1, xl));
        r.x1 = std::max(r.x0 + 1, std::min(ew, xh));
        r.y0 = std::max(0, std::min(eh - 1, yl));
        r.y1 = std::max(r.y0 + 1, std::min(eh, yh));
        rg[(size_t)L] = r;
    }
    return rg;
}

// ------------------------------------------------------------------ SIMD ---
// v0.3: acc[c] += k * p[c] for the 5 channels, 4 of them in one SSE register. SSE2 mul / add round exactly
// like the scalar float ops (no FMA: the plug-in is built without /arch:AVX2), so results are bit-identical.
// mac_port: three interchangeable implementations (platform/plat.h picks one; MBB_FORCE_SCALAR forces the scalar
// reference). Each lane: one single-precision multiply, then one add - identical rounding in all three (no FMA:
// -ffp-contract=off on clang, and NEON uses separate vmulq / vaddq). Windows x64 = the v0.3.1 SSE2 code unchanged.
#if MBB_SIMD_SSE2
struct Acc5
{
    __m128 v;
    float e;
    Acc5() : v(_mm_setzero_ps()), e(0.f) {}
    inline void add(float k, const float* p)
    {
        v = _mm_add_ps(v, _mm_mul_ps(_mm_set1_ps(k), _mm_loadu_ps(p)));
        e += k * p[4];
    }
    inline void addAcc(float k, const Acc5& a)
    {
        v = _mm_add_ps(v, _mm_mul_ps(_mm_set1_ps(k), a.v));
        e += k * a.e;
    }
    inline void store(float* o) const
    {
        _mm_storeu_ps(o, v);
        o[4] = e;
    }
};
#elif MBB_SIMD_NEON
// UNTESTED (written on Windows; syntax-checked only): arm64 path, must equal the scalar one bit for bit
struct Acc5
{
    float32x4_t v;
    float e;
    Acc5() : v(vdupq_n_f32(0.f)), e(0.f) {}
    inline void add(float k, const float* p)
    {
        v = vaddq_f32(v, vmulq_f32(vdupq_n_f32(k), vld1q_f32(p)));
        e += k * p[4];
    }
    inline void addAcc(float k, const Acc5& a)
    {
        v = vaddq_f32(v, vmulq_f32(vdupq_n_f32(k), a.v));
        e += k * a.e;
    }
    inline void store(float* o) const
    {
        vst1q_f32(o, v);
        o[4] = e;
    }
};
#else
// scalar reference (any CPU): the same operations lane by lane
struct Acc5
{
    float v[4];
    float e;
    Acc5() : v{0.f, 0.f, 0.f, 0.f}, e(0.f) {}
    inline void add(float k, const float* p)
    {
        for (int c = 0; c < 4; ++c)
        {
            const float m = k * p[c];
            v[c] = v[c] + m;
        }
        e += k * p[4];
    }
    inline void addAcc(float k, const Acc5& a)
    {
        for (int c = 0; c < 4; ++c)
        {
            const float m = k * a.v[c];
            v[c] = v[c] + m;
        }
        e += k * a.e;
    }
    inline void store(float* o) const
    {
        for (int c = 0; c < 4; ++c)
            o[c] = v[c];
        o[4] = e;
    }
};
#endif

// ------------------------------------------------------------ downsample ---
const float kBin6[6] = {1.f / 32, 5.f / 32, 10.f / 32, 10.f / 32, 5.f / 32, 1.f / 32};

// dst (level L) from a source of level L-1 given as src(cx, cy, float tmp[CH]) -> const float* (CH values,
// clamped inside; either a pointer into a level or `tmp` filled)
template <class Src>
void Down(bool bin6, const Src& src, int srcExtW, int srcExtH, Lvl& dst, Pool& pool)
{
    const int taps = bin6 ? 6 : 2, off = bin6 ? -2 : 0;
    float kb[6];
    for (int m = 0; m < taps; ++m)
        kb[m] = bin6 ? kBin6[m] : 0.5f;
    const int sy0 = std::max(0, 2 * dst.y0 + off), sy1 = std::min(srcExtH, 2 * (dst.y0 + dst.h - 1) + off + taps);
    const int th = sy1 - sy0, tw = dst.w;
    FVec tmp((size_t)tw * th * CH);
    pool.For(th, [&](int b, int e) {
        for (int j = b; j < e; ++j)
        {
            const int sy = sy0 + j;
            float* o = &tmp[(size_t)j * tw * CH];
            for (int i = 0; i < tw; ++i, o += CH)
            {
                const int cx = dst.x0 + i;
                Acc5 acc;
                for (int m = 0; m < taps; ++m)
                {
                    const int sx = std::min(srcExtW - 1, std::max(0, 2 * cx + off + m));
                    float pt[CH];
                    acc.add(kb[m], src(sx, sy, pt));
                }
                acc.store(o);
            }
        }
    });
    pool.For(dst.h, [&](int b, int e) {
        for (int j = b; j < e; ++j)
        {
            const int cy = dst.y0 + j;
            float* o = &dst.d[(size_t)j * dst.w * CH];
            int rows[6];
            for (int m = 0; m < taps; ++m)
                rows[m] = std::min(sy1 - 1, std::max(sy0, std::min(srcExtH - 1, std::max(0, 2 * cy + off + m)))) - sy0;
            for (int i = 0; i < dst.w; ++i, o += CH)
            {
                Acc5 acc;
                for (int m = 0; m < taps; ++m)
                    acc.add(kb[m], &tmp[((size_t)rows[m] * tw + i) * CH]);
                acc.store(o);
            }
        }
    });
}

// -------------------------------------------------------------- sampling ---
// sample a level: get(cx, cy, float tmp[CH]) -> const float* (CH). X, Y: pixel indices relative to D.
template <class Get>
inline void SampleLevel(int interp, bool antiRing, int L, int X, int Y, const Get& get, float out[CH])
{
    const double s = std::ldexp(1.0, L);
    const double u = (X + 0.5) / s - 0.5, v = (Y + 0.5) / s - 0.5;
    for (int c = 0; c < CH; ++c)
        out[c] = 0.f;
    float p[CH];
    if (interp == IN_HEX)
    {
        // nearest centre of a hexagonal lattice of cell area 1 (side a: 3 sqrt3 / 2 a^2 = 1), anchored at cell 0
        const double h = std::sqrt(2.0 / std::sqrt(3.0));  // lattice spacing
        const double rr = v / (h * std::sqrt(3.0) / 2.0);
        const double qq = u / h - rr / 2.0;
        double cq = std::round(qq), cr = std::round(rr), cs = std::round(-qq - rr);
        const double dq = std::fabs(cq - qq), dr = std::fabs(cr - rr), dsv = std::fabs(cs + qq + rr);
        if (dq > dr && dq > dsv)
            cq = -cr - cs;
        else if (dr > dsv)
            cr = -cq - cs;
        const double uc = h * (cq + cr / 2.0), vc = h * cr * std::sqrt(3.0) / 2.0;
        const int iu = (int)std::floor(uc), iv = (int)std::floor(vc);
        const float fu = (float)(uc - iu), fv = (float)(vc - iv);
        const float wx[2] = {1.f - fu, fu}, wy[2] = {1.f - fv, fv};
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i)
            {
                const float* q = get(iu + i, iv + j, p);
                const float w = wx[i] * wy[j];
                for (int c = 0; c < CH; ++c)
                    out[c] += w * q[c];
            }
        return;
    }
    const KernInfo ki = KI(interp);
    const int iu = (int)std::floor(u), iv = (int)std::floor(v);
    const float fu = (float)(u - iu), fv = (float)(v - iv);
    float wx[6], wy[6];
    Weights(interp, fu, wx);
    Weights(interp, fv, wy);
    const bool clampIt = ki.neg && antiRing;
    float mn[CH], mx[CH];
    if (clampIt)
        for (int c = 0; c < CH; ++c)
            mn[c] = 3e38f, mx[c] = -3e38f;
    Acc5 sum;
    for (int j = ki.lo; j <= ki.hi; ++j)
    {
        const float wyj = wy[j - ki.lo];
        Acc5 row;
        for (int i = ki.lo; i <= ki.hi; ++i)
        {
            const float w = wx[i - ki.lo];
            if (w == 0.f && !(clampIt && (i == 0 || i == 1) && (j == 0 || j == 1)))
                continue;
            const float* q = get(iu + i, iv + j, p);
            row.add(w, q);
            if (clampIt && (i == 0 || i == 1) && (j == 0 || j == 1))
                for (int c = 0; c < CH; ++c)
                {
                    mn[c] = std::min(mn[c], q[c]);
                    mx[c] = std::max(mx[c], q[c]);
                }
        }
        if (wyj != 0.f)
            sum.addAcc(wyj, row);
    }
    sum.store(out);
    if (clampIt)
        for (int c = 0; c < CH; ++c)
            out[c] = std::min(mx[c], std::max(mn[c], out[c]));
}

inline float Smoothstep(float a, float b, float x)
{
    float t = (x - a) / (b - a);
    t = std::min(1.f, std::max(0.f, t));
    return t * t * (3.f - 2.f * t);
}

inline unsigned Hash(unsigned x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// --------------------------------------------------------------- the job ---
struct Engine
{
    const Job& job;
    Settings s;
    Ladder lad;
    int Lp = 0;
    Rect O1;
    std::vector<Range> rg;
    Lvl base;                 // level 0 (q*rgba, q) over R0
    std::vector<Lvl> pyr;     // unweighted pyramid levels 1..Lp (pyr[L])
    // Protect Edges
    std::vector<float> lam, fr;     // per R0 pixel: ladder position, front membership
    std::vector<const Lvl*> far;    // far[k] (entries with level >= 1), may alias pyr
    std::vector<Lvl> farOwn;
    std::vector<const Lvl*> nearB;  // near band j (level >= 1), nullptr = absent
    std::vector<Lvl> nearOwn;
    std::vector<char> nearHas;      // band j present (any j)
    std::vector<Range> bandBox;     // px (rel. D) where band j can be non-zero
    std::vector<Lvl> agg;           // per level: (min ladder position, max front membership) over each cell's support
    bool aggBuilt = false;
    std::vector<Lvl> lamPyr;        // ((1 - f) q * lam, ..., (1 - f) q), levels 1..Lp
    float minLam = 0;
    bool anyFrontW = false;
    bool anyFront = false;
    bool clip = false;

    std::unique_ptr<Pool> pool;  // v0.3: the frame's worker threads
    explicit Engine(const Job& j) : job(j), s(j.s) {}

    inline const float* B0(int X, int Y) const { return base.at(X, Y); }
    inline size_t Idx0(int X, int Y) const
    {
        X = std::min(base.extW - 1, std::max(0, X));
        Y = std::min(base.extH - 1, std::max(0, Y));
        X = std::min(base.x0 + base.w - 1, std::max(base.x0, X));
        Y = std::min(base.y0 + base.h - 1, std::max(base.y0, Y));
        return (size_t)(Y - base.y0) * base.w + (X - base.x0);
    }

    double TargetVar(float a) const
    {
        const double sg = SigmaForRadius(std::fabs((double)a) * s.radius);
        return sg * sg;
    }
    double SharpVar() const
    {
        const double ss = SigmaForRadius(s.sharpRadius);
        return ss * ss;
    }

    bool Aborted() const { return job.abort && job.abort(); }

    // Steps / Dither: the blend towards entry k+1 becomes 0 or 1
    inline float BlendT(float t, int X, int Y) const
    {
        if (!(t > 0) || s.blend == BL_SMOOTH)
            return t;
        if (s.blend == BL_STEPS)
            return t >= 0.5f ? 1.f : 0.f;
        // full-res pixel coordinates (a Half preview picks one pixel of each 2x2: same statistics)
        const double inv = s.ds > 0 ? 1.0 / s.ds : 1.0;
        const int xf = (int)std::floor((X + job.D.l) * inv + 1e-6), yf = (int)std::floor((Y + job.D.t) * inv + 1e-6);
        const unsigned hsh = Hash((unsigned)xf * 0x9E3779B1U ^ Hash((unsigned)yf + 0x632BE5ABU) ^
                                  Hash(s.ditherSeed * 0x85EBCA77U + (unsigned)s.ditherFrame * 0xC2B2AE3DU));
        const float r = (hsh >> 8) * (1.f / 16777216.f);
        return r < t ? 1.f : 0.f;
    }

    Status Run();
    void BuildBase();
    void BuildPyramid();
    void BuildEdge();
    void BuildAgg();
    void Evaluate(float* o1);
    void SampleIdx(int k, int X, int Y, float out[CH]) const;
    void FarIdx(int k, int X, int Y, float out[CH]) const;
    void NearIdx(int j, int X, int Y, float out[CH]) const;
    void Plain(double v, int X, int Y, float rgba[4]) const;
    void FarValue(double v, int X, int Y, float rgba[4]) const;
    void FarVp(int j, int X, int Y, float rgba[4]) const;
    float FarLam(int X, int Y, float ownLam, float f) const;
    float NearSum(int X, int Y, float rgba[4]) const;
};

void Engine::BuildBase()
{
    const Rect& R0 = job.R0;
    Range r;
    r.x0 = R0.l - job.D.l;
    r.x1 = R0.r - job.D.l;
    r.y0 = R0.t - job.D.t;
    r.y1 = R0.b - job.D.t;
    if (job.base5 && job.base5->size() == (size_t)R0.w() * R0.h() * CH)
    {
        base.L = 0;
        base.x0 = r.x0;
        base.y0 = r.y0;
        base.w = R0.w();
        base.h = R0.h();
        base.extW = job.D.w();
        base.extH = job.D.h();
        base.d.swap(*job.base5);
        return;
    }
    base.alloc(0, r, job.D.w(), job.D.h());
    const int w = R0.w();
    pool->For(R0.h(), [&](int b, int e) {
        for (int y = b; y < e; ++y)
            for (int x = 0; x < w; ++x)
            {
                const size_t i = (size_t)y * w + x;
                const float* p = job.rgba + i * 4;
                const float q = job.q ? job.q[i] : 1.f;
                float* d = &base.d[i * CH];
                d[0] = q * p[0];
                d[1] = q * p[1];
                d[2] = q * p[2];
                d[3] = q * p[3];
                d[4] = q;
            }
    });
}

void Engine::BuildPyramid()
{
    pyr.resize((size_t)Lp + 1);
    for (int L = 1; L <= Lp; ++L)
    {
        Lvl& dst = pyr[(size_t)L];
        dst.alloc(L, rg[(size_t)L], Ext(job.D.w(), L), Ext(job.D.h(), L));
        const Lvl& src = L == 1 ? base : pyr[(size_t)L - 1];
        Down(s.downSmooth, [&](int x, int y, float*) { return src.at(x, y); }, src.extW, src.extH,
             dst, *pool);
    }
}

// min / max over the downsample support of each cell, level by level (ch 0 = min lam, ch 1 = max f)
void Engine::BuildAgg()
{
    if (aggBuilt)
        return;
    aggBuilt = true;
    agg.resize((size_t)Lp + 1);
    const int dl = DownLo(s.downSmooth), dh = DownHi(s.downSmooth);
    for (int L = 1; L <= Lp; ++L)
    {
        const Lvl& src = agg[(size_t)L - 1];
        auto get = [&](int x, int y, float& mn, float& mx) {
            if (L == 1)
            {
                const size_t i = Idx0(x, y);
                mn = lam[i];
                mx = fr[i];
            }
            else
            {
                const float* p = src.at(x, y);
                mn = p[0];
                mx = p[1];
            }
        };
        Lvl& dst = agg[(size_t)L];
        dst.alloc(L, rg[(size_t)L], Ext(job.D.w(), L), Ext(job.D.h(), L));
        pool->For(dst.h, [&](int b, int e) {
            for (int j = b; j < e; ++j)
                for (int i = 0; i < dst.w; ++i)
                {
                    const int cx = dst.x0 + i, cy = dst.y0 + j;
                    float mn = 1e9f, mx = 0.f;
                    for (int v = dl; v <= dh; ++v)
                        for (int u = dl; u <= dh; ++u)
                        {
                            float a, b2;
                            get(2 * cx + u, 2 * cy + v, a, b2);
                            mn = std::min(mn, a);
                            mx = std::max(mx, b2);
                        }
                    float* o = &dst.d[((size_t)j * dst.w + i) * CH];
                    o[0] = mn;
                    o[1] = mx;
                }
        });
    }
}

void Engine::BuildEdge()
{
    auto te = std::chrono::steady_clock::now();
    auto elap = [&](const char* what) {
        if (!g_profile)
            return;
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "    [edge] %-10s %7.1f ms\n", what, std::chrono::duration<double, std::milli>(t1 - te).count());
        te = t1;
    };

    const Rect& R0 = job.R0;
    const size_t n = (size_t)R0.w() * R0.h();
    lam.assign(n, 0.f);
    fr.assign(n, 0.f);
    std::atomic<bool> front{false};
    const double dsInv = s.ds > 0 ? 1.0 / s.ds : 1.0;
    pool->For(R0.h(), [&](int b, int e) {
        bool fl = false;
        for (int y = b; y < e; ++y)
            for (int x = 0; x < R0.w(); ++x)
            {
                const size_t i = (size_t)y * R0.w() + x;
                const float a = job.amount ? job.amount[i] : 1.f;
                int k;
                float t;
                Locate(lad, TargetVar(a), k, t);
                lam[i] = (float)k + t;
                if (a < 0)
                {
                    const float rFull = (float)(std::fabs(a) * s.radius * dsInv);
                    fr[i] = Smoothstep(0.5f, 2.0f, rFull);  // front pixels blurred less than ~1 px stay "far"
                    if (fr[i] > 0 && base.d[i * CH + 4] > 0)
                        fl = true;
                }
            }
        if (fl)
            front = true;
    });
    anyFront = front;

    far.assign((size_t)lad.top + 2, nullptr);
    farOwn.clear();
    farOwn.reserve((size_t)lad.top + 2);
    nearB.assign((size_t)lad.top + 2, nullptr);
    nearOwn.clear();
    nearOwn.reserve((size_t)lad.top + 2);
    nearHas.assign((size_t)lad.top + 2, 0);

    minLam = 1e9f;
    anyFrontW = false;
    bandBox.assign((size_t)lad.top + 2, Range{1 << 30, -(1 << 30), 1 << 30, -(1 << 30)});
    {
        // v0.3: in parallel; min / max / or are order-independent, so the result equals the serial scan
        std::vector<char> has((size_t)lad.top + 2, 0);
        std::mutex mm;
        pool->For(R0.h(), [&](int yb, int ye) {
        std::vector<char> hasL((size_t)lad.top + 2, 0);
        std::vector<Range> boxL((size_t)lad.top + 2, Range{1 << 30, -(1 << 30), 1 << 30, -(1 << 30)});
        float minL = 1e9f;
        bool frontL = false;
        for (int y = yb; y < ye; ++y)
            for (int x = 0; x < R0.w(); ++x)
            {
                const size_t i = (size_t)y * R0.w() + x;
                if (base.d[i * CH + 4] > 0)
                {
                    minL = std::min(minL, lam[i]);
                    if (fr[i] > 0)
                        frontL = true;
                }
                if (fr[i] > 0)
                {
                    const int k = (int)lam[i];
                    const float t = lam[i] - k;
                    const int X = base.x0 + x, Y = base.y0 + y;
                    for (int j = k; j <= std::min(lad.top, k + 1); ++j)
                    {
                        if (j == k + 1 && !(t > 0))
                            break;
                        hasL[(size_t)j] = 1;
                        Range& bb = boxL[(size_t)j];
                        bb.x0 = std::min(bb.x0, X);
                        bb.x1 = std::max(bb.x1, X + 1);
                        bb.y0 = std::min(bb.y0, Y);
                        bb.y1 = std::max(bb.y1, Y + 1);
                    }
                }
            }
        std::lock_guard<std::mutex> lk(mm);
        minLam = std::min(minLam, minL);
        anyFrontW = anyFrontW || frontL;
        for (size_t j = 0; j < has.size(); ++j)
        {
            has[j] = has[j] | hasL[j];
            Range& bb = bandBox[j];
            bb.x0 = std::min(bb.x0, boxL[j].x0);
            bb.x1 = std::max(bb.x1, boxL[j].x1);
            bb.y0 = std::min(bb.y0, boxL[j].y0);
            bb.y1 = std::max(bb.y1, boxL[j].y1);
        }
        });
        nearHas = has;
        // grow by the reach of a band-j sample (chain spread + kernel taps): values outside are exactly 0
        for (int j = 0; j <= lad.top; ++j)
        {
            const int L = std::max(0, lad.level(j));
            const int g = j == 0 ? 3 : 9 * (1 << L) + 4;
            Range& bb = bandBox[(size_t)j];
            bb.x0 -= g;
            bb.x1 += g;
            bb.y0 -= g;
            bb.y1 += g;
        }
    }

    elap("scan");
    // a chain: weighted level 0 -> level L (>= 1), stored at L only
    auto chain = [&](int L, const auto& wfun, std::vector<Lvl>& own) -> const Lvl* {
        own.emplace_back();
        Lvl cur;
        {
            Lvl& l1 = L == 1 ? own.back() : cur;
            l1.alloc(1, rg[1], Ext(job.D.w(), 1), Ext(job.D.h(), 1));
            const Lvl& b0 = base;
            auto src = [&](int x, int y, float* o) -> const float* {
                x = std::min(b0.x0 + b0.w - 1, std::max(b0.x0, x));
                y = std::min(b0.y0 + b0.h - 1, std::max(b0.y0, y));
                const size_t i = (size_t)(y - b0.y0) * b0.w + (x - b0.x0);
                float wc, wq;
                wfun(i, wc, wq);
                const float* p = &b0.d[i * CH];
                o[0] = wc * p[0];
                o[1] = wc * p[1];
                o[2] = wc * p[2];
                o[3] = wc * p[3];
                o[4] = wq * p[4];
                return o;
            };
            Down(s.downSmooth, src, b0.extW, b0.extH, l1, *pool);
        }
        for (int lv = 2; lv <= L; ++lv)
        {
            Lvl nx;
            Lvl& dst = lv == L ? own.back() : nx;
            dst.alloc(lv, rg[(size_t)lv], Ext(job.D.w(), lv), Ext(job.D.h(), lv));
            Down(s.downSmooth, [&](int x, int y, float*) { return cur.at(x, y); }, cur.extW, cur.extH,
                 dst, *pool);
            if (lv != L)
                cur = std::move(nx);
        }
        return &own.back();
    };

    // lamPyr: the background's ladder position around front pixels
    if (anyFrontW && Lp >= 1)
    {
        lamPyr.resize((size_t)Lp + 1);
        for (int L = 1; L <= Lp; ++L)
        {
            Lvl& dst = lamPyr[(size_t)L];
            dst.alloc(L, rg[(size_t)L], Ext(job.D.w(), L), Ext(job.D.h(), L));
            if (L == 1)
            {
                const Lvl& b0 = base;
                Down(s.downSmooth, [&](int x, int y, float* o) -> const float* {
                    x = std::min(b0.x0 + b0.w - 1, std::max(b0.x0, x));
                    y = std::min(b0.y0 + b0.h - 1, std::max(b0.y0, y));
                    const size_t i = (size_t)(y - b0.y0) * b0.w + (x - b0.x0);
                    const float w = (1.f - fr[i]) * b0.d[i * CH + 4];
                    o[0] = w * lam[i];
                    o[1] = o[2] = o[3] = 0.f;
                    o[4] = w;
                    return o;
                }, b0.extW, b0.extH, dst, *pool);
            }
            else
            {
                const Lvl& src = lamPyr[(size_t)L - 1];
                Down(s.downSmooth, [&](int x, int y, float*) { return src.at(x, y); }, src.extW, src.extH,
                     dst, *pool);
            }
        }
    }

    elap("lampyr");
    // far chains for entries with a pyramid level >= 1: weight (1 - f) * g_k, g_k = clamp(lam - (k - 2), 0, 1)
    for (int k = 1; k <= lad.top; ++k)
    {
        const int L = lad.level(k);
        if (L < 1)
            continue;
        if (Aborted())
            return;
        const float kk = (float)k;
        auto gk = [kk](float l) { return kk <= 1.f ? 1.f : std::min(1.f, std::max(0.f, l - (kk - 2.f))); };
        // every weight is 1 (no front pixel, none sharper than this entry): the unweighted level
        if (!anyFrontW && (k <= 1 || minLam >= kk - 1.f))
        {
            far[(size_t)k] = &pyr[(size_t)L];
            continue;
        }
        const int sLev = std::max(0, L - 2);
        if (sLev == 0)
        {
            far[(size_t)k] = chain(L, [&](size_t i, float& wc, float& wq) { wc = wq = (1.f - fr[i]) * gk(lam[i]); }, farOwn);
            continue;
        }
        // coarse start (speed): weight whole level-sLev cells of the unweighted pyramid, by the min ladder
        // position / max front membership over each cell's support (conservative: a cell whose support
        // touches a sharper or front pixel is left out, so nothing of the subject leaks; the hole grows by
        // the support, ~0.3 sigma of this entry, and is filled from the background around it)
        BuildAgg();
        farOwn.emplace_back();
        Lvl cur, nx;
        for (int lv = sLev + 1; lv <= L; ++lv)
        {
            Lvl& dst = lv == L ? farOwn.back() : nx;
            dst.alloc(lv, rg[(size_t)lv], Ext(job.D.w(), lv), Ext(job.D.h(), lv));
            if (lv == sLev + 1)
            {
                const Lvl& P = pyr[(size_t)sLev];
                const Lvl& A = agg[(size_t)sLev];
                Down(s.downSmooth, [&](int x, int y, float* o) -> const float* {
                    const float* p = P.at(x, y);
                    const float* a = A.at(x, y);
                    const float w = (1.f - a[1]) * gk(a[0]);
                    for (int c = 0; c < CH; ++c)
                        o[c] = w * p[c];
                    return o;
                }, P.extW, P.extH, dst, *pool);
            }
            else
                Down(s.downSmooth, [&](int x, int y, float*) { return cur.at(x, y); }, cur.extW, cur.extH,
                     dst, *pool);
            if (lv != L)
                cur = std::move(nx);
        }
        far[(size_t)k] = &farOwn.back();
    }
    elap("far");
    // near bands with a level >= 1: colour weight f * b_j, q unweighted (the normaliser)
    for (int j = 1; j <= lad.top; ++j)
    {
        if (!nearHas[(size_t)j] || lad.level(j) < 1)
            continue;
        if (Aborted())
            return;
        nearB[(size_t)j] = chain(lad.level(j), [&](size_t i, float& wc, float& wq) {
            const int k = (int)lam[i];
            const float t = lam[i] - k;
            const float b = k == j ? 1.f - t : (k + 1 == j ? t : 0.f);
            wc = fr[i] * b;
            wq = 1.f;
        }, nearOwn);
    }
    elap("near");
}

// unweighted ladder entry k at pixel (X, Y) (relative to D)
void Engine::SampleIdx(int k, int X, int Y, float out[CH]) const
{
    if (k <= 0)
    {
        std::memcpy(out, B0(X, Y), sizeof(float) * CH);
        return;
    }
    const int L = lad.level(k);
    const Lvl& l = L == 0 ? base : pyr[(size_t)L];
    SampleLevel(s.interp, s.antiRing, L, X, Y, [&](int cx, int cy, float*) -> const float* { return l.at(cx, cy); }, out);
}

void Engine::FarIdx(int k, int X, int Y, float out[CH]) const
{
    auto get0 = [&](int cx, int cy, float* o) -> const float* {
        const size_t i = Idx0(cx, cy);
        const float w = 1.f - fr[i];
        const float* p = &base.d[i * CH];
        for (int c = 0; c < CH; ++c)
            o[c] = w * p[c];
        return o;
    };
    if (k <= 0)
    {
        get0(X, Y, out);
        return;
    }
    const int L = lad.level(k);
    if (L == 0)
    {
        SampleLevel(s.interp, s.antiRing, 0, X, Y, get0, out);
        return;
    }
    const Lvl* l = far[(size_t)k];
    SampleLevel(s.interp, s.antiRing, L, X, Y, [&](int cx, int cy, float*) -> const float* { return l->at(cx, cy); }, out);
}

void Engine::NearIdx(int j, int X, int Y, float out[CH]) const
{
    auto get0 = [&](int cx, int cy, float* o) {
        const size_t i = Idx0(cx, cy);
        const int k = (int)lam[i];
        const float t = lam[i] - k;
        const float b = k == j ? 1.f - t : (k + 1 == j ? t : 0.f);
        const float w = fr[i] * b;
        const float* p = &base.d[i * CH];
        for (int c = 0; c < 4; ++c)
            o[c] = w * p[c];
        o[4] = p[4];
        return o;
    };
    if (j <= 0)
    {
        get0(X, Y, out);
        return;
    }
    const int L = lad.level(j);
    if (L == 0)
    {
        SampleLevel(s.interp, s.antiRing, 0, X, Y, get0, out);
        return;
    }
    const Lvl* l = nearB[(size_t)j];
    SampleLevel(s.interp, s.antiRing, L, X, Y, [&](int cx, int cy, float*) -> const float* { return l->at(cx, cy); }, out);
}

// without Protect Edges: blend of two ladder entries, normalised by the inside weight
void Engine::Plain(double v, int X, int Y, float rgba[4]) const
{
    int k;
    float t;
    Locate(lad, v, k, t);
    t = BlendT(t, X, Y);
    float a[CH], b[CH];
    if (t >= 1.f)
    {
        ++k;
        t = 0.f;
    }
    if (k == 0 && t == 0)
    {
        // exact identity (no normalisation round-off) where the pixel is inside
        const float* p = B0(X, Y);
        if (p[4] == 1.f)
        {
            for (int c = 0; c < 4; ++c)
                rgba[c] = p[c];
            return;
        }
    }
    SampleIdx(k, X, Y, a);
    if (t > 0)
    {
        SampleIdx(k + 1, X, Y, b);
        for (int c = 0; c < CH; ++c)
            a[c] += t * (b[c] - a[c]);
    }
    if (a[4] > 1e-12f)
    {
        const float inv = 1.f / a[4];
        for (int c = 0; c < 4; ++c)
            rgba[c] = a[c] * inv;
    }
    else
        rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.f;
}

const float kTheta = 0.15f;  // far layer: below this weight (fraction of the inside mass) look coarser

void Engine::FarVp(int j, int X, int Y, float rgba[4]) const
{
    float S[CH];
    FarIdx(j, X, Y, S);
    float Q = 1.f;
    if (clip)
    {
        float u[CH];
        SampleIdx(j, X, Y, u);
        Q = u[4];
    }
    const float thr = kTheta * Q;
    if (S[4] >= thr && S[4] > 1e-12f)
    {
        if (j == 0 && S[4] == 1.f)
        {
            for (int c = 0; c < 4; ++c)
                rgba[c] = S[c];
            return;
        }
        const float inv = 1.f / S[4];
        for (int c = 0; c < 4; ++c)
            rgba[c] = S[c] * inv;
        return;
    }
    float nx[4];
    if (j < lad.top)
        FarVp(j + 1, X, Y, nx);
    else
    {
        float u[CH];
        SampleIdx(lad.top, X, Y, u);
        if (u[4] > 1e-12f)
            for (int c = 0; c < 4; ++c)
                nx[c] = u[c] / u[4];
        else
            nx[0] = nx[1] = nx[2] = nx[3] = 0.f;
    }
    if (!(thr > 1e-12f))
    {
        for (int c = 0; c < 4; ++c)
            rgba[c] = nx[c];
        return;
    }
    const float m = thr - std::max(0.f, S[4]);
    for (int c = 0; c < 4; ++c)
        rgba[c] = (S[c] + m * nx[c]) / thr;
}

float Engine::FarLam(int X, int Y, float ownLam, float f) const
{
    if (!(f > 0) || lamPyr.empty())
        return ownLam;
    float bg = -1.f;
    for (int k = std::max(1, std::min(lad.top, (int)ownLam + 1)); k <= lad.top && bg < 0; ++k)
    {
        const int L = lad.level(k);
        if (L < 1 || L > Lp)
            continue;
        const Lvl& l = lamPyr[(size_t)L];
        float S[CH];
        SampleLevel(IN_BSPLINE, false, L, X, Y, [&](int cx, int cy, float*) -> const float* { return l.at(cx, cy); }, S);
        if (S[4] > 0.05f)
            bg = S[0] / S[4];
    }
    if (bg < 0)
        bg = ownLam;
    return (1.f - f) * ownLam + f * bg;
}

void Engine::FarValue(double v, int X, int Y, float rgba[4]) const
{
    int k;
    float t;
    Locate(lad, v, k, t);
    t = BlendT(t, X, Y);
    if (t >= 1.f)
    {
        ++k;
        t = 0.f;
    }
    FarVp(k, X, Y, rgba);
    if (t > 0)
    {
        float b[4];
        FarVp(k + 1, X, Y, b);
        for (int c = 0; c < 4; ++c)
            rgba[c] += t * (b[c] - rgba[c]);
    }
}

float Engine::NearSum(int X, int Y, float rgba[4]) const
{
    rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.f;
    if (!anyFront)
        return 0.f;
    bool any = false;
    for (int j = 0; j <= lad.top; ++j)
    {
        if (!nearHas[(size_t)j])
            continue;
        const Range& bb = bandBox[(size_t)j];
        if (X < bb.x0 || X >= bb.x1 || Y < bb.y0 || Y >= bb.y1)
            continue;
        float S[CH];
        NearIdx(j, X, Y, S);
        if (S[4] > 1e-12f && (S[0] != 0.f || S[1] != 0.f || S[2] != 0.f || S[3] != 0.f))
        {
            const float inv = 1.f / S[4];
            for (int c = 0; c < 4; ++c)
                rgba[c] += S[c] * inv;
            any = true;
        }
    }
    return any ? 1.f : 0.f;
}

void Engine::Evaluate(float* o1)
{
    const int w = O1.w();
    const bool unsharp = s.sharpType == SH_UNSHARP && s.sharpAmount > 0;
    const float gain = (float)s.sharpAmount;
    const double sv = SharpVar();
    const Rect& R0 = job.R0;
    pool->For(O1.h(), [&](int b, int e) {
        for (int y = b; y < e; ++y)
        {
            const int Y = O1.t + y - job.D.t;
            for (int x = 0; x < w; ++x)
            {
                const int X = O1.l + x - job.D.l;
                const int ax = std::min(R0.r - 1, std::max(R0.l, O1.l + x)) - R0.l;
                const int ay = std::min(R0.b - 1, std::max(R0.t, O1.t + y)) - R0.t;
                const float am = job.amount ? job.amount[(size_t)ay * R0.w() + ax] : 1.f;
                const double v = TargetVar(am);
                float* o = o1 + ((size_t)y * w + x) * 4;
                float res[4], res2[4];
                if (s.protect)
                {
                    float N[4];
                    const bool hasN = NearSum(X, Y, N) > 0;
                    double vf = v;
                    const size_t ai = (size_t)ay * R0.w() + ax;
                    if (anyFrontW && fr[ai] > 0)
                    {
                        const float lf = FarLam(X, Y, lam[ai], fr[ai]);
                        const int kk = std::min(lad.top, (int)lf);
                        const double tt = lf - kk;
                        vf = kk >= lad.top ? lad.var[lad.top] : lad.var[kk] + tt * (lad.var[kk + 1] - lad.var[kk]);
                    }
                    FarValue(vf, X, Y, res);
                    if (unsharp)
                        FarValue(vf + sv, X, Y, res2);
                    if (hasN)
                    {
                        const float occ = 1.f - std::min(1.f, std::max(0.f, N[3]));
                        for (int c = 0; c < 4; ++c)
                        {
                            res[c] = N[c] + occ * res[c];
                            if (unsharp)
                                res2[c] = N[c] + occ * res2[c];
                        }
                    }
                }
                else
                {
                    Plain(v, X, Y, res);
                    if (unsharp)
                        Plain(v + sv, X, Y, res2);
                }
                if (unsharp)
                {
                    // detail of the blurred image at the sharpen radius, soft-limited (bright side harder:
                    // no white halos around dark line art), in the layer's own (encoded) values
                    const float al = std::max(0.f, std::min(1.f, res[3]));
                    const float lp = 0.08f * al + 1e-6f, ln = 0.25f * al + 1e-6f;
                    for (int c = 0; c < 3; ++c)
                    {
                        const float d = gain * (res[c] - res2[c]);
                        const float lim = d > 0 ? lp : ln;
                        res[c] = res[c] + lim * std::tanh(d / lim);
                    }
                }
                // negative lobes without anti-ringing can undershoot: never negative, alpha <= 1
                o[0] = std::max(0.f, res[0]);
                o[1] = std::max(0.f, res[1]);
                o[2] = std::max(0.f, res[2]);
                o[3] = std::min(1.f, std::max(0.f, res[3]));
            }
        }
    });
}

Status Engine::Run()
{
    pool.reset(new Pool(s.threads));
    lad = MakeLadder(s);
    Lp = lad.top > 0 ? std::max(0, lad.level(lad.top)) : 0;
    clip = job.clip || job.q != nullptr;
    O1 = NeedsRing(s) ? Rect{job.O.l - 1, job.O.t - 1, job.O.r + 1, job.O.b + 1} : job.O;
    rg = MakeRanges(s, lad, Lp, job.D, O1);
    auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
        if (!g_profile)
            return;
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  [mbb] %-10s %7.1f ms\n", what, std::chrono::duration<double, std::milli>(t1 - t0).count());
        t0 = t1;
    };
    BuildBase();
    lap("base");
    if (Aborted())
        return ST_ABORTED;
    BuildPyramid();
    lap("pyramid");
    if (Aborted())
        return ST_ABORTED;
    if (s.protect)
    {
        BuildEdge();
        lap("edge");
        if (Aborted())
            return ST_ABORTED;
    }
    const bool ring = NeedsRing(s);
    FVec o1v;
    float* o1 = job.out;
    if (ring)
    {
        o1v.resize((size_t)O1.w() * O1.h() * 4);
        o1 = o1v.data();
    }
    Evaluate(o1);
    lap("evaluate");
    if (Aborted())
        return ST_ABORTED;
    if (ring)
    {
        // AE-style Sharpen: 3x3, centre 1 + 8a, the 8 neighbours -a, on premultiplied RGBA
        const float a = (float)s.sharpAmount;
        const int w1 = O1.w(), w = job.O.w();
        pool->For(job.O.h(), [&](int b, int e) {
            for (int y = b; y < e; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const float* c0 = o1 + ((size_t)(y + 1) * w1 + (x + 1)) * 4;
                    float sum[4] = {0, 0, 0, 0};
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (!dx && !dy)
                                continue;
                            const float* p = c0 + ((ptrdiff_t)dy * w1 + dx) * 4;
                            for (int c = 0; c < 4; ++c)
                                sum[c] += p[c];
                        }
                    float* o = job.out + ((size_t)y * w + x) * 4;
                    float al = (1.f + 8.f * a) * c0[3] - a * sum[3];
                    al = std::min(1.f, std::max(0.f, al));
                    for (int c = 0; c < 3; ++c)
                        o[c] = std::max(0.f, (1.f + 8.f * a) * c0[c] - a * sum[c]);
                    o[3] = al;
                }
        });
    }
    return ST_OK;
}

}  // namespace

int PyramidLevels(const Settings& s)
{
    const Ladder l = MakeLadder(s);
    return l.top > 0 ? std::max(0, l.level(l.top)) : 0;
}

int SpreadPx(const Settings& s)
{
    // 4 sigma of the largest blur (the Gaussian-like tail beyond is < 0.04 %)
    return (int)std::ceil(4.0 * SigmaForRadius(std::max(0.0, s.radius))) + 2;
}

bool NeedsRing(const Settings& s)
{
    return s.sharpType == SH_AE && s.sharpAmount > 0;
}

Rect InputRect(const Settings& s, const Rect& D, const Rect& O)
{
    const Ladder lad = MakeLadder(s);
    const int Lp = PyramidLevels(s);
    const Rect O1 = NeedsRing(s) ? Rect{O.l - 1, O.t - 1, O.r + 1, O.b + 1} : O;
    const std::vector<Range> rg = MakeRanges(s, lad, Lp, D, O1);
    const Range& r = rg[0];
    return Rect{D.l + r.x0, D.t + r.y0, D.l + r.x1, D.t + r.y1};
}

bool DescribeGpu(const Job& job, mbbgpu_desc& d)
{
    const Settings& s = job.s;
    if (job.O.empty() || job.D.empty() || !job.base5)
        return false;
    std::memset(&d, 0, sizeof d);
    d.size = sizeof d;
    const Ladder lad = MakeLadder(s);
    const int Lp = lad.top > 0 ? std::max(0, lad.level(lad.top)) : 0;
    if (lad.top + 2 > MBBGPU_MAX_ENTRIES || Lp + 1 > MBBGPU_MAX_ENTRIES)
        return false;
    const bool ring = NeedsRing(s);
    const Rect O1 = ring ? Rect{job.O.l - 1, job.O.t - 1, job.O.r + 1, job.O.b + 1} : job.O;
    const std::vector<Range> rg = MakeRanges(s, lad, Lp, job.D, O1);
    auto put = [](int32_t* a, const Rect& r) {
        a[0] = r.l;
        a[1] = r.t;
        a[2] = r.r;
        a[3] = r.b;
    };
    put(d.D, job.D);
    put(d.O, job.O);
    put(d.O1, O1);
    put(d.R0, job.R0);
    d.interp = s.interp;
    d.downSmooth = s.downSmooth ? 1 : 0;
    d.blend = s.blend;
    d.antiRing = s.antiRing ? 1 : 0;
    d.protect = s.protect ? 1 : 0;
    d.clip = (job.clip || job.q) ? 1 : 0;
    d.ditherSeed = s.ditherSeed;
    d.ditherFrame = s.ditherFrame;
    d.sharpType = s.sharpType;
    d.ring = ring ? 1 : 0;
    d.hasAmount = job.amount ? 1 : 0;
    d.top = lad.top;
    d.Lp = Lp;
    d.radius = s.radius;
    d.ds = s.ds;
    d.sharpAmount = s.sharpAmount;
    d.sharpRadius = s.sharpRadius;
    for (int k = 0; k < MBBGPU_MAX_ENTRIES; ++k)
    {
        d.lev[k] = k <= lad.top + 1 ? lad.lev[k] : 0;
        d.var[k] = k <= lad.top + 1 ? lad.var[k] : 0.0;
    }
    for (int L = 0; L <= Lp; ++L)
    {
        const Range& r = rg[(size_t)L];
        d.rg[L] = mbbgpu_range{r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0, Ext(job.D.w(), L), Ext(job.D.h(), L)};
    }
    return true;
}

Status Render(const Job& job)
{
    if (job.O.empty() || job.D.empty() || (!job.rgba && !job.base5) || !job.out)
        return ST_BAD;
    try
    {
        Engine e(job);
        return e.Run();
    }
    catch (const std::bad_alloc&)
    {
        return ST_OOM;
    }
    catch (...)
    {
        return ST_BAD;
    }
}

}  // namespace mbb
