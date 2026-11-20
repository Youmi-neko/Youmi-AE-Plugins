// Shitsukan engine helpers (header-only, internal): threads, noise, the recursive domain-transform filter,
// normalised smoothing, min / max filters. Included by sk_core.cpp only.
#pragma once
#include "sk_core.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
// mac_port: the Windows-only pieces (thread pool, SSE2, memory counters) behind sk_platform.h; the float math
// behind sk_pmath.h (std:: on Windows, the portable MSVC-exact functions elsewhere)
#include "sk_platform.h"
#include "sk_pmath.h"
#if !SK_PLAT_WIN32
#include <condition_variable>
#include <deque>
#endif

namespace sk {
extern int g_profile;
namespace {
std::atomic<long> g_pforCount{0};  // (profiling only)

struct AbortEx
{
};

// ------------------------------------------------------------------ helpers ---
#if SK_PLAT_WIN32
// 0.6.1: the chunks of a parallel loop run on the system thread pool (no thread creation per loop: that was
// ~3 ms per loop, about 160 loops per frame). The chunks are EXACTLY the old ones ([n*t/T, n*(t+1)/T)), so every
// result is bit-identical; who runs a chunk does not matter. The caller claims chunks too, so a loop always
// finishes even when no pool thread is free (nested loops, many concurrent frames: no deadlock). The job lives on
// the heap and is shared with the callbacks: a late callback finds nothing left to claim and just lets go.
struct PoolJob
{
    void (*call)(const void*, int, int) = nullptr;
    const void* fn = nullptr;
    int n = 0, T = 0;
    std::atomic<int> next{0}, done{0}, refs{1};
    std::exception_ptr ex;
    std::mutex m;
    HANDLE ev = nullptr;
    void release()
    {
        if (refs.fetch_sub(1) == 1)
        {
            CloseHandle(ev);
            delete this;
        }
    }
    void work()
    {
        for (;;)
        {
            const int t = next.fetch_add(1);
            if (t >= T)
                return;
            try
            {
                call(fn, (int)((long long)n * t / T), (int)((long long)n * (t + 1) / T));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(m);
                if (!ex)
                    ex = std::current_exception();
            }
            if (done.fetch_add(1) + 1 == T)
                SetEvent(ev);
        }
    }
};
inline void CALLBACK PoolCb(PTP_CALLBACK_INSTANCE, void* ctx)
{
    PoolJob* j = (PoolJob*)ctx;
    j->work();
    j->release();
}
inline PTP_CALLBACK_ENVIRON PoolEnv()
{
    // the callbacks keep this module loaded until they have returned
    static TP_CALLBACK_ENVIRON env;
    static std::once_flag once;
    std::call_once(once, [] {
        InitializeThreadpoolEnvironment(&env);
        HMODULE me = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCWSTR)&PoolCb, &me))
            SetThreadpoolCallbackLibrary(&env, me);
    });
    return &env;
}
#else
// mac_port: the portable equivalent of the Windows pool above (macOS and SK_FORCE_PORTABLE builds): the SAME chunks
// ([n*t/T, n*(t+1)/T)) claimed by the caller and by persistent worker threads, so every result is bit-identical to
// the Windows build; the caller claims chunks too (nested loops / many concurrent frames cannot deadlock). The job
// is shared with the workers by reference count; a late worker finds nothing left to claim and lets go.
struct PoolJob
{
    void (*call)(const void*, int, int) = nullptr;
    const void* fn = nullptr;
    int n = 0, T = 0;
    std::atomic<int> next{0}, done{0}, refs{1};
    std::exception_ptr ex;
    std::mutex m;
    std::condition_variable cv;
    void release()
    {
        if (refs.fetch_sub(1) == 1)
            delete this;
    }
    void work()
    {
        for (;;)
        {
            const int t = next.fetch_add(1);
            if (t >= T)
                return;
            try
            {
                call(fn, (int)((long long)n * t / T), (int)((long long)n * (t + 1) / T));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(m);
                if (!ex)
                    ex = std::current_exception();
            }
            if (done.fetch_add(1) + 1 == T)
            {
                std::lock_guard<std::mutex> lk(m);
                cv.notify_all();
            }
        }
    }
    void wait()
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [this] { return done.load() >= T; });
    }
};
// persistent workers (hardware_concurrency of them, started on first use, never joined: they only ever run engine
// code while a PFor of a render is waiting for them, so nothing of the plug-in is running when AE quits)
struct WorkerPool
{
    std::mutex m;
    std::condition_variable cv;
    std::deque<PoolJob*> q;
    bool ok = false;
    WorkerPool()
    {
        const unsigned hw = std::thread::hardware_concurrency();
        const int n = (int)std::max(1u, hw ? hw : 4u);
        try
        {
            for (int i = 0; i < n; ++i)
                std::thread([this] { loop(); }).detach();
            ok = true;
        }
        catch (...)
        {
            ok = false;  // (no threads: PFor falls back to threads per call)
        }
    }
    void loop()
    {
        for (;;)
        {
            PoolJob* j = nullptr;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this] { return !q.empty(); });
                j = q.front();
                q.pop_front();
            }
            j->work();
            j->release();
        }
    }
    void submit(PoolJob* j, int copies)
    {
        {
            std::lock_guard<std::mutex> lk(m);
            for (int i = 0; i < copies; ++i)
                q.push_back(j);
        }
        cv.notify_all();
    }
    static WorkerPool& Get()
    {
        static WorkerPool* p = new WorkerPool();  // (leaked on purpose: no destructor runs at unload)
        return *p;
    }
};
#endif

template <class F>
void PFor(int n, int threads, const F& f)  // f(begin, end); every index handled by exactly one call
{
    const int T = std::max(1, std::min(threads, n / 4));
    if (g_profile)
        g_pforCount.fetch_add(1);
    if (T <= 1)
    {
        if (n > 0)
            f(0, n);
        return;
    }
#if SK_PLAT_WIN32
    {
        PoolJob* j = new PoolJob;
        j->call = [](const void* p, int a, int b) { (*(const F*)p)(a, b); };
        j->fn = &f;
        j->n = n;
        j->T = T;
        j->ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (j->ev)
        {
            PTP_CALLBACK_ENVIRON env = PoolEnv();
            for (int t = 1; t < T; ++t)
            {
                j->refs.fetch_add(1);
                if (!TrySubmitThreadpoolCallback(PoolCb, j, env))
                {
                    j->refs.fetch_sub(1);
                    break;  // (the caller claims what is left)
                }
            }
            j->work();
            WaitForSingleObject(j->ev, INFINITE);
            std::exception_ptr ex = j->ex;
            j->release();
            if (ex)
                std::rethrow_exception(ex);
            return;
        }
        delete j;
    }
#else
    if (WorkerPool::Get().ok)
    {
        PoolJob* j = new PoolJob;
        j->call = [](const void* p, int a, int b) { (*(const F*)p)(a, b); };
        j->fn = &f;
        j->n = n;
        j->T = T;
        j->refs.fetch_add(T - 1);
        WorkerPool::Get().submit(j, T - 1);
        j->work();
        j->wait();
        std::exception_ptr ex = j->ex;
        j->release();
        if (ex)
            std::rethrow_exception(ex);
        return;
    }
#endif
    std::vector<std::thread> pool;
    std::exception_ptr ex;
    std::mutex m;
    pool.reserve((size_t)T);
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t] {
            try
            {
                f((int)((long long)n * t / T), (int)((long long)n * (t + 1) / T));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(m);
                if (!ex)
                    ex = std::current_exception();
            }
        });
    for (auto& th : pool)
        th.join();
    if (ex)
        std::rethrow_exception(ex);
}

inline float Clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float Smooth(float e0, float e1, float x)
{
    const float t = Clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}
inline float Luma(const float* c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }
inline double Clampd(double v, double lo, double hi) { return std::isfinite(v) ? std::min(hi, std::max(lo, v)) : lo; }

// ------------------------------------------------------------------- noise ---
// Gradient noise on an integer lattice hashed with the seed; evaluated at FULL-RES layer coordinates.
inline uint32_t Hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
inline uint32_t Hash2(int x, int y, uint32_t s) { return Hash((uint32_t)x * 0x9E3779B1u ^ Hash((uint32_t)y * 0x85EBCA77u ^ s)); }

struct GradTable
{
    float gx[256], gy[256];
    GradTable()
    {
#if SK_PORTABLE_MATH
        // mac_port: the Windows build's values (tools/gen_gradtab.cpp), not the platform's double cos / sin
#include "sk_gradtab.inc"
        std::memcpy(gx, kGradX, sizeof(gx));
        std::memcpy(gy, kGradY, sizeof(gy));
#else
        for (int i = 0; i < 256; ++i)
        {
            const double a = (i + 0.5) * 6.283185307179586 / 256.0;
            gx[i] = (float)pm::cos(a);
            gy[i] = (float)pm::sin(a);
        }
#endif
    }
};
const GradTable g_grad;

inline float Fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
inline float Perlin(double x, double y, uint32_t s)
{
    const double fx0 = std::floor(x), fy0 = std::floor(y);
    const int ix = (int)fx0, iy = (int)fy0;
    const float fx = (float)(x - fx0), fy = (float)(y - fy0);
    auto g = [&](int i, int j, float dx, float dy) {
        const uint32_t h = Hash2(ix + i, iy + j, s) & 255u;
        return g_grad.gx[h] * dx + g_grad.gy[h] * dy;
    };
    const float n00 = g(0, 0, fx, fy), n10 = g(1, 0, fx - 1.f, fy), n01 = g(0, 1, fx, fy - 1.f), n11 = g(1, 1, fx - 1.f, fy - 1.f);
    const float u = Fade(fx), v = Fade(fy);
    const float a = n00 + (n10 - n00) * u, b = n01 + (n11 - n01) * u;
    return 1.41f * (a + (b - a) * v);
}

// fBm, octaves rotated against each other (no axis-aligned grain). lambda: wavelength of the first octave,
// full-res px. ds: buffer px per full px: octaves shorter than ~1 buffer px fade out (preview = average look).
// Normalised so the result has roughly the spread of a single octave (about -1..1, sd ~0.35).
inline float Fbm(double X, double Y, double lambda, int oct, double gain, uint32_t seed, double ds)
{
    double f = 1.0 / std::max(0.5, lambda);
    float sum = 0.f, amp = 1.f, norm = 0.f;
    const double c = 0.8253356149, s = 0.5646424734;  // ~34.4 degrees per octave
    double x = X, y = Y;
    for (int o = 0; o < oct; ++o)
    {
        const double wlBuf = ds / f;  // wavelength in buffer px
        const float fade = (float)Clampd((wlBuf - 0.9) / 0.9, 0.0, 1.0);
        if (fade > 0.f)
            sum += amp * fade * Perlin(x * f + 17.31 * o, y * f - 9.17 * o, seed + 0x51ED27u * (uint32_t)o);
        norm += amp * amp;
        amp *= (float)gain;
        f *= 2.0;
        const double nx = c * x - s * y, ny = s * x + c * y;
        x = nx;
        y = ny;
    }
    return norm > 0.f ? sum / std::sqrt(norm) : 0.f;
}

inline int FloorDiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// A smooth noise field over the frame. When its shortest wavelength allows (>= ~6 nodes per wavelength), it is
// evaluated on a coarse grid anchored to LAYER coordinates (multiples of `step` buffer px) and interpolated
// bilinearly: the same values whatever the domain's origin / size (tile independent), much faster at UHD.
// fn(X, Y) takes full-res layer coordinates.
template <class F>
void NoiseField(std::vector<float>& out, int W, int H, int ox, int oy, double ds, double minLambdaFull, int threads, const F& fn)
{
    out.resize((size_t)W * H);
    // >= 4 nodes per wavelength of the finest octave (bilinear between them: the error is far below what shows)
    const int step = std::max(1, std::min(8, (int)std::floor(minLambdaFull * ds / 4.0)));
    if (step == 1)
    {
        PFor(H, threads, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                    out[(size_t)y * W + x] = fn((x + ox) / ds, (y + oy) / ds);
        });
        return;
    }
    const int gx0 = FloorDiv(ox, step), gy0 = FloorDiv(oy, step);
    const int gw = FloorDiv(ox + W - 1, step) - gx0 + 2, gh = FloorDiv(oy + H - 1, step) - gy0 + 2;
    std::vector<float> g((size_t)gw * gh);
    PFor(gh, threads, [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j)
            for (int i = 0; i < gw; ++i)
                g[(size_t)j * gw + i] = fn((double)(gx0 + i) * step / ds, (double)(gy0 + j) * step / ds);
    });
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
        {
            const int ly = y + oy, j = FloorDiv(ly, step) - gy0;
            const float fy = (float)(ly - (gy0 + j) * step) / (float)step;
            const float* r0 = &g[(size_t)j * gw];
            const float* r1 = r0 + gw;
            for (int x = 0; x < W; ++x)
            {
                const int lx = x + ox, i = FloorDiv(lx, step) - gx0;
                const float fx = (float)(lx - (gx0 + i) * step) / (float)step;
                const float a = r0[i] + (r0[i + 1] - r0[i]) * fx, b = r1[i] + (r1[i + 1] - r1[i]) * fx;
                out[(size_t)y * W + x] = a + (b - a) * fy;
            }
        }
    });
}

// --------------------------------------------- recursive domain-transform filter ---
// Edge cost between neighbours: d = 1 + k * |guide difference|_1 + (barrier on either side ? huge : 0).
struct Edges
{
    const float* gh = nullptr;  // difference to the left neighbour (x >= 1)
    const float* gv = nullptr;  // difference to the upper neighbour (y >= 1)
    const uint8_t* bar = nullptr;
    const uint8_t* cut = nullptr;  // per pixel: bit 0 = no flow to the left neighbour, bit 1 = to the upper one
    double k = 0;
};
const float kBarrierCost = 1.0e4f;

// In-place smoothing of a 4-float-per-pixel image (Gastal & Oliveira's recursive filter, 3 iterations of
// horizontal + vertical forward / backward passes; sigma = sigma_s in buffer px). Deterministic: every row /
// column is computed by the same arithmetic whatever the thread split.
void RF4(float* v, int W, int H, double sigma, int iters, const Edges& e, int threads)
{
    if (!(sigma > 0.05) || W <= 0 || H <= 0)
        return;
    const bool edgy = (e.gh && e.k > 0) || e.bar || e.cut;
    for (int it = 0; it < iters; ++it)
    {
        const double si = sigma * std::sqrt(3.0) * pm::pow(2.0, iters - it - 1) / std::sqrt(pm::pow(4.0, iters) - 1.0);
        const float lnA = (float)(-std::sqrt(2.0) / si);
        const float cConst = pm::exp(lnA);
        const float kk = (float)e.k;
        // horizontal
        PFor(H, threads, [&](int y0, int y1) {
            std::vector<float> cr((size_t)W);
            for (int y = y0; y < y1; ++y)
            {
                float* r = v + (size_t)y * W * 4;
                const size_t row = (size_t)y * W;
                if (edgy)
                {
                    for (int x = 1; x < W; ++x)
                    {
                        float d = 1.f;
                        if (e.gh && kk > 0)
                            d += kk * e.gh[row + x];
                        if (e.bar && (e.bar[row + x] | e.bar[row + x - 1]))
                            d += kBarrierCost;
                        if (e.cut && (e.cut[row + x] & 1))
                            d += kBarrierCost;
                        cr[(size_t)x] = pm::exp(lnA * d);
                    }
                }
                skplat::F4 prev = skplat::F4Load(r);
                for (int x = 1; x < W; ++x)
                {
                    const skplat::F4 c = skplat::F4Set1(edgy ? cr[(size_t)x] : cConst);
                    skplat::F4 cur = skplat::F4Load(r + (size_t)x * 4);
                    cur = skplat::F4Lerp(cur, c, prev);
                    skplat::F4Store(r + (size_t)x * 4, cur);
                    prev = cur;
                }
                prev = skplat::F4Load(r + (size_t)(W - 1) * 4);
                for (int x = W - 2; x >= 0; --x)
                {
                    const skplat::F4 c = skplat::F4Set1(edgy ? cr[(size_t)x + 1] : cConst);
                    skplat::F4 cur = skplat::F4Load(r + (size_t)x * 4);
                    cur = skplat::F4Lerp(cur, c, prev);
                    skplat::F4Store(r + (size_t)x * 4, cur);
                    prev = cur;
                }
            }
        });
        // vertical (row-major sweep over a strip of columns)
        PFor(W, threads, [&](int x0, int x1) {
            const int n = x1 - x0;
            std::vector<float> cr((size_t)n);
            auto coef = [&](int y) {  // edge between rows y-1 and y
                const size_t row = (size_t)y * W;
                for (int i = 0; i < n; ++i)
                {
                    const int x = x0 + i;
                    float d = 1.f;
                    if (e.gv && kk > 0)
                        d += kk * e.gv[row + x];
                    if (e.bar && (e.bar[row + x] | e.bar[row - W + x]))
                        d += kBarrierCost;
                    if (e.cut && (e.cut[row + x] & 2))
                        d += kBarrierCost;
                    cr[(size_t)i] = pm::exp(lnA * d);
                }
            };
            for (int y = 1; y < H; ++y)
            {
                if (edgy)
                    coef(y);
                float* r = v + ((size_t)y * W + x0) * 4;
                const float* p = r - (size_t)W * 4;
                for (int i = 0; i < n; ++i)
                {
                    const skplat::F4 c = skplat::F4Set1(edgy ? cr[(size_t)i] : cConst);
                    skplat::F4 cur = skplat::F4Load(r + (size_t)i * 4);
                    const skplat::F4 pr = skplat::F4Load(p + (size_t)i * 4);
                    cur = skplat::F4Lerp(cur, c, pr);
                    skplat::F4Store(r + (size_t)i * 4, cur);
                }
            }
            for (int y = H - 2; y >= 0; --y)
            {
                if (edgy)
                    coef(y + 1);
                float* r = v + ((size_t)y * W + x0) * 4;
                const float* p = r + (size_t)W * 4;
                for (int i = 0; i < n; ++i)
                {
                    const skplat::F4 c = skplat::F4Set1(edgy ? cr[(size_t)i] : cConst);
                    skplat::F4 cur = skplat::F4Load(r + (size_t)i * 4);
                    const skplat::F4 pr = skplat::F4Load(p + (size_t)i * 4);
                    cur = skplat::F4Lerp(cur, c, pr);
                    skplat::F4Store(r + (size_t)i * 4, cur);
                }
            }
        });
    }
}

// guide differences (L1 over 3 channels, guide clamped to 0..2) for the edge cost
void BuildGrad(const float* g3, int W, int H, std::vector<float>& gh, std::vector<float>& gv, int threads)
{
    const size_t N = (size_t)W * H;
    gh.assign(N, 0.f);
    gv.assign(N, 0.f);
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                const float* c = g3 + i * 3;
                auto cl = [](float v) { return Clampf(v, 0.f, 2.f); };
                if (x > 0)
                {
                    const float* l = c - 3;
                    gh[i] = std::fabs(cl(c[0]) - cl(l[0])) + std::fabs(cl(c[1]) - cl(l[1])) + std::fabs(cl(c[2]) - cl(l[2]));
                }
                if (y > 0)
                {
                    const float* u = c - (size_t)W * 3;
                    gv[i] = std::fabs(cl(c[0]) - cl(u[0])) + std::fabs(cl(c[1]) - cl(u[1])) + std::fabs(cl(c[2]) - cl(u[2]));
                }
            }
    });
}

// normalised smoothing of a 3-channel image with per-pixel weights (nullptr = 1): dst = S(src*w) / S(w);
// where S(w) ~ 0 the source value is kept. dst may equal src.
void SmoothNorm3(const float* src, const float* w, float* dst, int W, int H, double sigma, const Edges& e, int threads,
                 std::vector<float>& tmp, float* denOut = nullptr)
{
    const size_t N = (size_t)W * H;
    tmp.resize(N * 4);
    PFor(H, threads, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * W; i < (size_t)y1 * W; ++i)
        {
            const float ww = w ? w[i] : 1.f;
            tmp[i * 4 + 0] = src[i * 3 + 0] * ww;
            tmp[i * 4 + 1] = src[i * 3 + 1] * ww;
            tmp[i * 4 + 2] = src[i * 3 + 2] * ww;
            tmp[i * 4 + 3] = ww;
        }
    });
    RF4(tmp.data(), W, H, sigma, 3, e, threads);
    PFor(H, threads, [&](int y0, int y1) {
        for (size_t i = (size_t)y0 * W; i < (size_t)y1 * W; ++i)
        {
            const float d = tmp[i * 4 + 3];
            if (denOut)
                denOut[i] = d;
            if (d > 1e-5f)
            {
                const float inv = 1.f / d;
                dst[i * 3 + 0] = tmp[i * 4 + 0] * inv;
                dst[i * 3 + 1] = tmp[i * 4 + 1] * inv;
                dst[i * 3 + 2] = tmp[i * 4 + 2] * inv;
            }
            else if (dst != src)
            {
                dst[i * 3 + 0] = src[i * 3 + 0];
                dst[i * 3 + 1] = src[i * 3 + 1];
                dst[i * 3 + 2] = src[i * 3 + 2];
            }
        }
    });
}

// square min / max filter (separable, radius r), single channel
void MinMax(const float* src, float* dst, int W, int H, int r, bool isMax, int threads, std::vector<float>& tmp)
{
    const size_t N = (size_t)W * H;
    tmp.resize(N);
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
        {
            const float* s = src + (size_t)y * W;
            float* d = tmp.data() + (size_t)y * W;
            for (int x = 0; x < W; ++x)
            {
                float m = s[x];
                const int a = std::max(0, x - r), b = std::min(W - 1, x + r);
                for (int k = a; k <= b; ++k)
                    m = isMax ? std::max(m, s[k]) : std::min(m, s[k]);
                d[x] = m;
            }
        }
    });
    // vertical: row-major (each output row = element-wise min / max of 2r+1 input rows): cache friendly
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
        {
            float* d = dst + (size_t)y * W;
            std::memcpy(d, tmp.data() + (size_t)y * W, sizeof(float) * (size_t)W);
            const int a = std::max(0, y - r), b = std::min(H - 1, y + r);
            for (int k = a; k <= b; ++k)
            {
                if (k == y)
                    continue;
                const float* s = tmp.data() + (size_t)k * W;
                if (isMax)
                    for (int x = 0; x < W; ++x)
                        d[x] = std::max(d[x], s[x]);
                else
                    for (int x = 0; x < W; ++x)
                        d[x] = std::min(d[x], s[x]);
            }
        }
    });
}

struct Timer
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void lap(const char* what)
    {
        if (!g_profile)
            return;
        const auto t = std::chrono::steady_clock::now();
        size_t ws = 0, pk = 0;
        skplat::ProcessMemoryMB(ws, pk);
        std::fprintf(stderr, "  [sk] %-18s %7.1f ms   ws %zu MB  peak %zu MB  pfor %ld\n", what, std::chrono::duration<double, std::milli>(t - t0).count(), ws,
                     pk, g_pforCount.exchange(0));
        t0 = t;
    }
};

// ------------------------------------------------------------- round 2 helpers ---
// An uninitialised float buffer (no single-threaded zero fill of hundreds of MB): every element must be
// written before it is read; the writers below are parallel (first touch spread over the threads).
struct Buf
{
    std::unique_ptr<float[]> p;
    size_t n = 0;
    void alloc(size_t k)
    {
        p.reset(new float[k]);
        n = k;
    }
    void release()
    {
        p.reset();
        n = 0;
    }
    float* data() { return p.get(); }
    const float* data() const { return p.get(); }
    float& operator[](size_t i) { return p[i]; }
    const float& operator[](size_t i) const { return p[i]; }
    bool empty() const { return n == 0; }
};

// Recursive filter on an NC-channel interleaved image without edges (plain Gaussian-like smoothing), scalar.
template <int NC>
void RFPlain(float* v, int W, int H, double sigma, int iters, int threads)
{
    if (!(sigma > 0.05) || W <= 0 || H <= 0)
        return;
    for (int it = 0; it < iters; ++it)
    {
        const double si = sigma * std::sqrt(3.0) * pm::pow(2.0, iters - it - 1) / std::sqrt(pm::pow(4.0, iters) - 1.0);
        const float c = (float)pm::exp(-std::sqrt(2.0) / si);
        PFor(H, threads, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
            {
                float* r = v + (size_t)y * W * NC;
                for (int x = 1; x < W; ++x)
                    for (int k = 0; k < NC; ++k)
                        r[x * NC + k] += c * (r[(x - 1) * NC + k] - r[x * NC + k]);
                for (int x = W - 2; x >= 0; --x)
                    for (int k = 0; k < NC; ++k)
                        r[x * NC + k] += c * (r[(x + 1) * NC + k] - r[x * NC + k]);
            }
        });
        PFor(W, threads, [&](int x0, int x1) {
            for (int y = 1; y < H; ++y)
            {
                float* r = v + ((size_t)y * W) * NC;
                const float* p = r - (size_t)W * NC;
                for (int i = x0 * NC; i < x1 * NC; ++i)
                    r[i] += c * (p[i] - r[i]);
            }
            for (int y = H - 2; y >= 0; --y)
            {
                float* r = v + ((size_t)y * W) * NC;
                const float* p = r + (size_t)W * NC;
                for (int i = x0 * NC; i < x1 * NC; ++i)
                    r[i] += c * (p[i] - r[i]);
            }
        });
    }
}

// parallel loop over all N pixels (rows split over threads)
template <class F>
void ForPx(int W, int H, int threads, const F& f)  // f(i, x, y)
{
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                f((size_t)y * W + x, x, y);
    });
}

// Big frames (> kHalfResPx): the low-frequency watercolour smoothing runs at half resolution. 2x2 weighted box
// down (barriers max-pooled: a block with a line pixel is a wall), smoothing at sigma / 2 (edge cost from the half
// image), bilinear up of the values and the weights. Deterministic (fixed 2x2 grid of the buffer).
const size_t kHalfResPx = 4000000;
inline float Bilerp(const float* a, int w2, int h2, int nc, int c, float fx, float fy)
{
    fx = Clampf(fx, 0.f, (float)(w2 - 1));
    fy = Clampf(fy, 0.f, (float)(h2 - 1));
    const int x0 = std::min(w2 - 1, (int)fx), y0 = std::min(h2 - 1, (int)fy);
    const int x1 = std::min(w2 - 1, x0 + 1), y1 = std::min(h2 - 1, y0 + 1);
    const float tx = fx - x0, ty = fy - y0;
    const float p = a[((size_t)y0 * w2 + x0) * nc + c] + (a[((size_t)y0 * w2 + x1) * nc + c] - a[((size_t)y0 * w2 + x0) * nc + c]) * tx;
    const float q = a[((size_t)y1 * w2 + x0) * nc + c] + (a[((size_t)y1 * w2 + x1) * nc + c] - a[((size_t)y1 * w2 + x0) * nc + c]) * tx;
    return p + (q - p) * ty;
}
// sr: sigma_r of the edge cost (0 = no edges)
inline void SmoothNorm3Half(const float* src, const float* w, float* dst, int W, int H, double sigma, double sr, const uint8_t* bar,
                            int threads, std::vector<float>& tmp, float* denOut)
{
    const int w2 = (W + 1) / 2, h2 = (H + 1) / 2;
    const size_t N2 = (size_t)w2 * h2;
    std::vector<float> s2(N2 * 3), ww(N2);
    std::vector<uint8_t> b2(bar ? N2 : 0);
    PFor(h2, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w2; ++x)
            {
                float acc[3] = {0, 0, 0}, plain[3] = {0, 0, 0}, wsum = 0.f;
                int n = 0;
                uint8_t bb = 0;
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i)
                    {
                        const int xx = std::min(W - 1, 2 * x + i), yy = std::min(H - 1, 2 * y + j);
                        const size_t k = (size_t)yy * W + xx;
                        const float wk = w ? w[k] : 1.f;
                        for (int c = 0; c < 3; ++c)
                        {
                            acc[c] += wk * src[k * 3 + c];
                            plain[c] += src[k * 3 + c];
                        }
                        wsum += wk;
                        ++n;
                        if (bar)
                            bb |= bar[k];
                    }
                const size_t o = (size_t)y * w2 + x;
                for (int c = 0; c < 3; ++c)
                    s2[o * 3 + c] = wsum > 1e-6f ? acc[c] / wsum : plain[c] / n;
                ww[o] = wsum / n;
                if (bar)
                    b2[o] = bb;
            }
    });
    Edges e;
    std::vector<float> gh, gv;
    if (sr > 0)
    {
        BuildGrad(s2.data(), w2, h2, gh, gv, threads);
        e.gh = gh.data();
        e.gv = gv.data();
        e.k = (sigma * 0.5) / sr;
    }
    if (bar)
        e.bar = b2.data();
    std::vector<float> r2(N2 * 3), d2(N2);
    SmoothNorm3(s2.data(), ww.data(), r2.data(), w2, h2, sigma * 0.5, e, threads, tmp, d2.data());
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                const float fx = (x + 0.5f) * 0.5f - 0.5f, fy = (y + 0.5f) * 0.5f - 0.5f;
                for (int c = 0; c < 3; ++c)
                    dst[i * 3 + c] = Bilerp(r2.data(), w2, h2, 3, c, fx, fy);
                if (denOut)
                    denOut[i] = Bilerp(d2.data(), w2, h2, 1, 0, fx, fy);
            }
    });
}
// plain 3-channel smoothing at half resolution (in place)
inline void RFPlain3Half(float* v, int W, int H, double sigma, int threads)
{
    const int w2 = (W + 1) / 2, h2 = (H + 1) / 2;
    std::vector<float> s2((size_t)w2 * h2 * 3);
    PFor(h2, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w2; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    float s = 0.f;
                    for (int j = 0; j < 2; ++j)
                        for (int i = 0; i < 2; ++i)
                            s += v[((size_t)std::min(H - 1, 2 * y + j) * W + std::min(W - 1, 2 * x + i)) * 3 + c];
                    s2[((size_t)y * w2 + x) * 3 + c] = 0.25f * s;
                }
    });
    RFPlain<3>(s2.data(), w2, h2, sigma * 0.5, 3, threads);
    PFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c)
                    v[((size_t)y * W + x) * 3 + c] = Bilerp(s2.data(), w2, h2, 3, c, (x + 0.5f) * 0.5f - 0.5f, (y + 0.5f) * 0.5f - 0.5f);
    });
}

// cold-press paper height 0..1 at full-res layer coordinates (bumps + a faint cell network + undulation)
inline float PaperHeight(double gx, double gy, double ps, uint32_t seed, double ds)
{
    const float bump = Fbm(gx, gy, 2.2 * ps, 3, 0.55, seed ^ 0x9A9Eu, ds);
    const float cell = 1.f - 2.f * std::fabs(Fbm(gx, gy, 5.0 * ps, 2, 0.5, seed ^ 0x9A9Fu, ds));
    const float und = Fbm(gx, gy, 14.0 * ps, 2, 0.5, seed ^ 0x9AA0u, ds);
    return Clampf(0.5f + 0.40f * bump + 0.06f * cell + 0.08f * und, 0.f, 1.f);
}

}  // namespace
}  // namespace sk
