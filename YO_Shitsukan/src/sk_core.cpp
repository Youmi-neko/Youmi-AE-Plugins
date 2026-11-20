// Shitsukan render engine (round 2) - see sk_core.h and docs\watercolor_model.md for the design.
#include "sk_util.h"
#ifdef SK_GPU
#include <future>
#include <string>
#include "skgpu.h"
#include "sk_styles.h"  // (the self-test's cases)
#endif

#include <array>
#include <functional>
#ifdef SK_DEV_DUMP
#include <cstdlib>
#include <string>
#endif

namespace sk {
int g_profile = 0;

namespace {

const float kInkEps = 0.03f;
const float kPaper[3] = {0.985f, 0.975f, 0.95f};  // the painter's paper (a warm white)  // ink model: out = (fill + eps) * exp(-delta) - eps
const float kInvGamma = 1.f / 2.2f;

inline float ToGamma(float v) { return v > 0.f ? pm::pow(v, kInvGamma) : 0.f; }
inline float ToLinear(float v) { return v > 0.f ? pm::pow(v, 2.2f) : 0.f; }

// black / white top-hat coverage of one signal (closing / opening by a square of radius r)
// dark: (closing - s) / (closing - erosion); light: (s - opening) / (dilation - opening); gated by the contrast
void TopHat(const float* s, int W, int H, int r, bool dark, float thr, int T, std::vector<float>& tmp, std::vector<float>& outA,
            std::vector<float>* core)
{
    const size_t N = (size_t)W * H;
    std::vector<float> a1(N), a2(N);
    MinMax(s, a1.data(), W, H, r, dark, T, tmp);          // dark: dilation (max) ; light: erosion (min)
    MinMax(a1.data(), a2.data(), W, H, r, !dark, T, tmp);  // closing / opening
    MinMax(s, a1.data(), W, H, r, !dark, T, tmp);          // dark: erosion (core) ; light: dilation
    outA.resize(N);
    if (core)
        core->assign(a1.begin(), a1.end());
    ForPx(W, H, T, [&](size_t i, int, int) {
        const float th = dark ? a2[i] - s[i] : s[i] - a2[i];
        const float K = dark ? a2[i] - a1[i] : a1[i] - a2[i];
        const float gate = Smooth(0.5f * thr, thr, K);
        outA[i] = (K > 1e-4f && th > 0.f) ? Clampf(th / K, 0.f, 1.f) * gate : 0.f;
    });
}

#ifdef SK_GPU
// ---- 0.6.0.4: the GPU path (gpu\src\lib.rs). Bit-identical results by construction (gpu\shaders\lines.wgsl); any
// failure (no GPU, busy, device lost, out of memory) = the CPU code for that frame. SHITSUKAN_GPU=0 forces the CPU.
// 0.6.0.6: the context is made on a background thread by the first frame that could use it, and a start-up self-test
// (GpuSelfTest, after RenderImpl) runs every ported stage on small built-in frames against the CPU, bit for bit; a
// stage that differs is switched off for the session. Frames before it has finished use the CPU (the same picture).
// SHITSUKAN_GPU_LOG=<path>: one line per frame (size, the stages on the GPU, the path, why not, wait / render ms).
// Test hooks: SKGPU_SELFTEST=sync (the first frame waits for the self-test) | off (no self-test: every stage on).
enum : unsigned { GU_LINES = 1, GU_WET = 2, GU_REST = 4, GU_INK = 8 };
SkGpuCtx* g_gpu = nullptr;
std::atomic<int> g_gpuState{0};       // 0 not started, 1 context + self-test running, 2 ready, 3 no usable GPU
std::atomic<unsigned> g_gpuUnits{0};  // the stages the self-test passed
std::atomic<bool> g_gpuTail{false};   // the GPU can run the painter's tail at all (f64, the exact exp)
std::atomic<bool> g_gpuOff{false};    // after a lost device: CPU for the rest of the session
bool EnvIs(const char* name, const char* val)
{
    char b[32] = {0};
    const unsigned n = skplat::GetEnv(name, b, sizeof(b));
    return n > 0 && n < sizeof(b) && !strcmp(b, val);
}
// mac_port: on macOS the GPU path is OFF unless SHITSUKAN_GPU=1 (opt-in until the Metal self-test has been seen
// passing on real Macs: PORTING_NOTES.md "GPU"); SHITSUKAN_GPU=0 still forces the CPU everywhere. Windows: unchanged.
bool GpuEnvOff()
{
    char b[8] = {0};
    const unsigned n = skplat::GetEnv("SHITSUKAN_GPU", b, sizeof(b));
#if defined(__APPLE__) && !SK_MAC_GPU_DEFAULT_ON
    return !(n > 0 && n < sizeof(b) && b[0] == '1');
#else
    return n > 0 && n < sizeof(b) && b[0] == '0';
#endif
}
double NowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::string UnitStr(unsigned u)
{
    std::string s;
    const char* nm[4] = {"lines", "wet", "rest", "ink"};
    for (int k = 0; k < 4; ++k)
        if (u & (1u << k))
            s += (s.empty() ? "" : ",") + std::string(nm[k]);
    return s.empty() ? "-" : s;
}
const char* StatusName(int st)
{
    return st == SKGPU_BUSY ? "busy" : st == SKGPU_LOST ? "lost" : st == SKGPU_OOM ? "oom" : st == SKGPU_OK ? "ok" : "error";
}

// ---- the diagnostic log (off unless SHITSUKAN_GPU_LOG is set: then one line per frame, appended)
const skplat::LogPathT* GpuLogPath()
{
    static const skplat::LogPathT p = skplat::LogPathFromEnv("SHITSUKAN_GPU_LOG");
    return p.empty() ? nullptr : &p;
}
void GpuLogLine(const std::string& s)
{
    const skplat::LogPathT* path = GpuLogPath();
    if (!path)
        return;
    static std::mutex m;
    char head[96];
    skplat::LogHead(head, sizeof(head));
    std::lock_guard<std::mutex> lk(m);
    FILE* fp = skplat::OpenAppend(*path);
    if (fp)
    {
        fputs(head, fp);
        fputs(s.c_str(), fp);
        fputs("\n", fp);
        fclose(fp);
    }
}
struct GpuLogFrame
{
    unsigned elig = 0, used = 0;  // the stages this frame could run on the GPU / did
    std::string why;              // the reasons for the stages left on the CPU, '+' separated
    double waitMs = 0.0;
};
thread_local GpuLogFrame* t_log = nullptr;
void LogWhy(const std::string& tok)
{
    if (!t_log)
        return;
    std::string& w = t_log->why;
    if (("+" + w + "+").find("+" + tok + "+") == std::string::npos)
        w += (w.empty() ? "" : "+") + tok;
}
void LogUse(unsigned u)
{
    if (t_log)
        t_log->used |= u;
}
void LogElig(unsigned u)
{
    if (t_log)
        t_log->elig |= u;
}

// ---- the start-up self-test's bookkeeping: the GPU result of a stage is kept until the CPU reaches the same point
enum { ST_LINES = 0, ST_WET, ST_DENSITY, ST_SLIVERS, ST_PA, ST_PAPER, ST_INK, ST_COUNT };
const char* const kStName[ST_COUNT] = {"lines", "wet", "density", "slivers", "pablur", "paper", "ink"};
struct SelfTest
{
    SkGpuCtx* ctx = nullptr;
    int runs[ST_COUNT] = {}, bad[ST_COUNT] = {};
    std::string note[ST_COUNT];
    std::vector<float> g[ST_COUNT];
    bool have[ST_COUNT] = {};
    int beginFail = 0;
    void put(int s, int st, std::vector<float>&& v)
    {
        have[s] = false;
        if (st != SKGPU_OK)
        {
            ++runs[s];
            ++bad[s];
            note[s] = std::string("gpu status ") + StatusName(st);
            return;
        }
        g[s] = std::move(v);
        have[s] = true;
    }
    void cmp(int s, const float* cpu, size_t n)
    {
        if (!have[s])
            return;
        have[s] = false;
        ++runs[s];
        size_t d = 0, first = 0;
        const uint32_t* a = reinterpret_cast<const uint32_t*>(cpu);
        const uint32_t* b = reinterpret_cast<const uint32_t*>(g[s].data());
        if (g[s].size() != n)
            d = n + 1;
        else
            for (size_t i = 0; i < n; ++i)
                if (a[i] != b[i] && d++ == 0)
                    first = i;
        if (d)
        {
            ++bad[s];
            char t[160];
            if (g[s].size() != n)
                snprintf(t, sizeof(t), "size %zu vs %zu", g[s].size(), n);
            else
                snprintf(t, sizeof(t), "%zu of %zu values differ, first [%zu] cpu %08x gpu %08x", d, n, first, a[first], b[first]);
            note[s] = t;
        }
        g[s].clear();
        g[s].shrink_to_fit();
    }
};
thread_local SelfTest* t_st = nullptr;
inline void StCmp(int s, const float* p, size_t n)
{
    if (t_st)
        t_st->cmp(s, p, n);
}
void GpuInit();
// the context when the self-test has finished (starts it on the first call); null while it runs / without a GPU
SkGpuCtx* GpuReady()
{
    int s = g_gpuState.load(std::memory_order_acquire);
    if (s == 0)
    {
        int z = 0;
        if (g_gpuState.compare_exchange_strong(z, 1))
        {
            try
            {
                std::thread(GpuInit).detach();
            }
            catch (...)
            {
                g_gpuState.store(3, std::memory_order_release);
            }
        }
        s = g_gpuState.load(std::memory_order_acquire);
    }
    if (s == 1 && EnvIs("SKGPU_SELFTEST", "sync"))
        while ((s = g_gpuState.load(std::memory_order_acquire)) == 1)
            skplat::SleepMs(1);
    if (s == 2 && g_gpu)
        return g_gpu;
    LogWhy(s == 1 ? "pending" : "nodev");
    return nullptr;
}
// the recursive filters' coefficients, computed exactly as RFPlain / RF4 compute them
float RfPlainCoef(double sigma, int iters, int it)
{
    const double si = sigma * std::sqrt(3.0) * pm::pow(2.0, iters - it - 1) / std::sqrt(pm::pow(4.0, iters) - 1.0);
    return (float)pm::exp(-std::sqrt(2.0) / si);
}
float Rf4Coef(double sigma, int iters, int it)
{
    const double si = sigma * std::sqrt(3.0) * pm::pow(2.0, iters - it - 1) / std::sqrt(pm::pow(4.0, iters) - 1.0);
    const float lnA = (float)(-std::sqrt(2.0) / si);
    return pm::exp(lnA);
}
const size_t kGpuMinPx = 200000;  // smaller frames: the CPU is as fast
const uint32_t kGpuWaitMs = 300;  // MFR: a third concurrent frame waits this long for a GPU slot, then the CPU
// one GPU frame per render (gpu\src\lib.rs Frame: a pooled slot; C / A / alpha stay resident between the stages).
// Begun lazily by the first stage that wants it; any failure ends it and the rest of the frame is CPU.
struct GpuFrame
{
    SkGpuFrame* f = nullptr;
    bool tried = false;
    unsigned units = 0;  // the stages that may run on the GPU in this frame (the self-test's verdict)
    // want: the stages of the caller (GU_*); false = the caller runs them on the CPU (the reason goes to the log)
    bool begin(int W, int H, unsigned want)
    {
        if (f)
        {
            if (units & want)
                return true;
            LogOff(want);
            return false;
        }
        if (tried)
            return false;
        if (t_st)
        {
            // (the self-test: every stage, any size)
            tried = true;
            int32_t st = 0;
            f = skgpu_frame_begin(t_st->ctx, W, H, kGpuWaitMs, &st);
            if (!f)
                t_st->beginFail = st ? st : SKGPU_ERROR;
            units = GU_LINES | (skgpu_has_tail(t_st->ctx) ? GU_WET | GU_REST | GU_INK : 0u);
            return f != nullptr;
        }
        if (GpuEnvOff())
        {
            tried = true;
            LogWhy("env");
            return false;
        }
        if (g_gpuOff.load())
        {
            tried = true;
            LogWhy("lost");
            return false;
        }
        if ((size_t)W * H < kGpuMinPx)
        {
            tried = true;
            LogWhy("small");
            return false;
        }
        SkGpuCtx* g = GpuReady();
        if (!g)
        {
            tried = true;
            return false;
        }
        units = g_gpuUnits.load();
        if (!(units & want))
        {
            LogOff(want);  // (not "tried": a later stage may still want the GPU)
            return false;
        }
        tried = true;
        int32_t st = 0;
        const double t0 = t_log ? NowMs() : 0.0;
        f = skgpu_frame_begin(g, W, H, kGpuWaitMs, &st);
        if (t_log)
            t_log->waitMs += NowMs() - t0;
        if (!f)
        {
            LogWhy(StatusName(st));
            if (st == SKGPU_LOST || skgpu_lost(g))
                g_gpuOff = true;
        }
        return f != nullptr;
    }
    static void LogOff(unsigned want)
    {
        if (!t_log)
            return;
        const unsigned off = want & ~g_gpuUnits.load();
        if ((off & (GU_WET | GU_REST | GU_INK)) && !g_gpuTail.load())
            LogWhy("notail");
        else
            LogWhy("selftest-off:" + UnitStr(off));
    }
    void fail(int st)
    {
        if (st == SKGPU_LOST || (g_gpu && skgpu_lost(g_gpu)))
            g_gpuOff = true;
        if (f)
            skgpu_frame_end(f, 0);
        f = nullptr;
    }
    // done with the GPU for this frame: the slot goes back to the pool now (MFR: the next frame may take it)
    void release()
    {
        if (f)
            skgpu_frame_end(f, 1);
        f = nullptr;
    }
    ~GpuFrame() { release(); }
};
// 0.6.0.5: sk_core.cpp 5. ink (without a thickness change) after the painter's paper on the GPU
struct GpuInkIn
{
    const float* delta = nullptr;  // N*3
    const float* far_w = nullptr;  // N, or null: no depth line scaling
    float dk = 1.f, wc = 0.f, kLi = 0.f;
    bool inpainted = false;
    int style = 0;  // 0 as is, 1 pencil, 2 ink
    double soft = 0.0;
    int bkMode = 0;  // the line noise: 0 none, 1 a CPU grid (NoiseField step > 1), 2 per pixel on the GPU
    double bkLambda = 16.0;
    std::vector<float> bkGrid;
    int bkMeta[7] = {0, 0, 0, 0, 1, 0, 0};  // gx0 gy0 gw gh step ox oy
    bool* done = nullptr;
};
#endif

#include "sk_paint.inc"



Status RenderImpl(const Params& pin, Frame& f)
{
    const int W = f.w, H = f.h;
    const size_t N = (size_t)W * H;
    float* out = f.out;
    if (N == 0 || !f.rgba || !out)
        return ST_OK;
    const double ds = (f.ds > 0 && std::isfinite(f.ds)) ? f.ds : 1.0;
    const double par = (pin.par > 0.05 && pin.par < 20) ? pin.par : 1.0;
    const int T = std::max(1, f.threads);
    auto poll = [&] {
        if (f.abort && f.abort())
            throw AbortEx();
    };
    Timer tm;
#ifdef SK_GPU
    GpuFrame gpuFr;
#endif
    // full-res layer coordinates of buffer pixel x / y (x stretched by the pixel aspect: round on screen)
    auto Xf = [&](double x) { return (x + f.ox) / ds * par; };
    auto Yf = [&](double y) { return (y + f.oy) / ds; };

    // ---- derived settings (buffer px)
    const float amt = (float)Clampd(pin.amount / 100.0, 0, 1);
    float gains[3] = {(float)Clampd(pin.fine / 100.0, -1, 2), (float)Clampd(pin.mid / 100.0, -1, 2), (float)Clampd(pin.large / 100.0, -1, 2)};
    const bool texOn = gains[0] != 0.f || gains[1] != 0.f || gains[2] != 0.f;
    // depth matte: the map is depth, not a strength mask (no map = off)
    const bool depthOn = pin.depth && f.map != nullptr;
    const float dk = depthOn ? (float)Clampd(pin.dMaster / 100.0, 0, 1) : 0.f;
    auto dInf = [&](double v) { return (float)Clampd(v / 100.0, -1, 1) * dk; };
    const float kAm = dInf(pin.dAmount), kSc = dInf(pin.dScale), kWe = dInf(pin.dWet), kLi = dInf(pin.dLines), kPa = dInf(pin.dPale),
                kTx = dInf(pin.dTexture), kBl = dInf(pin.dBlur);
    const double fillBlurIn = Clampd(pin.fillBlur, 0, 500) * ds;
    const bool baseBlur = fillBlurIn > 0.05;
    // with no fill blur set, a positive depth blur blurs the far parts at 16 px
    const double fillBlur = baseBlur ? fillBlurIn : 16.0 * ds;
    const bool blurOn = baseBlur || kBl > 0.f;
    const float wc = (float)Clampd(pin.wc / 100.0, 0, 1);
    const bool wcOn = wc > 0.f;
    const bool lineOps = pin.lineThick != 0 || pin.lineDark != 0 || pin.lineSoft > 0 || pin.lineStyle != LS_ASIS;
    const int view = (pin.view >= 0 && pin.view < V_COUNT) ? pin.view : V_RESULT;
    const bool wcView = view == V_RIM || view == V_STROKES || view == V_GLAZE;
    const bool anyProc = texOn || blurOn || wcOn;
    const bool protect = pin.protectLines;
    const bool needLines0 = (protect && anyProc) || lineOps || view == V_LINES || view == V_FILL || pin.blackWalls;
    const bool depthLines = kLi != 0.f;  // depth: line density scaled by (1 + k * far) - the lines are needed, not inpainted
    const bool needLines = needLines0 || depthLines;
    const bool inpainted = protect && needLines0;
    const uint32_t seed = Hash((uint32_t)pin.seed * 0x2545F491u + 0x9E37u) ^ Hash((uint32_t)pin.paperFrame + 0x7F4A7C15u);
    const float* M = depthOn ? nullptr : f.map;  // (as a strength mask only when not used as depth)
    auto mapOf = [&](int target, size_t i) -> float { return (M && pin.mapTarget == target) ? Clampf(M[i], 0.f, 1.f) : 1.f; };
    // far: 0 at the reference depth, 1 at "range" from it (a continuous function of the map: no thresholds)
    std::vector<float> dmap, depthFar;
    if (depthOn)
    {
        dmap.resize(N);
        ForPx(W, H, T, [&](size_t i, int, int) { dmap[i] = std::isfinite(f.map[i]) ? Clampf(f.map[i], 0.f, 1.f) : 0.f; });
        const double mb = Clampd(pin.mapBlur, 0, 200) * ds;
        if (mb > 0.05)
            RFPlain<1>(dmap.data(), W, H, mb, 3, T);
        float ref = (float)Clampd(pin.depthRef / 100.0, 0, 1);
        if (pin.depthPick)
        {
            const float px = Clampf((float)(pin.pickX * ds - f.ox), 0.f, (float)(W - 1)), py = Clampf((float)(pin.pickY * ds - f.oy), 0.f, (float)(H - 1));
            ref = Bilerp(dmap.data(), W, H, 1, 0, px, py);
        }
        const float range = (float)Clampd(pin.depthRange / 100.0, 0.01, 1);
        depthFar.resize(N);
        ForPx(W, H, T, [&](size_t i, int, int) { depthFar[i] = Smooth(0.f, 1.f, std::fabs(dmap[i] - ref) / range); });
    }
    auto farAt = [&](size_t i) -> float { return depthFar.empty() ? 0.f : depthFar[i]; };
    // the painter's paper is a little coarser (calmer when the picture moves under the layer-fixed paper)
    const double ps = 3.2 * Clampd(pin.paperScale / 100.0, 0.25, 8) * (pin.wcMode == WCM_PAINT && pin.wc > 0 ? 1.5 : 1.0);
    auto paperAt = [&](size_t i, int x, int y) -> float {
        return f.paperTex ? Clampf(f.paperTex[i], 0.f, 1.f) : PaperHeight(Xf(x), Yf(y), ps, seed, ds);
    };

    // ---- 0. inputs (linear working space: processed in gamma 2.2)
    std::vector<float> C(N * 3), A(N);
    ForPx(W, H, T, [&](size_t i, int, int) {
        const float* s = f.rgba + i * 4;
        for (int c = 0; c < 3; ++c)
        {
            float v = std::isfinite(s[c]) ? Clampf(s[c], 0.f, 64.f) : 0.f;
            C[i * 3 + c] = pin.linear ? ToGamma(v) : v;
        }
        A[i] = std::isfinite(s[3]) ? Clampf(s[3], 0.f, 1.f) : 0.f;
    });
    std::vector<float> tmp4, tmp1;
    // 0.6.1 (memory): the scratch planes are given back between the stages (the next user allocates them again)
    auto freeScratch = [&] {
        tmp4.clear();
        tmp4.shrink_to_fit();
        tmp1.clear();
        tmp1.shrink_to_fit();
    };

    // ---- 1. line analysis -> alpha (coverage), bar (barrier), delta (ink density), Fl (fill layer, in place)
    std::vector<float> alpha, delta, Fl;
    std::vector<uint8_t> bar, light;
    int rc = 1;
    if (needLines)
    {
        rc = std::max(1, std::min(16, (int)std::ceil(Clampd(pin.lineWidth, 0.5, 32) * ds * 0.5 - 1e-6)));
        const float sens = (float)Clampd(pin.lineSens / 100.0, 0, 1);
        const float thr = 0.30f * pm::pow(0.03f / 0.30f, sens);
#ifdef SK_DEV_DUMP
        // (dev builds only: the GPU probe's reference - the line stage's inputs, constants and outputs)
        auto devDump = [&](const char* name, const void* d, size_t bytes) {
            const char* dir = std::getenv("SK_DUMP");
            if (!dir)
                return;
            FILE* fp = nullptr;
            if ((fp = skplat::OpenFile(std::string(dir) + skplat::PathSep() + name, "wb")) != nullptr)
            {
                fwrite(d, 1, bytes, fp);
                fclose(fp);
            }
        };
        devDump("in_C.raw", C.data(), N * 12);
        devDump("in_A.raw", A.data(), N * 4);
        {
            auto rf4c = [](double sigma, int iters, int it) {  // as RF4: exp of the float exponent
                const double si = sigma * std::sqrt(3.0) * pm::pow(2.0, iters - it - 1) / std::sqrt(pm::pow(4.0, iters) - 1.0);
                const float lnA = (float)(-std::sqrt(2.0) / si);
                return pm::exp(lnA);
            };
            auto rfc = [](double sigma, int iters, int it) {
                const double si = sigma * std::sqrt(3.0) * pm::pow(2.0, iters - it - 1) / std::sqrt(pm::pow(4.0, iters) - 1.0);
                return (float)pm::exp(-std::sqrt(2.0) / si);
            };
            const int rb = std::max(1, (int)std::lround(2.0 * ds));
            const float k[16] = {(float)W, (float)H, (float)rc, thr, (float)pin.lineColor, pin.bgDetailLines ? 1.f : 0.f, pin.blackWalls ? 1.f : 0.f,
                                 inpainted ? 1.f : 0.f, (float)rb, rfc(7.0 * ds, 3, 0), rfc(7.0 * ds, 3, 1), rfc(7.0 * ds, 3, 2),
                                 rf4c(rc + 1.0, 3, 0), rf4c(rc + 1.0, 3, 1), rf4c(rc + 1.0, 3, 2), 0.f};
            devDump("consts.raw", k, sizeof(k));
        }
#endif
        bool gpuLines = false;
        Buf FhG;  // (the GPU writes every value: no zero fill of N*3 floats)
#ifdef SK_GPU
        LogElig(GU_LINES);
        std::vector<float> stLines;  // (the self-test: the GPU's alpha, light, Fh - compared with the CPU's below)
        if (gpuFr.begin(W, H, GU_LINES))
            {
                alpha.resize(N);  // (every value is written by the read-back)
                if (pin.lineColor >= LC_ALL)
                    light.resize(N);
                FhG.alloc(N * 3);
                SkLinesArgs ga{};
                ga.w = W; ga.h = H; ga.rc = rc; ga.line_color = pin.lineColor; ga.bg_detail = pin.bgDetailLines ? 1 : 0;
                ga.rb = std::max(1, (int)std::lround(2.0 * ds));
                ga.thr = thr; ga.thr_a = 0.5f / 255.f;
                for (int it = 0; it < 3; ++it)
                {
                    ga.crf1[it] = RfPlainCoef(7.0 * ds, 3, it);
                    ga.crf4[it] = Rf4Coef(rc + 1.0, 3, it);
                }
                ga.c = C.data(); ga.a = A.data(); ga.out_alpha = alpha.data(); ga.out_light = light.empty() ? nullptr : light.data();
                ga.out_fh = FhG.data();
                const int st = skgpu_frame_lines(gpuFr.f, &ga);
                if (t_st)
                {
                    // the self-test: keep the GPU's result, let the CPU compute its own, compare after the in-paint;
                    // the painter starts a fresh GPU frame (nothing of this one stays resident)
                    if (st == SKGPU_OK)
                    {
                        stLines.assign(alpha.begin(), alpha.end());
                        if (!light.empty())
                            for (size_t i = 0; i < N; ++i) stLines.push_back(light[i] ? 1.f : 0.f);
                        stLines.insert(stLines.end(), FhG.data(), FhG.data() + N * 3);
                    }
                    t_st->put(ST_LINES, st, std::move(stLines));
                    gpuFr.release();
                    gpuFr.tried = false;
                    alpha.clear(); light.clear(); FhG.release();
                }
                else if (st == SKGPU_OK)
                {
                    gpuLines = true;
                    LogUse(GU_LINES);
                }
                else
                {
                    LogWhy(std::string("lines:") + StatusName(st));
                    gpuFr.fail(st);
                    alpha.clear(); light.clear(); FhG.release();
                }
                tm.lap("lines gpu");
            }
#endif
        std::vector<float> Y, core, aS;
        if (!gpuLines)
        {
        Y.resize(N);
        ForPx(W, H, T, [&](size_t i, int, int) { Y[i] = A[i] < (0.5f / 255.f) ? 1.f : Clampf(Luma(&C[i * 3]), 0.f, 1.f); });
        TopHat(Y.data(), W, H, rc, true, thr, T, tmp1, alpha, &core);
        // luma lines: dark cores are more line-like
        ForPx(W, H, T, [&](size_t i, int, int) { alpha[i] *= 1.f - 0.45f * Smooth(0.45f, 0.85f, core[i]); });
        if (pin.lineColor >= LC_COLOR)
        {
            // colour cue: a line darker than both sides in ANY channel (brown / red / blue outlines on light fills)
            // (one signal: the smallest channel - a red / brown line on skin drops in green and blue)
            std::vector<float> ch(N);
            ForPx(W, H, T, [&](size_t i, int, int) {
                const float* c = &C[i * 3];
                ch[i] = A[i] < (0.5f / 255.f) ? 1.f : Clampf(std::min(c[0], std::min(c[1], c[2])), 0.f, 1.f);
            });
            TopHat(ch.data(), W, H, rc, true, thr * 1.3f, T, tmp1, aS, nullptr);
            ForPx(W, H, T, [&](size_t i, int, int) { alpha[i] = std::max(alpha[i], 0.85f * aS[i]); });        }
        if (pin.lineColor >= LC_ALL)
        {
            // light lines on dark (white top-hat on luma): drawn back as negative ink
            TopHat(Y.data(), W, H, rc, false, thr * 1.2f, T, tmp1, aS, nullptr);
            light.assign(N, 0);
            ForPx(W, H, T, [&](size_t i, int, int) {
                if (aS[i] > alpha[i])
                {
                    alpha[i] = aS[i];
                    light[i] = 1;
                }
            });
        }
        tm.lap("lines tophat");
#ifdef SK_DEV_DUMP
        devDump("mid_alpha0.raw", alpha.data(), N * 4);
#endif
        // dense clusters of thin dark specks (foliage, brush texture) are not lines unless asked for
        if (!pin.bgDetailLines)
        {
            std::vector<float> v(N);
            ForPx(W, H, T, [&](size_t i, int, int) { v[i] = alpha[i] > 0.3f ? 1.f : 0.f; });
            // isolated specks (a dark grain, not a stroke): a line crossing a (2r+1)^2 window has >= 2r+1 pixels
            std::vector<float> cnt(N), tmpc(N);
            const int rb = std::max(1, (int)std::lround(2.0 * ds));
            ForPx(W, H, T, [&](size_t i, int x, int) {
                float s = 0.f;
                for (int k = std::max(0, x - rb); k <= std::min(W - 1, x + rb); ++k)
                    s += v[i - x + k];
                tmpc[i] = s;
            });
            ForPx(W, H, T, [&](size_t i, int x, int y) {
                float s = 0.f;
                for (int k = std::max(0, y - rb); k <= std::min(H - 1, y + rb); ++k)
                    s += tmpc[(size_t)k * W + x];
                cnt[i] = s;
            });
            RFPlain<1>(v.data(), W, H, 7.0 * ds, 3, T);
#ifdef SK_DEV_DUMP
            devDump("mid_vb.raw", v.data(), N * 4);
            devDump("mid_cnt.raw", cnt.data(), N * 4);
#endif
            const float side = 2.f * rb + 1.f;
            ForPx(W, H, T, [&](size_t i, int, int) {
                alpha[i] *= (1.f - Smooth(0.30f, 0.48f, v[i])) * Smooth(0.4f * side, 0.8f * side, cnt[i]);
            });
        }
        ForPx(W, H, T, [&](size_t i, int, int) {
            if (A[i] < (0.5f / 255.f))
                alpha[i] = 0.f;
        });
        }  // (!gpuLines)
#ifdef SK_DEV_DUMP
        // SK_LOAD=<dir>: continue from the GPU's alpha / light (hybrid: the GPU filters, the CPU does the rest)
        const char* devLd = std::getenv("SK_LOAD");
        auto devLoad = [&](const char* name, void* d, size_t bytes) {
            FILE* fp = nullptr;
            if ((fp = skplat::OpenFile(std::string(devLd) + skplat::PathSep() + name, "rb")) != nullptr)
            {
                const size_t r = fread(d, 1, bytes, fp);
                fclose(fp);
                if (r != bytes)
                    std::abort();
            }
            else
                std::abort();
        };
        if (devLd)
        {
            devLoad("gpu_alpha.raw", alpha.data(), N * 4);
            if (!light.empty())
            {
                std::vector<float> tl(N);
                devLoad("gpu_light.raw", tl.data(), N * 4);
                for (size_t i = 0; i < N; ++i) light[i] = tl[i] > 0.5f ? 1 : 0;
            }
        }
#endif
        bar.resize(N);
        std::vector<float> beta(N), wIn(N);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float a = alpha[i];
            bar[i] = (a > 0.25f || (pin.blackWalls && A[i] > 0.5f &&
                                    (Y.empty() ? (A[i] < (0.5f / 255.f) ? 1.f : Clampf(Luma(&C[i * 3]), 0.f, 1.f)) : Y[i]) < 0.10f)) ? 1 : 0;
            beta[i] = Smooth(0.03f, 0.35f, a);
            const float ob = 1.f - beta[i];
            wIn[i] = A[i] * ob * ob;
        });
        Y.clear(); Y.shrink_to_fit();
        core.clear(); core.shrink_to_fit();
        aS.clear(); aS.shrink_to_fit();
        tm.lap("lines clutter");
        // in-painted fill under the lines
        std::vector<float> Fh;
        if (!gpuLines)
        {
            Fh.resize(N * 3);
            SmoothNorm3(C.data(), wIn.data(), Fh.data(), W, H, rc + 1.0, Edges(), T, tmp4);
        }
        const float* FhP = gpuLines ? FhG.data() : Fh.data();
#ifdef SK_GPU
        if (t_st)
        {
            std::vector<float> c(alpha.begin(), alpha.end());
            if (!light.empty())
                for (size_t i = 0; i < N; ++i) c.push_back(light[i] ? 1.f : 0.f);
            c.insert(c.end(), Fh.begin(), Fh.end());
            StCmp(ST_LINES, c.data(), c.size());
        }
#endif
#ifdef SK_DEV_DUMP
        if (devLd)
            devLoad("gpu_Fh.raw", Fh.data(), N * 12);
#endif
        delta.resize(N * 3);
        const bool hasLight = !light.empty();
        ForPx(W, H, T, [&](size_t i, int, int) {
            for (int c = 0; c < 3; ++c)
            {
                float d = 0.f;
                if (beta[i] > 0.f)
                {
                    d = beta[i] * pm::log((FhP[i * 3 + c] + kInkEps) / (C[i * 3 + c] + kInkEps));
                    if (!(hasLight && light[i]) && d < 0.f)
                        d = 0.f;  // dark lines only darken
                    else if (hasLight && light[i] && d > 0.f)
                        d = 0.f;  // light lines only lighten
                }
                delta[i * 3 + c] = d;
                if (inpainted && d != 0.f)  // in place: C becomes the fill layer
                    C[i * 3 + c] = (C[i * 3 + c] + kInkEps) * pm::exp(d) - kInkEps;
            }
        });
        tm.lap("lines");
        freeScratch();
#ifdef SK_DEV_DUMP
        devDump("out_alpha.raw", alpha.data(), N * 4);
        devDump("out_bar.raw", bar.data(), N);
        if (!light.empty())
            devDump("out_light.raw", light.data(), N);
        devDump("out_delta.raw", delta.data(), N * 12);
        devDump("out_C.raw", C.data(), N * 12);
#endif
        poll();
        if (!needLines0)
        {
            // (only the depth line scaling asked for the lines: nothing else may see them, so the rest stays exact)
            alpha.clear();
            bar.clear();
            light.clear();
        }
    }
#ifdef SK_GPU
    if (!(wcOn && pin.wcMode == WCM_PAINT && !wcView))
        gpuFr.release();  // (only the painter uses the GPU after the line stage)
#endif
    Fl.swap(C);  // the fill layer (the input itself when lines are not protected)

    // ---- 2. texture bands on the fill layer
    std::vector<float> Tx;
    std::vector<float> bandView;
    const int viewBand = view == V_FINE ? 0 : view == V_MID ? 1 : view == V_LARGE ? 2 : -1;
    if (texOn || viewBand >= 0 || view == V_FILL)
        Tx = Fl;  // the bands read the untouched fill
    else
        Tx.swap(Fl);
    if (texOn || viewBand >= 0)
    {
        std::vector<float> gh, gv;
        const float ea = (float)Clampd(pin.edgeAware / 100.0, 0, 1);
        if (ea > 0.f)
            BuildGrad(Fl.data(), W, H, gh, gv, T);
        const double sr = 0.5 * pm::pow(0.03 / 0.5, (double)ea);
        // fine | mid | large: sigma s, 4 s, and a "clarity" scale of 24 s (36 px at the default 1.5 px)
        const double s1 = Clampd(pin.detailSize, 0.3, 50) * ds, sig[3] = {s1, 4.0 * s1, 24.0 * s1};
        const float halo = (float)Clampd(pin.halo / 100.0, 0, 1);
        const float tau0 = 0.6f * pm::pow(0.03f / 0.6f, halo);
        const float tauK[3] = {1.f, 2.f, 6.f};  // bigger bands have bigger natural amplitudes
        const float cd = (float)Clampd(pin.colorDetail / 100.0, 0, 1);
        bool active[3], needS[4] = {true, false, false, false};
        for (int b = 0; b < 3; ++b)
        {
            active[b] = gains[b] != 0.f || viewBand == b;
            if (active[b])
                needS[b] = needS[b + 1] = true;
        }
        if (viewBand >= 0)
            bandView.assign(N, 0.f);
        // upper = S_b (S_0 = the fill itself: no copy), lower = S_(b+1); two scratch planes are swapped
        std::vector<float> bufA, bufB;
        const float* upper = Fl.data();
        for (int b = 0; b < 3; ++b)
        {
            if (!needS[b + 1])
                continue;
            Edges e;
            if (ea > 0.f)
            {
                e.gh = gh.data();
                e.gv = gv.data();
                // the large band tolerates bigger edges (clarity = local contrast across shapes), the lines and
                // silhouettes still stay in the base (no halos)
                const double srb = sr * (b == 0 ? 1.0 : b == 1 ? 1.6 : 3.5);
                e.k = sig[b] / srb;
            }
            std::vector<float>& lowerV = (upper == bufA.data() && !bufA.empty()) ? bufB : bufA;
            lowerV.resize(N * 3);
            float* lower = lowerV.data();
            SmoothNorm3(Fl.data(), A.data(), lower, W, H, sig[b], e, T, tmp4);
            if (active[b])
            {
                const float g = gains[b];
                const bool vb = viewBand == b;
                const float tau = tau0 * tauK[b];
                ForPx(W, H, T, [&](size_t i, int, int) {
                    float d[3] = {upper[i * 3] - lower[i * 3], upper[i * 3 + 1] - lower[i * 3 + 1], upper[i * 3 + 2] - lower[i * 3 + 2]};
                    const float dY = Luma(d);
                    if (vb)
                        bandView[i] = dY;
                    if (g == 0.f)
                        return;
                    const float gm = g * mapOf(MT_TEXTURE, i) * std::max(0.f, 1.f + kTx * farAt(i));
                    float gY, gC;
                    if (gm > 0.f)
                    {
                        const float q = dY / tau;
                        gY = gm / (1.f + q * q);  // big differences (edges) get little gain: no halos
                        gC = gY * cd;
                    }
                    else
                        gY = gC = gm;
                    for (int c = 0; c < 3; ++c)
                        Tx[i * 3 + c] += gY * dY + gC * (d[c] - dY);
                });
            }
            upper = lower;
            poll();
        }
        tm.lap("texture");
        freeScratch();
    }
    if (view != V_FILL)
    {
        Fl.clear();
        Fl.shrink_to_fit();
    }

    // ---- 3. fill blur (lines are barriers, line pixels get no weight, then are in-painted)
    std::vector<float>& B = Tx;
    const bool useBar = !bar.empty() && (protect || pin.blackWalls);
    auto inpaintBarrier = [&](std::vector<float>& R, const std::vector<float>& den) {
        // barrier pixels (and pixels nothing reached) take the smoothed value around them
        std::vector<float> w2(N), Pq(N * 3);
        ForPx(W, H, T, [&](size_t i, int, int) { w2[i] = (bar[i] || den[i] < 1e-4f) ? 0.f : A[i]; });
        SmoothNorm3(R.data(), w2.data(), Pq.data(), W, H, rc + 1.0, Edges(), T, tmp4);
        ForPx(W, H, T, [&](size_t i, int, int) {
            if (bar[i] || den[i] < 1e-4f)
                for (int c = 0; c < 3; ++c)
                    R[i * 3 + c] = Pq[i * 3 + c];
        });
    };
    if (blurOn)
    {
        std::vector<float> R(N * 3), gh, gv, w(N), den(N);
        const float ke = (float)Clampd(pin.keepEdges / 100.0, 0, 1);
        Edges e;
        if (ke > 0.f)
        {
            BuildGrad(Tx.data(), W, H, gh, gv, T);
            e.gh = gh.data();
            e.gv = gv.data();
            e.k = (fillBlur * 0.5) / (1.0 * pm::pow(0.04, (double)ke));
        }
        if (useBar)
            e.bar = bar.data();
        ForPx(W, H, T, [&](size_t i, int, int) { w[i] = A[i] * ((useBar && bar[i]) ? 0.f : 1.f); });
        SmoothNorm3(Tx.data(), w.data(), R.data(), W, H, fillBlur * 0.5, e, T, tmp4, den.data());
        if (useBar)
            inpaintBarrier(R, den);
        // depth +: with a fill blur already set, the far parts blend toward a stronger blur (up to 3x the radius)
        std::vector<float> R2;
        if (baseBlur && kBl > 0.f && !depthFar.empty())
        {
            R2.resize(N * 3);
            std::vector<float> den2(N);
            SmoothNorm3(Tx.data(), w.data(), R2.data(), W, H, fillBlur * 0.5 * (1.0 + 2.0 * kBl), e, T, tmp4, den2.data());
            if (useBar)
                inpaintBarrier(R2, den2);
        }
        ForPx(W, H, T, [&](size_t i, int, int) {
            // depth: near keeps the set amount, far more (+) or less (-); with no fill blur set, only the far parts blur
            const float fa = farAt(i);
            const float m = mapOf(MT_FILLBLUR, i) * Clampf((baseBlur ? 1.f : 0.f) + kBl * fa, 0.f, 1.f);
            const float m2 = R2.empty() ? 0.f : Clampf(kBl * fa, 0.f, 1.f);
            for (int c = 0; c < 3; ++c)
            {
                const float r = R[i * 3 + c] + (m2 > 0.f ? m2 * (R2[i * 3 + c] - R[i * 3 + c]) : 0.f);
                B[i * 3 + c] += m * (r - B[i * 3 + c]);
            }
        });
        tm.lap("fill blur");
        freeScratch();
        poll();
    }

    // ---- 4. watercolour (docs\watercolor_model.md)
    freeScratch();
    std::vector<float> viewWc;
    std::vector<float> paintBusy;  // the painter: busy texture (foliage) at the buffer size - its line specks are dropped
    bool gpuInked = false;  // 0.6.0.5: 5. ink ran on the GPU after the painter's paper
    if (wcOn || wcView)
    {
        const bool copyP = !(wc >= 1.f && (!M || pin.mapTarget != MT_WATERCOLOR));
        std::vector<float> Pown;
        if (copyP)
            Pown = B;
        float* P = copyP ? Pown.data() : B.data();
        struct WcFrame
        {
            int ox, oy;
            Rect layer;
        };
        // part 1 - the washes (flatten, strokes, glazes, wobble, bleed, blooms, unevenness, rims) -> total density.
        // Runs at the working resolution: full, or half for big frames (then it is what a Half preview computes).
        auto washes = [&](const int W, const int H, const double ds, const WcFrame& f, float* P, const float* B, const std::vector<float>& A,
                          const std::vector<uint8_t>& bar, const std::vector<float>& alpha, Buf& Dt, Buf& prot, Buf& busy, Buf& strokeTex,
                          std::vector<float>& viewWc) {
            const size_t N = (size_t)W * H;
            auto inpaintBarrier = [&](std::vector<float>& R, const std::vector<float>& den) {
                std::vector<float> w2(N), Pq(N * 3);
                ForPx(W, H, T, [&](size_t i, int, int) { w2[i] = (bar[i] || den[i] < 1e-4f) ? 0.f : A[i]; });
                SmoothNorm3(R.data(), w2.data(), Pq.data(), W, H, rc + 1.0, Edges(), T, tmp4);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    if (bar[i] || den[i] < 1e-4f)
                        for (int c = 0; c < 3; ++c)
                            R[i * 3 + c] = Pq[i * 3 + c];
                });
            };
            std::vector<float> w(N);
            ForPx(W, H, T, [&](size_t i, int, int) { w[i] = A[i] * ((useBar && bar[i]) ? 0.f : 1.f); });

            // (0) small-detail map: density of lines / strong edges nearby (eyes, mouths, fingers, small props).
            //     prot: how much the random parts (wobble, blooms, strokes, granulation) back off; busy: wash size cue
            prot.alloc(N);
            busy.alloc(N);
            {
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float* c = P + i * 3;
                    const float* r = x + 1 < W ? c + 3 : c;
                    const float* d = y + 1 < H ? c + (size_t)W * 3 : c;
                    const float g = std::fabs(Luma(r) - Luma(c)) + std::fabs(Luma(d) - Luma(c));
                    const float e = std::max(alpha.empty() ? 0.f : alpha[i], Clampf((g - 0.04f) * 8.f, 0.f, 1.f));
                    prot[i] = e * A[i];
                    busy[i] = Clampf((g - 0.02f) * 10.f, 0.f, 1.f) * A[i];
                });
                RFPlain<1>(prot.data(), W, H, 4.0 * ds, 3, T);
                RFPlain<1>(busy.data(), W, H, 10.0 * ds, 3, T);
                const float pd = (float)Clampd(pin.protectDetail / 100.0, 0, 1);
                ForPx(W, H, T, [&](size_t i, int, int) { prot[i] = pd * Smooth(0.10f, 0.32f, prot[i]); });
            }
            tm.lap("wc prep");
            // (a) flatten washes: strongly edge-aware smoothing, lines are barriers
            const float flat = (float)Clampd(pin.flatten / 100.0, 0, 1);
            if (flat > 0.f)
            {
                std::vector<float> E(N * 3), den(N);
                const double sf = Clampd(pin.flattenSize, 0.5, 200) * ds;
                if (N > kHalfResPx && sf >= 3.0)
                    SmoothNorm3Half(P, w.data(), E.data(), W, H, sf, 0.07, useBar ? bar.data() : nullptr, T, tmp4, den.data());
                else
                {
                    std::vector<float> gh, gv;
                    BuildGrad(P, W, H, gh, gv, T);
                    Edges e;
                    e.gh = gh.data();
                    e.gv = gv.data();
                    e.k = sf / 0.07;
                    if (useBar)
                        e.bar = bar.data();
                    SmoothNorm3(P, w.data(), E.data(), W, H, sf, e, T, tmp4, den.data());
                }
                if (useBar)
                    inpaintBarrier(E, den);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    // small features keep more of their own shape
                    const float k = flat * (1.f - 0.5f * prot[i]);
                    for (int c = 0; c < 3; ++c)
                        P[i * 3 + c] += k * (E[i * 3 + c] - P[i * 3 + c]);
                });
                tm.lap("wc flatten");
                poll();
            }
            // (b) brush strokes: a direction field (structure tensor along shapes, a fixed slanted hand where the
            //     picture has no direction) and a walk along it: the wash is smoothed along the stroke (stopping at
            //     colour changes) and a stroke texture (two noise images averaged along the walk) is collected
            const float stk = (float)Clampd(pin.strokes / 100.0, 0, 1);
            if (stk > 0.f || view == V_STROKES)
            {
                const double L = Clampd(pin.strokeLen, 2, 200) * ds;
                Buf dx, dy;
                dx.alloc(N);
                dy.alloc(N);
                {
                    Buf st;
                    st.alloc(N * 3);
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        const float* c = P + i * 3;
                        const float l = Luma(x > 0 ? c - 3 : c), r = Luma(x + 1 < W ? c + 3 : c);
                        const float u = Luma(y > 0 ? c - (size_t)W * 3 : c), d = Luma(y + 1 < H ? c + (size_t)W * 3 : c);
                        const float gx = 0.5f * (r - l) * A[i], gy = 0.5f * (d - u) * A[i];
                        st[i * 3] = gx * gx;
                        st[i * 3 + 1] = gx * gy;
                        st[i * 3 + 2] = gy * gy;
                    });
                    RFPlain<3>(st.data(), W, H, std::max(1.5, L * 0.3), 3, T);
                    const double a0 = Clampd(pin.strokeAngle, -360, 360) * 3.14159265358979 / 180.0;
                    std::vector<float> ang;
                    NoiseField(ang, W, H, f.ox, f.oy, ds, 320.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 320.0, 3, 0.5, seed ^ 0x57A0u, ds); });
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        const float jxx = st[i * 3], jxy = st[i * 3 + 1], jyy = st[i * 3 + 2];
                        const float tr = jxx + jyy, disc = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.f * jxy * jxy);
                        const float coh = disc / (tr + 1e-9f);
                        const float gate = Smooth(2e-5f, 3e-4f, tr) * Smooth(0.15f, 0.6f, coh);
                        // doubled angles: the stroke runs ALONG the shape (perpendicular to the gradient)
                        float c2 = disc > 1e-12f ? -(jxx - jyy) / disc : 0.f, s2 = disc > 1e-12f ? -2.f * jxy / disc : 0.f;
                        const double a = a0 + 0.55 * ang[i];  // the hand's default slant, varying slowly (+-30 deg)
                        // screen angle counter-clockwise with y down: direction (cos a, -sin a)
                        const float c0 = (float)pm::cos(2.0 * a), s0 = (float)-pm::sin(2.0 * a);
                        const float vx = gate * c2 + (1.f - gate) * c0, vy = gate * s2 + (1.f - gate) * s0;
                        const float th = 0.5f * pm::atan2(vy, vx);
                        dx[i] = pm::cos(th);
                        dy[i] = pm::sin(th);
                    });
                }
                // two noise images (anchored to the layer): faint bristle streaks (3 px) and soft stroke bodies (14 px)
                std::vector<float> n0, n1;
                NoiseField(n0, W, H, f.ox, f.oy, ds, 3.0 / 2, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 3.0, 2, 0.5, seed ^ 0xB415u, ds); });
                NoiseField(n1, W, H, f.ox, f.oy, ds, 14.0 / 2, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 14.0, 2, 0.5, seed ^ 0xB416u, ds); });
                const int n = std::max(1, std::min(12, (int)std::ceil(L)));
                const float step = (float)std::max(1.0, L / n);
                strokeTex.alloc(N);
                std::vector<float> Po(N * 3);
                const float sc2 = 1.f / (2.f * 0.07f * 0.07f);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float* c0 = P + i * 3;
                    float acc[3] = {c0[0], c0[1], c0[2]}, wsum = 1.f, ns = 0.25f * n0[i] + 0.75f * n1[i], nw = 1.f, nw2 = 1.f;
                    for (int sgn = -1; sgn <= 1; sgn += 2)
                    {
                        float px = x + 0.5f, py = y + 0.5f, ddx = dx[i] * sgn, ddy = dy[i] * sgn;
                        for (int k = 1; k <= n; ++k)
                        {
                            px += ddx * step;
                            py += ddy * step;
                            const int xi = (int)std::floor(px), yi = (int)std::floor(py);
                            if (xi < 0 || yi < 0 || xi >= W || yi >= H)
                                break;
                            const size_t j = (size_t)yi * W + xi;
                            if (useBar && bar[j])
                                break;  // strokes stop at lines
                            float ndx = dx[j], ndy = dy[j];
                            if (ndx * ddx + ndy * ddy < 0.f)
                            {
                                ndx = -ndx;
                                ndy = -ndy;
                            }
                            ddx = ndx;
                            ddy = ndy;
                            const float* cj = P + j * 3;
                            const float d0 = cj[0] - c0[0], d1 = cj[1] - c0[1], d2 = cj[2] - c0[2];
                            const float t = (float)k / n;
                            const float wk = pm::exp(-2.f * t * t) * pm::exp(-(d0 * d0 + d1 * d1 + d2 * d2) * sc2) * A[j];
                            acc[0] += wk * cj[0];
                            acc[1] += wk * cj[1];
                            acc[2] += wk * cj[2];
                            wsum += wk;
                            ns += wk * (0.25f * n0[j] + 0.75f * n1[j]);
                            nw += wk;
                            nw2 += wk * wk;
                        }
                    }
                    for (int c = 0; c < 3; ++c)
                        Po[i * 3 + c] = acc[c] / wsum;
                    // the averaged noise has sd ~ 0.3 * sqrt(sum w^2) / sum w: normalise to ~N(0,1)
                    strokeTex[i] = Clampf((ns / nw) / (0.3f * std::sqrt(nw2) / nw + 1e-3f), -2.5f, 2.5f);
                });
                const float k = 0.7f * stk;
                ForPx(W, H, T, [&](size_t i, int, int) {
                    for (int c = 0; c < 3; ++c)
                        P[i * 3 + c] += k * (Po[i * 3 + c] - P[i * 3 + c]);
                });
                if (view == V_STROKES)
                {
                    viewWc.resize(N * 3);
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        const float v = Clampf(0.5f + 0.18f * strokeTex[i], 0.f, 1.f);
                        viewWc[i * 3] = v * (0.6f + 0.4f * std::fabs(dx[i]));
                        viewWc[i * 3 + 1] = v;
                        viewWc[i * 3 + 2] = v * (0.6f + 0.4f * std::fabs(dy[i]));
                    });
                }
                tm.lap("wc strokes");
                poll();
            }

            // (c) optical density (paper white = 0); values above 1 (HDR) pass through unchanged.
            //     tend: how much this pigment granulates (ultramarine-like blues, earths: yes; vivid staining hues: no)
            //     ink: near-black areas stay as they are (lashes, black hair shadows), like ink, not a grey wash
            Buf D1, ink, Yw;
            D1.alloc(N * 3);
            ink.alloc(N);
            Yw.alloc(N);
            ForPx(W, H, T, [&](size_t i, int, int) {
                const float* c = P + i * 3;
                Yw[i] = std::min(c[0], std::min(c[1], c[2]));
                ink[i] = 1.f - Smooth(0.07f, 0.17f, Luma(&B[i * 3]));
                for (int k = 0; k < 3; ++k)
                    D1[i * 3 + k] = -pm::log(Clampf(c[k], 1e-3f, 1.f));
            });
            // (d) glazing: a light local-colour wash + a shadow glaze on top (tonal separation): base density =
            //     the lightest density nearby (min filter, smoothed), the glaze is the rest. Each layer gets its own
            //     wobble, bleed and rim; densities add (transparent layers multiply)
            const float gz = (float)Clampd(pin.glaze / 100.0, 0, 1);
            Buf D2;
            if (gz > 0.f || view == V_GLAZE)
            {
                std::vector<float> dm(N), env(N);
                ForPx(W, H, T, [&](size_t i, int, int) { dm[i] = (D1[i * 3] + D1[i * 3 + 1] + D1[i * 3 + 2]) * (1.f / 3.f); });
                const int R = std::max(1, (int)std::lround(14.0 * ds));
                MinMax(dm.data(), env.data(), W, H, R, false, T, tmp1);
                RFPlain<1>(env.data(), W, H, R * 0.6, 3, T);
                D2.alloc(N * 3);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    const float base = std::min(dm[i], env[i]);
                    const float s = dm[i] > 1e-4f ? 1.f - gz * (1.f - base / dm[i]) : 1.f;
                    for (int c = 0; c < 3; ++c)
                    {
                        const float d = D1[i * 3 + c];
                        D1[i * 3 + c] = d * s;
                        D2[i * 3 + c] = d * (1.f - s);
                    }
                });
                if (view == V_GLAZE)
                {
                    viewWc.resize(N * 3);
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        for (int c = 0; c < 3; ++c)
                            viewWc[i * 3 + c] = pm::exp(-D2[i * 3 + c] * 1.5f);
                    });
                }
            }
            Buf* layers[2] = {&D1, D2.empty() ? nullptr : &D2};
            // (e) wobble per layer (different fields: the glaze does not sit exactly on the base = gaps and overlaps)
            const double wob = Clampd(pin.wobble, 0, 50) * ds;
            if (wob > 0.01 && W >= 2 && H >= 2)
            {
                const double ws = Clampd(pin.wobbleScale, 2, 1000);
                for (int li = 0; li < 2; ++li)
                {
                    if (!layers[li])
                        continue;
                    Buf& Dl = *layers[li];
                    std::vector<float> VX, VY;
                    const uint32_t s2 = seed ^ (li ? 0x51A9Eu : 0u);
                    NoiseField(VX, W, H, f.ox, f.oy, ds, ws / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, ws, 3, 0.45, s2 ^ 0xA11CEu, ds); });
                    NoiseField(VY, W, H, f.ox, f.oy, ds, ws / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, ws, 3, 0.45, s2 ^ 0xB0B5u, ds); });
                    Buf Q;
                    Q.alloc(N * 3);
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        const float am = (float)(wob * 1.6) * (1.f - prot[i]);
                        const double sx = Clampd(x + am * VX[i], 0, W - 1), sy = Clampd(y + am * VY[i], 0, H - 1);
                        const int x0 = std::min(W - 2, (int)sx), y0 = std::min(H - 2, (int)sy);
                        const float fx = (float)(sx - x0), fy = (float)(sy - y0);
                        float acc[3] = {0, 0, 0}, wa = 0.f;
                        const float wxs[2] = {1.f - fx, fx}, wys[2] = {1.f - fy, fy};
                        for (int jj = 0; jj < 2; ++jj)
                            for (int ii = 0; ii < 2; ++ii)
                            {
                                const size_t k = (size_t)(y0 + jj) * W + (x0 + ii);
                                const float ww = wxs[ii] * wys[jj] * w[k];
                                wa += ww;
                                for (int c = 0; c < 3; ++c)
                                    acc[c] += ww * Dl[k * 3 + c];
                            }
                        const float eps = 0.05f;
                        for (int c = 0; c < 3; ++c)
                            Q[i * 3 + c] = (acc[c] + eps * Dl[i * 3 + c]) / (wa + eps);
                    });
                    std::swap(Dl.p, Q.p);
                }
                tm.lap("wc wobble");
                poll();
            }
            // (f) bleeding, lost and found: density diffuses across boundaries where the two sides are close in
            //     value (edges get lost), much less where the contrast is high (edges stay found); varied by noise
            const float bl = (float)Clampd(pin.bleed / 100.0, 0, 1);
            const double bw = Clampd(pin.bleedWidth, 0.5, 300);
            if (bl > 0.f)
            {
                std::vector<float> NB;
                NoiseField(NB, W, H, f.ox, f.oy, ds, bw * 5.0 / 8, T, [&](double gx, double gy) { return Fbm(gx * par, gy, bw * 5.0, 4, 0.55, seed ^ 0xB1EEDu, ds); });
                // one smoothing of the total density; each layer keeps its share of the result
                Buf Dt;
                Dt.alloc(N * 3);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    for (int c = 0; c < 3; ++c)
                        Dt[i * 3 + c] = D1[i * 3 + c] + (layers[1] ? D2[i * 3 + c] : 0.f);
                });
                std::vector<float> Dg(N * 3);
                if (N > kHalfResPx && bw * ds >= 6.0)
                    SmoothNorm3Half(Dt.data(), w.data(), Dg.data(), W, H, bw * ds * 0.5, 0.0, useBar ? bar.data() : nullptr, T, tmp4, nullptr);
                else
                {
                    Edges e;
                    if (useBar)
                        e.bar = bar.data();
                    SmoothNorm3(Dt.data(), w.data(), Dg.data(), W, H, bw * ds * 0.5, e, T, tmp4);
                }
                ForPx(W, H, T, [&](size_t i, int, int) {
                    const float dm = (Dt[i * 3] + Dt[i * 3 + 1] + Dt[i * 3 + 2]) * (1.f / 3.f);
                    const float gm = (Dg[i * 3] + Dg[i * 3 + 1] + Dg[i * 3 + 2]) * (1.f / 3.f);
                    const float contrast = 2.f * std::fabs(dm - gm);  // ~ the density step of a nearby boundary
                    const float lost = 1.f - Smooth(0.06f, 0.40f, contrast);
                    const float k = Clampf(bl * (0.2f + 0.95f * lost) * (0.55f + 0.6f * Smooth(-0.4f, 0.5f, NB[i])), 0.f, 1.f) *
                                    (1.f - 0.6f * prot[i]);
                    for (int c = 0; c < 3; ++c)
                    {
                        const float t = Dt[i * 3 + c], t2 = t + k * (Dg[i * 3 + c] - t);
                        if (t > 1e-5f)
                        {
                            const float r = t2 / t;
                            D1[i * 3 + c] *= r;
                            if (layers[1])
                                D2[i * 3 + c] *= r;
                        }
                        else
                            D1[i * 3 + c] = t2;  // bled into bare paper: the base wash
                    }
                });
                tm.lap("wc bleed");
                poll();
            }
            // (g) backruns (blooms) grow FROM wet edges: wash boundaries inside "wet" patches of the picture seed a
            //     bloom that spreads into the drying wash with a fractal (cauliflower) front: lighter inside, a dark
            //     ring of pushed pigment at the front. Driven by the picture's own boundaries, not free noise blots.
            const float br = (float)Clampd(pin.backrun / 100.0, 0, 1);
            const float un = (float)Clampd(pin.uneven / 100.0, 0, 1);
            const float dil = (float)Clampd(pin.dilution / 100.0, 0, 1);
            const float kw = (float)Clampd(pin.keepWhite / 100.0, 0, 1);
            Buf kf;  // density factor shared by both layers
            kf.alloc(N);
            {
                std::vector<float> S, NF, NU, WET;
                if (br > 0.f)
                {
                    const double bloomR = std::max(10.0, bw * 2.5);
                    NoiseField(WET, W, H, f.ox, f.oy, ds, 300.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 300.0, 3, 0.5, seed ^ 0x3E7u, ds); });
                    NoiseField(NF, W, H, f.ox, f.oy, ds, bloomR / 4, T,
                               [&](double gx, double gy) { return Fbm(gx * par, gy, bloomR, 3, 0.6, seed ^ 0xCA11u, ds); });
                    S.resize(N);
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        auto dmAt = [&](size_t j) {
                            float s = (D1[j * 3] + D1[j * 3 + 1] + D1[j * 3 + 2]);
                            if (layers[1])
                                s += D2[j * 3] + D2[j * 3 + 1] + D2[j * 3 + 2];
                            return s * (1.f / 3.f);
                        };
                        const size_t r = x + 1 < W ? i + 1 : i, d = y + 1 < H ? i + W : i;
                        const float g = std::fabs(dmAt(r) - dmAt(i)) + std::fabs(dmAt(d) - dmAt(i));
                        // seeds: clear wash boundaries in wet patches, not the busy texture of small shapes
                        S[i] = Smooth(0.04f, 0.14f, g) * Smooth(0.05f, 0.35f, WET[i]) * (1.f - Smooth(0.03f, 0.15f, busy[i])) * A[i];
                    });
                    RFPlain<1>(S.data(), W, H, bloomR * ds * 0.6, 3, T);
                }
                if (un > 0.f)  // pigment flow: slow, smooth density variation inside washes
                    NoiseField(NU, W, H, f.ox, f.oy, ds, 190.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 190.0, 3, 0.45, seed ^ 0x0E7Eu, ds); });
                ForPx(W, H, T, [&](size_t i, int, int) {
                    float dm = (D1[i * 3] + D1[i * 3 + 1] + D1[i * 3 + 2]) * (1.f / 3.f);
                    if (layers[1])
                        dm += (D2[i * 3] + D2[i * 3 + 1] + D2[i * 3 + 2]) * (1.f / 3.f);
                    float k = 1.f;
                    if (br > 0.f)
                    {
                        // the bloom: where the edge proximity beats a fractal threshold, inside a painted wash
                        const float tau = 0.08f * (1.f + 0.8f * NF[i]);
                        // mid-value washes only (a bloom in near-black paint or next to ink reads as a stain), never on small features
                    const float m = Smooth(tau, tau + 0.035f, S[i]) * Smooth(0.12f, 0.35f, dm) * (1.f - Smooth(0.7f, 1.1f, dm)) * (1.f - ink[i]) *
                                    (1.f - prot[i]) * (1.f - Smooth(0.05f, 0.2f, busy[i]));
                        const float ring = Clampf(4.f * m * (1.f - m), 0.f, 1.f);
                        k *= (1.f - 0.45f * br * m) * (1.f + 0.35f * br * ring);  // the rim step adds the rest of the dark front
                    }
                    if (un > 0.f)
                        k *= std::max(0.f, 1.f + 0.65f * un * NU[i]);
                    k *= (1.f - 0.6f * dil) * (1.f - kw * Smooth(0.86f, 0.97f, Yw[i]));
                    kf[i] = k;
                });
            }
            for (int li = 0; li < 2; ++li)
                if (layers[li])
                {
                    Buf& Dl = *layers[li];
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        for (int c = 0; c < 3; ++c)
                            Dl[i * 3 + c] *= kf[i];
                    });
                }
            kf.release();
            poll();
            // (h) pigment pools at each wash's edge: positive part of (density - its blur) = the "sharpen" signal,
            //     mapped to a relative density increase. The width follows the wash size (busy = small shapes = thin
            //     rims, big quiet washes = wide rims); "gravity" makes the bottom edges (where pigment ran) stronger.
            const float rim = (float)Clampd(pin.rim / 100.0, 0, 2);
            if (rim > 0.f || view == V_RIM)
            {
                const double rw = Clampd(pin.rimWidth, 0.3, 100) * ds;
                const bool paperOutside = pin.edgeMode == EM_PAPER;
                const float grav = (float)Clampd(pin.gravity / 100.0, 0, 1);
                const int NL = layers[1] ? 2 : 1;
                // channels: layer means (premultiplied by the "inside" weight), the weight
                auto fill3 = [&](Buf& v) {
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        const bool inLayer = x >= f.layer.l && x < f.layer.r && y >= f.layer.t && y < f.layer.b;
                        // transparent pixels inside the layer = bare paper (rim at a cel's silhouette)
                        const float om = (inLayer || paperOutside) ? 1.f : 0.f;
                        for (int li = 0; li < 2; ++li)
                        {
                            float dmv = 0.f;
                            if (li < NL)
                            {
                                const Buf& Dl = *layers[li];
                                dmv = (Dl[i * 3] + Dl[i * 3 + 1] + Dl[i * 3 + 2]) * (1.f / 3.f) * A[i];
                            }
                            v[i * 3 + li] = dmv * om;
                        }
                        v[i * 3 + 2] = om;
                    });
                };
                Buf vt, vw;  // thin (rw / 2) and wide (rw * 2) blurs
                vt.alloc(N * 3);
                vw.alloc(N * 3);
                fill3(vt);
                fill3(vw);
                RFPlain<3>(vt.data(), W, H, std::max(0.6, rw * 0.5), 3, T);
                if (N > kHalfResPx && rw * 2.0 >= 3.0)
                    RFPlain3Half(vw.data(), W, H, rw * 2.0, T);
                else
                    RFPlain<3>(vw.data(), W, H, rw * 2.0, 3, T);
                const float rk = 2.4f * std::max(rim, view == V_RIM ? 0.7f : 0.f);
                if (view == V_RIM)
                    viewWc.assign(N * 3, 0.f);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float t = Smooth(0.04f, 0.25f, busy[i]);  // 0 = big quiet wash, 1 = busy small shapes
                    float total = 0.f;
                    for (int li = 0; li < NL; ++li)
                    {
                        Buf& Dl = *layers[li];
                        const float dm = (Dl[i * 3] + Dl[i * 3 + 1] + Dl[i * 3 + 2]) * (1.f / 3.f);
                        const float da = dm * A[i];
                        auto norm = [&](const Buf& v, size_t j) { return v[j * 3 + 2] > 1e-6f ? v[j * 3 + li] / v[j * 3 + 2] : da; };
                        const float gt = norm(vt, i), gw = norm(vw, i);
                        const float Rt = std::max(0.f, da - gt), Rw = std::max(0.f, da - gw);
                        float R = 0.45f * Rt + 0.55f * (t * Rt + (1.f - t) * Rw);
                        if (grav > 0.f)
                        {
                            // the wide blur falls off downwards at a wash's bottom edge (y grows downwards)
                            const size_t u = y > 0 ? i - W : i, d = y + 1 < H ? i + W : i;
                            const size_t l = x > 0 ? i - 1 : i, r = x + 1 < W ? i + 1 : i;
                            const float gy = 0.5f * (norm(vw, d) - norm(vw, u)), gx = 0.5f * (norm(vw, r) - norm(vw, l));
                            const float down = -gy / (std::sqrt(gx * gx + gy * gy) + 1e-5f);
                            R *= 1.f + grav * 0.8f * down;
                        }
                        const float rf = std::min(4.f, 1.f + rk * std::max(0.f, R) / (dm + 0.08f));
                        const float k = 1.f + (rf - 1.f) * (1.f - ink[i]);
                        if (rim > 0.f)
                            for (int c = 0; c < 3; ++c)
                                Dl[i * 3 + c] *= k;
                        total += rf - 1.f;
                    }
                    if (view == V_RIM)
                    {
                        const float v = Clampf(total * 0.8f, 0.f, 1.f);
                        viewWc[i * 3] = v;
                        viewWc[i * 3 + 1] = 0.6f * v;
                        viewWc[i * 3 + 2] = 0.2f * v;
                    }
                });
                tm.lap("wc rim");
                poll();
            }
            Dt.alloc(N * 3);
            ForPx(W, H, T, [&](size_t i, int, int) {
                for (int c = 0; c < 3; ++c)
                    Dt[i * 3 + c] = D1[i * 3 + c] + (layers[1] ? D2[i * 3 + c] : 0.f);
            });
        };
        Buf Dt, prot, busy, strokeTex;
        const bool painter = pin.wcMode == WCM_PAINT;
        bool gpuPainted = false;  // 0.6.0.5: the painter's tail and its paper ran on the GPU (P holds the colour)
#ifdef SK_GPU
        GpuInkIn gpuInk;
#endif
        if (painter)
        {
            // round 6: the region painter -> total density at buffer resolution
            Dt.alloc(N * 3);
            PaintIn pi{W, H, ds, par, f.ox, f.oy, f.layer, seed, T, P, A.data(), bar.empty() ? nullptr : bar.data(),
                       alpha.empty() ? nullptr : alpha.data(), depthFar.empty() ? nullptr : depthFar.data(), kSc, kWe};
#ifdef SK_GPU
            // the ink on the GPU too when it is the plain one (no thickness change, no line map, B = P)
            const bool composeW = needLines && ((inpainted && anyProc) || lineOps || depthLines);
            const double thW = Clampd(pin.lineThick, -10, 10) * ds;
            const bool inkElig = composeW && !delta.empty() && !(thW > 0.01) && !(thW < -0.01) && !(M && pin.mapTarget == MT_LINES) && !copyP;
            if (!wcView)
                LogElig(GU_REST | (inkElig ? GU_INK : 0u));  // (wet: PaintWatercolour, when there is a wet-in-wet mix)
            if (!wcView && gpuFr.begin(W, H, GU_WET | GU_REST))
            {
                pi.gpu = &gpuFr;
                pi.gpuOut = P;
                pi.gpuDone = &gpuPainted;
                pi.gr = (float)Clampd(pin.gran / 100.0, 0, 1);
                pi.pp = (float)Clampd(pin.paper / 100.0, 0, 1);
                pi.pd = (float)Clampd(pin.protectDetail / 100.0, 0, 1);
                pi.paperTex = f.paperTex;
                pi.ps = ps;
                if (inkElig && !(gpuFr.units & GU_INK))
                    GpuFrame::LogOff(GU_INK);
                if (inkElig && (gpuFr.units & GU_INK))
                {
                    gpuInk.delta = delta.data();
                    gpuInk.far_w = (kLi != 0.f && !depthFar.empty()) ? depthFar.data() : nullptr;
                    gpuInk.dk = 1.f + (float)Clampd(pin.lineDark / 100.0, -1, 2);
                    gpuInk.wc = wc;
                    gpuInk.kLi = kLi;
                    gpuInk.inpainted = inpainted;
                    gpuInk.style = pin.lineStyle == LS_PENCIL ? 1 : pin.lineStyle == LS_INK ? 2 : 0;
                    gpuInk.soft = Clampd(pin.lineSoft, 0, 100) * ds * 0.5 + (pin.lineStyle == LS_INK ? 0.35 * ds : 0.0);
                    if (gpuInk.style)
                    {
                        // NoiseField(BK, ...): the same grid, or (step 1) the same fBm per pixel on the GPU
                        const double lam = gpuInk.style == 1 ? 16.0 : 40.0;
                        gpuInk.bkLambda = lam;
                        const int step = std::max(1, std::min(8, (int)std::floor(lam / 4 * ds / 4.0)));
                        if (step == 1)
                            gpuInk.bkMode = 2;
                        else
                        {
                            gpuInk.bkMode = 1;
                            const int gx0 = FloorDiv(f.ox, step), gy0 = FloorDiv(f.oy, step);
                            const int gw = FloorDiv(f.ox + W - 1, step) - gx0 + 2, gh = FloorDiv(f.oy + H - 1, step) - gy0 + 2;
                            gpuInk.bkGrid.resize((size_t)gw * gh);
                            PFor(gh, T, [&](int j0, int j1) {
                                for (int j = j0; j < j1; ++j)
                                    for (int i = 0; i < gw; ++i)
                                    {
                                        const double gx = (double)(gx0 + i) * step / ds, gy = (double)(gy0 + j) * step / ds;
                                        gpuInk.bkGrid[(size_t)j * gw + i] = Fbm(gx * par, gy, lam, 3, 0.55, seed ^ 0x9E7C1u, ds);
                                    }
                            });
                            const int mt[7] = {gx0, gy0, gw, gh, step, f.ox, f.oy};
                            std::copy(mt, mt + 7, gpuInk.bkMeta);
                        }
                    }
                    gpuInk.done = &gpuInked;
                    pi.ink = &gpuInk;
                }
                for (size_t i = 0; i < N * 3 && !pi.anyHdr; ++i)
                    pi.anyHdr = P[i] > 1.f;
            }
#endif
            PaintWatercolour(pin, pi, Dt.data(), wcView ? &viewWc : nullptr, poll, &paintBusy);
        }
#ifdef SK_GPU
        gpuFr.release();
#endif
        if (gpuPainted)
        {
            tm.lap("wc painter + paper (gpu)");

        }
        else if (painter)
        {
            prot.alloc(N);
            busy.alloc(N);
            std::vector<float> pa(N, 0.f);
            if (!alpha.empty())
                ForPx(W, H, T, [&](size_t i, int, int) { pa[i] = alpha[i]; });
            RFPlain<1>(pa.data(), W, H, 4.0 * ds, 3, T);
#ifdef SK_GPU
            StCmp(ST_PA, pa.data(), N);
#endif
            const float pd = (float)Clampd(pin.protectDetail / 100.0, 0, 1);
            ForPx(W, H, T, [&](size_t i, int, int) {
                prot[i] = pd * Smooth(0.05f, 0.2f, pa[i]);
                busy[i] = 0.f;
            });
            tm.lap("wc painter");
        }
        else if (N > kHalfResPx)
        {
            // big frame: the washes at half resolution (2x2 boxes; walls and lines max-pooled), then bilinear up
            const int w2 = (W + 1) / 2, h2 = (H + 1) / 2;
            const size_t N2 = (size_t)w2 * h2;
            std::vector<float> P2(N2 * 3), B2(N2 * 3), A2(N2), al2(alpha.empty() ? 0 : N2);
            std::vector<uint8_t> bar2(bar.empty() ? 0 : N2);
            PFor(h2, T, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w2; ++x)
                    {
                        const size_t o = (size_t)y * w2 + x;
                        float pa[3] = {0, 0, 0}, pp[3] = {0, 0, 0}, bb[3] = {0, 0, 0}, as = 0.f, am = 0.f;
                        uint8_t br2 = 0;
                        for (int j = 0; j < 2; ++j)
                            for (int ii = 0; ii < 2; ++ii)
                            {
                                const size_t k = (size_t)std::min(H - 1, 2 * y + j) * W + std::min(W - 1, 2 * x + ii);
                                for (int c = 0; c < 3; ++c)
                                {
                                    pa[c] += A[k] * P[k * 3 + c];
                                    pp[c] += P[k * 3 + c];
                                    bb[c] += B[k * 3 + c];
                                }
                                as += A[k];
                                if (!alpha.empty())
                                    am = std::max(am, alpha[k]);
                                if (!bar.empty())
                                    br2 |= bar[k];
                            }
                        for (int c = 0; c < 3; ++c)
                        {
                            P2[o * 3 + c] = as > 1e-6f ? pa[c] / as : 0.25f * pp[c];
                            B2[o * 3 + c] = 0.25f * bb[c];
                        }
                        A2[o] = 0.25f * as;
                        if (!alpha.empty())
                            al2[o] = am;
                        if (!bar.empty())
                            bar2[o] = br2;
                    }
            });
            Buf Dt2, prot2, busy2, st2;
            std::vector<float> vw2;
            const Rect L2{FloorDiv(f.layer.l, 2), FloorDiv(f.layer.t, 2), (f.layer.r + 1) / 2, (f.layer.b + 1) / 2};
            washes(w2, h2, ds * 0.5, WcFrame{FloorDiv(f.ox, 2), FloorDiv(f.oy, 2), L2}, P2.data(), B2.data(), A2, bar2, al2, Dt2, prot2,
                   busy2, st2, vw2);
            auto up = [&](const Buf& s2, Buf& d, int nc) {
                d.alloc(N * nc);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float fx = (x + 0.5f) * 0.5f - 0.5f, fy = (y + 0.5f) * 0.5f - 0.5f;
                    for (int c = 0; c < nc; ++c)
                        d[i * nc + c] = Bilerp(s2.data(), w2, h2, nc, c, fx, fy);
                });
            };
            up(Dt2, Dt, 3);
            up(prot2, prot, 1);
            up(busy2, busy, 1);
            if (!st2.empty())
                up(st2, strokeTex, 1);
            if (!vw2.empty())
            {
                viewWc.resize(N * 3);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    for (int c = 0; c < 3; ++c)
                        viewWc[i * 3 + c] = Bilerp(vw2.data(), w2, h2, 3, c, (x + 0.5f) * 0.5f - 0.5f, (y + 0.5f) * 0.5f - 0.5f);
                });
            }
        }
        else
            washes(W, H, ds, WcFrame{f.ox, f.oy, f.layer}, P, B.data(), A, bar, alpha, Dt, prot, busy, strokeTex, viewWc);
        if (!gpuPainted)
        {
        // part 2 (full resolution): pigment tendency, ink areas, HDR excess from the full picture
        const float stk = (float)Clampd(pin.strokes / 100.0, 0, 1);
        Buf ink, tend;
        ink.alloc(N);
        tend.alloc(N);
        std::vector<float> hdr;
        bool anyHdr = false;
        for (size_t i = 0; i < N * 3 && !anyHdr; ++i)
            anyHdr = P[i] > 1.f;
        if (anyHdr)
            hdr.assign(N * 3, 0.f);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float* c = P + i * 3;
            const float mx = std::max(c[0], std::max(c[1], c[2])), mn = std::min(c[0], std::min(c[1], c[2]));
            ink[i] = painter ? 0.f : 1.f - Smooth(0.07f, 0.17f, Luma(&B[i * 3]));  // the painter lifts darks itself
            const float sat = (mx - mn) / (mx + 1e-4f);
            const float blue = Smooth(0.0f, 0.15f, c[2] - std::max(c[0], c[1]));
            const float earth = (1.f - Smooth(0.2f, 0.6f, sat)) * Smooth(0.0f, 0.10f, c[0] - c[2]);
            tend[i] = Clampf(0.2f + 0.8f * blue + 0.5f * earth, 0.f, 1.f);
            if (anyHdr)
                for (int k = 0; k < 3; ++k)
                    hdr[i * 3 + k] = std::max(0.f, c[k] - 1.f);
        });
        // (i) strokes in the pigment, granulation in the paper's valleys (pigment dependent, never on small
        //     features), the paper's relief; back to colour. Near-black ink areas keep the untouched picture.
        const float gr = (float)Clampd(pin.gran / 100.0, 0, 1), pp = (float)Clampd(pin.paper / 100.0, 0, 1);
        std::vector<float> FN;
        if (gr > 0.f)
            NoiseField(FN, W, H, f.ox, f.oy, ds, 1.7 / 2, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 1.7, 2, 0.6, seed ^ 0x6A11u, ds); });
        ForPx(W, H, T, [&](size_t i, int x, int y) {
            float d[3];
            for (int c = 0; c < 3; ++c)
                d[c] = Dt[i * 3 + c];
            const float dm = (d[0] + d[1] + d[2]) * (1.f / 3.f);
            const float h = (gr > 0.f || pp > 0.f) ? paperAt(i, x, y) : 0.5f;
            const float valley = (0.5f - h) * 2.f;
            const float capD = std::min(dm, 0.55f) * (1.f - ink[i]);
            // the painter: calmer, coarser paper (the fine grain noise reads as boiling when the picture moves under it)
            float addD = painter ? capD * (gr * tend[i] * 0.24f * valley * (1.f - 0.85f * prot[i]) + pp * 0.07f * valley)
                                 : capD * (gr * tend[i] * (0.30f * valley + 0.10f * (FN.empty() ? 0.f : FN[i])) * (1.f - 0.85f * prot[i]) + pp * 0.10f * valley);
            if (!strokeTex.empty())
                addD += std::min(dm, 0.8f) * (1.f - ink[i]) * stk * 0.10f * strokeTex[i] * (1.f - 0.6f * prot[i]) * (1.f - 0.6f * Smooth(0.04f, 0.25f, busy[i]));
            // painter: dry-brush sparkle - a light, fast wash skips the paper's peaks
            if (painter && pp > 0.f)
                addD -= dm * pp * 0.3f * Smooth(0.75f, 0.92f, h) * (1.f - Smooth(0.12f, 0.35f, dm)) * (1.f - prot[i]);
            const float shade = 1.f + pp * (painter ? 0.04f : 0.07f) * (-valley);
            for (int c = 0; c < 3; ++c)
            {
                const float share = dm > 1e-4f ? d[c] / dm : 1.f;
                float v = pm::exp(-std::max(0.f, d[c] + addD * share)) * shade * (painter ? kPaper[c] : 1.f);
                if (anyHdr)
                    v += hdr[i * 3 + c];
                v += ink[i] * (B[i * 3 + c] - v);
                P[i * 3 + c] = v;
            }
        });
#ifdef SK_GPU
        if (painter)
            StCmp(ST_PAPER, P, N * 3);
#endif
        }  // (!gpuPainted)
        if (copyP && wcOn)
            ForPx(W, H, T, [&](size_t i, int, int) {
                const float m = wc * mapOf(MT_WATERCOLOR, i);
                for (int c = 0; c < 3; ++c)
                    B[i * 3 + c] += m * (P[i * 3 + c] - B[i * 3 + c]);
            });
        tm.lap("wc paper");
        poll();
    }

    // ---- 5. ink: line controls / line style, then the lines multiplied back over the processed fill
    const bool compose = needLines && ((inpainted && anyProc) || lineOps || depthLines);
    if (compose && gpuInked)
        tm.lap("ink (gpu)");
    else if (compose)
    {
        std::vector<float> dl = delta;
        const double th = Clampd(pin.lineThick, -10, 10) * ds;
        if (th > 0.01)
        {
            const int ri = (int)std::ceil(th - 1e-6);
            const float frac = (float)(th - (ri - 1));
            std::vector<float> amax(N);
            MinMax(alpha.data(), amax.data(), W, H, ri, true, T, tmp1);
            ForPx(W, H, T, [&](size_t i, int x, int y) {
                if (amax[i] <= alpha[i] + 1e-4f)
                    return;
                float best = alpha[i], bw = 1.f;
                size_t bi = i;
                for (int dy = -ri; dy <= ri; ++dy)
                    for (int dx = -ri; dx <= ri; ++dx)
                    {
                        const int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                            continue;
                        const float dist = std::sqrt((float)(dx * dx + dy * dy));
                        if (dist > (float)ri + 0.5f)
                            continue;
                        const float wr = dist > (float)(ri - 1) + 0.5f ? frac : 1.f;
                        const size_t k = (size_t)yy * W + xx;
                        if (alpha[k] * wr > best * bw)
                        {
                            best = alpha[k];
                            bi = k;
                            bw = wr;
                        }
                    }
                if (bi != i)
                    for (int c = 0; c < 3; ++c)
                    {
                        const float v = delta[bi * 3 + c] * bw;
                        dl[i * 3 + c] = std::fabs(v) > std::fabs(dl[i * 3 + c]) ? v : dl[i * 3 + c];
                    }
            });
        }
        else if (th < -0.01)
        {
            const int ri = (int)std::ceil(-th - 1e-6);
            std::vector<float> amin(N);
            MinMax(alpha.data(), amin.data(), W, H, ri, false, T, tmp1);
            const float frac = (float)std::min(1.0, -th);
            ForPx(W, H, T, [&](size_t i, int, int) {
                if (alpha[i] <= 0.f)
                    return;
                const float s = Clampf(amin[i] / std::max(alpha[i], 1e-3f), 0.f, 1.f);
                const float k = 1.f + frac * (s - 1.f);
                for (int c = 0; c < 3; ++c)
                    dl[i * 3 + c] *= k;
            });
        }
        // line style: pencil = lighter warm grey graphite, broken here and there, catching the paper's peaks;
        // ink = a slightly varied, soft-edged black
        if (pin.lineStyle == LS_PENCIL || pin.lineStyle == LS_INK)
        {
            std::vector<float> BK;
            const bool pencil = pin.lineStyle == LS_PENCIL;
            NoiseField(BK, W, H, f.ox, f.oy, ds, (pencil ? 16.0 : 40.0) / 4, T,
                       [&](double gx, double gy) { return Fbm(gx * par, gy, pencil ? 16.0 : 40.0, 3, 0.55, seed ^ 0x9E7C1u, ds); });
            const float tint[3] = {pencil ? 0.88f : 1.f, pencil ? 1.f : 1.f, pencil ? 1.22f : 1.06f};
            ForPx(W, H, T, [&](size_t i, int x, int y) {
                const float m = (dl[i * 3] + dl[i * 3 + 1] + dl[i * 3 + 2]) * (1.f / 3.f);
                if (m <= 0.f)
                    return;
                float k;
                if (pencil)
                {
                    const float keep = Smooth(-0.75f, -0.3f, BK[i]);          // rare soft breaks (a continuous stroke)
                    const float h = paperAt(i, x, y);                          // graphite sits on the peaks
                    k = 0.55f * (0.35f + 0.65f * keep) * (0.7f + 0.6f * h);
                }
                else
                    k = 1.05f * (0.85f + 0.3f * Smooth(-0.5f, 0.5f, BK[i]));
                for (int c = 0; c < 3; ++c)
                    dl[i * 3 + c] = m * k * tint[c];
            });
        }
        const double soft = Clampd(pin.lineSoft, 0, 100) * ds * 0.5 + (pin.lineStyle == LS_INK ? 0.35 * ds : 0.0);
        if (soft > 0.05)
            RFPlain<3>(dl.data(), W, H, soft, 3, T);
        const float dk = 1.f + (float)Clampd(pin.lineDark / 100.0, -1, 2);
        ForPx(W, H, T, [&](size_t i, int, int) {
            // the painter: specks of "line" inside busy foliage are part of the wash, not pen strokes
            const float m = mapOf(MT_LINES, i) * (paintBusy.empty() ? 1.f : 1.f - 0.85f * wc * paintBusy[i]);
            // depth: the whole line density (the drawn lines too) scales with (1 + k * far); exact at the reference
            const float lf = std::max(0.f, 1.f + kLi * farAt(i));
            for (int c = 0; c < 3; ++c)
            {
                float d = dl[i * 3 + c] * dk;
                d = (delta[i * 3 + c] + m * (d - delta[i * 3 + c])) * lf;
                if (!inpainted)
                    d -= delta[i * 3 + c];
                B[i * 3 + c] = (B[i * 3 + c] + kInkEps) * pm::exp(-d) - kInkEps;
            }
        });
#ifdef SK_GPU
        StCmp(ST_INK, B.data(), N * 3);
#endif
        tm.lap("ink");
    }
    // ---- 5b. (painter) hand-painted distortion: the whole painting - washes AND lines - is resampled through a slow,
    //      layer-anchored displacement (edges wander, straight lines bow a little, shapes sit a bit off). Continuous
    //      and fixed to the layer, so it does not swim; none next to transparency (the colour stays under its alpha).
    const float warp = (float)Clampd(pin.warp / 100.0, 0, 1);
    if (warp > 0.f && wcOn && pin.wcMode == WCM_PAINT && view == V_RESULT)
    {
        std::vector<float> wx, wy, amin(N), tmpm;
        const double lw = 180.0;
        NoiseField(wx, W, H, f.ox, f.oy, ds, lw / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, lw, 2, 0.45, seed ^ 0xB01u, ds); });
        NoiseField(wy, W, H, f.ox, f.oy, ds, lw / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, lw, 2, 0.45, seed ^ 0xB02u, ds); });
        const float amp = warp * 24.f * (float)ds;
        MinMax(A.data(), amin.data(), W, H, std::max(1, (int)std::ceil(amp + 1.f)), false, T, tmpm);
        std::vector<float> Bw(B);
        ForPx(W, H, T, [&](size_t i, int x, int y) {
            const float a = amp * Smooth(0.5f, 0.99f, amin[i]);
            const float sx = Clampf(x + a * wx[i], 0.f, (float)(W - 1)), sy = Clampf(y + a * wy[i], 0.f, (float)(H - 1));
            for (int c = 0; c < 3; ++c) Bw[i * 3 + c] = Bilerp(B.data(), W, H, 3, c, sx, sy);
        });
        B.swap(Bw);
        tm.lap("warp");
    }
    // ---- 5c. depth: paleness (far toward the paper white, atmospheric perspective; negative = denser)
    if (kPa != 0.f && view == V_RESULT)
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float t = kPa * farAt(i);
            for (int c = 0; c < 3; ++c)
            {
                float& b = B[i * 3 + c];
                b = t >= 0.f ? b + 0.75f * t * (kPaper[c] - b) : b * (1.f + 0.5f * t);
            }
        });
    // ---- 6. output (straight colour, the input's alpha) / views. out may be f.rgba (in place).
    ForPx(W, H, T, [&](size_t i, int, int) {
        const float* s = f.rgba + i * 4;
        float in[4] = {s[0], s[1], s[2], s[3]};
        float* o = out + i * 4;
        if (view == V_RESULT)
        {
            const float m = Clampf(amt * mapOf(MT_ALL, i) * (1.f + kAm * farAt(i)), 0.f, 1.f);
            for (int c = 0; c < 3; ++c)
            {
                const float iv = std::isfinite(in[c]) ? in[c] : 0.f;
                const float bv = pin.linear ? ToLinear(B[i * 3 + c]) : B[i * 3 + c];
                const float v = iv + m * (bv - iv);
                o[c] = std::isfinite(v) ? std::max(v, std::min(0.f, iv)) : 0.f;
            }
            o[3] = in[3];
            return;
        }
        float r = 0, g = 0, b = 0;
        switch (view)
        {
        case V_LINES:
        {
            const float a = alpha.empty() ? 0.f : alpha[i];
            const float yl = Luma(in);
            const float base = 0.25f * (std::isfinite(yl) ? Clampf(yl, 0.f, 1.f) : 0.f);
            const bool lt = !light.empty() && light[i];
            r = base + ((lt ? 0.4f : 1.f) - base) * a;
            g = base + ((lt ? 0.9f : 1.f) - base) * a;
            b = base + ((lt ? 1.f : 0.6f) - base) * a;
            if (!bar.empty() && bar[i])
                b = std::max(b, 0.15f);
            break;
        }
        case V_FINE:
        case V_MID:
        case V_LARGE:
        {
            const float k = view == V_FINE ? 4.f : view == V_MID ? 3.f : 2.f;
            r = g = b = Clampf(0.5f + k * (bandView.empty() ? 0.f : bandView[i]), 0.f, 1.f);
            break;
        }
        case V_RIM:
        case V_STROKES:
        case V_GLAZE:
            if (!viewWc.empty())
            {
                r = viewWc[i * 3];
                g = viewWc[i * 3 + 1];
                b = viewWc[i * 3 + 2];
            }
            break;
        case V_MAP:
            r = g = b = dmap.empty() ? (f.map ? Clampf(f.map[i], 0.f, 1.f) : 0.f) : dmap[i];
            break;
        case V_DEPTHW:
            r = g = b = farAt(i);
            break;
        case V_FILL:
            r = Fl[i * 3];
            g = Fl[i * 3 + 1];
            b = Fl[i * 3 + 2];
            break;
        default: break;
        }
        o[0] = r;
        o[1] = g;
        o[2] = b;
        o[3] = 1.f;
    });
    tm.lap("output");
    return ST_OK;
}

#ifdef SK_GPU
// ---- 0.6.0.6: the start-up self-test. Small built-in frames (cel regions with soft gradients and noise, black /
// brown / red lines of 1-3 px, light lines on a dark block, foliage specks, transparent and half-transparent pixels,
// HDR values, a depth map, a paper texture, a negative layer offset) through RenderImpl with t_st set: every GPU
// stage is run and read back on its own, from the CPU's input of that stage, and compared bit for bit with the CPU.
void SynthFrame(int W, int H, int v, std::vector<float>& rgba, std::vector<float>& map, std::vector<float>& ptex)
{
    const size_t N = (size_t)W * H;
    rgba.assign(N * 4, 0.f);
    map.assign(N, 0.f);
    ptex.assign(N, 0.f);
    auto hf = [](int x, int y, uint32_t s) { return (float)(Hash((uint32_t)x * 0x9E3779B1u ^ Hash((uint32_t)y * 0x85EBCA77u ^ s)) >> 8) * (1.f / 16777216.f); };
    const float pal[6][3] = {{0.92f, 0.78f, 0.62f}, {0.35f, 0.55f, 0.85f}, {0.30f, 0.62f, 0.32f},
                             {0.85f, 0.35f, 0.30f}, {0.95f, 0.92f, 0.80f}, {0.55f, 0.40f, 0.65f}};
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            const size_t i = (size_t)y * W + x;
            float c[3];
            const int reg = ((x + 7 * v) / 38 + (y / 29) * 2 + (x + y) / 61) % 6;
            for (int k = 0; k < 3; ++k)
                c[k] = pal[reg][k] * (0.85f + 0.15f * pm::sin(0.045f * x + 0.03f * y + k)) + 0.05f * (hf(x, y, v * 3 + k) - 0.5f);
            if (x > W * 0.6 && x < W * 0.85 && y > H * 0.55 && y < H * 0.85)
            {
                const bool lt = (x % 9) == 0 || (y % 11) == 0;  // light lines on a dark block
                for (int k = 0; k < 3; ++k)
                    c[k] = lt ? 0.9f : 0.06f + 0.03f * hf(x, y, 77);
            }
            if (x < W / 4 && y > H / 2 && hf(x, y, 99) < 0.18f)  // foliage specks
                for (int k = 0; k < 3; ++k)
                    c[k] *= 0.15f;
            auto set = [&](float r, float g, float b) { c[0] = r; c[1] = g; c[2] = b; };
            if (std::fabs(x - 0.7f * y - 0.15f * W) < 1.3f)
                set(0.04f, 0.04f, 0.05f);  // a black slanted line
            if (std::abs(y - H / 3) <= 1 && x > 10 && x < W - 20)
                set(0.35f, 0.20f, 0.12f);  // brown, 3 px
            const float cx = x - W * 0.45f, cy = y - H * 0.4f;
            if (std::fabs(std::sqrt(cx * cx + cy * cy) - H * 0.22f) < 0.9f)
                set(0.55f, 0.12f, 0.10f);  // a red circle
            if (x == W / 2 + 3)
                set(0.1f, 0.1f, 0.1f);  // 1 px
            if (v == 1 && x > W * 0.1 && x < W * 0.3 && y < H * 0.25)
                for (int k = 0; k < 3; ++k)
                    c[k] *= 1.7f;  // HDR
            float a = 1.f;
            if (x >= W - 10)
                a = 0.f;
            else if (y < 6)
                a = (y + 0.5f) / 6.f;
            else if (v == 2 && hf(x, y, 5) < 0.02f)
                a = 0.5f;
            for (int k = 0; k < 3; ++k)
                rgba[i * 4 + k] = c[k];
            rgba[i * 4 + 3] = a;
            map[i] = (float)(0.5 + 0.45 * pm::sin(x / 23.0) * pm::cos(y / 17.0));
            ptex[i] = Clampf(0.5f + 0.3f * pm::sin(x * 0.3f) * pm::cos(y * 0.21f) + 0.2f * (hf(x, y, 11) - 0.5f), 0.f, 1.f);
        }
}

unsigned GpuSelfTestRun(SkGpuCtx* ctx, std::string& msg)
{
    const bool tail = skgpu_has_tail(ctx) != 0;
    const unsigned all = GU_LINES | (tail ? GU_WET | GU_REST | GU_INK : 0u);
    if (EnvIs("SKGPU_SELFTEST", "off"))
    {
        msg = "skipped (SKGPU_SELFTEST=off)";
        return all;
    }
    SelfTest st;
    st.ctx = ctx;
    t_st = &st;
    for (int k = 0; k < 3; ++k)
    {
        // 0: wet-in-wet (pencil lines: the per-pixel line noise, the paper height in f64), light lines, the depth matte
        //    (far plane in the ink), blooms, margin; 1: line-and-wash (ink lines: the line noise grid, soft lines),
        //    colour lines, HDR, a paper texture; 2: background art at Half (lines as they are, background detail
        //    lines, strong granulation / paper, linear space, pixel aspect)
        Params p = MakeStyle(k == 0 ? 7 : k == 1 ? 4 : 6);
        p.seed = 3 + k;
        int W = 200, H = 136, ox = -5, oy = 3;
        double ds = 1.0;
        if (k == 0)
        {
            p.lineColor = LC_ALL;
            p.depth = true;
            p.dLines = -70;
            p.dWet = 50;
            p.dScale = 40;
            p.depthRef = 40;
            p.depthRange = 35;
        }
        else if (k == 1)
        {
            p.lineColor = LC_COLOR;
            p.lineSoft = 1.0;
            W = 176; H = 120; ox = 7; oy = -2;
        }
        else
        {
            p.lineColor = LC_DARK;
            p.bgDetailLines = true;
            p.gran = 60;
            p.paper = 70;
            p.linear = true;
            p.par = 0.9;
            ds = 0.5;
            W = 160; H = 104; ox = 3; oy = 5;
        }
        std::vector<float> rgba, map, ptex, out;
        SynthFrame(W, H, k, rgba, map, ptex);
        out.resize(rgba.size());
        Frame f;
        f.w = W; f.h = H; f.ox = ox; f.oy = oy; f.ds = ds;
        f.rgba = rgba.data();
        f.map = k == 0 ? map.data() : nullptr;
        f.paperTex = k == 1 ? ptex.data() : nullptr;
        f.layer = Rect{1, 2, W - 3, H - 1};
        f.out = out.data();
        f.threads = 4;
        try
        {
            RenderImpl(p, f);
        }
        catch (...)
        {
            msg += " [case " + std::to_string(k) + ": exception]";
        }
    }
    t_st = nullptr;
    auto ok = [&](int s) { return st.runs[s] > 0 && st.bad[s] == 0; };
    unsigned u = 0;
    if (ok(ST_LINES))
        u |= GU_LINES;
    if (tail)
    {
        if (ok(ST_WET))
            u |= GU_WET;
        if (ok(ST_DENSITY) && ok(ST_SLIVERS) && ok(ST_PA) && ok(ST_PAPER))
            u |= GU_REST;
        if ((u & GU_REST) && ok(ST_INK))
            u |= GU_INK;
    }
    for (int s = 0; s < ST_COUNT; ++s)
    {
        msg += std::string(msg.empty() ? "" : " ") + kStName[s] + "=";
        if (st.runs[s] == 0)
            msg += (s == ST_LINES || tail) ? "not-run" : "n/a";
        else if (st.bad[s])
            msg += "FAIL(" + st.note[s] + ")";
        else
            msg += "ok";
        msg += "/" + std::to_string(st.runs[s]);
    }
    if (st.beginFail)
        msg += std::string(" begin=") + StatusName(st.beginFail);
    return u & all;
}

// the background thread: the context, then the self-test; frames use the GPU from then on
void GpuInit()
{
    const double t0 = NowMs();
    SkGpuCtx* c = nullptr;
    unsigned u = 0;
    std::string msg;
    double tc = 0.0;
    try
    {
        c = skgpu_create();
        tc = NowMs() - t0;
        if (c)
        {
            g_gpuTail = skgpu_has_tail(c) != 0;
            u = GpuSelfTestRun(c, msg);
            if (skgpu_lost(c))
            {
                u = 0;
                msg += " (device lost)";
            }
        }
    }
    catch (...)
    {
        u = 0;
        msg += " (exception)";
    }
    if (GpuLogPath())
    {
        char d[512] = "-";
        if (c)
            skgpu_describe(c, d, sizeof(d));
        char h[160];
        snprintf(h, sizeof(h), "gpu-init context_ms=%.0f selftest_ms=%.0f units=%s ", tc, NowMs() - t0 - tc, UnitStr(u).c_str());
        GpuLogLine(std::string(h) + (c ? "selftest: " + msg + " | adapter: " + d : std::string("no usable GPU (CPU only)")));
    }
    g_gpu = c;
    g_gpuUnits = u;
    g_gpuState.store(c ? 2 : 3, std::memory_order_release);
}
#endif

}  // namespace

// 0.6.0.3: the round-6 engine (sk_r6.cpp)
bool UseRound6(const Params& p);
bool IsIdentityRound6(const Params& p);
Status RenderRound6(const Params& p, Frame& f);

bool IsIdentity(const Params& p)
{
    if (UseRound6(p))
        return IsIdentityRound6(p);
    if (!(p.amount > 0))
        return true;
    if (p.view != V_RESULT)
        return false;
    const bool tex = p.fine != 0 || p.mid != 0 || p.large != 0;
    const bool blur = p.fillBlur > 0;
    const bool wc = p.wc > 0;
    const bool lineOps = p.lineThick != 0 || p.lineDark != 0 || p.lineSoft > 0 || p.lineStyle != LS_ASIS;
    const bool depthOps = p.depth && p.dMaster > 0 && (p.dBlur > 0 || p.dPale != 0);
    return !tex && !blur && !wc && !lineOps && !depthOps;
}

static Status RenderGuarded(const Params& p, Frame& f)
{
    if (UseRound6(p))
        return RenderRound6(p, f);
    try
    {
        return RenderImpl(p, f);
    }
    catch (const AbortEx&)
    {
        return ST_ABORTED;
    }
    catch (const std::bad_alloc&)
    {
        return ST_OOM;
    }
    catch (...)
    {
        return ST_ERROR;
    }
}

Status Render(const Params& p, Frame& f)
{
#ifdef SK_GPU
    if (GpuLogPath() && !t_st)
    {
        // 0.6.0.6: SHITSUKAN_GPU_LOG - one line per frame (the picture is not touched)
        GpuLogFrame lg;
        t_log = &lg;
        const double t0 = NowMs();
        const Status s = RenderGuarded(p, f);
        const double ms = NowMs() - t0;
        t_log = nullptr;
        if (UseRound6(p))
            lg.why = "r6";
        const char* path = lg.elig == 0 ? "cpu" : lg.used == lg.elig ? "gpu" : lg.used ? "partial" : "cpu";
        const int gs = g_gpuState.load();
        char b[512];
        snprintf(b, sizeof(b), "frame %dx%d ds=%.4g elig=%s gpu=%s path=%s reason=%s wait_ms=%.1f render_ms=%.1f status=%d gpu_state=%s units=%s",
                 f.w, f.h, f.ds, UnitStr(lg.elig).c_str(), UnitStr(lg.used).c_str(), path,
                 lg.why.empty() ? (lg.elig ? "-" : "no-gpu-stage") : lg.why.c_str(), lg.waitMs, ms, (int)s,
                 gs == 0 ? "idle" : gs == 1 ? "selftest" : gs == 2 ? "ready" : "none", gs == 2 ? UnitStr(g_gpuUnits.load()).c_str() : "-");
        GpuLogLine(b);
        return s;
    }
#endif
    return RenderGuarded(p, f);
}

void GpuShutdown()
{
#ifdef SK_GPU
    // (the self-test thread runs code of this module: wait for it before the module may go away)
    for (int i = 0; i < 20000 && g_gpuState.load() == 1; ++i)
        skplat::SleepMs(1);
#endif
}

}  // namespace sk
