// Optic Bokeh v2.4: GPU blur stage wrapper (see ob_gpu.h).
#include "ob_gpu.h"
#if defined(OB_OLD_ENGINE) || defined(OB_NO_GPU)
// timing builds against the v2.2.1 engine copy (no describeBlur), and (mac port) builds without the Rust GPU library
// (CMake -DOB_WITH_GPU=OFF): the CPU-only stubs
namespace obgpuw {
void Start() {}
void Shutdown() {}
bool Eligible(const lenscore::Params&, int, bool, int, int, int) { return false; }
int Blur(const lenscore::Prepared&, const lenscore::Params&, bool, float*, int, float*, lenscore::AbortFn, void*) { return 0; }
std::string StatusLine() { return "GPU: not in this build"; }
long long GpuFrames() { return 0; }
long long FallbackFrames() { return 0; }
int LastStatus() { return 0; }
int TestInitSync(int, unsigned long long, unsigned, int, int) { return 3; }
int TestHook(int, unsigned long long) { return 3; }
void TestReset() {}
int Frame(const float*, int, int, const float*, const lenscore::Params&, int, int, int, int, const lenscore::Params&, float*, int,
          std::vector<float>&, lenscore::Prepared&, lenscore::Params&, double& wu, lenscore::AbortFn, void*) { wu = 0; return 0; }
void ReportCpu(double, double) {}
std::string CountersLine() { return "no counters"; }
long long PostJobs() { return 0; }
long long La34Jobs() { return 0; }
void TestLa34(bool) {}
void TestRouting(bool) {}
void TestGpuPost(bool) {}
double TestLastWorkUnits() { return 0; }
long long BusyFrames() { return 0; }
long long CpuRoutedFrames() { return 0; }
long long FrameJobs() { return 0; }
void TestFramePath(bool) {}
int TestSelftest(unsigned long long) { return 3; }
std::string SelftestLine() { return "self-test not in this build"; }
bool FrameLogOn() { return false; }
void FrameLogBegin() {}
void FrameLogEnd(int, int, int, int, double, int, const char*) {}
std::string FrameLogPath() { return std::string(); }
}  // namespace obgpuw
#else
#include "ob_gpu_desc.h"
#include "obgpu.h"
#include "../cpp/obgpu_desc.h"  // build\gpu_snap\cpp (on the include path as build\gpu_snap\include\..): obgpu_prepare_desc_build
#include "../platform/ob_platform.h"  // mac port: getenv / fopen (relative to this file: src/plugin/..)
#ifndef OB_MAC_GPU_DEFAULT_ON
#define OB_MAC_GPU_DEFAULT_ON 0  // mac port: GPU opt-in on macOS (see ReadEnv); no effect on Windows
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace obgpuw {
namespace {

enum State { ST_IDLE = 0, ST_INIT, ST_READY, ST_UNAVAILABLE, ST_DISABLED, ST_OFF_ENV, ST_SHUTDOWN };

std::mutex g_m;                          // guards g_ctx / g_thread / g_losses / g_adapter
obgpu_context* g_ctx = nullptr;
std::thread g_thread;
std::atomic<int> g_state{ST_IDLE};
std::atomic<int> g_env{-1};              // OPTICBOKEH_GPU: -1 unset, 0 off, 1 force
std::atomic<int> g_lastStatus{OBGPU_OK};
std::atomic<long long> g_gpuFrames{0}, g_fallbacks{0}, g_frameJobs{0};
std::atomic<bool> g_framePath{true};     // ABI 1.1 obgpu_render_frame (tests / timing can switch it off)
std::atomic<bool> g_routing{true};       // ABI 1.2 obgpu_route / obgpu_report_cpu (tests switch it off: deterministic paths)
std::atomic<bool> g_gpuPost{true};       // ABI 1.3 obgpu_render_frame_post
std::atomic<long long> g_postJobs{0}, g_busy{0}, g_cpuRouted{0};
std::atomic<double> g_lastWu{0};
std::atomic<bool> g_la34{true};           // ABI 1.4 LA modes 3 / 4 on the GPU (tests can switch it off)
std::atomic<long long> g_la34Jobs{0};
obgpu_selftest_report g_selftest = {};   // under g_m
int g_selftestStatus = -1;               // -1 not run, else the obgpu status
std::deque<std::chrono::steady_clock::time_point> g_losses;
std::string g_adapter;
std::once_flag g_envOnce;
std::atomic<unsigned long long> g_envVramBytes{0};  // OPTICBOKEH_GPU_VRAM_MB (aetest)
std::string g_statsPath;                 // OPTICBOKEH_GPU_STATS (aetest): Shutdown() writes the counters there
// v2.4.2 OPTICBOKEH_GPU_LOG=<file>: one line per frame (FrameLogEnd). Off (empty) = nothing recorded.
std::string g_logPath;
std::atomic<bool> g_logOn{false};
std::mutex g_logM;
// what the GPU did for the frame this render thread is running (only filled while the log is on)
struct FrameNote
{
    int frame = -9, frameStatus = -1;  // Frame(): its return value (-9 = not called) and the obgpu status
    int blur = -9, blurStatus = -1;    // Blur(): the same
    int cw = 0, ch = 0;
    double wu = 0;
    bool haveJob = false;
    obgpu_job_info job;
};
thread_local FrameNote t_note;

// a still-joinable init thread at static destruction (host never sent GLOBAL_SETDOWN) would call
// std::terminate: detach it then; the device is leaked at process exit on purpose (no driver teardown)
struct ThreadGuard
{
    ~ThreadGuard()
    {
        if (g_thread.joinable())
            g_thread.detach();
    }
} g_guard;

void ReadEnv()
{
    std::call_once(g_envOnce, [] {
        // mac port: obp::getenv (Windows: the same _dupenv_s call as v2.4.2; macOS: getenv)
        std::string v;
#if defined(__APPLE__) && !OB_MAC_GPU_DEFAULT_ON
        // mac port: the GPU is OFF on macOS unless the user opts in, until the Metal self-test and the accuracy gate
        // (GPU within 1 8-bit level of the CPU, tests/run_offline_suite.py + the AE GPU checks) have passed on a Mac.
        //   unset / 0 -> off (no GPU initialisation at all);  1 -> on, the "Use GPU" popup decides (= unset on Windows);
        //   2 -> forced on (= OPTICBOKEH_GPU=1 on Windows: also when the popup is Off).
        // Shipping switch: build with -DOB_MAC_GPU_DEFAULT_ON=ON (CMake) to get the Windows behaviour back.
        if (!obp::getenv("OPTICBOKEH_GPU", v))
            g_env.store(0);
        else
            g_env.store(v.c_str()[0] == '1' ? -1 : v.c_str()[0] == '2' ? 1 : 0);
#else
        if (obp::getenv("OPTICBOKEH_GPU", v))
            g_env.store(v.c_str()[0] == '0' ? 0 : 1);
#endif
        if (obp::getenv("OPTICBOKEH_GPU_STATS", v))
            g_statsPath = v;
        // aetest only (documented in aetest\run_host.ps1): OPTICBOKEH_GPU_ROUTE=0 = no routing (every eligible frame
        // tries the GPU: deterministic paths), OPTICBOKEH_GPU_VRAM_MB=<n> = the library's total VRAM budget (a tiny
        // budget forces the out-of-VRAM fallback)
        if (obp::getenv("OPTICBOKEH_GPU_ROUTE", v) && v.c_str()[0] == '0')
            g_routing.store(false);
        if (obp::getenv("OPTICBOKEH_GPU_LOG", v) && !v.empty())
        {
            g_logPath = v;
            g_logOn.store(true);
        }
        if (obp::getenv("OPTICBOKEH_GPU_VRAM_MB", v))
        {
            const long long mb = std::atoll(v.c_str());
            if (mb > 0)
                g_envVramBytes.store((unsigned long long)mb << 20);
        }
    });
}

obgpu_config Config(int forceNoAdapter, unsigned long long vram, unsigned timeoutMs, int maxJobs, int allowSoftware)
{
    obgpu_config c;
    std::memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.backends = 0;              // 0 = the library's platform default: Windows Vulkan preferred, DX12 fallback; macOS Metal (gpu/src/gpu.rs)
    c.allow_software = allowSoftware;
    c.force_no_adapter = forceNoAdapter;
    c.vram_budget_bytes = vram;  // 0 = library default (1.5 GB, capped by the free budget)
    c.timeout_ms = timeoutMs;    // 0 = 4 s
    // v2.4.2 (obgpu ABI 1.5.2): 0 = the library's 15 s upper bound; a waiting frame goes to the CPU as soon as the
    // GPU is no longer expected to finish it first (v2.4.1 passed 2000: on an idle GPU 4 of 8 UHD MFR frames waited
    // 2 s behind the two job slots and were rendered on the CPU, ~10 x slower than on the GPU)
    c.wait_ms = 0;
    c.max_jobs = maxJobs;        // 0 = 2
    c.verbose = 0;
    return c;
}

void InitBody(obgpu_config cfg)
{
    obgpu_status st = OBGPU_OK;
    obgpu_context* ctx = obgpu_init(&cfg, &st);
    if (!ctx)
    {
        g_lastStatus.store(st);
        g_state.store(ST_UNAVAILABLE);
        return;
    }
    // ABI 1.1: the adapter self-test (embedded tiny cases vs stored CPU results, 0.1-0.3 s on Vulkan) before any
    // render may use the device; a failure disables the context (every call returns SELFTEST_FAILED -> CPU)
    obgpu_selftest_report rep;
    std::memset(&rep, 0, sizeof rep);
    rep.struct_size = sizeof rep;
    const obgpu_status sst = obgpu_selftest(ctx, &rep);
    {
        std::lock_guard<std::mutex> lk(g_m);
        g_selftest = rep;
        g_selftestStatus = sst;
    }
    // aetest: OPTICBOKEH_GPU_VRAM_MB = the exact total budget (the config value is clamped to >= 0.75 GB by the
    // library; the test hook is not), so a tiny value forces the out-of-VRAM fallback on every job
    if (g_envVramBytes.load() > 0)
        obgpu_test_hook(ctx, 2, g_envVramBytes.load());
    // compile the common pipeline variants before any render may use the device (DX12 / FXC: ~1 min)
    const obgpu_status w = sst == OBGPU_OK ? obgpu_warmup(ctx) : sst;
    char buf[256] = {0};
    obgpu_adapter_info(ctx, buf, sizeof buf - 1);
    {
        std::lock_guard<std::mutex> lk(g_m);
        g_ctx = ctx;
        // About has 256 bytes for everything: "NVIDIA GeForce RTX 4070 SUPER (Vulkan, DiscreteGpu, driver ...)"
        // -> "NVIDIA GeForce RTX 4070 SUPER (Vulkan)"
        std::string a = buf;
        const size_t paren = a.find(" (");
        const size_t comma = paren == std::string::npos ? std::string::npos : a.find(", ", paren);
        if (comma != std::string::npos)
            a = a.substr(0, comma) + ")";
        g_adapter = a;
    }
    if (g_state.load() == ST_SHUTDOWN)
        return;  // shut down while warming up: Shutdown() could not release it; the thread owner leaks it
    g_lastStatus.store(w);
    g_state.store(w == OBGPU_OK ? ST_READY : ST_UNAVAILABLE);
}

inline double Clamp(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

struct AbortThunk
{
    lenscore::AbortFn fn;
    void* user;
};
int AbortTrampoline(void* u)
{
    const AbortThunk* a = static_cast<const AbortThunk*>(u);
    return a->fn && a->fn(a->user) ? 1 : 0;
}

void NoteLoss()
{
    std::lock_guard<std::mutex> lk(g_m);
    const auto now = std::chrono::steady_clock::now();
    g_losses.push_back(now);
    while (!g_losses.empty() && now - g_losses.front() > std::chrono::seconds(60))
        g_losses.pop_front();
    if (g_losses.size() >= 3)
        g_state.store(ST_DISABLED);  // three device losses / timeouts within a minute: CPU for the session
}

}  // namespace

void Start()
{
    ReadEnv();
    if (g_env.load() == 0)
    {
        g_state.store(ST_OFF_ENV);
        return;
    }
    int expected = ST_IDLE;
    if (!g_state.compare_exchange_strong(expected, ST_INIT))
        return;
    std::lock_guard<std::mutex> lk(g_m);
    try
    {
        g_thread = std::thread(InitBody, Config(0, 0, 0, 0, 0));
    }
    catch (...)
    {
        g_state.store(ST_UNAVAILABLE);  // no thread: CPU only
    }
}

static void WriteStats()
{
    if (g_statsPath.empty())
        return;
    FILE* f = obp::fopen(g_statsPath.c_str(), "ab");
    if (!f)
        return;
    std::fprintf(f, "%s|gpu %lld|fallback %lld|last %d|%s\r\n", StatusLine().c_str(), g_gpuFrames.load(), g_fallbacks.load(),
                 g_lastStatus.load(), CountersLine().c_str());
    std::fclose(f);
}

void Shutdown()
{
    ReadEnv();
    WriteStats();  // tests (OPTICBOKEH_GPU_STATS): one line per process, appended
    const int prev = g_state.exchange(ST_SHUTDOWN);
    std::thread t;
    {
        std::lock_guard<std::mutex> lk(g_m);
        t.swap(g_thread);
    }
    if (t.joinable())
    {
        if (prev == ST_INIT)
            t.detach();  // still compiling pipelines (DX12 can take a minute): do not block AE's shutdown
        else
            t.join();
    }
    obgpu_context* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_m);
        if (prev != ST_INIT)
        {
            ctx = g_ctx;
            g_ctx = nullptr;
        }
    }
    if (ctx)
        obgpu_shutdown(ctx);  // waits <= 2 s for in-flight work
}

bool Eligible(const lenscore::Params& p, int renderVersion, bool bestQuality, int popup, int w, int h)
{
    if (renderVersion < 2 || bestQuality || g_state.load() != ST_READY)
        return false;
    const int env = g_env.load();
    if (env == 0 || (popup == USE_OFF && env != 1))
        return false;
    if (!(p.maxRadius >= 8.f) || (long long)w * h < 250000)
        return false;
    // a plain disc (no fringe, no cat eye, legacy profile) is already cheap on the CPU at large radii
    const bool fringe = std::fabs(Clamp(p.chromaFringe, -1, 1)) > 5e-4;
    const bool cat = Clamp(p.catEye, 0, 1) > 1e-3;
    const bool general = Clamp(p.edgeSoftness, 0, 1) > 1e-4 || Clamp(p.apodization, 0, 1) > 1e-4 ||
                         std::fabs(Clamp(p.coreWeight, -1, 1)) > 1e-4 || Clamp(p.obstruction, 0, 0.9) > 1e-4 ||
                         Clamp(p.onionRings, 0, 1) > 1e-4;
    const bool plainDisc = !fringe && !cat && !general;
    return !(plainDisc && p.maxRadius > 150.f);
}

static void NoteJob(obgpu_context* ctx, bool frame, int gs)
{
    // (log on only) the library's record of this thread's job: queue wait, run time, why BUSY
    FrameNote& n = t_note;
    (frame ? n.frameStatus : n.blurStatus) = gs;
    std::memset(&n.job, 0, sizeof n.job);
    n.job.struct_size = sizeof n.job;
    n.haveJob = obgpu_last_job(ctx, &n.job) == OBGPU_OK;
}

static int BlurBody(const lenscore::Prepared& prep, const lenscore::Params& p, bool needEL, float* dst, int dstStride, float* eLOut,
                    lenscore::AbortFn abortFn, void* user)
{
    obgpu_context* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_m);
        ctx = g_state.load() == ST_READY ? g_ctx : nullptr;
    }
    if (!ctx)
        return 0;
    try
    {
        Desc D;
        lenscore::Status st = lenscore::Status::Ok;
        if (!BuildDesc(p, prep.w, prep.h, needEL, D, abortFn, user, st))
        {
            if (st == lenscore::Status::Aborted)
                return -1;
            g_fallbacks.fetch_add(1);
            return 0;
        }
        obgpu_input in;
        in.struct_size = sizeof in;
        in.rgba = prep.rgba.data();
        in.radius = prep.radius.data();
        in.side = reinterpret_cast<const int8_t*>(prep.side.data());
        in.el = prep.eL.data();
        obgpu_output out;
        out.struct_size = sizeof out;
        out.rgba = dst;
        out.rgba_stride = dstStride;
        out.el = needEL ? eLOut : nullptr;
        AbortThunk at = {abortFn, user};
        const obgpu_status gs = obgpu_blur(ctx, &in, &D.d, &out, AbortTrampoline, &at, nullptr);
        g_lastStatus.store(gs);
        if (g_logOn.load(std::memory_order_relaxed))
            NoteJob(ctx, false, gs);
        if (gs == OBGPU_OK)
        {
            g_gpuFrames.fetch_add(1);
            return 1;
        }
        if (gs == OBGPU_ABORTED)
            return -1;
        if (gs == OBGPU_DEVICE_LOST)
            NoteLoss();  // (ABI 1.3: TIMEOUT / BUSY are congestion: the device is kept, the frame uses the CPU blur)
        else if (gs == OBGPU_BUSY)
            g_busy.fetch_add(1);
        else if (gs == OBGPU_NO_ADAPTER || gs == OBGPU_SELFTEST_FAILED)
            g_state.store(ST_UNAVAILABLE);
        g_fallbacks.fetch_add(1);
        return 0;  // the CPU blur runs for this frame (nothing of dst / eLOut is relied upon)
    }
    catch (...)
    {
        g_fallbacks.fetch_add(1);
        return 0;
    }
}

static int FrameBody(const float* src, int w, int h, const float* map, const lenscore::Params& p, int cx, int cy, int cw, int ch,
                     const lenscore::Params& pq, float* dst, int dstStride, std::vector<float>& eLc, lenscore::Prepared& pc,
                     lenscore::Params& prest, double& workUnits, lenscore::AbortFn abortFn, void* user)
{
    workUnits = 0;
    if (!g_framePath.load())
        return 0;
    obgpu_context* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_m);
        ctx = g_state.load() == ST_READY ? g_ctx : nullptr;
    }
    if (!ctx)
        return 0;
    try
    {
        obgpu_prepare_desc pd;
        std::memset(&pd, 0, sizeof pd);
        if (obgpuPrepareDescBuild(p, w, h, map != nullptr, &pd) != 0)
            return 0;  // renderVersion 1 / highlight layer source: CPU prepare (+ the phase-1 GPU blur)
        const bool needEL = pq.highlightRolloff > 1e-4f;  // upper bound: rollRelevant decides after the frame
        Desc D;
        lenscore::Status st = lenscore::Status::Ok;
        if (!BuildDesc(pq, cw, ch, needEL, D, abortFn, user, st))
            return st == lenscore::Status::Aborted ? -1 : 0;
        // ABI 1.2 routing: on a GPU that other applications saturate the frame may be faster on the CPU
        workUnits = obgpu_work_units(&D.d);
        g_lastWu.store(workUnits);
        if (g_routing.load() && !obgpu_route(ctx, workUnits))
        {
            g_cpuRouted.fetch_add(1);
            return 3;
        }
        // ABI 1.3: a prefix of postProcess on the GPU (glow + vignetting -> halation -> LA 0-2 -> rolloff)
        obgpu_post_desc post;
        std::memset(&post, 0, sizeof post);
        // ABI 1.4: also luminance aberration modes 3 / 4 (Contrast Rim v2.1, Channel Drift); only grain stays on the CPU
        const bool gpuPost = g_gpuPost.load() && obgpuPostDescBuildEx(pq, cw, ch, g_la34.load() ? OBGPU_POST_REQ_LA34 : 0u, &post) == 0;
        const bool glow = pq.glow > 1e-4f;  // the CPU glow (fallback) reads Prepared::rgba / eL
        std::vector<float> prgba(glow ? (size_t)cw * ch * 4 : 0), pel(glow ? (size_t)cw * ch : 0);
        eLc.assign(needEL ? (size_t)cw * ch : 0, 0.f);
        obgpu_frame_input fi;
        std::memset(&fi, 0, sizeof fi);
        fi.struct_size = sizeof fi;
        fi.src = src;
        fi.src_stride = w * 4;
        fi.map = map;
        fi.map_stride = w;
        fi.crop_x = cx;
        fi.crop_y = cy;
        fi.crop_w = cw;
        fi.crop_h = ch;
        obgpu_frame_output fo;
        std::memset(&fo, 0, sizeof fo);
        fo.struct_size = sizeof fo;
        fo.rgba = dst;
        fo.rgba_stride = dstStride;
        fo.el = needEL ? eLc.data() : nullptr;
        fo.prepared_rgba = glow ? prgba.data() : nullptr;
        fo.prepared_el = glow ? pel.data() : nullptr;
        AbortThunk at = {abortFn, user};
        uint32_t done = 0;
        const int gs = gpuPost ? obgpu_render_frame_post(ctx, &fi, &pd, &D.d, &post, &fo, AbortTrampoline, &at, nullptr, &done)
                               : obgpu_render_frame(ctx, &fi, &pd, &D.d, &fo, AbortTrampoline, &at, nullptr);
        g_lastStatus.store(gs);
        if (g_logOn.load(std::memory_order_relaxed))
            NoteJob(ctx, true, gs);
        if (gs == OBGPU_OK)
        {
            prest = pq;
            if (done)
                obgpuPostRestParams(prest, done);  // the CPU postProcess runs only what the GPU did not
            pc = lenscore::Prepared();
            pc.w = cw;
            pc.h = ch;
            pc.rollRelevant = fo.roll_relevant != 0;
            if (prest.glow > 0)
            {
                pc.rgba.swap(prgba);
                pc.eL.swap(pel);
            }
            g_gpuFrames.fetch_add(1);
            g_frameJobs.fetch_add(1);
            if (done)
                g_postJobs.fetch_add(1);
            if (done & OBGPU_POST_LA34)
                g_la34Jobs.fetch_add(1);
            return 1;
        }
        eLc.clear();
        if (gs == OBGPU_ABORTED)
            return -1;
        if (gs == OBGPU_UNSUPPORTED)
            return 0;  // not a failure: the CPU prepare + phase-1 blur
        if (gs == OBGPU_BUSY)
        {
            g_busy.fetch_add(1);  // routing deadline / no job slot: this frame on the CPU, not an error
            return 2;
        }
        if (gs == OBGPU_DEVICE_LOST)
            NoteLoss();  // (ABI 1.3: a timeout is congestion, the device is kept: not counted as a loss)
        else if (gs == OBGPU_NO_ADAPTER || gs == OBGPU_SELFTEST_FAILED)
            g_state.store(ST_UNAVAILABLE);
        g_fallbacks.fetch_add(1);
        return 2;  // a GPU failure: this frame runs entirely on the CPU (no phase-1 retry on a device just lost)
    }
    catch (...)
    {
        eLc.clear();
        g_fallbacks.fetch_add(1);
        return 2;
    }
}

int Blur(const lenscore::Prepared& prep, const lenscore::Params& p, bool needEL, float* dst, int dstStride, float* eLOut,
         lenscore::AbortFn abortFn, void* user)
{
    const int r = BlurBody(prep, p, needEL, dst, dstStride, eLOut, abortFn, user);
    if (g_logOn.load(std::memory_order_relaxed))
    {
        t_note.blur = r;
        if (t_note.cw == 0)
        {
            t_note.cw = prep.w;
            t_note.ch = prep.h;
        }
    }
    return r;
}

int Frame(const float* src, int w, int h, const float* map, const lenscore::Params& p, int cx, int cy, int cw, int ch,
          const lenscore::Params& pq, float* dst, int dstStride, std::vector<float>& eLc, lenscore::Prepared& pc,
          lenscore::Params& prest, double& workUnits, lenscore::AbortFn abortFn, void* user)
{
    const int r = FrameBody(src, w, h, map, p, cx, cy, cw, ch, pq, dst, dstStride, eLc, pc, prest, workUnits, abortFn, user);
    if (g_logOn.load(std::memory_order_relaxed))
    {
        t_note.frame = r;
        t_note.cw = cw;
        t_note.ch = ch;
        t_note.wu = workUnits;
    }
    return r;
}

void ReportCpu(double workUnits, double ms)
{
    obgpu_context* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_m);
        ctx = g_state.load() == ST_READY ? g_ctx : nullptr;
    }
    if (ctx && workUnits > 0 && ms >= 0 && g_routing.load())
        obgpu_report_cpu(ctx, workUnits, ms);
}

std::string CountersLine()
{
    obgpu_counters c;
    std::memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    {
        std::lock_guard<std::mutex> lk(g_m);
        if (!g_ctx || obgpu_get_counters(g_ctx, &c) != OBGPU_OK)
            return "no counters";
    }
    char b[600];
    std::snprintf(b, sizeof b,
                  "route %s, gpu jobs %llu, failed %llu, cpu routed %llu, probes %llu, cpu reports %llu, ema gpu %.3g / cpu %.3g ms/wu, "
                  "last wall %.1f ms busy %.1f ms, busy fraction %.2f, seed %.3g ms/wu, deadline aborts %llu, timeouts %llu, prep bands %d (x%.2f), vram x%.2f, "
                  "slot-wait busy %llu (limit %llu, queue %llu, saturated %llu); "
                  "wrapper: frame jobs %lld, post on GPU %lld (LA 3/4 %lld), busy %lld, cpu routed %lld",
                  c.route_gpu ? "GPU" : "CPU", (unsigned long long)c.gpu_jobs, (unsigned long long)c.gpu_failed, (unsigned long long)c.cpu_routed,
                  (unsigned long long)c.probes, (unsigned long long)c.cpu_reports, c.ema_gpu_ms_per_wu, c.ema_cpu_ms_per_wu, c.last_wall_ms,
                  c.last_gpu_busy_ms, c.ema_busy_fraction, c.seed_ms_per_wu, (unsigned long long)c.deadline_aborts, (unsigned long long)c.timeouts, c.last_prep_bands,
                  c.last_prep_rows_factor, c.vram_scale, (unsigned long long)(c.busy_wait_limit + c.busy_queue + c.busy_saturated),
                  (unsigned long long)c.busy_wait_limit, (unsigned long long)c.busy_queue, (unsigned long long)c.busy_saturated, g_frameJobs.load(),
                  g_postJobs.load(), g_la34Jobs.load(), g_busy.load(), g_cpuRouted.load());
    return b;
}

// ---- v2.4.2 per-frame log (OPTICBOKEH_GPU_LOG) ----
bool FrameLogOn()
{
    return g_logOn.load(std::memory_order_relaxed);
}
std::string FrameLogPath()
{
    return g_logPath;
}
void FrameLogBegin()
{
    if (g_logOn.load(std::memory_order_relaxed))
        t_note = FrameNote();
}
static const char* BusyReasonName(int r)
{
    switch (r)
    {
    case OBGPU_BUSY_WAIT_LIMIT: return "busy-wait-limit";
    case OBGPU_BUSY_QUEUE: return "busy-queue";
    case OBGPU_BUSY_SATURATED: return "busy-saturated";
    case OBGPU_BUSY_DEADLINE: return "busy-deadline";
    case OBGPU_BUSY_COLD_DEADLINE: return "busy-cold-deadline";
    default: return "busy";
    }
}
static void Token(std::string& s)
{
    for (char& ch : s)
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == '(' || ch == ')' || ch == ',')
            ch = '-';
}
void FrameLogEnd(int w, int h, int cw, int ch, double ms, int err, const char* why)
{
    if (!g_logOn.load(std::memory_order_relaxed))
        return;
    const FrameNote n = t_note;
    const char* path = n.frame == 1 ? "GPU" : (n.blur == 1 ? "GPU-blur" : "CPU");
    std::string reason = "-";
    if (n.frame != 1)
    {
        // the first thing that kept this frame off the GPU frame path
        auto status = [&](int gs) -> std::string {
            if (gs == OBGPU_BUSY)
                return BusyReasonName(n.haveJob ? n.job.busy_reason : 0);
            std::string s = obgpu_status_string((obgpu_status)gs);
            Token(s);
            return s;
        };
        if (n.frame == 3)
            reason = "routed-to-cpu";
        else if (n.frame == 2)
            reason = n.frameStatus >= 0 ? status(n.frameStatus) : std::string("gpu-error");
        else if (n.frame == 0 && n.blur == 1)
            reason = "-";  // (frame path not for this frame: CPU prepare + GPU blur)
        else if (n.blur == 0 && n.blurStatus >= 0)
            reason = status(n.blurStatus);
        else if (why)
            reason = why;
        else
        {
            const int st = g_state.load();
            reason = st == ST_READY ? "not-eligible" : st == ST_OFF_ENV ? "gpu-off-env" : st == ST_INIT ? "gpu-starting"
                   : st == ST_DISABLED ? "gpu-disabled" : st == ST_UNAVAILABLE ? "gpu-unavailable" : "gpu-not-started";
        }
        if (n.frame == 0 && n.blur != 1 && reason == "-")
            reason = "frame-path-not-used";
    }
    const bool job = n.haveJob;
    char line[640];
    const auto now = std::chrono::system_clock::now();
    const long long msEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::snprintf(line, sizeof line,
                  "t=%lld.%03lld tid=%u size=%dx%d crop=%dx%d wu=%.2f path=%s reason=%s wait_ms=%.1f gpu_ms=%.1f render_ms=%.1f ahead=%d "
                  "bands=%d budget_mb=%llu deadline_ms=%.0f%s est_gpu_ms=%.0f est_cpu_ms=%.0f err=%d\r\n",
                  msEpoch / 1000, msEpoch % 1000, (unsigned)std::hash<std::thread::id>()(std::this_thread::get_id()) % 100000u, w, h,
                  n.cw > 0 ? n.cw : cw, n.cw > 0 ? n.ch : ch, job ? n.job.work_units : n.wu, path, reason.c_str(),
                  job ? n.job.slot_wait_ms : 0.0, job ? n.job.run_ms : 0.0, ms, job ? n.job.ahead : 0, job ? n.job.prep_bands : 0,
                  job ? (unsigned long long)(n.job.budget_bytes >> 20) : 0ull, job ? n.job.deadline_ms : -1.0,
                  job && n.job.deadline_ms == n.job.deadline_ms ? (n.job.deadline_own ? "(own)" : "(call)") : "",
                  job ? n.job.est_gpu_ms : -1.0, job ? n.job.est_cpu_ms : -1.0, err);
    std::lock_guard<std::mutex> lk(g_logM);
    FILE* f = obp::fopen(g_logPath.c_str(), "ab");
    if (f)
    {
        std::fputs(line, f);
        std::fclose(f);
    }
}

long long PostJobs()
{
    return g_postJobs.load();
}
long long La34Jobs()
{
    return g_la34Jobs.load();
}
void TestLa34(bool on)
{
    g_la34.store(on);
}
void TestRouting(bool on)
{
    g_routing.store(on);
}
void TestGpuPost(bool on)
{
    g_gpuPost.store(on);
}
double TestLastWorkUnits()
{
    return g_lastWu.load();
}
long long BusyFrames()
{
    return g_busy.load();
}
long long CpuRoutedFrames()
{
    return g_cpuRouted.load();
}
long long FrameJobs()
{
    return g_frameJobs.load();
}
void TestFramePath(bool on)
{
    g_framePath.store(on);
}
int TestSelftest(unsigned long long ppm)
{
    std::lock_guard<std::mutex> lk(g_m);
    if (!g_ctx)
        return OBGPU_NO_ADAPTER;
    obgpu_test_hook(g_ctx, 6, ppm);
    obgpu_selftest_report rep;
    std::memset(&rep, 0, sizeof rep);
    rep.struct_size = sizeof rep;
    const obgpu_status s = obgpu_selftest(g_ctx, &rep);
    g_selftest = rep;
    g_selftestStatus = s;
    if (s != OBGPU_OK)
    {
        g_lastStatus.store(OBGPU_SELFTEST_FAILED);
        g_state.store(ST_UNAVAILABLE);
    }
    return s;
}
std::string SelftestLine()
{
    std::lock_guard<std::mutex> lk(g_m);
    if (g_selftestStatus < 0)
        return "self-test not run";
    char b[160];
    std::snprintf(b, sizeof b, "self-test %s: %d cases, max %.2g, mean %.2g, deterministic %d, %.0f ms (status %d), router seed %.3g ms/wu",
                  g_selftest.pass ? "passed" : "FAILED", g_selftest.cases, g_selftest.max_rel, g_selftest.mean_abs,
                  g_selftest.deterministic, g_selftest.ms, g_selftestStatus, g_selftest.seed_ms_per_wu);
    return b;
}
std::string StatusLine()
{
    ReadEnv();
    std::string a;
    {
        std::lock_guard<std::mutex> lk(g_m);
        a = g_adapter;
    }
    switch (g_state.load())
    {
    case ST_OFF_ENV: return "GPU: off (OPTICBOKEH_GPU=0)";
    case ST_INIT: return "GPU: starting (CPU until ready)";
    case ST_READY: return "GPU active (Fast / Standard): " + a;
    case ST_DISABLED: return "GPU off after repeated errors: " + a;
    case ST_UNAVAILABLE:
        if (g_lastStatus.load() == OBGPU_SELFTEST_FAILED)
            return "GPU off (failed its self-test): " + a;
        return std::string("GPU: not available (") + obgpu_status_string((obgpu_status)g_lastStatus.load()) + ") - CPU";
    case ST_SHUTDOWN: return "GPU: shut down";
    default: return "GPU: not started";
    }
}

long long GpuFrames()
{
    return g_gpuFrames.load();
}
long long FallbackFrames()
{
    return g_fallbacks.load();
}
int LastStatus()
{
    return g_lastStatus.load();
}

int TestInitSync(int forceNoAdapter, unsigned long long vramBudget, unsigned timeoutMs, int maxJobs, int allowSoftware)
{
    TestReset();
    g_state.store(ST_INIT);
    InitBody(Config(forceNoAdapter, vramBudget, timeoutMs, maxJobs, allowSoftware));
    return g_lastStatus.load();
}

int TestHook(int what, unsigned long long value)
{
    std::lock_guard<std::mutex> lk(g_m);
    return g_ctx ? (int)obgpu_test_hook(g_ctx, what, value) : (int)OBGPU_NO_ADAPTER;
}

void TestReset()
{
    Shutdown();
    std::lock_guard<std::mutex> lk(g_m);
    g_losses.clear();
    g_state.store(ST_IDLE);
    g_lastStatus.store(OBGPU_OK);
    g_gpuFrames.store(0);
    g_fallbacks.store(0);
    g_frameJobs.store(0);
    g_postJobs.store(0);
    g_la34Jobs.store(0);
    g_busy.store(0);
    g_cpuRouted.store(0);
    g_selftestStatus = -1;
}

}  // namespace obgpuw
#endif  // OB_OLD_ENGINE
