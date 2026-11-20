// Optic Bokeh v2.4: the GPU blur stage (gpu\obgpu, Rust + wgpu, linked statically) in the wrapper.
// Fast / Standard, render version >= 2. Two levels: Frame() = prepare + blur on the GPU (ABI 1.1, automatic
// highlight source, Result view), else Blur() = the blur stage only after the CPU prepare (phase 1; the cropped
// path and the uncropped Result path). Any failure falls back to
// the CPU blur for that frame (lenscore::blurAlpha: identical semantics, results within ~1.5e-4).
//
// Lifetime: Start() (GLOBAL_SETUP, after the host check) begins a background thread that creates the
// device, runs the adapter self-test (obgpu ABI 1.1, ~0.2 s; a failure keeps the GPU off for the session) and
// compiles the pipelines (Vulkan ~1 s, DX12 up to ~60 s); renders use the CPU until it is
// ready, so nothing ever waits for it. Shutdown() (GLOBAL_SETDOWN) releases the device; a host that
// never sends GLOBAL_SETDOWN leaks it at exit (no driver teardown from static destructors).
// Thread-safe (multi-frame rendering): one context for all render threads.
//
// Cache / MFR semantics (v2.4 decision): the GPU result is the same image as the CPU one within 2e-4 in
// linear light (measured: at most 1 output code at 8 / 16 bpc, a few values per million). Which path ran is
// NOT part of AE's cache key: the popup is an ordinary parameter (switching it re-renders), but a frame
// that fell back (device busy past wait_ms, lost, out of memory) or a render node without a GPU produces
// the CPU result for the same settings. Mixing such frames in one sequence is invisible (<= 1 code), so the
// effect declares nothing extra (no GPU flags, no non-param-vary). Each render thread calls Blur() on the
// shared context; obgpu serialises / queues jobs (max_jobs); v2.4.2 (obgpu ABI 1.5.2): a waiting frame goes to the CPU only
// when the GPU is no longer expected to finish it first (v2.4.1: after a fixed 2 s wait - 4 of 8 UHD MFR frames on the CPU). Abort is polled by obgpu between bands. OPTICBOKEH_GPU=0 disables everything
// (no device is created); =1 uses the GPU even where the popup says Off (diagnostics).
#pragma once
#include "lenscore.h"

#include <string>
#include <vector>

namespace obgpuw {

enum UsePopup { USE_AUTO = 1, USE_OFF = 2 };

// GLOBAL_SETUP: reads OPTICBOKEH_GPU once (0 = never use the GPU, 1 = use it even where the popup says
// Off) and starts the background initialisation (unless disabled). Safe to call more than once.
// Shutdown(): with OPTICBOKEH_GPU_STATS=<file> (aetest only) it first appends "status|gpu N|fallback M|last S".
void Start();
void Shutdown();

// Would this frame go to the GPU? (the cheap checks; Blur() may still fall back)
//   renderVersion >= 2, quality Fast / Standard, popup Auto (or the environment forces it), the device
//   is ready, and it is worth it: maxRadius >= 8, layer area >= 250 000 px (w x h = the LAYER at the current
//   downsample, so every tile of a frame takes the same path), not a plain disc above R 150
bool Eligible(const lenscore::Params& p, int renderVersion, bool bestQuality, int popup, int w, int h);

// Runs the blur stage of `prep` on the GPU. Returns 1 = done (dst / eLOut written), 0 = not done (the
// caller runs the CPU blur), -1 = aborted (the abort callback said so).
int Blur(const lenscore::Prepared& prep, const lenscore::Params& p, bool needEL, float* dst, int dstStride, float* eLOut,
         lenscore::AbortFn abortFn, void* user);

// ABI 1.1-1.4: prepareHighlights, the blur AND postProcess up to grain (glow + vignetting -> halation -> luminance
// aberration, all modes incl. Contrast Rim v2.1 / Channel Drift -> rolloff; grain stays on the CPU) on the GPU (obgpu_render_frame_post: one upload of the window + map, one readback
// of the crop). `p` = the window params (as for prepareHighlights), (cx, cy, cw, ch) = the crop the wrapper would
// pass to cropPrepared, `pq` = the crop params (frame rect / output rows shifted). Returns
//   1 = done: dst (cw x ch, stride dstStride) = the crop after the GPU stages; the caller runs
//       lenscore::postProcess(pc, prest, dst, ..., postNeedsEL(prest, pc) ? eLc : nullptr): `prest` = pq with the
//       stages the GPU ran zeroed, pc = what that needs (w, h, rollRelevant; rgba / eL when the glow is left to it)
//   0 = not supported here (highlight-layer source, renderVersion 1, path switched off): the caller runs the CPU
//       prepare and then Blur() (phase 1) / the CPU blur
//   2 = the GPU did not render it (BUSY at the routing deadline, lost, out of VRAM, ...): the whole frame on the CPU
//   3 = routed to the CPU (ABI 1.2: the GPU is predicted slower now, e.g. another application saturates it)
//  -1 = aborted.
// workUnits: the frame's work measure; after a CPU frame (2 / 3) the caller reports its prepare + blur + post
// time with ReportCpu(workUnits, ms), which drives the routing (and the deadline that abandons slow GPU jobs).
int Frame(const float* src, int w, int h, const float* map, const lenscore::Params& p, int cx, int cy, int cw, int ch,
          const lenscore::Params& pq, float* dst, int dstStride, std::vector<float>& eLc, lenscore::Prepared& pc,
          lenscore::Params& prest, double& workUnits, lenscore::AbortFn abortFn, void* user);
void ReportCpu(double workUnits, double ms);
std::string CountersLine();     // diagnostics (obgpu_get_counters + the wrapper's own counters)
long long PostJobs();           // frames whose post stages (some) ran on the GPU
long long La34Jobs();           // ... of which luminance aberration mode 3 / 4 ran on the GPU (ABI 1.4)
void TestLa34(bool on);         // tests: false = the ABI 1.3 post prefix (stops before LA modes 3 / 4)
void TestRouting(bool on);      // tests: false = no obgpu_route / report (every eligible frame tries the GPU)
void TestGpuPost(bool on);      // tests / timing: false = obgpu_render_frame (post all on the CPU)
double TestLastWorkUnits();     // tests: the work units of the last Frame() call
long long BusyFrames();         // frames the library returned BUSY for (routing deadline / no slot) -> CPU
long long CpuRoutedFrames();    // frames the router sent to the CPU
long long FrameJobs();         // frames that went through Frame() (also counted in GpuFrames)
void TestFramePath(bool on);   // tests / timing: false = phase-1 path only (CPU prepare + GPU blur)
std::string SelftestLine();    // diagnostics: the adapter self-test report
int TestSelftest(unsigned long long ppm);  // tests: re-run the self-test with the result scaled by 1 + ppm * 1e-6

// v2.4.2 per-frame diagnostic log (AE tests read it without the UI): with OPTICBOKEH_GPU_LOG=<file> (read once at
// GLOBAL_SETUP) every RunEngine call appends one ASCII line: size, path (GPU / GPU-blur / CPU), why not the GPU, GPU
// queue wait and run ms, render ms. Unset (the default): nothing is recorded or written (no effect on pixels or speed).
bool FrameLogOn();
void FrameLogBegin();  // the render thread starts a frame: forget the previous frame's GPU result
// the frame is done: w x h = engine window, cw x ch = the crop (0 = none), ms = RunEngine wall time, err = its PF_Err,
// why = the reason RunEngine knows when the GPU was not tried (e.g. "use-gpu-off", "quality-best"; nullptr = worked out here)
void FrameLogEnd(int w, int h, int cw, int ch, double ms, int err, const char* why);
std::string FrameLogPath();

// For About: "GPU: NVIDIA ... (Vulkan) - active" / "GPU: not available" / "GPU: off (OPTICBOKEH_GPU=0)" ...
std::string StatusLine();
// counters (tests): frames done on the GPU, frames that fell back
long long GpuFrames();
long long FallbackFrames();
int LastStatus();

// tests only: run initialisation synchronously with this config (struct obgpu_config fields), wait for
// it; hook: forward to obgpu_test_hook. Return the obgpu status.
int TestInitSync(int forceNoAdapter, unsigned long long vramBudget, unsigned timeoutMs, int maxJobs, int allowSoftware);
int TestHook(int what, unsigned long long value);
void TestReset();  // shuts down and forgets the session state (next TestInitSync starts fresh)

}  // namespace obgpuw
