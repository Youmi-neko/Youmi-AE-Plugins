//! obgpu - Optic Bokeh GPU blur stage (wgpu). C ABI in include/obgpu.h.
//!
//! Every exported function catches panics (they become OBGPU_INTERNAL) and never blocks without a
//! timeout. A context owns at most one device; when the device is lost (driver reset, removal, a
//! timeout, or the test hook) jobs return OBGPU_DEVICE_LOST and a later job re-creates the device
//! (at most once per `RECREATE_COOLDOWN`).

mod ffi;
mod gpu;
mod job;
mod vram;
mod selftest;
mod frame;
mod load;
mod post;

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ffi::*;
use gpu::{Gpu, InitError, Variant};

/// dev tool support (examples/dump_hlsl.rs): the WGSL of one pipeline variant
#[doc(hidden)]
pub fn debug_shader_source(fr: bool, cat: bool, roll: bool, raw: bool, k: u32, max_ws: u32) -> String {
    let v = Variant { fr, cat, roll, raw, k };
    gpu::shader_source(v, gpu::seg_for(v.nc(), max_ws))
}

const RECREATE_COOLDOWN: Duration = Duration::from_secs(2);

pub struct Context {
    cfg: obgpu_config,
    gpu: Mutex<Option<Arc<Gpu>>>,
    last_fail: Mutex<Option<Instant>>,
    info: Mutex<String>,
    slots: Mutex<i32>,
    slots_cv: Condvar,
    budget_override: AtomicU64,
    band_rows: AtomicU32,
    delay_ms: AtomicU32,
    timeout_override: AtomicU32,
    losses: Mutex<Vec<Instant>>,
    disabled: std::sync::atomic::AtomicBool,
    selftest_failed: std::sync::atomic::AtomicBool,
    inject_ppm: AtomicU64,
    router: Mutex<Router>,
    waiting: std::sync::atomic::AtomicI32,
    /// self-tuning VRAM factor: halved after an allocation failure (other applications hold memory the OS
    /// budget did not show yet), recovers by 5 % per successful job
    vram_scale: Mutex<f64>,
}

/// adaptive GPU / CPU routing state (obgpu_route)
struct Router {
    gpu: bool,
    ema_g: f64,
    ema_c: f64,
    ema_frac: f64,
    since_probe: u32,
    cpu_learning: bool,
    /// a GPU job without an estimate is in flight (cold start: only one learns, the others go to the CPU)
    gpu_learning: bool,
    /// calls between probes while on the CPU: 8, doubled after every probe that hit the deadline (cap 64),
    /// back to 8 once a probe is fast (a GPU saturated for hours costs < 2 % of the CPU throughput)
    probe_every: u32,
    /// GPU samples since the last deadline abort (a young estimate does not justify queueing for a slot)
    n_g: u32,
    /// deadline aborts so far: a job in flight when another one hit the deadline shares its fate sooner
    epoch: u64,
    /// ABI 1.5 cold start: GPU ms per work unit from the self-test (NaN = none) and whether the wrapper routes at
    /// all (no obgpu_route call = no routing deadline, the phase-1/2 behaviour)
    seed_g: f64,
    routed: bool,
    /// EMA of GPU busy / (wall - slot wait) of GPU jobs (timestamps; NaN = unknown)
    ema_act: f64,
    /// EMA of the GPU jobs' own ms per work unit (slot wait excluded): compared with the self-test seed
    ema_own: f64,
    c: obgpu_counters,
}
const EMA_A: f64 = 0.3;
fn ema(old: f64, v: f64) -> f64 {
    if old.is_nan() {
        v
    } else {
        old + EMA_A * (v - old)
    }
}
/// GPU estimate: follows improvements faster (0.6) than slowdowns (0.3): one slow frame from a burst of
/// other applications' load must not keep routing on the CPU for long; the deadline bounds the opposite error
fn ema_gpu(old: f64, v: f64) -> f64 {
    if old.is_nan() {
        v
    } else if v < old {
        old + 0.6 * (v - old)
    } else {
        old + EMA_A * (v - old)
    }
}

impl Context {
    /// records a finished GPU job (wall ms, GPU busy ms or NaN, work units)
    fn note_gpu(&self, ok: bool, wall: f64, busy: f64, wu: f64, slot_ms: f64) {
        if let Ok(mut r) = self.router.lock() {
            if ok {
                r.c.gpu_jobs += 1;
                r.c.last_wall_ms = wall;
                r.c.last_gpu_busy_ms = busy;
                if wu > 0.0 {
                    r.ema_g = ema_gpu(r.ema_g, wall / wu);
                    r.n_g = r.n_g.saturating_add(1);
                    // a fast GPU job (probe or not) resets the probe back-off
                    if r.ema_c.is_finite() && wall / wu < r.ema_c {
                        r.probe_every = 8;
                    }
                }
                if busy.is_finite() && wall > 0.0 {
                    r.ema_frac = ema(r.ema_frac, (busy / wall).min(1.0));
                    // GPU activity of the job itself (slot wait excluded): low = it mostly waited for other work
                    // on the GPU (another application) -> the cold start learns the CPU side (route)
                    let own = (wall - slot_ms).max(1e-3);
                    r.ema_act = ema(r.ema_act, (busy / own).min(1.0));
                }
                if wu > 0.0 {
                    r.ema_own = ema_gpu(r.ema_own, (wall - slot_ms).max(0.0) / wu);
                }
            } else {
                r.c.gpu_failed += 1;
            }
        }
    }

    /// routing deadline for a GPU job of `wu` work units: (ms, own). own = false: counted from the call (slot
    /// wait included); own = true (cold start): counted from the moment the job got its job slot. None = no
    /// deadline (a wrapper that never routes).
    fn deadline_ms(&self, wu: f64, epoch0: u64) -> Option<(f64, bool)> {
        let r = self.router.lock().ok()?;
        if !r.routed || !(wu > 0.0) {
            return None;
        }
        if !r.ema_c.is_finite() {
            // GPU-first cold start (no CPU time known yet): abandon a job that takes 4x what the GPU itself needs
            // (its own estimate; before the first sample the self-test seed, which is an upper bound: tiny frames
            // carry the fixed per-pass cost, measured 10-20x the per-work-unit cost of real frames), within
            // [2 s, 15 s] - a saturated GPU then hands the frame to the CPU, whose report enables the normal
            // deadline below.
            // ABI 1.5.2: measured on the job's OWN time (since it got its slot) and from its own-time estimate.
            // 1.5.1 counted the slot wait too: in an MFR burst of UHD frames queueing behind the two job slots the
            // last ones would have spent most of "4 x one job" waiting for the jobs ahead of them (AE 2026 / 2027,
            // v2.4.1). The wait itself is bounded by the queue rules (Context::acquire).
            let g = if r.ema_own.is_finite() {
                r.ema_own
            } else if r.ema_g.is_finite() {
                r.ema_g
            } else {
                r.seed_g
            };
            let k = if r.epoch != epoch0 { 2.0 } else { 4.0 };
            return Some((if g.is_finite() { (k * g * wu).clamp(2000.0, 15000.0) } else { 15000.0 }, true));
        }
        // ski rental: abandoning at ~the CPU time bounds a wrong GPU decision to ~2.25 x a CPU frame
        // (measured with 2 x: 3 x, and K queued jobs all paid it at once). Another job hit the deadline
        // since this one started (same saturated GPU): 0.5 x.
        let k = if r.epoch != epoch0 { 0.5 } else { 1.25 };
        Some(((k * r.ema_c * wu).max(if r.epoch != epoch0 { 500.0 } else { 1000.0 }), false))
    }

    /// the router's per-job estimates (ms) for `wu` work units: GPU own time (slot wait excluded; before the first
    /// sample the self-test seed), CPU time (NaN = unknown)
    fn estimates(&self, wu: f64) -> (f64, f64) {
        match self.router.lock() {
            Ok(r) => {
                let g = if r.ema_own.is_finite() { r.ema_own } else { r.seed_g };
                (g * wu, r.ema_c * wu)
            }
            Err(_) => (f64::NAN, f64::NAN),
        }
    }

    fn timeouts(&self) -> u64 {
        self.router.lock().map(|r| r.c.timeouts).unwrap_or(0)
    }

    /// waits for a job slot (bounded concurrency => bounded VRAM: each job plans for budget / max_jobs).
    /// ABI 1.5.2: a waiting frame goes to the CPU (OBGPU_BUSY) only when the GPU is no longer expected to be
    /// the faster path, instead of after a fixed wait (the wrapper's 2 s sent 4 of 8 UHD MFR frames to the CPU on
    /// an idle GPU in AE, each then taking ~10 x a GPU frame):
    ///  - another job hit the routing deadline / a submission timed out since this call arrived (saturated or
    ///    stuck GPU: the frame goes to the CPU at once; the bounded-time guarantee of the running jobs' deadlines
    ///    then covers the waiting ones too),
    ///  - a CPU time is known and wait so far + the GPU's own time for the job would pass the routing deadline
    ///    (1.25 x the CPU time: the GPU can no longer win),
    ///  - the queue moves far slower than the GPU's own pace: wait > max(2 s, 3 x the rounds ahead x the own
    ///    time per job), and never more than wait_ms (0 = 15 s).
    /// The abort callback (user cancel, and a normal-state deadline counted from the call) is polled every 10 ms.
    fn acquire<'a>(&'a self, call: &CallState, abort: &dyn Fn() -> bool) -> Result<SlotGuard<'a>, c_int> {
        let tw = Instant::now();
        let mut nslots = self.slots.lock().map_err(|_| OBGPU_INTERNAL)?;
        let cap = Duration::from_millis(if self.cfg.wait_ms > 0 { self.cfg.wait_ms as u64 } else { 15000 });
        let mj = self.max_jobs().max(1);
        let ahead = (*nslots + self.waiting.load(Ordering::SeqCst)).max(0);
        call.ahead.set(ahead);
        // rounds of max_jobs jobs that have to finish before this one can start (itself included)
        let rounds = ahead / mj + 1;
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let _wg = WaitGuard(self);
        let to0 = call.timeouts0;
        let busy = |reason: i32| -> c_int {
            call.reason.set(reason);
            if let Ok(mut r) = self.router.lock() {
                match reason {
                    OBGPU_BUSY_WAIT_LIMIT => r.c.busy_wait_limit += 1,
                    OBGPU_BUSY_QUEUE => r.c.busy_queue += 1,
                    _ => r.c.busy_saturated += 1,
                }
            }
            OBGPU_BUSY
        };
        while *nslots >= self.max_jobs() {
            if abort() {
                return Err(OBGPU_ABORTED);
            }
            if self.epoch() != call.ep0 || self.timeouts() != to0 {
                return Err(busy(OBGPU_BUSY_SATURATED));
            }
            let waited = tw.elapsed().as_secs_f64() * 1e3;
            let (own, _) = self.estimates(call.wu);
            let pace = if own.is_finite() { (3.0 * rounds as f64 * own).max(2000.0) } else { f64::INFINITY };
            if tw.elapsed() > cap || waited > pace {
                return Err(busy(OBGPU_BUSY_WAIT_LIMIT));
            }
            if let Some((d, false)) = self.deadline_ms(call.wu, call.ep0) {
                if own.is_finite() && call.t0.elapsed().as_secs_f64() * 1e3 + own > d {
                    return Err(busy(OBGPU_BUSY_QUEUE));
                }
            }
            nslots = match self.slots_cv.wait_timeout(nslots, Duration::from_millis(10)) {
                Ok((g, _)) => g,
                Err(_) => return Err(OBGPU_INTERNAL),
            };
        }
        *nslots += 1;
        call.slot_ms.set(tw.elapsed().as_secs_f64() * 1e3);
        call.slot_t.set(Some(Instant::now()));
        Ok(SlotGuard(self))
    }

    /// a job abandoned at the deadline: its time so far is a lower bound of what the GPU needs now -> the
    /// estimate jumps to at least that (x 1.5), so routing goes to the CPU at once
    fn epoch(&self) -> u64 {
        self.router.lock().map(|r| r.epoch).unwrap_or(0)
    }

    fn note_deadline(&self, wall: f64, wu: f64) {
        if let Ok(mut r) = self.router.lock() {
            r.c.deadline_aborts += 1;
            r.epoch += 1;
            if wu > 0.0 {
                r.ema_g = r.ema_g.max(1.5 * wall / wu);
            }
            r.gpu = false;
            r.n_g = 0;
            r.probe_every = (r.probe_every * 2).clamp(8, 64);
        }
    }

    /// final status bookkeeping shared by obgpu_blur / obgpu_render_frame
    fn finish(&self, status: c_int, hit: bool, wall: f64, wu: f64) -> c_int {
        if let Ok(mut r) = self.router.lock() {
            r.gpu_learning = false;
        }
        if let Ok(mut s) = self.vram_scale.lock() {
            if matches!(status, OBGPU_OUT_OF_VRAM | OBGPU_DEVICE_LOST) {
                *s = (*s * 0.5).max(0.125);
                log(self, &format!("vram: allocation failure / loss -> job budgets x {:.3}", *s));
            } else if status == OBGPU_OK {
                *s = (*s * 1.05).min(1.0);
            }
        }
        if status == OBGPU_ABORTED && hit {
            log(self, &format!("GPU job abandoned at the routing deadline ({wall:.0} ms): render on the CPU"));
            self.note_deadline(wall, wu);
            return OBGPU_BUSY;
        }
        if status == OBGPU_TIMEOUT {
            if let Ok(mut r) = self.router.lock() {
                r.c.timeouts += 1;
                if wu > 0.0 {
                    r.ema_g = r.ema_g.max(1.5 * wall / wu);
                }
                r.gpu = false;
            }
        }
        status
    }
}

/// per-call routing state of one GPU job (obgpu_blur / obgpu_render_frame[_post]), shared by its abort
/// callback, the slot wait and the diagnostics (obgpu_last_job)
struct CallState {
    t0: Instant,
    wu: f64,
    ep0: u64,
    timeouts0: u64,
    slot_t: std::cell::Cell<Option<Instant>>,
    slot_ms: std::cell::Cell<f64>,
    hit: std::cell::Cell<bool>,
    reason: std::cell::Cell<i32>,
    ahead: std::cell::Cell<i32>,
    budget: std::cell::Cell<u64>,
}
impl CallState {
    fn new(ctx: &Context, wu: f64) -> Self {
        CallState {
            t0: Instant::now(),
            wu,
            ep0: ctx.epoch(),
            timeouts0: ctx.timeouts(),
            slot_t: std::cell::Cell::new(None),
            slot_ms: std::cell::Cell::new(0.0),
            hit: std::cell::Cell::new(false),
            reason: std::cell::Cell::new(OBGPU_BUSY_NONE),
            ahead: std::cell::Cell::new(0),
            budget: std::cell::Cell::new(0),
        }
    }
    /// true (and remembered) once the routing deadline has passed: the job is abandoned -> OBGPU_BUSY. A cold
    /// (own-time) deadline does not run while the job still waits for its slot.
    fn deadline_passed(&self, ctx: &Context) -> bool {
        if let Some((ms, own)) = ctx.deadline_ms(self.wu, self.ep0) {
            let el = if own {
                match self.slot_t.get() {
                    Some(s) => s.elapsed(),
                    None => return false,
                }
            } else {
                self.t0.elapsed()
            };
            if el.as_secs_f64() * 1e3 > ms {
                self.hit.set(true);
                self.reason.set(if own { OBGPU_BUSY_COLD_DEADLINE } else { OBGPU_BUSY_DEADLINE });
                return true;
            }
        }
        false
    }
    /// records the call for obgpu_last_job (the calling thread's last GPU job)
    fn record(&self, ctx: &Context, status: c_int, prep_bands: i32) {
        let mut j = obgpu_job_info::default();
        j.status = status;
        j.busy_reason = if status == OBGPU_BUSY { self.reason.get() } else { OBGPU_BUSY_NONE };
        j.ahead = self.ahead.get();
        j.prep_bands = prep_bands;
        j.slot_wait_ms = self.slot_ms.get();
        j.total_ms = self.t0.elapsed().as_secs_f64() * 1e3;
        j.run_ms = self.slot_t.get().map(|s| s.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        if self.slot_t.get().is_none() {
            j.slot_wait_ms = j.total_ms;
        }
        if let Some((d, own)) = ctx.deadline_ms(self.wu, self.ep0) {
            j.deadline_ms = d;
            j.deadline_own = own as i32;
        }
        let (g, c) = ctx.estimates(self.wu);
        j.est_gpu_ms = g;
        j.est_cpu_ms = c;
        j.budget_bytes = self.budget.get();
        j.work_units = self.wu;
        LAST_JOB.with(|l| l.set(j));
    }
}

thread_local! {
    static LAST_JOB: std::cell::Cell<obgpu_job_info> = std::cell::Cell::new(obgpu_job_info::default());
}

fn log(ctx: &Context, s: &str) {
    if ctx.cfg.verbose != 0 {
        eprintln!("[obgpu] {s}");
    }
}

impl Context {
    /// VRAM for one job: the total (configured, or 50 % of the OS budget for this process, 0.75 - 6 GB)
    /// divided by the jobs actually running or waiting (a lone UHD frame gets the whole share: fewer prepare
    /// bands), and at most 60 % of what is free right now for this process (other applications included),
    /// split the same way. The test override (hook 2) is a fixed total split by max_jobs.
    fn job_budget(&self, g: &Gpu) -> u64 {
        let o = self.budget_override.load(Ordering::SeqCst);
        if o > 0 {
            return o / self.max_jobs().max(1) as u64;
        }
        // ABI 1.5.2: at most max_jobs jobs hold VRAM at once (the others wait for a slot and hold none), so the
        // share is total / min(running + waiting, max_jobs). 1.5.1 divided by every waiting caller too: in an MFR
        // burst of 8 UHD frames each running job planned for 3 GB / 8 = 384 MB and prepared in 3 row bands
        // (x 1.35-1.9 prepare rows, AE and offline), with nothing else using that VRAM.
        let jobs = (self.slots.lock().map(|n| *n).unwrap_or(1) + self.waiting.load(Ordering::SeqCst)).clamp(1, self.max_jobs().max(1)) as u64;
        let info = vram::local_budget(g.vendor, g.device_id);
        let total = if self.cfg.vram_budget_bytes > 0 {
            self.cfg.vram_budget_bytes
        } else {
            // phase 3: half the OS budget, up to 6 GB (was 40 %, 3 GB): on a free 12 GB card two concurrent UHD
            // frames each get ~2.7 GB and prepare untiled (at 0.75 GB / job: 3 bands, 1.6x prepare rows);
            // the 60 %-of-free cap below keeps it modest whenever other applications hold the memory
            // phase 5: back to max 3 GB (phase 3 had raised it to 6 GB): the OS budget for this process does not
            // see other applications' allocations soon enough - a UHD Film frame planned for 5.6 GB next to another
            // application's 4 GB lost the device ("out of memory"). 3 GB = 2 concurrent UHD frames untiled.
            info.map(|(b, _)| (b / 2).clamp(768 << 20, 3u64 << 30)).unwrap_or(1536u64 << 20)
        };
        let share = total / jobs;
        let scale = self.vram_scale.lock().map(|s| *s).unwrap_or(1.0);
        let r = match info {
            Some((b, u)) => share.min(b.saturating_sub(u) / 10 * 6 / jobs),
            None => share,
        };
        let r = ((r as f64 * scale) as u64).max(64 << 20);
        log(self, &format!("vram: job budget {} MB ({} jobs)", r >> 20, jobs));
        r
    }

    fn budget(&self) -> u64 {
        let o = self.budget_override.load(Ordering::SeqCst);
        if o > 0 {
            return o;
        }
        if self.cfg.vram_budget_bytes > 0 {
            self.cfg.vram_budget_bytes
        } else {
            1536u64 << 20
        }
    }
    fn max_jobs(&self) -> i32 {
        if self.cfg.max_jobs > 0 {
            self.cfg.max_jobs
        } else {
            2
        }
    }
    fn timeout(&self) -> Duration {
        let o = self.timeout_override.load(Ordering::SeqCst);
        let ms = if o > 0 { o } else if self.cfg.timeout_ms > 0 { self.cfg.timeout_ms } else { 4000 };
        Duration::from_millis(ms as u64)
    }
    /// the current device, re-created if it was lost (cooldown); Err = status
    fn device(&self) -> Result<Arc<Gpu>, c_int> {
        let mut g = self.gpu.lock().map_err(|_| OBGPU_INTERNAL)?;
        if let Some(d) = g.as_ref() {
            if !d.is_lost() {
                return Ok(d.clone());
            }
            // the job that hit the loss already returned DEVICE_LOST (its frame went to the CPU); this
            // call re-creates the device at once. Repeated losses (3 within a minute: a driver that keeps
            // resetting) disable the GPU path for the rest of the session.
            log(self, "dropping the lost device");
            *g = None;
            let mut lh = self.losses.lock().map_err(|_| OBGPU_INTERNAL)?;
            lh.retain(|t| t.elapsed() < Duration::from_secs(60));
            lh.push(Instant::now());
            if lh.len() >= 3 {
                log(self, "3 device losses within a minute: GPU path disabled for this session");
                self.disabled.store(true, Ordering::SeqCst);
            }
        }
        if self.disabled.load(Ordering::SeqCst) {
            return Err(OBGPU_DEVICE_LOST);
        }
        // a failed creation is not retried for RECREATE_COOLDOWN (no per-frame stalls on a broken driver)
        if let Some(t) = *self.last_fail.lock().map_err(|_| OBGPU_INTERNAL)? {
            if t.elapsed() < RECREATE_COOLDOWN {
                return Err(OBGPU_DEVICE_LOST);
            }
        }
        match Gpu::create(&self.cfg) {
            Ok(d) => {
                let d = Arc::new(d);
                *self.info.lock().map_err(|_| OBGPU_INTERNAL)? = d.info.clone();
                log(self, &format!("device: {}", d.info));
                *g = Some(d.clone());
                Ok(d)
            }
            Err(e) => {
                *self.last_fail.lock().map_err(|_| OBGPU_INTERNAL)? = Some(Instant::now());
                match e {
                    InitError::NoAdapter(m) => {
                        log(self, &m);
                        Err(OBGPU_NO_ADAPTER)
                    }
                    InitError::Device(m) => {
                        log(self, &m);
                        Err(OBGPU_DEVICE_LOST)
                    }
                }
            }
        }
    }
}

fn guard<F: FnOnce() -> c_int>(f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => OBGPU_INTERNAL,
    }
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_init(cfg: *const obgpu_config, status: *mut c_int) -> *mut Context {
    let mut st = OBGPU_INTERNAL;
    let r = catch_unwind(AssertUnwindSafe(|| -> *mut Context {
        let mut c = obgpu_config::default();
        if !cfg.is_null() {
            let n = ((*cfg).struct_size as usize).min(std::mem::size_of::<obgpu_config>());
            std::ptr::copy_nonoverlapping(cfg as *const u8, &mut c as *mut _ as *mut u8, n);
        }
        let ctx = Box::new(Context {
            cfg: c,
            gpu: Mutex::new(None),
            last_fail: Mutex::new(None),
            info: Mutex::new(String::new()),
            slots: Mutex::new(0),
            slots_cv: Condvar::new(),
            budget_override: AtomicU64::new(0),
            band_rows: AtomicU32::new(0),
            delay_ms: AtomicU32::new(0),
            timeout_override: AtomicU32::new(0),
            losses: Mutex::new(Vec::new()),
            disabled: std::sync::atomic::AtomicBool::new(false),
            selftest_failed: std::sync::atomic::AtomicBool::new(false),
            inject_ppm: AtomicU64::new(0),
            router: Mutex::new(Router { gpu: true, ema_g: f64::NAN, ema_c: f64::NAN, ema_frac: f64::NAN, since_probe: 0, cpu_learning: false, gpu_learning: false, probe_every: 8, n_g: 0, epoch: 0, seed_g: f64::NAN, routed: false, ema_act: f64::NAN, ema_own: f64::NAN, c: obgpu_counters::default() }),
            waiting: std::sync::atomic::AtomicI32::new(0),
            vram_scale: Mutex::new(1.0),
        });
        match ctx.device() {
            Ok(_) => {
                st = OBGPU_OK;
                Box::into_raw(ctx)
            }
            Err(s) => {
                st = s;
                std::ptr::null_mut()
            }
        }
    }));
    if !status.is_null() {
        *status = if r.is_ok() { st } else { OBGPU_INTERNAL };
    }
    r.unwrap_or(std::ptr::null_mut())
}

/// compiles the pipelines of the common variants (avoids a first-frame stall)
#[no_mangle]
pub unsafe extern "C" fn obgpu_warmup(ctx: *mut Context) -> c_int {
    if ctx.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let ctx = &*ctx;
    guard(|| {
        let g = match ctx.device() {
            Ok(g) => g,
            Err(s) => return s,
        };
        if let Err(e) = g.setup_pipes() {
            log(ctx, &e);
            return OBGPU_INTERNAL;
        }
        for fr in [false, true] {
            for cat in [false, true] {
                for roll in [false, true] {
                    for raw in [false, true] {
                        if let Err(e) = g.accum_pipe(Variant { fr, cat, roll, raw, k: 1 }) {
                            log(ctx, &e);
                            return OBGPU_INTERNAL;
                        }
                    }
                }
            }
        }
        OBGPU_OK
    })
}

struct WaitGuard<'a>(&'a Context);
impl Drop for WaitGuard<'_> {
    fn drop(&mut self) {
        self.0.waiting.fetch_sub(1, Ordering::SeqCst);
    }
}

struct SlotGuard<'a>(&'a Context);
impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut n) = self.0.slots.lock() {
            *n -= 1;
        }
        self.0.slots_cv.notify_one();
    }
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_blur(
    ctx: *mut Context,
    inp: *const obgpu_input,
    desc: *const obgpu_blur_desc,
    out: *const obgpu_output,
    abort_fn: obgpu_abort_fn,
    user: *mut c_void,
    stats: *mut obgpu_stats,
) -> c_int {
    if ctx.is_null() || inp.is_null() || desc.is_null() || out.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let ctx = &*ctx;
    let t0 = Instant::now();
    let mut st = obgpu_stats::default();
    let wu = obgpu_work_units(desc);
    let call = CallState::new(ctx, wu);
    let r = guard(|| {
        if ctx.selftest_failed.load(Ordering::SeqCst) {
            return OBGPU_SELFTEST_FAILED;
        }
        let d = &*desc;
        if let Err(f) = job::check_desc(d) {
            log(ctx, &f.msg);
            return f.status;
        }
        let i = &*inp;
        let o = &*out;
        let n = (d.w as usize) * (d.h as usize);
        if i.struct_size as usize != std::mem::size_of::<obgpu_input>() || o.struct_size as usize != std::mem::size_of::<obgpu_output>() {
            return OBGPU_BAD_ARGS;
        }
        if i.rgba.is_null() || i.radius.is_null() || i.side.is_null() || o.rgba.is_null() || (o.rgba_stride as i64) < 4 * d.w as i64 {
            return OBGPU_BAD_ARGS;
        }
        if d.need_el != 0 && (i.el.is_null() || o.el.is_null()) {
            return OBGPU_BAD_ARGS;
        }
        let input = job::Input {
            rgba: std::slice::from_raw_parts(i.rgba, n * 4),
            radius: std::slice::from_raw_parts(i.radius, n),
            side: std::slice::from_raw_parts(i.side, n),
            el: if d.need_el != 0 { Some(std::slice::from_raw_parts(i.el, n)) } else { None },
        };
        let output = job::Output { rgba: o.rgba, stride: o.rgba_stride as usize, el: o.el };
        let abort = || -> bool {
            let user_abort = match abort_fn {
                Some(f) => catch_unwind(AssertUnwindSafe(|| f(user) != 0)).unwrap_or(true),
                None => false,
            };
            if user_abort {
                return true;
            }
            call.deadline_passed(ctx)
        };
        // a job slot (bounded concurrency => bounded VRAM: each job plans for budget / max_jobs)
        if std::env::var("OBGPU_TRACE").is_ok() { eprintln!("[obgpu.trace] entry {:.1} ms", t0.elapsed().as_secs_f64() * 1e3); }
        // (waiting, abortable, beats a CPU fallback while the GPU frame is several times faster than the CPU one:
        // measured MFR UHD R120, K=4, 2 s limit -> 3 of 4 frames fell back and throughput = CPU)
        let _slot = match ctx.acquire(&call, &abort) {
            Ok(s) => s,
            Err(s) => return s,
        };
        let slot_ms = call.slot_ms.get();
        st.wait_ms += slot_ms;
        let g = match ctx.device() {
            Ok(g) => g,
            Err(s) => return s,
        };
        let budget = ctx.job_budget(&g);
        call.budget.set(budget);
        let tun = job::Tuning {
            budget,
            timeout: ctx.timeout(),
            band_rows: ctx.band_rows.load(Ordering::SeqCst),
            delay_ms: ctx.delay_ms.load(Ordering::SeqCst),
            target_ms: 25.0,
        };
        job::BUSY.with(|b| b.set(0.0));
        let res = job::run(&g, &input, d, &output, &abort, &tun, &mut st);
        if !matches!(res, Err(ref f) if f.status == OBGPU_ABORTED) {
            // wall from the call (slot wait included: it is part of what routing has to predict)
            ctx.note_gpu(res.is_ok(), t0.elapsed().as_secs_f64() * 1e3, job::BUSY.with(|b| b.get()), obgpu_work_units(d), slot_ms);
        }
        match res {
            Ok(()) => OBGPU_OK,
            Err(f) => {
                if f.status != OBGPU_ABORTED {
                    log(ctx, &format!("job failed ({}): {}", f.status, f.msg));
                }
                // GPU-side failures: the device is re-created (after the cooldown) before it is used again.
                // A timeout is congestion (abandoned, device kept): a hung GPU is reset by the OS -> DEVICE_LOST.
                if matches!(f.status, OBGPU_DEVICE_LOST | OBGPU_INTERNAL) {
                    g.lost.store(true, Ordering::SeqCst);
                }
                f.status
            }
        }
    });
    let r = ctx.finish(r, call.hit.get(), t0.elapsed().as_secs_f64() * 1e3, wu);
    call.record(ctx, r, 0);
    st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    if !stats.is_null() {
        *stats = st;
    }
    r
}

/// ABI 1.5.2: what happened to the calling thread's last GPU job (obgpu_blur / obgpu_render_frame[_post]):
/// status, why it returned OBGPU_BUSY, slot wait, run time, the deadline and the router's estimates. Diagnostics
/// only (the wrapper's per-frame log); reading it changes nothing.
#[no_mangle]
pub unsafe extern "C" fn obgpu_last_job(ctx: *mut Context, out: *mut obgpu_job_info) -> c_int {
    if ctx.is_null() || out.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let mut j = LAST_JOB.with(|l| l.get());
    let n = ((*out).struct_size as usize).min(std::mem::size_of::<obgpu_job_info>());
    j.struct_size = (*out).struct_size;
    std::ptr::copy_nonoverlapping(&j as *const _ as *const u8, out as *mut u8, n);
    OBGPU_OK
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_shutdown(ctx: *mut Context) {
    if ctx.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let c = Box::from_raw(ctx);
        if let Ok(mut g) = c.gpu.lock() {
            if let Some(d) = g.take() {
                let _ = d.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(2)) });
            }
        }
        drop(c);
    }));
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_adapter_info(ctx: *mut Context, buf: *mut c_char, cap: usize) -> usize {
    if ctx.is_null() {
        return 0;
    }
    let s = (*ctx).info.lock().map(|s| s.clone()).unwrap_or_default();
    if !buf.is_null() && cap > 0 {
        let n = s.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    s.len()
}

#[no_mangle]
pub extern "C" fn obgpu_status_string(s: c_int) -> *const c_char {
    let t: &'static [u8] = match s {
        OBGPU_OK => b"ok\0",
        OBGPU_BAD_ARGS => b"bad arguments\0",
        OBGPU_ABORTED => b"aborted\0",
        OBGPU_NO_ADAPTER => b"no suitable GPU\0",
        OBGPU_DEVICE_LOST => b"GPU device lost\0",
        OBGPU_OUT_OF_VRAM => b"out of GPU memory\0",
        OBGPU_TIMEOUT => b"GPU timeout\0",
        OBGPU_BUSY => b"GPU busy\0",
        OBGPU_HOST_OOM => b"out of host memory\0",
        OBGPU_UNSUPPORTED => b"unsupported\0",
        OBGPU_SELFTEST_FAILED => b"GPU failed the self-test (disabled)\0",
        _ => b"internal GPU error\0",
    };
    t.as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_test_hook(ctx: *mut Context, what: c_int, value: u64) -> c_int {
    if ctx.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let ctx = &*ctx;
    guard(|| match what {
        1 => {
            if let Ok(g) = ctx.gpu.lock() {
                if let Some(d) = g.as_ref() {
                    d.device.destroy();
                }
            }
            OBGPU_OK
        }
        2 => {
            ctx.budget_override.store(value, Ordering::SeqCst);
            OBGPU_OK
        }
        3 => {
            ctx.band_rows.store(value as u32, Ordering::SeqCst);
            OBGPU_OK
        }
        4 => {
            ctx.delay_ms.store(value as u32, Ordering::SeqCst);
            OBGPU_OK
        }
        5 => {
            ctx.timeout_override.store(value as u32, Ordering::SeqCst);
            OBGPU_OK
        }
        6 => {
            ctx.inject_ppm.store(value, Ordering::SeqCst);
            OBGPU_OK
        }
        7 => {
            // test-only GPU load generator on its own device: submissions of ~value ms back to back (0 = stop)
            if load::set(value as u32, gpu::backends_of(ctx.cfg.backends)) { OBGPU_OK } else { OBGPU_UNSUPPORTED }
        }
        8 => {
            // test-only: print the per-pass GPU time profile collected with OBGPU_PROFILE=1 (stderr) and reset it
            if let Ok(mut p) = gpu::PROFILE.lock() {
                let tot: f64 = p.iter().map(|e| e.1).sum();
                let mut v = p.clone();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (l, ms, n) in v {
                    eprintln!("[obgpu.profile] {l:<14} {ms:9.2} ms {:5.1} % passes {n}", 100.0 * ms / tot.max(1e-9));
                }
                eprintln!("[obgpu.profile] total {tot:.2} ms");
                p.clear();
            }
            OBGPU_OK
        }
        _ => OBGPU_BAD_ARGS,
    })
}

/// ABI 1.1: adapter self-test (see obgpu.h). Disables the context when the adapter is out of tolerance.
#[no_mangle]
pub unsafe extern "C" fn obgpu_selftest(ctx: *mut Context, report: *mut obgpu_selftest_report) -> c_int {
    if ctx.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let ctx = &*ctx;
    let mut rep = obgpu_selftest_report::default();
    let s = guard(|| {
        let g = match ctx.device() {
            Ok(g) => g,
            Err(s) => return s,
        };
        let tun = job::Tuning { budget: 256 << 20, timeout: ctx.timeout(), band_rows: 0, delay_ms: 0, target_ms: 25.0 };
        match selftest::run(&g, &tun, ctx.inject_ppm.load(Ordering::SeqCst)) {
            Ok(r) => {
                rep = r;
                if r.pass != 0 && r.seed_ms_per_wu.is_finite() && r.seed_ms_per_wu > 0.0 {
                    if let Ok(mut rt) = ctx.router.lock() {
                        rt.seed_g = r.seed_ms_per_wu;
                        rt.c.seed_ms_per_wu = r.seed_ms_per_wu;
                    }
                }
                log(ctx, &format!("self-test: pass={} max_rel={:.3e} mean={:.3e} det={} {:.0} ms", r.pass, r.max_rel, r.mean_abs, r.deterministic, r.ms));
                if r.pass != 0 {
                    OBGPU_OK
                } else {
                    ctx.selftest_failed.store(true, Ordering::SeqCst);
                    OBGPU_SELFTEST_FAILED
                }
            }
            Err(f) => {
                log(ctx, &format!("self-test could not run ({}): {}", f.status, f.msg));
                ctx.selftest_failed.store(true, Ordering::SeqCst);
                f.status
            }
        }
    });
    if !report.is_null() {
        let n = ((*report).struct_size as usize).min(std::mem::size_of::<obgpu_selftest_report>());
        rep.struct_size = (*report).struct_size;
        std::ptr::copy_nonoverlapping(&rep as *const _ as *const u8, report as *mut u8, n);
    }
    s
}
/// ABI 1.1: prepare + blur on the GPU for one frame (see obgpu.h).
#[no_mangle]
pub unsafe extern "C" fn obgpu_render_frame(
    ctx: *mut Context,
    inp: *const obgpu_frame_input,
    prep: *const obgpu_prepare_desc,
    blur: *const obgpu_blur_desc,
    out: *mut obgpu_frame_output,
    abort_fn: obgpu_abort_fn,
    user: *mut c_void,
    stats: *mut obgpu_stats,
) -> c_int {
    render_frame_impl(ctx, inp, prep, blur, std::ptr::null(), out, abort_fn, user, stats, std::ptr::null_mut())
}

/// ABI 1.3: prepare + blur + a prefix of the post chain on the GPU (see obgpu.h).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn obgpu_render_frame_post(
    ctx: *mut Context,
    inp: *const obgpu_frame_input,
    prep: *const obgpu_prepare_desc,
    blur: *const obgpu_blur_desc,
    post: *const obgpu_post_desc,
    out: *mut obgpu_frame_output,
    abort_fn: obgpu_abort_fn,
    user: *mut c_void,
    stats: *mut obgpu_stats,
    post_done: *mut u32,
) -> c_int {
    render_frame_impl(ctx, inp, prep, blur, post, out, abort_fn, user, stats, post_done)
}

#[allow(clippy::too_many_arguments)]
unsafe fn render_frame_impl(
    ctx: *mut Context,
    inp: *const obgpu_frame_input,
    prep: *const obgpu_prepare_desc,
    blur: *const obgpu_blur_desc,
    post: *const obgpu_post_desc,
    out: *mut obgpu_frame_output,
    abort_fn: obgpu_abort_fn,
    user: *mut c_void,
    stats: *mut obgpu_stats,
    post_done: *mut u32,
) -> c_int {
    if !post_done.is_null() {
        *post_done = 0;
    }
    if ctx.is_null() || inp.is_null() || prep.is_null() || blur.is_null() || out.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let ctx = &*ctx;
    let t0 = Instant::now();
    let mut st = obgpu_stats::default();
    let wu = obgpu_work_units(blur);
    let call = CallState::new(ctx, wu);
    let bands = std::cell::Cell::new(0i32);
    let r = guard(|| {
        if ctx.selftest_failed.load(Ordering::SeqCst) {
            return OBGPU_SELFTEST_FAILED;
        }
        let (fi, pd, bd, fo) = (&*inp, &*prep, &*blur, &mut *out);
        if fi.struct_size as usize != std::mem::size_of::<obgpu_frame_input>() || fo.struct_size as usize != std::mem::size_of::<obgpu_frame_output>() {
            return OBGPU_BAD_ARGS;
        }
        fo.stages = 0;
        if let Err(f) = frame::check_prep(pd) {
            log(ctx, &f.msg);
            return f.status;
        }
        if let Err(f) = job::check_desc(bd) {
            log(ctx, &f.msg);
            return f.status;
        }
        let (w, h) = (pd.w as usize, pd.h as usize);
        if fi.src.is_null() || (fi.src_stride as usize) < 4 * w || fo.rgba.is_null() || (fo.rgba_stride as i64) < 4 * bd.w as i64 {
            return OBGPU_BAD_ARGS;
        }
        if bd.need_el != 0 && fo.el.is_null() {
            return OBGPU_BAD_ARGS;
        }
        // an ABI 1.3 (version 1) desc is shorter: widen it (the 1.4 fields stay zero = la34 off)
        let postw: Option<obgpu_post_desc> = if post.is_null() {
            None
        } else {
            let sz = ((*post).struct_size as usize).min(std::mem::size_of::<obgpu_post_desc>());
            if sz < POST_DESC_V1_SIZE {
                return OBGPU_BAD_ARGS;
            }
            let mut v = obgpu_post_desc::default();
            std::ptr::copy_nonoverlapping(post as *const u8, &mut v as *mut _ as *mut u8, sz);
            Some(v)
        };
        let postd = postw.as_ref();
        if let Some(pd) = postd {
            if let Err(f) = post::check(pd, bd.w, bd.h) {
                log(ctx, &f.msg);
                return f.status;
            }
        }
        let src = std::slice::from_raw_parts(fi.src, (h - 1) * fi.src_stride as usize + 4 * w);
        let map = if fi.map.is_null() || pd.has_map == 0 {
            None
        } else {
            if (fi.map_stride as usize) < w {
                return OBGPU_BAD_ARGS;
            }
            Some((std::slice::from_raw_parts(fi.map, (h - 1) * fi.map_stride as usize + w), fi.map_stride as usize))
        };
        let fin = frame::FrameIn { src, src_stride: fi.src_stride as usize, map, crop: (fi.crop_x, fi.crop_y, fi.crop_w, fi.crop_h) };
        let output = job::Output { rgba: fo.rgba, stride: fo.rgba_stride as usize, el: fo.el };
        let abort = || -> bool {
            let user_abort = match abort_fn {
                Some(f) => catch_unwind(AssertUnwindSafe(|| f(user) != 0)).unwrap_or(true),
                None => false,
            };
            if user_abort {
                return true;
            }
            call.deadline_passed(ctx)
        };
        if std::env::var("OBGPU_TRACE").is_ok() { eprintln!("[obgpu.trace] entry {:.1} ms", t0.elapsed().as_secs_f64() * 1e3); }
        let _slot = match ctx.acquire(&call, &abort) {
            Ok(s) => s,
            Err(s) => return s,
        };
        let slot_ms = call.slot_ms.get();
        let g = match ctx.device() {
            Ok(g) => g,
            Err(s) => return s,
        };
        let pipes = match g.prep_pipes() {
            Ok(p) => p,
            Err(e) => {
                log(ctx, &e);
                return OBGPU_INTERNAL;
            }
        };
        let budget = ctx.job_budget(&g);
        call.budget.set(budget);
        let tun = job::Tuning { budget, timeout: ctx.timeout(), band_rows: ctx.band_rows.load(Ordering::SeqCst), delay_ms: ctx.delay_ms.load(Ordering::SeqCst), target_ms: 25.0 };
        let pout = if fo.prepared_rgba.is_null() { None } else { Some(fo.prepared_rgba) };
        let pel = if fo.prepared_el.is_null() { None } else { Some(fo.prepared_el) };
        job::BUSY.with(|b| b.set(0.0));
        job::PREP.with(|c| c.set((0, 0.0)));
        job::POST.with(|c| c.set(0));
        let ppipes = match postd {
            Some(p) if p.stages != 0 => match g.post_pipes() {
                Ok(pp) => Some(pp),
                Err(e) => {
                    log(ctx, &format!("post pipelines: {e} (CPU post)"));
                    None
                }
            },
            _ => None,
        };
        let parg = match (postd, ppipes.as_ref()) {
            (Some(p), Some(pp)) => Some((p, &**pp)),
            _ => None,
        };
        let res = frame::run(&g, &pipes, &fin, pd, bd, &output, pout, pel, parg, &abort, &tun, &mut st);
        if res.is_ok() && !post_done.is_null() {
            *post_done = job::POST.with(|c| c.get());
        }
        if let Ok(mut r) = ctx.router.lock() {
            let (b, f) = job::PREP.with(|c| c.get());
            bands.set(b);
            r.c.last_prep_bands = b;
            r.c.last_prep_rows_factor = f;
        }
        if !matches!(res, Err(ref f) if f.status == OBGPU_ABORTED) {
            ctx.note_gpu(res.is_ok(), t0.elapsed().as_secs_f64() * 1e3, job::BUSY.with(|b| b.get()), obgpu_work_units(bd), slot_ms);
        }
        match res {
            Ok((roll, stages)) => {
                fo.roll_relevant = roll as i32;
                fo.stages = stages;
                OBGPU_OK
            }
            Err(f) => {
                if f.status != OBGPU_ABORTED {
                    log(ctx, &format!("frame failed ({}): {}", f.status, f.msg));
                }
                if matches!(f.status, OBGPU_DEVICE_LOST | OBGPU_INTERNAL) {
                    g.lost.store(true, Ordering::SeqCst);
                }
                f.status
            }
        }
    });
    let r = ctx.finish(r, call.hit.get(), t0.elapsed().as_secs_f64() * 1e3, wu);
    call.record(ctx, r, bands.get());
    st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    if !stats.is_null() {
        *stats = st;
    }
    r
}


/// work units of a blur desc: output-relevant pixels (Mpx) x a radius cost factor (the blur's span work grows
/// with the radius up to the first resolution-level switch, ~2 x levelBase). Same measure for GPU and CPU.
#[no_mangle]
pub unsafe extern "C" fn obgpu_work_units(desc: *const obgpu_blur_desc) -> f64 {
    if desc.is_null() {
        return 0.0;
    }
    let d = &*desc;
    let rows = (d.row_end - d.row_begin).max(0) as f64;
    let r = d.max_radius.min(2.0 * d.level_base.max(1.0));
    (d.w.max(0) as f64) * rows * 1e-6 * (1.0 + r / 16.0)
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_route(ctx: *mut Context, work_units: f64) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    let ctx = &*ctx;
    let r = catch_unwind(AssertUnwindSafe(|| {
        if ctx.selftest_failed.load(Ordering::SeqCst) || ctx.disabled.load(Ordering::SeqCst) {
            return 0;
        }
        let active = ctx.slots.lock().map(|n| *n).unwrap_or(0);
        let waiting = ctx.waiting.load(Ordering::SeqCst).max(0);
        let mut rt = match ctx.router.lock() {
            Ok(g) => g,
            Err(_) => return 0,
        };
        rt.routed = true;
        if !(work_units > 0.0) {
            return 1;
        }
        if rt.ema_c.is_nan() {
            // ABI 1.5 GPU-first cold start: no CPU frame is spent on learning (in MFR a CPU frame takes K x
            // longer than a GPU one: phase 3 sent 1 of 8 frames to the GPU in AE). Every frame goes to the GPU
            // (queueing for the job slots) under the cold deadline (deadline_ms: 4 x the GPU's own / seeded time);
            // a frame that misses it is rendered on the CPU by the wrapper, whose obgpu_report_cpu switches to
            // the normal comparison below.
            let _ = (active, waiting);
            if rt.c.deadline_aborts > 0 || rt.c.timeouts > 0 {
                // the GPU already missed the cold deadline: CPU until the first CPU report arrives
                rt.c.cpu_routed += 1;
                return 0;
            }
            // contended GPU: learn the CPU side with one CPU frame (the phase-3 rule). Signal: the jobs' own time
            // per work unit against the self-test seed (tiny frames: their per-unit cost is 3-4x a free GPU's on
            // real frames, 3x below a saturated one's: measured idle 0.1-0.25 x seed, saturated 1-1.5 x). Without
            // a seed (no self-test): after 2 GPU samples, as in phase 3. A free GPU never spends a frame on it.
            // (Timestamp busy / wall is no signal: other processes' work interleaved into our passes counts as ours.)
            let contended = match (rt.seed_g.is_finite(), rt.ema_own.is_finite()) {
                (true, true) => rt.ema_own > 0.6 * rt.seed_g,
                (true, false) => false,
                (false, _) => rt.n_g >= 2,
            };
            if rt.n_g >= 1 && contended && !rt.cpu_learning {
                rt.cpu_learning = true;
                rt.c.cpu_routed += 1;
                return 0;
            }
            rt.c.route_gpu = 1;
            return 1;
        }
        // jobs ahead of this one: running ones beyond the free slots, and the ones already waiting
        let mj = ctx.max_jobs().max(1);
        let ahead = (active + waiting - mj + 1).max(0) as f64 / mj as f64;
        // state (hysteresis) from the learned rates; the queue only decides this call
        let pred_g = rt.ema_g * work_units;
        let pred_c = rt.ema_c * work_units;
        if rt.gpu && pred_g > pred_c {
            rt.gpu = false;
        } else if !rt.gpu && pred_g < 0.75 * pred_c {
            rt.gpu = true;
        }
        rt.c.route_gpu = rt.gpu as i32;
        if rt.gpu {
            rt.since_probe = 0;
            // queueing for a slot only on a confirmed estimate (>= 3 samples): one optimistic sample followed
            // by a burst of other load sent K queued jobs into the deadline together (FHD K=4, measured)
            let queue_ok = ahead == 0.0 || rt.n_g >= 3;
            if queue_ok && pred_g * (1.0 + ahead) <= pred_c {
                return 1;
            }
            rt.c.cpu_routed += 1; // the GPU is faster but its queue is full: this frame on the CPU
            return 0;
        }
        rt.since_probe += 1;
        if rt.since_probe >= rt.probe_every && ahead == 0.0 {
            // probe: refreshes the GPU estimate once other applications release the GPU
            rt.since_probe = 0;
            rt.c.probes += 1;
            return 1;
        }
        rt.c.cpu_routed += 1;
        0
    }));
    r.unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_report_cpu(ctx: *mut Context, work_units: f64, ms: f64) {
    if ctx.is_null() || !(work_units > 0.0) || !(ms > 0.0) {
        return;
    }
    let ctx = &*ctx;
    if let Ok(mut r) = ctx.router.lock() {
        r.ema_c = ema(r.ema_c, ms / work_units);
        r.c.cpu_reports += 1;
        // a wrapper that reports CPU times uses the routing contract even on paths it does not route
        // (obgpu_blur for highlight-layer sources): the deadline applies to every GPU job from now on. ABI 1.5
        // had keyed the deadline on obgpu_route calls only, so such a path ran without one (587 s measured
        // with 1.5 s per submission instead of BUSY at ~1.6 s).
        r.routed = true;
    }
}

#[no_mangle]
pub unsafe extern "C" fn obgpu_get_counters(ctx: *mut Context, out: *mut obgpu_counters) -> c_int {
    if ctx.is_null() || out.is_null() {
        return OBGPU_BAD_ARGS;
    }
    let ctx = &*ctx;
    let mut c = match ctx.router.lock() {
        Ok(r) => {
            let mut c = r.c;
            c.route_gpu = r.gpu as i32;
            c.ema_gpu_ms_per_wu = r.ema_g;
            c.ema_cpu_ms_per_wu = r.ema_c;
            c.ema_busy_fraction = r.ema_frac;
            c
        }
        Err(_) => return OBGPU_INTERNAL,
    };
    c.vram_scale = ctx.vram_scale.lock().map(|s| *s).unwrap_or(1.0);
    c.active_jobs = ctx.slots.lock().map(|n| *n).unwrap_or(0);
    c.waiting_jobs = ctx.waiting.load(Ordering::SeqCst);
    c.timestamps = ctx.gpu.lock().ok().and_then(|g| g.as_ref().map(|d| d.ts as i32)).unwrap_or(0);
    let n = ((*out).struct_size as usize).min(std::mem::size_of::<obgpu_counters>());
    c.struct_size = (*out).struct_size;
    std::ptr::copy_nonoverlapping(&c as *const _ as *const u8, out as *mut u8, n);
    OBGPU_OK
}
