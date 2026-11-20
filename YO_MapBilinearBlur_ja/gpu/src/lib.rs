//! Map Bilinear Blur - GPU engine (wgpu). C ABI in include/mbbgpu.h.
//!
//! A port of src/mbb_core.cpp: the C++ side describes the frame (ladder, level ranges, settings) and
//! passes the level-0 buffer; this library builds the pyramid, the Protect-Edges chains and evaluates
//! every output pixel on the GPU (mbb.wgsl), then reads the premultiplied RGBA result back.
//! Pattern (device choice, loss / busy / VRAM fallbacks, non-blocking waits) follows Optic Bokeh's obgpu
//! (LensBlur\gpu, read only). Any status other than OK / ABORTED means: render this frame on the CPU.

use std::ffi::{c_char, c_int, c_void};
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::task::{Context as TaskCx, Poll, Waker};
use std::time::{Duration, Instant};

pub const OK: c_int = 0;
pub const NO_ADAPTER: c_int = 1;
pub const DEVICE_LOST: c_int = 2;
pub const OUT_OF_VRAM: c_int = 3;
pub const BUSY: c_int = 4;
pub const TIMEOUT: c_int = 5;
pub const ABORTED: c_int = 6;
pub const BAD_ARGS: c_int = 7;
pub const UNSUPPORTED: c_int = 8;
pub const INTERNAL: c_int = 9;
pub const DISABLED: c_int = 10;
pub const NOT_READY: c_int = 11;

const MAXE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Range {
    pub x0: i32,
    pub y0: i32,
    pub w: i32,
    pub h: i32,
    pub ext_w: i32,
    pub ext_h: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Desc {
    pub size: u32,
    pub d: [i32; 4],
    pub o: [i32; 4],
    pub o1: [i32; 4],
    pub r0: [i32; 4],
    pub interp: i32,
    pub down_smooth: i32,
    pub blend: i32,
    pub anti_ring: i32,
    pub protect: i32,
    pub clip: i32,
    pub dither_seed: u32,
    pub dither_frame: i32,
    pub sharp_type: i32,
    pub ring: i32,
    pub has_amount: i32,
    pub top: i32,
    pub lp: i32,
    pub pad0: i32,
    pub radius: f64,
    pub ds: f64,
    pub sharp_amount: f64,
    pub sharp_radius: f64,
    pub lev: [i32; MAXE],
    pub var: [f64; MAXE],
    pub rg: [Range; MAXE],
}

pub type AbortFn = Option<unsafe extern "C" fn(*mut c_void) -> c_int>;

fn block_on<F: Future>(f: F) -> Option<F::Output> {
    let mut f = pin!(f);
    let mut cx = TaskCx::from_waker(Waker::noop());
    let t0 = Instant::now();
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return Some(v);
        }
        if t0.elapsed() > Duration::from_secs(10) {
            return None;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
}

// ------------------------------------------------------------------ device ---
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: String,
    lost: Arc<AtomicBool>,
    bgl: wgpu::BindGroupLayout,
    pipes: Vec<wgpu::ComputePipeline>, // lamfr, down_h, down_v, agg, eval, sharpen
    max_binding: u64,
    max_buffer: u64,
    ubo_align: u64,
}

const ENTRIES: [&str; 6] = ["lamfr", "down_h", "down_v", "agg", "eval", "sharpen"];
const P_LAMFR: usize = 0;
const P_DOWNH: usize = 1;
const P_DOWNV: usize = 2;
const P_AGG: usize = 3;
const P_EVAL: usize = 4;
const P_SHARP: usize = 5;

fn create_gpu() -> Result<Gpu, (c_int, String)> {
    // mac_port: backend per platform. Windows: Vulkan + DX12 (v0.3.1). Apple: Metal only (no MoltenVK, no GL).
    // NOTE (Metal): Metal Shading Language has no 64-bit floats, so wgpu never reports SHADER_F64 on Metal and the
    // adapter check below rejects every Apple adapter -> NO_ADAPTER -> the plug-in renders on the CPU for the
    // session. That is the intended first-port behaviour (PORTING_NOTES.md "GPU").
    let backends = if cfg!(windows) {
        wgpu::Backends::VULKAN | wgpu::Backends::DX12
    } else if cfg!(target_vendor = "apple") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    desc.flags = wgpu::InstanceFlags::empty();
    desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::Fxc; // read by the DX12 backend only
    let instance = wgpu::Instance::new(desc);
    let adapters = block_on(instance.enumerate_adapters(backends)).unwrap_or_default();
    let mut best: Option<(i32, wgpu::Adapter)> = None;
    let mut seen = Vec::new();
    for a in adapters {
        let info = a.get_info();
        let lname = info.name.to_lowercase();
        let mut score = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 300,
            wgpu::DeviceType::IntegratedGpu => 200,
            wgpu::DeviceType::VirtualGpu => 100,
            wgpu::DeviceType::Other => 50,
            wgpu::DeviceType::Cpu => -1,
        };
        if lname.contains("microsoft basic render") || lname.contains("llvmpipe") || lname.contains("swiftshader") {
            score = -1;
        }
        if cfg!(windows) && score >= 0 && info.backend == wgpu::Backend::Vulkan {
            score += 5; // Windows only: Vulkan compiles WGSL much faster than DX12 / FXC (Optic Bokeh measurement)
        }
        let lim = a.limits();
        if !a.features().contains(wgpu::Features::SHADER_F64)
            || !a.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            || lim.max_storage_buffers_per_shader_stage < 4
            || lim.max_compute_invocations_per_workgroup < 256
        {
            score = -1;
        }
        seen.push(format!("{} ({:?}) {}", info.name, info.backend, score));
        if score >= 0 && best.as_ref().map(|b| score > b.0).unwrap_or(true) {
            best = Some((score, a));
        }
    }
    let Some((_, adapter)) = best else {
        return Err((NO_ADAPTER, format!("no suitable adapter (64-bit floats in shaders required): [{}]", seen.join("; "))));
    };
    let info = adapter.get_info();
    let al = adapter.limits();
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size.min(2u64 << 30);
    limits.max_buffer_size = al.max_buffer_size.min(4u64 << 30);
    limits.max_storage_buffers_per_shader_stage = 4;
    let ddesc = wgpu::DeviceDescriptor {
        label: Some("mbbgpu"),
        required_features: wgpu::Features::SHADER_F64,
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = match block_on(adapter.request_device(&ddesc)) {
        Some(Ok(dq)) => dq,
        // v0.3.1: INTERNAL (retried once), not NO_ADAPTER (that one stays off for the session)
        Some(Err(e)) => return Err((INTERNAL, format!("request_device: {e}"))),
        None => return Err((INTERNAL, "request_device did not complete".into())),
    };
    let lost = Arc::new(AtomicBool::new(false));
    {
        let l1 = lost.clone();
        device.set_device_lost_callback(move |_r, _m| l1.store(true, Ordering::SeqCst));
        let l2 = lost.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            if format!("{e}").to_lowercase().contains("lost") {
                l2.store(true, Ordering::SeqCst);
            }
        }));
    }
    let storage = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mbbgpu.bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(64) },
                count: None,
            },
            storage(1, false),
            storage(2, true),
            storage(3, true),
            storage(4, false),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mbbgpu.layout"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let iscope = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("mbb.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("mbb.wgsl").into()) });
    let mut pipes = Vec::new();
    for e in ENTRIES {
        pipes.push(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(e),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(e),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &[], zero_initialize_workgroup_memory: false },
            cache: None,
        }));
    }
    let ierr = block_on(iscope.pop()).flatten();
    if let Some(Some(err)) = block_on(scope.pop()) {
        return Err((INTERNAL, format!("shader: {err}")));
    }
    if let Some(err) = ierr {
        return Err((INTERNAL, format!("shader (backend): {err}")));
    }
    Ok(Gpu {
        info: format!("{} ({:?}, driver {})", info.name, info.backend, info.driver_info),
        ubo_align: limits.min_uniform_buffer_offset_alignment as u64,
        max_binding: limits.max_storage_buffer_binding_size,
        max_buffer: limits.max_buffer_size,
        device,
        queue,
        lost,
        bgl,
        pipes,
    })
}

// --------------------------------------------------------------- the state ---
struct State {
    gpu: Option<Arc<Gpu>>,
    status: c_int, // OK when gpu is set; else why not
    initing: bool,
    losses: Vec<Instant>,
    info: String,
    init_started: Option<Instant>, // v0.3.1: the current / last initialisation began here
    init_ms: f64,                  // duration of the last finished initialisation (-1 = none yet)
    init_failures: u32,            // initialisations that failed for a reason other than "no adapter"
}
static STATE: OnceLock<(Mutex<State>, Condvar)> = OnceLock::new();
fn state() -> &'static (Mutex<State>, Condvar) {
    STATE.get_or_init(|| {
        (
            Mutex::new(State { gpu: None, status: NOT_READY, initing: false, losses: Vec::new(), info: String::new(), init_started: None, init_ms: -1.0, init_failures: 0 }),
            Condvar::new(),
        )
    })
}
static HOOK: AtomicI32 = AtomicI32::new(0);
static HOOK_INIT_DELAY_MS: AtomicU32 = AtomicU32::new(0); // tests: make the initialisation this much slower
static ACTIVE: AtomicI32 = AtomicI32::new(0);
const MAX_JOBS: i32 = 2;
static SLOT_WAIT_MS: AtomicU32 = AtomicU32::new(2000); // how long a job waits for one of the MAX_JOBS slots
static INFO_C: Mutex<Vec<u8>> = Mutex::new(Vec::new());
thread_local! {
    static LAST_SLOT_WAIT_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}
// an initialisation that failed for another reason than "no adapter / no 64-bit floats" (a device request that
// failed or did not complete, a backend compile error) is retried this many times, then the GPU stays off
const MAX_INIT_FAILURES: u32 = 2;

fn start_init() {
    let (m, cv) = state();
    let mut st = m.lock().unwrap();
    if st.initing || st.gpu.is_some() || st.status == DISABLED || st.status == NO_ADAPTER || st.init_failures >= MAX_INIT_FAILURES {
        return;
    }
    st.initing = true;
    st.status = NOT_READY;
    let t_start = Instant::now();
    st.init_started = Some(t_start);
    drop(st);
    std::thread::spawn(move || {
        let delay = HOOK_INIT_DELAY_MS.load(Ordering::SeqCst);
        if delay > 0 {
            std::thread::sleep(Duration::from_millis(delay as u64));
        }
        let r = match HOOK.load(Ordering::SeqCst) {
            1 => Err((NO_ADAPTER, "test hook: no adapter".to_string())),
            5 => {
                HOOK.store(0, Ordering::SeqCst); // fails once, the retry works
                Err((INTERNAL, "test hook: initialisation failed once".to_string()))
            }
            6 => Err((INTERNAL, "test hook: initialisation always fails".to_string())),
            _ => create_gpu(),
        };
        let mut st = m.lock().unwrap();
        st.initing = false;
        st.init_ms = t_start.elapsed().as_secs_f64() * 1e3;
        match r {
            Ok(g) => {
                st.info = g.info.clone();
                st.gpu = Some(Arc::new(g));
                st.status = OK;
            }
            Err((s, msg)) => {
                st.info = msg;
                st.status = s;
                if s != NO_ADAPTER {
                    st.init_failures += 1;
                    if st.init_failures >= MAX_INIT_FAILURES {
                        st.status = DISABLED;
                    }
                }
            }
        }
        cv.notify_all();
    });
}

fn note_loss() {
    let (m, _) = state();
    let mut st = m.lock().unwrap();
    st.gpu = None;
    let now = Instant::now();
    st.losses.retain(|t| now.duration_since(*t) < Duration::from_secs(60));
    st.losses.push(now);
    st.status = if st.losses.len() >= 3 { DISABLED } else { NOT_READY };
    drop(st);
    if m.lock().unwrap().status != DISABLED {
        start_init(); // re-create in the background; frames meanwhile on the CPU
    }
}

// ----------------------------------------------------------------- the job ---
#[derive(Clone, Copy)]
struct Img {
    off: u32, // in floats
    x0: i32,
    y0: i32,
    w: i32,
    h: i32,
    ext_w: i32,
    ext_h: i32,
    ch: i32,
}

struct Plan {
    imgs: Vec<Img>,
    floats: u64,
}
impl Plan {
    fn add(&mut self, x0: i32, y0: i32, w: i32, h: i32, ext_w: i32, ext_h: i32, ch: i32) -> i32 {
        let n = (w.max(0) as u64) * (h.max(0) as u64) * ch as u64;
        self.floats = (self.floats + 63) / 64 * 64;
        self.imgs.push(Img { off: self.floats as u32, x0, y0, w, h, ext_w, ext_h, ch });
        self.floats += n.max(1);
        (self.imgs.len() - 1) as i32
    }
    fn alias(&mut self, of: i32, r: &Range) -> i32 {
        let o = self.imgs[of as usize].off;
        self.imgs.push(Img { off: o, x0: r.x0, y0: r.y0, w: r.w, h: r.h, ext_w: r.ext_w, ext_h: r.ext_h, ch: 5 });
        (self.imgs.len() - 1) as i32
    }
}

fn ext(n: i32, l: i32) -> i32 {
    ((n + (1 << l) - 1) >> l).max(1)
}

struct Disp {
    pipe: usize,
    p: [i32; 16],
    gx: u32,
    gy: u32,
}

fn le_i32(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}

struct Glob {
    pyr: [i32; MAXE],
    far: [i32; MAXE],
    near: [i32; MAXE],
    near_has: [i32; MAXE],
    band: [[i32; 4]; MAXE],
    lam: [i32; MAXE],
    any_front: i32,
    lam_built: i32,
    img_base: i32,
    img_amount: i32,
    img_lam: i32,
    img_fr: i32,
    img_out1: i32,
    img_out: i32,
}

fn glob_bytes(d: &Desc, g: &Glob) -> Vec<u8> {
    let mut b = Vec::with_capacity(1760);
    for v in [d.d, d.o, d.o1, d.r0] {
        for x in v {
            le_i32(&mut b, x);
        }
    }
    for x in [d.interp, d.down_smooth, d.blend, d.anti_ring] {
        le_i32(&mut b, x);
    }
    for x in [d.protect, d.clip, d.dither_seed as i32, d.dither_frame] {
        le_i32(&mut b, x);
    }
    for x in [d.sharp_type, d.ring, d.has_amount, d.top] {
        le_i32(&mut b, x);
    }
    for x in [d.lp, g.any_front, g.img_base, g.img_amount] {
        le_i32(&mut b, x);
    }
    for x in [g.img_lam, g.img_fr, g.img_out1, g.img_out] {
        le_i32(&mut b, x);
    }
    for x in [g.lam_built, g.any_front, 0, 0] {
        le_i32(&mut b, x);
    }
    debug_assert_eq!(b.len(), 160);
    let s3 = 3.0f64.sqrt();
    let hx = (2.0f64 / s3).sqrt();
    for x in [d.radius, d.ds, d.sharp_amount, d.sharp_radius, hx, s3, 0.0, 0.0] {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for x in d.lev {
        le_i32(&mut b, x);
    }
    for x in d.var {
        b.extend_from_slice(&x.to_le_bytes());
    }
    for arr in [&g.pyr, &g.far, &g.near, &g.near_has] {
        for x in arr.iter() {
            le_i32(&mut b, *x);
        }
    }
    for bb in g.band.iter() {
        for x in bb {
            le_i32(&mut b, *x);
        }
    }
    for x in g.lam {
        le_i32(&mut b, x);
    }
    debug_assert_eq!(b.len(), 1760);
    b
}

struct Job<'a> {
    gpu: &'a Gpu,
    abort: &'a dyn Fn() -> bool,
}

impl Job<'_> {
    fn wait(&self, flag: &Arc<AtomicBool>, timeout: Duration) -> Result<(), c_int> {
        let t0 = Instant::now();
        loop {
            if flag.load(Ordering::SeqCst) {
                return Ok(());
            }
            if self.gpu.device.poll(wgpu::PollType::Poll).is_err() {
                self.gpu.lost.store(true, Ordering::SeqCst);
                return Err(DEVICE_LOST);
            }
            if flag.load(Ordering::SeqCst) {
                return Ok(());
            }
            if self.gpu.lost.load(Ordering::SeqCst) {
                return Err(DEVICE_LOST);
            }
            if (self.abort)() {
                return Err(ABORTED);
            }
            if t0.elapsed() > timeout {
                return Err(TIMEOUT);
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }
    fn submit(&self, cb: wgpu::CommandBuffer) -> Arc<AtomicBool> {
        self.gpu.queue.submit(Some(cb));
        let f = Arc::new(AtomicBool::new(false));
        let f2 = f.clone();
        self.gpu.queue.on_submitted_work_done(move || f2.store(true, Ordering::SeqCst));
        f
    }
    fn read(&self, src: &wgpu::Buffer, off: u64, bytes: u64) -> Result<Vec<u8>, c_int> {
        let mut v = vec![0u8; bytes as usize];
        self.read_into(src, off, &mut v)?;
        Ok(v)
    }
    /// reads straight into `dst` (no intermediate copy)
    fn read_into(&self, src: &wgpu::Buffer, off: u64, dst: &mut [u8]) -> Result<(), c_int> {
        let bytes = dst.len() as u64;
        let rb = self.gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mbbgpu.read"), size: bytes.max(4), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut e = self.gpu.device.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(src, off, &rb, 0, bytes);
        let f = self.submit(e.finish());
        self.wait(&f, Duration::from_secs(20))?;
        let sl = rb.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let t0 = Instant::now();
        loop {
            let _ = self.gpu.device.poll(wgpu::PollType::Poll);
            if let Ok(ok) = rx.try_recv() {
                if !ok {
                    return Err(if self.gpu.lost.load(Ordering::SeqCst) { DEVICE_LOST } else { INTERNAL });
                }
                break;
            }
            if t0.elapsed() > Duration::from_secs(10) {
                return Err(TIMEOUT);
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        {
            let m = sl.get_mapped_range().map_err(|_| INTERNAL)?;
            dst.copy_from_slice(&m);
        }
        rb.destroy();
        Ok(())
    }
}

fn bytes_f32(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

// a Down call (Engine's Down): horizontal pass into tmp, vertical pass into dst
#[allow(clippy::too_many_arguments)]
fn push_down(ds: &mut Vec<Disp>, plan: &Plan, d: &Desc, dst: i32, src: i32, aux: i32, mode: i32, k: i32, j: i32, tmp: i32, lsrc: i32) {
    let im = plan.imgs[dst as usize];
    let sm = d.down_smooth != 0;
    let (taps, off) = if sm { (6, -2) } else { (2, 0) };
    let src_ext_w = ext(d.d[2] - d.d[0], lsrc);
    let src_ext_h = ext(d.d[3] - d.d[1], lsrc);
    let sy0 = (2 * im.y0 + off).max(0);
    let sy1 = src_ext_h.min(2 * (im.y0 + im.h - 1) + off + taps);
    let th = sy1 - sy0;
    if th <= 0 || im.w <= 0 || im.h <= 0 {
        return;
    }
    let mut p = [0i32; 16];
    p[0] = mode;
    p[1] = dst;
    p[2] = src;
    p[3] = aux;
    p[4] = k;
    p[5] = j;
    p[6] = sm as i32;
    p[7] = tmp;
    p[8] = sy0;
    p[9] = th;
    p[10] = src_ext_w;
    p[11] = src_ext_h;
    ds.push(Disp { pipe: P_DOWNH, p, gx: (im.w as u32).div_ceil(16), gy: (th as u32).div_ceil(16) });
    let mut q = p;
    q[9] = sy1;
    ds.push(Disp { pipe: P_DOWNV, p: q, gx: (im.w as u32).div_ceil(16), gy: (im.h as u32).div_ceil(16) });
}

fn run_dispatches(job: &Job, ds: &[Disp], bg_for: &dyn Fn(&wgpu::Buffer) -> wgpu::BindGroup) -> Result<Arc<AtomicBool>, c_int> {
    let g = job.gpu;
    let stride = 256u64.max(g.ubo_align);
    let n = ds.len().max(1) as u64;
    let ubo = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mbbgpu.disp"), size: n * stride, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut bytes = vec![0u8; (n * stride) as usize];
    for (i, d) in ds.iter().enumerate() {
        for (q, v) in d.p.iter().enumerate() {
            let o = i * stride as usize + q * 4;
            bytes[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    g.queue.write_buffer(&ubo, 0, &bytes);
    let bg = bg_for(&ubo);
    let mut e = g.device.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for (i, d) in ds.iter().enumerate() {
            if d.gx == 0 || d.gy == 0 {
                continue;
            }
            pass.set_pipeline(&g.pipes[d.pipe]);
            pass.set_bind_group(0, &bg, &[(i as u64 * stride) as u32]);
            pass.dispatch_workgroups(d.gx, d.gy, 1);
        }
    }
    Ok(job.submit(e.finish()))
}

fn prof_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MBBGPU_PROFILE").is_ok())
}

fn render(g: &Gpu, d: &Desc, base5: &[f32], amount: Option<&[f32]>, out: &mut [f32], abort: &dyn Fn() -> bool) -> Result<(), c_int> {
    let job = Job { gpu: g, abort };
    let t_start = Instant::now();
    let mut t_last = t_start;
    let mut lap = |what: &str| {
        if prof_on() {
            let n = Instant::now();
            eprintln!("    [gpu] {:<10} {:7.1} ms", what, (n - t_last).as_secs_f64() * 1e3);
            t_last = n;
        }
    };
    let top = d.top as usize;
    let lp = d.lp;
    if !(0..MAXE as i32 - 1).contains(&d.top) || !(0..MAXE as i32).contains(&lp) {
        return Err(BAD_ARGS);
    }
    if !matches!(d.interp, 0 | 1 | 2 | 3 | 5 | 7) {
        return Err(UNSUPPORTED); // Mitchell / Lanczos 3 are not offered in the UI: CPU only
    }
    let (r0w, r0h) = (d.r0[2] - d.r0[0], d.r0[3] - d.r0[1]);
    let (dw, dh) = (d.d[2] - d.d[0], d.d[3] - d.d[1]);
    let (ow, oh) = (d.o[2] - d.o[0], d.o[3] - d.o[1]);
    let (o1w, o1h) = (d.o1[2] - d.o1[0], d.o1[3] - d.o1[1]);
    let n0 = (r0w as usize) * (r0h as usize);
    if r0w <= 0 || r0h <= 0 || ow <= 0 || oh <= 0 || base5.len() != n0 * 5 || out.len() != (ow * oh * 4) as usize {
        return Err(BAD_ARGS);
    }
    let protect = d.protect != 0;
    // ---- memory plan (worst case: every chain built)
    let mut plan = Plan { imgs: Vec::new(), floats: 0 };
    let rg = |l: i32| d.rg[l as usize];
    let base = plan.add(d.r0[0] - d.d[0], d.r0[1] - d.d[1], r0w, r0h, dw, dh, 5);
    let amount_img = if d.has_amount != 0 { plan.add(0, 0, r0w, r0h, dw, dh, 1) } else { -1 };
    let (lam_img, fr_img) = if protect { (plan.add(0, 0, r0w, r0h, dw, dh, 1), plan.add(0, 0, r0w, r0h, dw, dh, 1)) } else { (-1, -1) };
    let mut pyr = [base; MAXE];
    for l in 1..=lp {
        let r = rg(l);
        pyr[l as usize] = plan.add(r.x0, r.y0, r.w, r.h, r.ext_w, r.ext_h, 5);
    }
    let out1 = if d.ring != 0 { plan.add(0, 0, o1w, o1h, 1, 1, 4) } else { -1 };
    let outi = plan.add(0, 0, ow, oh, 1, 1, 4);
    // tmp of the horizontal pass: dst width x source rows
    let mut tmp_n = 1u64;
    for l in 1..=lp {
        let r = rg(l);
        tmp_n = tmp_n.max(r.w as u64 * (2 * r.h as u64 + 6) * 5);
    }
    let tmp = plan.add(0, 0, tmp_n as i32, 1, 1, 1, 1);
    let (mut sa, mut sb) = (-1, -1);
    let mut chain_far = [-1i32; MAXE];
    let mut chain_near = [-1i32; MAXE];
    let mut aggs = [-1i32; MAXE];
    let mut lamp = [-1i32; MAXE];
    let mut scr_a = [-1i32; MAXE];
    let mut scr_b = [-1i32; MAXE];
    if protect && lp >= 1 {
        let r1 = rg(1);
        sa = plan.add(r1.x0, r1.y0, r1.w, r1.h, r1.ext_w, r1.ext_h, 5);
        sb = plan.add(r1.x0, r1.y0, r1.w, r1.h, r1.ext_w, r1.ext_h, 5);
        for l in 1..=lp {
            let r = rg(l);
            scr_a[l as usize] = plan.alias(sa, &r);
            scr_b[l as usize] = plan.alias(sb, &r);
            aggs[l as usize] = plan.add(r.x0, r.y0, r.w, r.h, r.ext_w, r.ext_h, 5);
            lamp[l as usize] = plan.add(r.x0, r.y0, r.w, r.h, r.ext_w, r.ext_h, 5);
        }
        for k in 1..=top {
            let l = d.lev[k];
            if l >= 1 {
                let r = rg(l);
                chain_far[k] = plan.add(r.x0, r.y0, r.w, r.h, r.ext_w, r.ext_h, 5);
                chain_near[k] = plan.add(r.x0, r.y0, r.w, r.h, r.ext_w, r.ext_h, 5);
            }
        }
    }
    let _ = (sa, sb);
    let bytes = plan.floats * 4;
    let budget = (1536u64 << 20).min(g.max_binding).min(g.max_buffer);
    if HOOK.load(Ordering::SeqCst) == 3 || bytes > budget {
        return Err(OUT_OF_VRAM);
    }
    // ---- buffers
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mem = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mbbgpu.mem"), size: bytes, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let mut tb = Vec::with_capacity(plan.imgs.len() * 32);
    for im in &plan.imgs {
        tb.extend_from_slice(&im.off.to_le_bytes());
        for v in [im.x0, im.y0, im.w, im.h, im.ext_w, im.ext_h, im.ch] {
            le_i32(&mut tb, v);
        }
    }
    let tab = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mbbgpu.tab"), size: tb.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let globb = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mbbgpu.glob"), size: 1760, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let scan = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some("mbbgpu.scan"), size: 162 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    if let Some(Some(_)) = block_on(scope.pop()) {
        return Err(OUT_OF_VRAM);
    }
    let free = |bs: &[&wgpu::Buffer]| {
        for b in bs {
            b.destroy();
        }
    };
    let result = (|| -> Result<(), c_int> {
        g.queue.write_buffer(&tab, 0, &tb);
        g.queue.write_buffer(&mem, plan.imgs[base as usize].off as u64 * 4, bytes_f32(base5));
        if let (Some(a), true) = (amount, amount_img >= 0) {
            if a.len() != n0 {
                return Err(BAD_ARGS);
            }
            g.queue.write_buffer(&mem, plan.imgs[amount_img as usize].off as u64 * 4, bytes_f32(a));
        }
        let mut sc = vec![0i32; 162];
        sc[1] = i32::MAX;
        for jj in 0..MAXE {
            sc[34 + jj * 4] = 1 << 30;
            sc[35 + jj * 4] = -(1 << 30);
            sc[36 + jj * 4] = 1 << 30;
            sc[37 + jj * 4] = -(1 << 30);
        }
        g.queue.write_buffer(&scan, 0, unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const u8, sc.len() * 4) });
        let mut gl = Glob {
            pyr,
            far: [0; MAXE],
            near: [-1; MAXE],
            near_has: [0; MAXE],
            band: [[1 << 30, -(1 << 30), 1 << 30, -(1 << 30)]; MAXE],
            lam: [0; MAXE],
            any_front: 0,
            lam_built: 0,
            img_base: base,
            img_amount: amount_img.max(0),
            img_lam: lam_img.max(0),
            img_fr: fr_img.max(0),
            img_out1: out1.max(0),
            img_out: outi,
        };
        g.queue.write_buffer(&globb, 0, &glob_bytes(d, &gl));
        let bg_for = |ubo: &wgpu::Buffer| {
            g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &g.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: ubo, offset: 0, size: wgpu::BufferSize::new(64) }) },
                    wgpu::BindGroupEntry { binding: 1, resource: mem.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: tab.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: globb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: scan.as_entire_binding() },
                ],
            })
        };
        // ---- pass 1: lam / fr + scan, the unweighted pyramid
        let mut ds1: Vec<Disp> = Vec::new();
        if protect {
            ds1.push(Disp { pipe: P_LAMFR, p: [0; 16], gx: (r0w as u32).div_ceil(16), gy: (r0h as u32).div_ceil(16) });
        }
        for l in 1..=lp {
            push_down(&mut ds1, &plan, d, pyr[l as usize], pyr[l as usize - 1], 0, 0, 0, 0, tmp, l - 1);
        }
        lap("setup+upload");
        let f1 = run_dispatches(&job, &ds1, &bg_for)?;
        let mut ds2: Vec<Disp> = Vec::new();
        if protect {
            job.wait(&f1, Duration::from_secs(20))?;
            let sv = job.read(&scan, 0, 162 * 4)?;
            let s = |i: usize| i32::from_le_bytes(sv[4 * i..4 * i + 4].try_into().unwrap());
            let any_front = s(0) != 0;
            let min_lam = if s(1) == i32::MAX { 1e9f32 } else { f32::from_bits(s(1) as u32) };
            gl.any_front = any_front as i32;
            for jj in 0..=top {
                gl.near_has[jj] = (s(2 + jj) != 0) as i32;
                let lv = d.lev[jj].max(0);
                let gg = if jj == 0 { 3 } else { 9 * (1 << lv) + 4 };
                gl.band[jj] = [s(34 + jj * 4) - gg, s(35 + jj * 4) + gg, s(36 + jj * 4) - gg, s(37 + jj * 4) + gg];
            }
            // lamPyr
            if any_front && lp >= 1 {
                for l in 1..=lp {
                    let (src, mode) = if l == 1 { (base, 3) } else { (lamp[l as usize - 1], 0) };
                    push_down(&mut ds2, &plan, d, lamp[l as usize], src, 0, mode, 0, 0, tmp, l - 1);
                }
                gl.lam = lamp;
                gl.lam_built = 1;
            }
            // far chains
            let mut agg_built = false;
            for k in 1..=top {
                let l = d.lev[k];
                if l < 1 {
                    continue;
                }
                let kk = k as f32;
                if !any_front && (k <= 1 || min_lam >= kk - 1.0) {
                    gl.far[k] = pyr[l as usize];
                    continue;
                }
                let slev = (l - 2).max(0);
                let fin = chain_far[k];
                if slev == 0 {
                    for lv in 1..=l {
                        let dst = if lv == l { fin } else if lv % 2 == 1 { scr_a[lv as usize] } else { scr_b[lv as usize] };
                        let (src, mode) = if lv == 1 { (base, 1) } else { (if (lv - 1) % 2 == 1 { scr_a[lv as usize - 1] } else { scr_b[lv as usize - 1] }, 0) };
                        push_down(&mut ds2, &plan, d, dst, src, 0, mode, k as i32, 0, tmp, lv - 1);
                    }
                } else {
                    if !agg_built {
                        for la in 1..=lp {
                            let mut p = [0i32; 16];
                            p[0] = la;
                            p[1] = aggs[la as usize];
                            p[2] = if la == 1 { 0 } else { aggs[la as usize - 1] };
                            p[6] = d.down_smooth;
                            let r = rg(la);
                            ds2.push(Disp { pipe: P_AGG, p, gx: (r.w as u32).div_ceil(16), gy: (r.h as u32).div_ceil(16) });
                        }
                        agg_built = true;
                    }
                    for lv in slev + 1..=l {
                        let dst = if lv == l { fin } else if lv % 2 == 1 { scr_a[lv as usize] } else { scr_b[lv as usize] };
                        if lv == slev + 1 {
                            push_down(&mut ds2, &plan, d, dst, pyr[slev as usize], aggs[slev as usize], 4, k as i32, 0, tmp, lv - 1);
                        } else {
                            let src = if (lv - 1) % 2 == 1 { scr_a[lv as usize - 1] } else { scr_b[lv as usize - 1] };
                            push_down(&mut ds2, &plan, d, dst, src, 0, 0, 0, 0, tmp, lv - 1);
                        }
                    }
                }
                gl.far[k] = fin;
            }
            // near bands
            for j in 1..=top {
                let l = d.lev[j];
                if gl.near_has[j] == 0 || l < 1 {
                    continue;
                }
                let fin = chain_near[j];
                for lv in 1..=l {
                    let dst = if lv == l { fin } else if lv % 2 == 1 { scr_a[lv as usize] } else { scr_b[lv as usize] };
                    let (src, mode) = if lv == 1 { (base, 2) } else { (if (lv - 1) % 2 == 1 { scr_a[lv as usize - 1] } else { scr_b[lv as usize - 1] }, 0) };
                    push_down(&mut ds2, &plan, d, dst, src, 0, mode, 0, j as i32, tmp, lv - 1);
                }
                gl.near[j] = fin;
            }
            g.queue.write_buffer(&globb, 0, &glob_bytes(d, &gl));
            lap("pass1+scan");
        }
        if (job.abort)() {
            return Err(ABORTED);
        }
        ds2.push(Disp { pipe: P_EVAL, p: [0; 16], gx: (o1w as u32).div_ceil(16), gy: (o1h as u32).div_ceil(16) });
        if d.ring != 0 {
            ds2.push(Disp { pipe: P_SHARP, p: [0; 16], gx: (ow as u32).div_ceil(16), gy: (oh as u32).div_ceil(16) });
        }
        // chains write the scratch levels in submission order: one pass, dispatches serialised by wgpu's
        // storage-buffer barriers (all dispatches use the same read_write buffer)
        let f2 = run_dispatches(&job, &ds2, &bg_for)?;
        job.wait(&f2, Duration::from_secs(30))?;
        lap("compute");
        let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        job.read_into(&mem, plan.imgs[outi as usize].off as u64 * 4, ob)?;
        lap("readback");
        Ok(())
    })();
    free(&[&mem, &tab, &globb, &scan]);
    let _ = g.device.poll(wgpu::PollType::Poll);
    result
}

// ------------------------------------------------------------------- C ABI ---
#[no_mangle]
pub extern "C" fn mbbgpu_init_async() {
    start_init();
}

#[no_mangle]
pub extern "C" fn mbbgpu_wait_ready() -> c_int {
    start_init();
    let (m, cv) = state();
    let mut st = m.lock().unwrap();
    while st.initing {
        st = cv.wait(st).unwrap();
    }
    st.status
}

/// v0.3.1: waits while the device is being created, but only until `window_ms` after that initialisation began
/// (so a short render does not go to the CPU just because it started first, and a slow initialisation costs
/// each frame at most what is left of the window). Starts the initialisation if needed (also again after a
/// device loss, or once more after a failed attempt). Returns OK, NOT_READY (window over), ABORTED, or why the
/// GPU cannot be used (NO_ADAPTER, DISABLED, INTERNAL). Polls `abort` on the calling thread.
///
/// # Safety
/// `abort` (may be null) is called with `user`.
#[no_mangle]
pub unsafe extern "C" fn mbbgpu_wait_init(window_ms: u32, abort: AbortFn, user: *mut c_void) -> c_int {
    start_init();
    let (m, cv) = state();
    let window = Duration::from_millis(window_ms as u64);
    let mut st = m.lock().unwrap();
    loop {
        if st.gpu.is_some() {
            return OK;
        }
        if !st.initing {
            return if st.status == OK { NOT_READY } else { st.status };
        }
        let since = st.init_started.map(|t| t.elapsed()).unwrap_or_default();
        if since >= window {
            return NOT_READY;
        }
        let step = (window - since).min(Duration::from_millis(10));
        st = cv.wait_timeout(st, step).unwrap().0;
        if let Some(f) = abort {
            drop(st);
            if f(user) != 0 {
                return ABORTED;
            }
            st = m.lock().unwrap();
        }
    }
}

/// v0.3.1: how long one job may wait for a GPU slot before it is BUSY (default 2000 ms)
#[no_mangle]
pub extern "C" fn mbbgpu_set_slot_wait_ms(ms: u32) {
    SLOT_WAIT_MS.store(ms.max(1), Ordering::SeqCst);
}

/// v0.3.1: the slot wait of the calling thread's last mbbgpu_render call (ms)
#[no_mangle]
pub extern "C" fn mbbgpu_last_slot_wait_ms() -> f64 {
    LAST_SLOT_WAIT_MS.with(|c| c.get())
}

/// v0.3.1: duration of the last finished initialisation in ms (-1 = none finished yet)
#[no_mangle]
pub extern "C" fn mbbgpu_init_ms() -> f64 {
    state().0.lock().map(|s| s.init_ms).unwrap_or(-1.0)
}

/// v0.3.1: tests only: delay every following initialisation by `ms` (to imitate a slow start in a busy host)
#[no_mangle]
pub extern "C" fn mbbgpu_test_init_delay_ms(ms: u32) {
    HOOK_INIT_DELAY_MS.store(ms, Ordering::SeqCst);
}

/// # Safety
/// `desc` must point to an mbbgpu_desc; base5 / amount / out must hold the sizes it implies.
#[no_mangle]
pub unsafe extern "C" fn mbbgpu_render(desc: *const Desc, base5: *const f32, amount: *const f32, out: *mut f32, abort: AbortFn, user: *mut c_void) -> c_int {
    if desc.is_null() || base5.is_null() || out.is_null() {
        return BAD_ARGS;
    }
    let d = &*desc;
    if d.size as usize != std::mem::size_of::<Desc>() {
        return BAD_ARGS;
    }
    let ab = move || -> bool {
        match abort {
            Some(f) => f(user) != 0,
            None => false,
        }
    };
    let gpu = {
        let (m, _) = state();
        let st = m.lock().unwrap();
        match (&st.gpu, st.status) {
            (Some(g), _) => g.clone(),
            (None, s) => {
                drop(st);
                if s == NOT_READY {
                    start_init();
                }
                return if s == OK { NOT_READY } else { s };
            }
        }
    };
    if gpu.lost.load(Ordering::SeqCst) {
        note_loss();
        return DEVICE_LOST;
    }
    // job slots (MFR): at most MAX_JOBS at once; wait up to SLOT_WAIT_MS (2 s by default) polling the abort callback
    let hook = HOOK.load(Ordering::SeqCst);
    let t0 = Instant::now();
    let slot_wait = Duration::from_millis(SLOT_WAIT_MS.load(Ordering::SeqCst) as u64);
    LAST_SLOT_WAIT_MS.with(|c| c.set(0.0));
    loop {
        if hook == 2 {
            return BUSY;
        }
        let a = ACTIVE.load(Ordering::SeqCst);
        if a < MAX_JOBS && ACTIVE.compare_exchange(a, a + 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            break;
        }
        if ab() {
            return ABORTED;
        }
        if t0.elapsed() > slot_wait {
            LAST_SLOT_WAIT_MS.with(|c| c.set(t0.elapsed().as_secs_f64() * 1e3));
            return BUSY;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    LAST_SLOT_WAIT_MS.with(|c| c.set(t0.elapsed().as_secs_f64() * 1e3));
    struct Slot;
    impl Drop for Slot {
        fn drop(&mut self) {
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _slot = Slot;
    if hook == 4 {
        gpu.lost.store(true, Ordering::SeqCst);
        HOOK.store(0, Ordering::SeqCst);
    }
    let (r0w, r0h) = (d.r0[2] - d.r0[0], d.r0[3] - d.r0[1]);
    let (ow, oh) = (d.o[2] - d.o[0], d.o[3] - d.o[1]);
    if r0w <= 0 || r0h <= 0 || ow <= 0 || oh <= 0 {
        return BAD_ARGS;
    }
    let n0 = r0w as usize * r0h as usize;
    let b = std::slice::from_raw_parts(base5, n0 * 5);
    let a = if d.has_amount != 0 && !amount.is_null() { Some(std::slice::from_raw_parts(amount, n0)) } else { None };
    let o = std::slice::from_raw_parts_mut(out, ow as usize * oh as usize * 4);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| render(&gpu, d, b, a, o, &ab)));
    let st = match r {
        Ok(Ok(())) => OK,
        Ok(Err(s)) => s,
        Err(_) => INTERNAL,
    };
    if st == DEVICE_LOST || gpu.lost.load(Ordering::SeqCst) {
        note_loss();
        return DEVICE_LOST;
    }
    st
}

#[no_mangle]
pub extern "C" fn mbbgpu_info() -> *const c_char {
    let (m, _) = state();
    let s = m.lock().unwrap().info.clone();
    let mut v = INFO_C.lock().unwrap();
    *v = s.into_bytes();
    v.push(0);
    v.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn mbbgpu_status_string(s: c_int) -> *const c_char {
    let t: &'static [u8] = match s {
        OK => b"ok\0",
        NO_ADAPTER => b"no usable GPU\0",
        DEVICE_LOST => b"GPU device lost\0",
        OUT_OF_VRAM => b"not enough GPU memory\0",
        BUSY => b"GPU busy\0",
        TIMEOUT => b"GPU timeout\0",
        ABORTED => b"aborted\0",
        BAD_ARGS => b"bad arguments\0",
        UNSUPPORTED => b"unsupported settings\0",
        DISABLED => b"GPU disabled after errors\0",
        NOT_READY => b"GPU not ready yet\0",
        _ => b"internal error\0",
    };
    t.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn mbbgpu_shutdown() {
    let (m, _) = state();
    if let Ok(mut st) = m.lock() {
        st.gpu = None;
        if st.status == OK {
            st.status = NOT_READY;
        }
    }
}

#[no_mangle]
pub extern "C" fn mbbgpu_test_hook(h: c_int) {
    HOOK.store(h, Ordering::SeqCst);
    if h == 1 {
        let (m, _) = state();
        let mut st = m.lock().unwrap();
        st.gpu = None;
        st.status = NO_ADAPTER;
        st.info = "test hook: no adapter".into();
    }
}
