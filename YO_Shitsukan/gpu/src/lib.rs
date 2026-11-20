//! Shitsukan GPU path (wgpu, Vulkan). C ABI: include/skgpu.h.
//! 0.6.0.4: the line stage's heavy, exact filtering. 0.6.0.5: one GPU "frame" per rendered frame - a pooled arena of
//! planes (kept across frames, one per concurrent frame), one upload of the inputs, the results stay on the GPU
//! between the stages, only what the CPU needs is read back; and the painter's tail (wet-in-wet, density, slivers,
//! the paper) on the GPU. Every result is bit-identical to the CPU engine (shaders/lines.wgsl, shaders/tail.wgsl:
//! the arithmetic rules). Any non-OK status = the caller runs the CPU code for that stage.
use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

pub const SKGPU_OK: i32 = 0;
pub const SKGPU_BUSY: i32 = 1;
pub const SKGPU_LOST: i32 = 2;
pub const SKGPU_OOM: i32 = 3;
pub const SKGPU_ERROR: i32 = 4;

fn block_on<F: Future>(f: F, limit: Duration) -> Option<F::Output> {
    let mut f = pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    let t0 = Instant::now();
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return Some(v);
        }
        if t0.elapsed() > limit {
            return None;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct U {
    w: u32,
    h: u32,
    r: u32,
    mode: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    zero: u32,
    k: [f32; 8],
}

// ---- the arena: planes of S floats (S = N rounded up to 64: every plane starts 256-byte aligned)
const PL_C: usize = 0; // 3 planes: C (rgb) | tail: P (rgb)
const PL_A: usize = 3;
const PL_ALPHA: usize = 4;
const PL_Y: usize = 5; // tail: Dm
const PL_CH: usize = 6; // .. 13: line scratch | tail: aux slots 2..9 (aux = planes 4..13)
const PL_T1: usize = 7;
const PL_T2: usize = 8;
const PL_T3: usize = 9;
const PL_AS: usize = 10;
const PL_LIGHT: usize = 11;
const PL_V: usize = 12;
const PL_CNT: usize = 13;
const PL_WIN: usize = 14;
const PL_Q: usize = 15; // 4 planes (vec4 per pixel)
const PL_FH: usize = 19; // 3 planes: Fh | tail: Dw
const PLANES: usize = 22;
const UP_PLANES: usize = 7; // upload staging (bigger uploads go in several batches)
const DOWN_PLANES: usize = 5;
const TAIL_UNI: usize = 640; // TU after its "ps" vec4

struct Slot {
    s: usize,
    arena: wgpu::Buffer,
    up: wgpu::Buffer,
    down: wgpu::Buffer,
}

pub struct Ctx {
    dev: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    tail_bgl: Option<wgpu::BindGroupLayout>,
    tail_pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    etab: Option<(wgpu::Buffer, u32)>,
    dummy: wgpu::Buffer,
    lost: Arc<AtomicBool>,
    err: Arc<AtomicBool>,
    jobs: Mutex<u32>,
    cv: Condvar,
    pool: Mutex<Vec<Slot>>,
    max_binding: u64,
    max_buffer: u64,
    // 0.6.0.6: the adapter (for the diagnostic log) and the exp self-test: 1 passed, 0 failed, -1 not run (no f64)
    desc: String,
    exp_state: i32,
}

/// one frame: a slot of the pool (exclusive while the frame lives) and what is resident in it
pub struct Frame {
    ctx: *const Ctx,
    slot: Option<Slot>,
    w: u32,
    h: u32,
    a_ok: bool,
    alpha_ok: bool,
    prof: bool,
}

/// arguments of one line-stage job (see skgpu.h)
#[repr(C)]
pub struct SkLinesArgs {
    pub w: i32,
    pub h: i32,
    pub rc: i32,
    pub line_color: i32,
    pub bg_detail: i32,
    pub rb: i32,
    pub thr: f32,
    pub thr_a: f32,
    pub crf1: [f32; 3],
    pub crf4: [f32; 3],
    pub c: *const f32,
    pub a: *const f32,
    pub out_alpha: *mut f32,
    pub out_light: *mut u8,
    pub out_fh: *mut f32,
}

/// the painter's tail (see skgpu.h)
#[repr(C)]
pub struct SkTailArgs {
    pub stop: i32,
    pub wet_on: i32,
    pub pa_blur: i32,
    pub cwet: [f32; 3],
    pub crim: [f32; 3],
    pub cpa: [f32; 3],
    pub uni: *const u8,
    pub uni_bytes: u32,
    pub d64: *const f64,
    pub d64_count: u32,
    pub grids: *const f32,
    pub grids_count: u32,
    pub a: *const f32,
    pub alpha: *const f32,
    pub dw: *const f32,
    pub dm: *const f32,
    pub p: *const f32,
    pub ev: *const f32,
    pub wet: *const f32,
    pub bz: *const f32,
    pub dl: *const f32,
    pub pw: *const f32,
    pub dline: *const f32,
    pub ptex: *const f32,
    pub out_dw: *mut f32,
    pub out_dm: *mut f32,
    pub out_p: *mut f32,
    pub ink_on: i32,
    pub ink_soft: i32,
    pub cink: [f32; 3],
    pub delta: *const f32,
    pub far_w: *const f32,
}

const ENTRIES: [&str; 17] = [
    "k_prep", "k_mm_h", "k_mm_v", "k_tophat", "k_core", "k_merge", "k_v", "k_sum_h", "k_sum_v", "k_rf1_rows", "k_rf1_cols", "k_clutter", "k_bar", "k_fill4",
    "k_rf4_rows", "k_rf4_cols", "k_fh",
];
const TAIL_ENTRIES: [&str; 14] = [
    "k_inkprep", "k_ink",
    "k_wetprep", "k_q_rows", "k_q_cols", "k_wet", "k_rimprep", "k_density", "k_slivers", "k_paprep", "k_pa_rows", "k_pa_cols", "k_paper", "k_exptest",
];

// mac_port: the backend per platform (Cargo.toml enables only that one). Metal has no f64: tail_ok below is then false and
// only the line stage runs on the GPU; the engine's start-up self-test gates it bit for bit against the CPU.
#[cfg(target_os = "macos")]
const BACKENDS: wgpu::Backends = wgpu::Backends::METAL;
#[cfg(not(target_os = "macos"))]
const BACKENDS: wgpu::Backends = wgpu::Backends::VULKAN;

fn create() -> Option<Ctx> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = BACKENDS;
    desc.flags = wgpu::InstanceFlags::empty();
    let instance = wgpu::Instance::new(desc);
    let adapters = block_on(instance.enumerate_adapters(BACKENDS), Duration::from_secs(10))?;
    let mut best: Option<(i32, wgpu::Adapter)> = None;
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
        if lname.contains("llvmpipe") || lname.contains("swiftshader") || lname.contains("basic render") {
            score = -1;
        }
        let dl = a.get_downlevel_capabilities();
        if !dl.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) || a.limits().max_storage_buffers_per_shader_stage < 8 {
            score = -1;
        }
        if score >= 0 && best.as_ref().map(|b| score > b.0).unwrap_or(true) {
            best = Some((score, a));
        }
    }
    let (_, adapter) = best?;
    let desc = {
        let i = adapter.get_info();
        let s = format!("{} | vendor 0x{:04x} device 0x{:04x} | driver {} {}", i.name, i.vendor, i.device, i.driver, i.driver_info);
        s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect::<String>()
    };
    let al = adapter.limits();
    // the tail needs f64 (the paper noise coordinates, the exact exp) and 9 storage buffers per stage
    let tail_ok = adapter.features().contains(wgpu::Features::SHADER_F64) && al.max_storage_buffers_per_shader_stage >= 9;
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size.min(2u64 << 30);
    limits.max_buffer_size = al.max_buffer_size.min(4u64 << 30);
    limits.max_storage_buffers_per_shader_stage = if tail_ok { 9 } else { 8 };
    let (dev, queue) = block_on(
        adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("skgpu"),
            required_features: if tail_ok { wgpu::Features::SHADER_F64 } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }),
        Duration::from_secs(10),
    )?
    .ok()?;
    let lost = Arc::new(AtomicBool::new(false));
    let err = Arc::new(AtomicBool::new(false));
    {
        let l = lost.clone();
        dev.set_device_lost_callback(move |_r, _m| l.store(true, Ordering::SeqCst));
        let e = err.clone();
        dev.on_uncaptured_error(Arc::new(move |_e: wgpu::Error| e.store(true, Ordering::SeqCst)));
    }
    use wgpu::BindingType as T;
    let ent = |b: u32, ty: T| wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
    let st = |ro: bool| T::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: ro }, has_dynamic_offset: false, min_binding_size: None };
    let uni = T::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[ent(0, uni), ent(1, st(false)), ent(2, st(false)), ent(3, st(false)), ent(4, st(false)), ent(5, st(false)), ent(6, st(false)), ent(7, st(false))],
    });
    let mk_pipes = |bgl: &wgpu::BindGroupLayout, src: &str, name: &str, entries: &[&'static str]| {
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let mut pipes = HashMap::new();
        for e in entries.iter() {
            let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(e),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &[], zero_initialize_workgroup_memory: false },
                cache: None,
            });
            pipes.insert(*e, p);
        }
        pipes
    };
    let pipes = mk_pipes(&bgl, include_str!("../shaders/lines.wgsl"), "lines", &ENTRIES);
    if err.load(Ordering::SeqCst) {
        return None;
    }
    let mut tail_bgl = None;
    let mut tail_pipes = HashMap::new();
    let mut etab = None;
    if tail_ok {
        let tb = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                ent(0, uni),
                ent(1, st(true)),
                ent(2, st(false)),
                ent(3, st(false)),
                ent(4, st(false)),
                ent(5, st(false)),
                ent(6, st(false)),
                ent(7, st(false)),
                ent(8, st(true)),
                ent(9, st(true)),
            ],
        });
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let tp = mk_pipes(&tb, include_str!("../shaders/tail.wgsl"), "tail", &TAIL_ENTRIES);
        let popped = block_on(scope.pop(), Duration::from_secs(10));
        if std::env::var("SKGPU_PROFILE").is_ok() {
            if let Some(Some(e)) = &popped {
                let m = format!("{:?}", e);
                let k = m.find("labels").unwrap_or(0);
                eprintln!("  [skgpu] tail shader: {}", &m[k..]);
            }
        }
        let bad = popped.map(|e| e.is_some()).unwrap_or(true);
        if !bad && !err.load(Ordering::SeqCst) {
            let tab: &[u8] = include_bytes!("../shaders/exp_tab.bin");
            let b = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("exp table"), size: tab.len().max(8) as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
            {
                let mut v = b.slice(..).get_mapped_range_mut().ok()?;
                v.slice(..tab.len()).copy_from_slice(tab);
            }
            b.unmap();
            etab = Some((b, (tab.len() / 8) as u32));
            tail_bgl = Some(tb);
            tail_pipes = tp;
        }
        err.store(false, Ordering::SeqCst);
    }
    let dummy = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let ctx = Ctx {
        dev,
        queue,
        bgl,
        pipes,
        tail_bgl,
        tail_pipes,
        etab,
        dummy,
        lost,
        err,
        jobs: Mutex::new(0),
        cv: Condvar::new(),
        pool: Mutex::new(Vec::new()),
        max_binding: limits.max_storage_buffer_binding_size,
        max_buffer: limits.max_buffer_size,
        desc,
        exp_state: -1,
    };
    let mut ctx = ctx;
    if ctx.tail_bgl.is_some() {
        if ctx.exp_selftest() {
            ctx.exp_state = 1;
        } else {
            // (a driver whose f64 or float arithmetic is not the exact IEEE one: the tail stays on the CPU)
            ctx.exp_state = 0;
            ctx.tail_bgl = None;
            ctx.tail_pipes.clear();
            ctx.etab = None;
        }
    }
    Some(ctx)
}

/// copy src -> dst with a few threads (big planes: one thread alone is the bottleneck of an upload / read-back)
fn par_copy(dst: &mut [u8], src: &[u8]) {
    let n = dst.len().min(src.len());
    if n < (8 << 20) {
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    let parts = 8usize;
    let chunk = ((n + parts - 1) / parts + 63) & !63;
    std::thread::scope(|s| {
        for (d, sr) in dst[..n].chunks_mut(chunk).zip(src[..n].chunks(chunk)) {
            s.spawn(move || d.copy_from_slice(sr));
        }
    });
}

/// the same into a mapped (write-only) upload range
fn par_copy_w(mut dst: wgpu::WriteOnly<'_, [u8]>, src: &[u8]) {
    let n = src.len();
    if n < (8 << 20) {
        dst.copy_from_slice(src);
        return;
    }
    // (WriteOnly<[u8]> is not Send: the chunks are written through the raw pointer; ordinary writes, no reads)
    struct P(*mut u8);
    unsafe impl Send for P {}
    unsafe impl Sync for P {}
    let base = P(dst.as_raw_ptr().as_ptr() as *mut u8);
    let base = &base;
    let chunk = ((n + 7) / 8 + 63) & !63;
    std::thread::scope(|s| {
        let mut off = 0usize;
        while off < n {
            let len = chunk.min(n - off);
            let sr = &src[off..off + len];
            let o = off;
            s.spawn(move || unsafe { std::ptr::copy_nonoverlapping(sr.as_ptr(), base.0.add(o), len) });
            off += len;
        }
    });
}
/// test hook: SKGPU_FAULT=<stage>[,<stage>] (begin | lines | tail | ink) makes that call fail as if out of memory, so the
/// fake host can check that the CPU takes over with the same picture
fn fault(stage: &str) -> bool {
    match std::env::var("SKGPU_FAULT") {
        Ok(v) => v.split(',').any(|s| s == stage),
        Err(_) => false,
    }
}

/// 0.6.0.6 test hook: SKGPU_PERTURB=<stage>[,<stage>] (lines | wet | density | slivers | pablur | paper | ink): 64 values
/// of that stage's result are overwritten on the GPU right after its last pass (a wrong shader result), so the start-up
/// self-test can be shown to switch exactly that stage off with the picture unchanged
fn perturb_on(stage: &str) -> bool {
    match std::env::var("SKGPU_PERTURB") {
        Ok(v) => v.split(',').any(|s| s == stage),
        Err(_) => false,
    }
}

fn d64_fixed() -> Vec<f64> {
    // tail.wgsl's d64 table head: 1 / n!, 2^k, the exp constants, a zero (the host appends the frame's values)
    let mut v = vec![0f64; 339];
    let mut f = 1.0f64;
    for n in 0..14 {
        if n > 0 {
            f *= n as f64;
        }
        v[n] = 1.0 / f;
    }
    for k in -160i32..=160 {
        v[14 + (k + 160) as usize] = 2f64.powi(k);
    }
    v[335] = f64::from_bits(0x3FF71547652B82FE);
    v[336] = f64::from_bits(0x3FE62E42FEE00000);
    v[337] = f64::from_bits(0x3DEA39EF35793C76);
    v[338] = 0.0;
    v
}

impl Ctx {
    fn wait_done(&self, limit: Duration) -> bool {
        let t0 = Instant::now();
        loop {
            match self.dev.poll(wgpu::PollType::Poll) {
                Ok(s) if s.is_queue_empty() => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
            if self.lost.load(Ordering::SeqCst) || t0.elapsed() > limit {
                return false;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    fn map_wait(&self, sl: wgpu::BufferSlice, mode: wgpu::MapMode, limit: Duration) -> i32 {
        let mapped = Arc::new(AtomicBool::new(false));
        let ok = Arc::new(AtomicBool::new(false));
        {
            let (m2, o2) = (mapped.clone(), ok.clone());
            sl.map_async(mode, move |r| {
                o2.store(r.is_ok(), Ordering::SeqCst);
                m2.store(true, Ordering::SeqCst);
            });
        }
        let t0 = Instant::now();
        while !mapped.load(Ordering::SeqCst) {
            if self.dev.poll(wgpu::PollType::Poll).is_err() || self.lost.load(Ordering::SeqCst) {
                return SKGPU_LOST;
            }
            if t0.elapsed() > limit {
                self.lost.store(true, Ordering::SeqCst);
                return SKGPU_LOST;
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        if ok.load(Ordering::SeqCst) { SKGPU_OK } else { SKGPU_ERROR }
    }

    fn uniform(&self, bytes: &[u8]) -> wgpu::Buffer {
        let size = ((bytes.len() + 15) & !15) as u64;
        let ub = self.dev.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: true });
        {
            let mut v = ub.slice(..).get_mapped_range_mut().expect("map");
            v.slice(..bytes.len()).copy_from_slice(bytes);
        }
        ub.unmap();
        ub
    }
    fn storage_init(&self, bytes: &[u8]) -> wgpu::Buffer {
        let size = (((bytes.len() + 15) & !15).max(16)) as u64;
        let b = self.dev.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: true });
        {
            let mut v = b.slice(..).get_mapped_range_mut().expect("map");
            v.slice(..bytes.len()).copy_from_slice(bytes);
        }
        b.unmap();
        b
    }

    /// (test hook, see perturb_on) overwrite 64 floats in the middle of the arena range [off, off + len) with 0.4375
    fn poke(&self, enc: &mut wgpu::CommandEncoder, slot: &Slot, pk: &mut Vec<wgpu::Buffer>, off: usize, len: usize) {
        let cnt = 64usize.min(len);
        if cnt == 0 {
            return;
        }
        let at = off + (len / 2).min(len - cnt);
        let vals = vec![0.4375f32; cnt];
        let b = self.dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: (cnt * 4) as u64, usage: wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true });
        {
            let mut v = b.slice(..).get_mapped_range_mut().expect("map");
            v.copy_from_slice(unsafe { std::slice::from_raw_parts(vals.as_ptr() as *const u8, cnt * 4) });
        }
        b.unmap();
        enc.copy_buffer_to_buffer(&b, 0, &slot.arena, (at * 4) as u64, Some((cnt * 4) as u64));
        pk.push(b);
    }

    /// a slot with planes of at least s floats (from the pool when one is big enough)
    fn take_slot(&self, s: usize) -> Result<Slot, i32> {
        {
            let mut p = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(k) = p.iter().position(|sl| sl.s >= s) {
                return Ok(p.swap_remove(k));
            }
            // a smaller slot is dropped (its memory comes back before the bigger one is made)
            if !p.is_empty() {
                p.remove(0);
            }
        }
        let bytes = |planes: usize| (planes * s * 4) as u64;
        if bytes(PLANES) > self.max_buffer || bytes(10) > self.max_binding {
            return Err(SKGPU_OOM);
        }
        let scope = self.dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let arena = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arena"),
            size: bytes(PLANES),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let up = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("up"),
            size: bytes(UP_PLANES),
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let down = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("down"),
            size: bytes(DOWN_PLANES),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(Some(_e)) = block_on(scope.pop(), Duration::from_secs(5)) {
            return Err(SKGPU_OOM);
        }
        Ok(Slot { s, arena, up, down })
    }

    /// upload (float offset in the arena, source, floats); batches of at most UP_PLANES planes, each its own submit
    fn upload(&self, slot: &Slot, items: &[(usize, *const f32, usize)]) -> i32 {
        let cap = UP_PLANES * slot.s;
        let mut k = 0;
        while k < items.len() {
            let mut end = k;
            let mut tot = 0usize;
            while end < items.len() && (tot + items[end].2 <= cap || end == k) {
                tot += items[end].2;
                end += 1;
            }
            if tot > cap {
                return SKGPU_ERROR;
            }
            let sl = slot.up.slice(0..(tot * 4) as u64);
            let st = self.map_wait(sl, wgpu::MapMode::Write, Duration::from_secs(4));
            if st != SKGPU_OK {
                return st;
            }
            {
                let Ok(mut v) = slot.up.slice(0..(tot * 4) as u64).get_mapped_range_mut() else { return SKGPU_ERROR };
                let mut o = 0usize;
                for it in &items[k..end] {
                    let src = unsafe { std::slice::from_raw_parts(it.1 as *const u8, it.2 * 4) };
                    par_copy_w(v.slice(o..o + it.2 * 4), src);
                    o += it.2 * 4;
                }
            }
            slot.up.unmap();
            let mut enc = self.dev.create_command_encoder(&Default::default());
            let mut o = 0u64;
            for it in &items[k..end] {
                enc.copy_buffer_to_buffer(&slot.up, o, &slot.arena, (it.0 * 4) as u64, Some((it.2 * 4) as u64));
                o += (it.2 * 4) as u64;
            }
            self.queue.submit([enc.finish()]);
            k = end;
        }
        SKGPU_OK
    }

    /// submit enc + copies of (arena float offset, floats) to the read-back buffer, wait, then hand each range to out
    fn download(&self, slot: &Slot, mut enc: wgpu::CommandEncoder, items: &[(usize, usize)], out: &mut dyn FnMut(usize, &[u8])) -> i32 {
        let tot: usize = items.iter().map(|i| i.1).sum();
        if tot > DOWN_PLANES * slot.s {
            return SKGPU_ERROR;
        }
        let mut o = 0u64;
        for it in items {
            enc.copy_buffer_to_buffer(&slot.arena, (it.0 * 4) as u64, &slot.down, o, Some((it.1 * 4) as u64));
            o += (it.1 * 4) as u64;
        }
        self.queue.submit([enc.finish()]);
        let st = self.map_wait(slot.down.slice(0..(tot * 4) as u64), wgpu::MapMode::Read, Duration::from_secs(4));
        if st != SKGPU_OK {
            return st;
        }
        if self.err.load(Ordering::SeqCst) {
            slot.down.unmap();
            return SKGPU_ERROR;
        }
        {
            let Ok(v) = slot.down.slice(0..(tot * 4) as u64).get_mapped_range() else { return SKGPU_ERROR };
            let mut o = 0usize;
            for (k, it) in items.iter().enumerate() {
                out(k, &v[o..o + it.1 * 4]);
                o += it.1 * 4;
            }
        }
        slot.down.unmap();
        SKGPU_OK
    }

    fn rng<'a>(slot: &'a Slot, plane: usize, planes: usize) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &slot.arena,
            offset: (plane * slot.s * 4) as u64,
            size: NonZeroU64::new((planes * slot.s * 4) as u64),
        })
    }

    fn lines(&self, fr: &mut Frame, a: &SkLinesArgs) -> i32 {
        let (w, h) = (a.w as u32, a.h as u32);
        let n = (w as usize) * (h as usize);
        if n == 0 || a.c.is_null() || a.a.is_null() || a.out_alpha.is_null() || a.out_fh.is_null() || w != fr.w || h != fr.h {
            return SKGPU_ERROR;
        }
        let tp = Instant::now();
        let slot = fr.slot.as_ref().unwrap();
        let s = slot.s;
        let st = self.upload(slot, &[(PL_C * s, a.c, n * 3), (PL_A * s, a.a, n)]);
        if st != SKGPU_OK {
            return st;
        }
        fr.a_ok = true;
        let t_up = tp.elapsed();
        let dev = &self.dev;
        let mut enc = dev.create_command_encoder(&Default::default());
        // a plane argument: (first plane, planes); None = an unused binding
        type Pl = Option<(usize, usize)>;
        let g2 = ((w + 15) / 16, (h + 15) / 16);
        let gr = ((h + 63) / 64, 1);
        let gc = ((w + 63) / 64, 1);
        let rc = a.rc.max(1) as u32;
        let mk = |r: u32, mode: u32, f0: f32, f1: f32| U { w, h, r, mode, f0, f1, f2: 0.0, zero: 0, k: [0.45, 0.85, 0.30, 0.48, 0.03, 0.35, 0.0, 0.0] };
        let p1 = |pl: usize| Some((pl, 1usize));
        let (c, aa, y, ch, t1, t2, t3) = (Some((PL_C, 3)), p1(PL_A), p1(PL_Y), p1(PL_CH), p1(PL_T1), p1(PL_T2), p1(PL_T3));
        let (alpha, as_, light, v, cnt, win, q, fh) = (p1(PL_ALPHA), p1(PL_AS), p1(PL_LIGHT), p1(PL_V), p1(PL_CNT), p1(PL_WIN), Some((PL_Q, 4)), Some((PL_FH, 3)));
        // light lines are only written by the "also light lines" pass: start at 0
        enc.clear_buffer(&slot.arena, (PL_LIGHT * s * 4) as u64, Some((s * 4) as u64));
        let disp2 = |enc: &mut wgpu::CommandEncoder, e: &str, u: U, b: [Pl; 7], groups: (u32, u32)| {
            let ub = self.uniform(unsafe { std::slice::from_raw_parts(&u as *const U as *const u8, 64) });
            let res = |p: Pl| match p {
                Some((pl, np)) => Self::rng(slot, pl, np),
                None => self.dummy.as_entire_binding(),
            };
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: res(b[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: res(b[1]) },
                    wgpu::BindGroupEntry { binding: 3, resource: res(b[2]) },
                    wgpu::BindGroupEntry { binding: 4, resource: res(b[3]) },
                    wgpu::BindGroupEntry { binding: 5, resource: res(b[4]) },
                    wgpu::BindGroupEntry { binding: 6, resource: res(b[5]) },
                    wgpu::BindGroupEntry { binding: 7, resource: res(b[6]) },
                ],
            });
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.pipes[e]);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(groups.0, groups.1, 1);
            }
            (ub, bg)
        };
        let mut keep = Vec::new();
        let mut d2 = |enc: &mut wgpu::CommandEncoder, e: &str, u: U, b: [Pl; 7], groups: (u32, u32)| keep.push(disp2(enc, e, u, b, groups));
        d2(&mut enc, "k_prep", mk(0, 0, a.thr_a, 0.0), [c, aa, None, y, ch, None, None], g2);
        let mm2 = |enc: &mut wgpu::CommandEncoder, d2: &mut dyn FnMut(&mut wgpu::CommandEncoder, &str, U, [Pl; 7], (u32, u32)), src: Pl, dst: Pl, is_max: bool| {
            d2(enc, "k_mm_h", mk(rc, is_max as u32, 0.0, 0.0), [src, None, None, t3, None, None, None], g2);
            d2(enc, "k_mm_v", mk(rc, is_max as u32, 0.0, 0.0), [t3, None, None, dst, None, None, None], g2);
        };
        let tophat2 = |enc: &mut wgpu::CommandEncoder, d2: &mut dyn FnMut(&mut wgpu::CommandEncoder, &str, U, [Pl; 7], (u32, u32)), sig: Pl, dark: bool, th: f32, out: Pl| {
            mm2(enc, d2, sig, t1, dark);
            mm2(enc, d2, t1, t2, !dark);
            mm2(enc, d2, sig, t1, !dark);
            d2(enc, "k_tophat", mk(0, dark as u32, th, 0.0), [sig, t2, t1, out, None, None, None], g2);
        };
        tophat2(&mut enc, &mut d2, y, true, a.thr, alpha);
        d2(&mut enc, "k_core", mk(0, 0, 0.0, 0.0), [t1, None, None, alpha, None, None, None], g2);
        if a.line_color >= 1 {
            tophat2(&mut enc, &mut d2, ch, true, a.thr * 1.3f32, as_);
            d2(&mut enc, "k_merge", mk(0, 0, 0.0, 0.0), [as_, None, None, alpha, light, None, None], g2);
        }
        if a.line_color >= 2 {
            tophat2(&mut enc, &mut d2, y, false, a.thr * 1.2f32, as_);
            d2(&mut enc, "k_merge", mk(0, 1, 0.0, 0.0), [as_, None, None, alpha, light, None, None], g2);
        }
        let bg_detail = a.bg_detail != 0;
        if !bg_detail {
            let rb = a.rb.max(1) as u32;
            d2(&mut enc, "k_v", mk(0, 0, 0.0, 0.0), [alpha, None, None, v, None, None, None], g2);
            d2(&mut enc, "k_sum_h", mk(rb, 0, 0.0, 0.0), [v, None, None, t3, None, None, None], g2);
            d2(&mut enc, "k_sum_v", mk(rb, 0, 0.0, 0.0), [t3, None, None, cnt, None, None, None], g2);
            for it in 0..3 {
                d2(&mut enc, "k_rf1_rows", mk(0, 0, a.crf1[it], 0.0), [None, None, None, v, None, None, None], gr);
                d2(&mut enc, "k_rf1_cols", mk(0, 0, a.crf1[it], 0.0), [None, None, None, v, None, None, None], gc);
            }
        }
        let side = (2 * a.rb.max(1) + 1) as f32;
        d2(&mut enc, "k_clutter", mk(0, (!bg_detail) as u32, side, a.thr_a), [v, cnt, aa, alpha, None, None, None], g2);
        // wIn (= the CPU's, bit for bit) and the in-paint smoothing; the CPU recomputes bar / beta from alpha itself
        d2(&mut enc, "k_bar", mk(0, 0, 0.0, 0.0), [alpha, aa, y, win, t1, t2, None], g2);
        d2(&mut enc, "k_fill4", mk(0, 0, 0.0, 0.0), [c, win, None, None, None, None, q], g2);
        for it in 0..3 {
            d2(&mut enc, "k_rf4_rows", mk(0, 0, a.crf4[it], 0.0), [None, None, None, None, None, None, q], gr);
            d2(&mut enc, "k_rf4_cols", mk(0, 0, a.crf4[it], 0.0), [None, None, None, None, None, None, q], gc);
        }
        d2(&mut enc, "k_fh", mk(0, 0, 0.0, 0.0), [c, None, None, fh, None, None, q], g2);
        let mut pk = Vec::new();
        if perturb_on("lines") {
            self.poke(&mut enc, slot, &mut pk, PL_ALPHA * s, n);
        }
        let t_enc = tp.elapsed();
        let (oa, ol, of) = (a.out_alpha, a.out_light, a.out_fh);
        let st = self.download(slot, enc, &[(PL_ALPHA * s, n), (PL_LIGHT * s, n), (PL_FH * s, n * 3)], &mut |k, b| unsafe {
            match k {
                0 => par_copy(std::slice::from_raw_parts_mut(oa as *mut u8, n * 4), b),
                1 => {
                    if !ol.is_null() {
                        let f: &[f32] = std::slice::from_raw_parts(b.as_ptr() as *const f32, n);
                        let lt = std::slice::from_raw_parts_mut(ol, n);
                        for i in 0..n {
                            lt[i] = (f[i] > 0.5) as u8;
                        }
                    }
                }
                _ => par_copy(std::slice::from_raw_parts_mut(of as *mut u8, n * 12), b),
            }
        });
        drop(keep);
        drop(pk);
        if st != SKGPU_OK {
            return st;
        }
        fr.alpha_ok = true;
        if fr.prof {
            eprintln!("  [skgpu] lines: upload {:?}, encode {:?}, gpu+read-back {:?}", t_up, t_enc - t_up, tp.elapsed() - t_enc);
        }
        SKGPU_OK
    }

    fn tail(&self, fr: &mut Frame, a: &SkTailArgs) -> i32 {
        let (Some(tbgl), Some((etab, ecount))) = (self.tail_bgl.as_ref(), self.etab.as_ref()) else { return SKGPU_ERROR };
        let (w, h) = (fr.w, fr.h);
        let n = (w as usize) * (h as usize);
        if a.uni.is_null() || a.uni_bytes as usize != TAIL_UNI || a.d64.is_null() || a.grids.is_null() || a.dw.is_null() || a.dm.is_null() || a.p.is_null() {
            return SKGPU_ERROR;
        }
        if a.ev.is_null() || a.wet.is_null() || a.bz.is_null() || a.pw.is_null() {
            return SKGPU_ERROR;
        }
        // stop (0.6.0.6: the self-test reads every stage back): 1 after (5b) -> out_dw, out_dm; 2 after (5) -> out_dw;
        // 4 after (6) -> out_dw; 5 the paper's blurred line coverage -> out_dm (w*h); otherwise through the paper (and
        // the ink) -> out_p
        let st_dw = matches!(a.stop, 1 | 2 | 4);
        let st_dm = matches!(a.stop, 1 | 5);
        if (st_dw && a.out_dw.is_null()) || (st_dm && a.out_dm.is_null()) || (!st_dw && !st_dm && a.out_p.is_null()) {
            return SKGPU_ERROR;
        }
        let tp = Instant::now();
        let slot = fr.slot.as_ref().unwrap();
        let s = slot.s;
        let mut items: Vec<(usize, *const f32, usize)> = vec![(PL_C * s, a.p, n * 3), (PL_FH * s, a.dw, n * 3), (PL_Y * s, a.dm, n)];
        if !fr.a_ok {
            if a.a.is_null() {
                return SKGPU_ERROR;
            }
            items.push((PL_A * s, a.a, n));
        }
        if !fr.alpha_ok && !a.alpha.is_null() {
            items.push((PL_ALPHA * s, a.alpha, n));
        }
        // aux slot k = plane 4 + k
        items.push((6 * s, a.ev, n));
        items.push((7 * s, a.wet, n));
        items.push((8 * s, a.bz, n));
        if !a.dl.is_null() {
            items.push((9 * s, a.dl, n));
        }
        items.push((10 * s, a.pw, n));
        if !a.dline.is_null() {
            items.push((11 * s, a.dline, n));
        }
        if !a.ptex.is_null() {
            items.push((12 * s, a.ptex, n));
        }
        let st = self.upload(slot, &items);
        if st != SKGPU_OK {
            return st;
        }
        fr.a_ok = true;
        let t_up = tp.elapsed();
        let dev = &self.dev;
        let mut d64 = d64_fixed();
        let host = unsafe { std::slice::from_raw_parts(a.d64, a.d64_count as usize) };
        d64.extend_from_slice(host);
        let b64 = self.storage_init(unsafe { std::slice::from_raw_parts(d64.as_ptr() as *const u8, d64.len() * 8) });
        let bgr = self.storage_init(unsafe { std::slice::from_raw_parts(a.grids as *const u8, a.grids_count as usize * 4) });
        let base = unsafe { std::slice::from_raw_parts(a.uni, TAIL_UNI) };
        let mut uni = vec![0u8; 16 + TAIL_UNI];
        uni[16..].copy_from_slice(base);
        // n.z: the plane stride; e2.w: the exp table's entries
        uni[24..28].copy_from_slice(&(s as u32).to_le_bytes());
        uni[76..80].copy_from_slice(&ecount.to_le_bytes());
        let mut enc = dev.create_command_encoder(&Default::default());
        let mut keep = Vec::new();
        let mut disp = |enc: &mut wgpu::CommandEncoder, e: &str, coef: f32, groups: (u32, u32)| {
            uni[0..4].copy_from_slice(&coef.to_le_bytes());
            let ub = self.uniform(&uni);
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: tbgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: bgr.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: Self::rng(slot, PL_FH, 3) },
                    wgpu::BindGroupEntry { binding: 3, resource: Self::rng(slot, PL_Y, 1) },
                    wgpu::BindGroupEntry { binding: 4, resource: Self::rng(slot, PL_Q, 4) },
                    wgpu::BindGroupEntry { binding: 5, resource: Self::rng(slot, PL_A, 1) },
                    wgpu::BindGroupEntry { binding: 6, resource: Self::rng(slot, PL_C, 3) },
                    wgpu::BindGroupEntry { binding: 7, resource: Self::rng(slot, PL_ALPHA, 10) },
                    wgpu::BindGroupEntry { binding: 8, resource: etab.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: b64.as_entire_binding() },
                ],
            });
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.tail_pipes[e]);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(groups.0, groups.1, 1);
            }
            keep.push((ub, bg));
        };
        let g2 = ((w + 15) / 16, (h + 15) / 16);
        let gr = ((h + 63) / 64, 1);
        let gc = ((w + 63) / 64, 1);
        let mut pk = Vec::new();
        if a.wet_on != 0 {
            disp(&mut enc, "k_wetprep", 0.0, g2);
            for it in 0..3 {
                disp(&mut enc, "k_q_rows", a.cwet[it], gr);
                disp(&mut enc, "k_q_cols", a.cwet[it], gc);
            }
            disp(&mut enc, "k_wet", 0.0, g2);
            if perturb_on("wet") {
                self.poke(&mut enc, slot, &mut pk, PL_FH * s, n * 3);
            }
        }
        let items_out: Vec<(usize, usize)> = 'run: {
            if a.stop == 1 {
                // through (5b) only: the CPU goes on from (5)
                break 'run vec![(PL_FH * s, n * 3), (PL_Y * s, n)];
            }
            disp(&mut enc, "k_rimprep", 0.0, g2);
            for it in 0..3 {
                disp(&mut enc, "k_q_rows", a.crim[it], gr);
                disp(&mut enc, "k_q_cols", a.crim[it], gc);
            }
            disp(&mut enc, "k_density", 0.0, g2);
            if perturb_on("density") {
                self.poke(&mut enc, slot, &mut pk, PL_FH * s, n * 3);
            }
            if a.stop == 2 {
                break 'run vec![(PL_FH * s, n * 3)];
            }
            disp(&mut enc, "k_slivers", 0.0, g2);
            if perturb_on("slivers") {
                self.poke(&mut enc, slot, &mut pk, PL_FH * s, n * 3);
            }
            if a.stop == 4 {
                break 'run vec![(PL_FH * s, n * 3)];
            }
            disp(&mut enc, "k_paprep", 0.0, g2);
            if a.pa_blur != 0 {
                for it in 0..3 {
                    disp(&mut enc, "k_pa_rows", a.cpa[it], gr);
                    disp(&mut enc, "k_pa_cols", a.cpa[it], gc);
                }
            }
            if perturb_on("pablur") {
                self.poke(&mut enc, slot, &mut pk, (PL_ALPHA + 9) * s, n);
            }
            if a.stop == 5 {
                break 'run vec![((PL_ALPHA + 9) * s, n)];
            }
            disp(&mut enc, "k_paper", 0.0, g2);
            if perturb_on("paper") {
                self.poke(&mut enc, slot, &mut pk, PL_C * s, n * 3);
            }
            if a.ink_on != 0 {
                // 5. ink: delta (and the depth far plane) go up once the paper has read its inputs
                if a.delta.is_null() {
                    return SKGPU_ERROR;
                }
                let done = std::mem::replace(&mut enc, dev.create_command_encoder(&Default::default()));
                self.queue.submit([done.finish()]);
                let mut up2: Vec<(usize, *const f32, usize)> = vec![(PL_FH * s, a.delta, n * 3)];
                if !a.far_w.is_null() {
                    up2.push((6 * s, a.far_w, n));
                }
                let st = if fault("ink") { SKGPU_OOM } else { self.upload(slot, &up2) };
                if st != SKGPU_OK {
                    return st;
                }
                disp(&mut enc, "k_inkprep", 0.0, g2);
                if a.ink_soft != 0 {
                    for it in 0..3 {
                        disp(&mut enc, "k_q_rows", a.cink[it], gr);
                        disp(&mut enc, "k_q_cols", a.cink[it], gc);
                    }
                }
                disp(&mut enc, "k_ink", 0.0, g2);
                if perturb_on("ink") {
                    self.poke(&mut enc, slot, &mut pk, PL_C * s, n * 3);
                }
            }
            vec![(PL_C * s, n * 3)]
        };
        let t_enc = tp.elapsed();
        let (odw, odm, op) = (a.out_dw, a.out_dm, a.out_p);
        let stop = a.stop;
        let st = self.download(slot, enc, &items_out, &mut |k, b| unsafe {
            match (stop, k) {
                (1 | 2 | 4, 0) => par_copy(std::slice::from_raw_parts_mut(odw as *mut u8, n * 12), b),
                (1, _) | (5, _) => par_copy(std::slice::from_raw_parts_mut(odm as *mut u8, n * 4), b),
                _ => par_copy(std::slice::from_raw_parts_mut(op as *mut u8, n * 12), b),
            }
        });
        drop(keep);
        drop(pk);
        if st != SKGPU_OK {
            return st;
        }
        if fr.prof {
            eprintln!("  [skgpu] tail: upload {:?}, encode {:?}, gpu+read-back {:?}", t_up, t_enc - t_up, tp.elapsed() - t_enc);
        }
        SKGPU_OK
    }

    /// the exact exp against this process's float exp (the same MSVC runtime as the engine): every table entry and a
    /// spread of other inputs. false = the tail stays on the CPU.
    fn exp_selftest(&self) -> bool {
        let Some(tbgl) = self.tail_bgl.as_ref() else { return false };
        let Some((etab, ecount)) = self.etab.as_ref() else { return false };
        let tab: &[u8] = include_bytes!("../shaders/exp_tab.bin");
        let mut xs: Vec<f32> = Vec::new();
        for k in 0..(*ecount as usize) {
            xs.push(f32::from_bits(u32::from_le_bytes([tab[k * 8], tab[k * 8 + 1], tab[k * 8 + 2], tab[k * 8 + 3]])));
        }
        let mut r = 0x12345678u32;
        for _ in 0..200000 {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            xs.push(-20.0 + 30.0 * ((r >> 8) as f32 / 16777216.0));
        }
        xs.push(0.0);
        xs.push(-0.0);
        let n = xs.len();
        let s = (n + 63) & !63;
        let scope = self.dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut arena_bytes = vec![0u8; s * 10 * 4];
        arena_bytes[2 * s * 4..2 * s * 4 + n * 4].copy_from_slice(unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, n * 4) });
        let ar = self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: arena_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        {
            let Ok(mut v) = ar.slice(..).get_mapped_range_mut() else { return false };
            v.copy_from_slice(&arena_bytes);
        }
        ar.unmap();
        let rd = self.dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut uni = vec![0u8; 16 + TAIL_UNI];
        uni[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        uni[20..24].copy_from_slice(&1u32.to_le_bytes());
        uni[24..28].copy_from_slice(&(s as u32).to_le_bytes());
        uni[76..80].copy_from_slice(&ecount.to_le_bytes());
        let ub = self.uniform(&uni);
        let d64 = d64_fixed();
        let b64 = self.storage_init(unsafe { std::slice::from_raw_parts(d64.as_ptr() as *const u8, d64.len() * 8) });
        let bgr = self.storage_init(&[0u8; 16]);
        let b16 = self.storage_init(&[0u8; 64]);
        let bg = self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: tbgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bgr.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b16.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b16.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b16.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b16.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b16.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: ar.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: etab.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: b64.as_entire_binding() },
            ],
        });
        let mut enc = self.dev.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.tail_pipes["k_exptest"]);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(((n + 63) / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&ar, (3 * s * 4) as u64, &rd, 0, Some((n * 4) as u64));
        self.queue.submit([enc.finish()]);
        if block_on(scope.pop(), Duration::from_secs(10)).map(|e| e.is_some()).unwrap_or(true) {
            return false;
        }
        if self.map_wait(rd.slice(..), wgpu::MapMode::Read, Duration::from_secs(10)) != SKGPU_OK {
            return false;
        }
        let Ok(v) = rd.slice(..).get_mapped_range() else { return false };
        let got: &[f32] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, n) };
        let mut bad = 0usize;
        for k in 0..n {
            let x = std::hint::black_box(xs[k]);
            if got[k].to_bits() != x.exp().to_bits() {
                bad += 1;
            }
        }
        if std::env::var("SKGPU_PROFILE").is_ok() {
            eprintln!("  [skgpu] exp self-test: {} of {} differ", bad, n);
        }
        bad == 0
    }
}

// ------------------------------------------------------------------------------------------------ C ABI ---
/// Create the context (instance, Vulkan adapter, device, pipelines). NULL = no usable GPU.
#[no_mangle]
pub extern "C" fn skgpu_create() -> *mut Ctx {
    match std::panic::catch_unwind(create) {
        Ok(Some(c)) => Box::into_raw(Box::new(c)),
        _ => std::ptr::null_mut(),
    }
}

/// 1 when the painter's tail can run on this GPU (f64, the exact exp checked)
#[no_mangle]
pub extern "C" fn skgpu_has_tail(ctx: *mut Ctx) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    unsafe { (*ctx).tail_bgl.is_some() as i32 }
}

/// 0.6.0.6: one line about the context for the diagnostic log ("<adapter> | vendor .. | driver .. | exp <ok|FAIL|n/a>
/// | tail <yes|no>"), NUL-terminated in buf; returns its length
#[no_mangle]
pub extern "C" fn skgpu_describe(ctx: *mut Ctx, buf: *mut u8, len: u32) -> i32 {
    if ctx.is_null() || buf.is_null() || len == 0 {
        return 0;
    }
    let c = unsafe { &*ctx };
    let s = format!(
        "{} | exp {} | tail {}",
        c.desc,
        match c.exp_state { 1 => "ok", 0 => "FAIL", _ => "n/a" },
        if c.tail_bgl.is_some() { "yes" } else { "no" }
    );
    let b = s.as_bytes();
    let k = b.len().min(len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), buf, k);
        *buf.add(k) = 0;
    }
    k as i32
}

/// Begin a frame of w x h px: waits up to wait_ms when two frames already run (MFR), then BUSY. NULL + status on failure.
#[no_mangle]
pub extern "C" fn skgpu_frame_begin(ctx: *mut Ctx, w: i32, h: i32, wait_ms: u32, status: *mut i32) -> *mut Frame {
    let set = |s: i32| {
        if !status.is_null() {
            unsafe { *status = s };
        }
    };
    if ctx.is_null() || w <= 0 || h <= 0 {
        set(SKGPU_ERROR);
        return std::ptr::null_mut();
    }
    let c = unsafe { &*ctx };
    if c.lost.load(Ordering::SeqCst) {
        set(SKGPU_LOST);
        return std::ptr::null_mut();
    }
    {
        let g = c.jobs.lock().unwrap_or_else(|e| e.into_inner());
        let (mut g, _) = c.cv.wait_timeout_while(g, Duration::from_millis(wait_ms as u64), |j| *j >= 2).unwrap_or_else(|e| e.into_inner());
        if *g >= 2 {
            set(SKGPU_BUSY);
            return std::ptr::null_mut();
        }
        *g += 1;
    }
    let n = (w as usize) * (h as usize);
    let s = (n + 63) & !63;
    let r = if fault("begin") { Err(SKGPU_OOM) } else { std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.take_slot(s))).unwrap_or(Err(SKGPU_ERROR)) };
    match r {
        Ok(slot) => {
            set(SKGPU_OK);
            Box::into_raw(Box::new(Frame { ctx, slot: Some(slot), w: w as u32, h: h as u32, a_ok: false, alpha_ok: false, prof: std::env::var("SKGPU_PROFILE").is_ok() }))
        }
        Err(e) => {
            let mut g = c.jobs.lock().unwrap_or_else(|e| e.into_inner());
            *g -= 1;
            c.cv.notify_one();
            set(e);
            std::ptr::null_mut()
        }
    }
}

/// End a frame: its slot goes back to the pool (kept for the next frame). ok = 0 after a failure: the slot is dropped.
#[no_mangle]
pub extern "C" fn skgpu_frame_end(fr: *mut Frame, ok: i32) {
    if fr.is_null() {
        return;
    }
    let mut f = unsafe { Box::from_raw(fr) };
    let c = unsafe { &*f.ctx };
    if let Some(slot) = f.slot.take() {
        if ok != 0 && !c.lost.load(Ordering::SeqCst) {
            c.pool.lock().unwrap_or_else(|e| e.into_inner()).push(slot);
        }
    }
    let mut g = c.jobs.lock().unwrap_or_else(|e| e.into_inner());
    *g -= 1;
    c.cv.notify_one();
}

/// The line stage's GPU part (see SkLinesArgs); C, A and alpha stay resident in the frame.
#[no_mangle]
pub extern "C" fn skgpu_frame_lines(fr: *mut Frame, args: *const SkLinesArgs) -> i32 {
    if fr.is_null() || args.is_null() {
        return SKGPU_ERROR;
    }
    let f = unsafe { &mut *fr };
    let c = unsafe { &*f.ctx };
    if c.lost.load(Ordering::SeqCst) {
        return SKGPU_LOST;
    }
    if fault("lines") {
        return SKGPU_OOM;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.lines(f, unsafe { &*args }))).unwrap_or(SKGPU_ERROR)
}

/// The painter's tail (see SkTailArgs).
#[no_mangle]
pub extern "C" fn skgpu_frame_tail(fr: *mut Frame, args: *const SkTailArgs) -> i32 {
    if fr.is_null() || args.is_null() {
        return SKGPU_ERROR;
    }
    let f = unsafe { &mut *fr };
    let c = unsafe { &*f.ctx };
    if c.lost.load(Ordering::SeqCst) {
        return SKGPU_LOST;
    }
    if fault("tail") {
        return SKGPU_OOM;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.tail(f, unsafe { &*args }))).unwrap_or(SKGPU_ERROR)
}

/// 1 when the device was lost (the caller stops using the GPU).
#[no_mangle]
pub extern "C" fn skgpu_lost(ctx: *mut Ctx) -> i32 {
    if ctx.is_null() {
        return 1;
    }
    unsafe { (*ctx).lost.load(Ordering::SeqCst) as i32 }
}

/// Give the pooled frame memory back (no frame may be running).
#[no_mangle]
pub extern "C" fn skgpu_trim(ctx: *mut Ctx) {
    if ctx.is_null() {
        return;
    }
    let c = unsafe { &*ctx };
    c.pool.lock().unwrap_or_else(|e| e.into_inner()).clear();
    let _ = c.wait_done(Duration::from_millis(500));
}
