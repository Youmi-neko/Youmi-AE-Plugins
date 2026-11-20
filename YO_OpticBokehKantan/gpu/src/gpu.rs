//! Device creation, adapter selection, pipelines.

use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskCx, Poll, Waker};
use std::time::{Duration, Instant};

use crate::ffi;

/// Drives a wgpu future to completion on this thread (native wgpu futures resolve on poll).
pub fn block_on<F: Future>(f: F) -> Option<F::Output> {
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

/// Accumulation pipeline variant (see blur.wgsl header).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Variant {
    pub fr: bool,
    pub cat: bool,
    pub roll: bool,
    pub raw: bool,
    /// magnitude tiers of the colour accumulators (1 = SDR range, 2-3 = HDR), see blur.wgsl tierOf
    pub k: u32,
}

impl Variant {
    /// accumulator slots without the extra magnitude tiers
    pub fn ncb(&self) -> u32 {
        (if self.fr { 7 } else { 5 }) + (self.roll as u32)
    }
    /// colour slots per extra tier (R, G, B, eL)
    pub fn ncc(&self) -> u32 {
        3 + self.roll as u32
    }
    pub fn nc(&self) -> u32 {
        self.ncb() + (self.k.max(1) - 1) * self.ncc()
    }
}

/// largest supported segment width (= workgroup size); a variant uses less if its slots do not fit the
/// device's workgroup memory (2 x u32 per slot and bin)
pub type Sub = (wgpu::SubmissionIndex, Arc<AtomicBool>);

pub const SEG_MAX: u32 = 256;
pub fn seg_for(nc: u32, max_ws: u32) -> u32 {
    let mut s = SEG_MAX;
    while s > 32 && s * nc * 8 > max_ws {
        s /= 2;
    }
    s
}

pub struct Pipes {
    pub setup_fine: wgpu::ComputePipeline,
    pub setup_coarse: wgpu::ComputePipeline,
}

#[allow(dead_code)] // backend / errors: diagnostics
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: String,
    pub backend: wgpu::Backend,
    pub vendor: u32,
    pub device_id: u32,
    pub limits: wgpu::Limits,
    pub lost: Arc<AtomicBool>,
    pub errors: Arc<Mutex<Vec<String>>>,
    pub bgl: wgpu::BindGroupLayout,
    pub layout: wgpu::PipelineLayout,
    setup: Mutex<Option<Arc<Pipes>>>,
    accum: Mutex<HashMap<Variant, Arc<wgpu::ComputePipeline>>>,
    pub verbose: bool,
    pub checks: String,
    prep: Mutex<Option<Arc<crate::frame::PrepPipes>>>,
    post: Mutex<Option<Arc<crate::post::PostPipes>>>,
    /// phase 5: reusable host staging buffers (upload: MAP_WRITE, readback: MAP_READ). A fresh 1 GB staging
    /// buffer per UHD frame cost ~0.3-0.5 s of page faults on first touch.
    pool_up: Mutex<Vec<wgpu::Buffer>>,
    pool_down: Mutex<Vec<wgpu::Buffer>>,
    pub ts: bool,
}

/// Per-job GPU timestamps: every compute pass gets a begin / end slot; busy = sum of (end - begin).
pub struct Ts {
    qs: wgpu::QuerySet,
    n: u32,
    cap: u32,
    labels: Vec<&'static str>,
}

/// per-pass GPU time profile (test hook 8 prints and resets it): label -> (ms, passes)
pub static PROFILE: std::sync::Mutex<Vec<(&'static str, f64, u32)>> = std::sync::Mutex::new(Vec::new());
pub fn profile_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBGPU_PROFILE").is_ok())
}
impl Ts {
    pub fn new(gpu: &Gpu, passes: u32) -> Option<Ts> {
        if !gpu.ts {
            return None;
        }
        let cap = passes.clamp(1, 2048);
        let qs = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("obgpu.ts"), ty: wgpu::QueryType::Timestamp, count: 2 * cap });
        Some(Ts { qs, n: 0, cap, labels: Vec::new() })
    }
    /// timestamp writes for the next pass (None once the set is full)
    pub fn writes(&mut self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.writes_l("?")
    }
    /// as writes(), with a profile label for this pass
    pub fn writes_l(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if self.n >= self.cap {
            return None;
        }
        self.labels.push(label);
        let k = self.n;
        self.n += 1;
        Some(wgpu::ComputePassTimestampWrites { query_set: &self.qs, beginning_of_pass_write_index: Some(2 * k), end_of_pass_write_index: Some(2 * k + 1) })
    }
    /// resolves and reads back; GPU busy milliseconds of the recorded passes
    pub fn busy_ms(&self, gpu: &Gpu) -> Option<f64> {
        if self.n == 0 {
            return Some(0.0);
        }
        let bytes = 16 * self.n as u64;
        let rb = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let st = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut e = gpu.device.create_command_encoder(&Default::default());
        e.resolve_query_set(&self.qs, 0..2 * self.n, &rb, 0);
        e.copy_buffer_to_buffer(&rb, 0, &st, 0, bytes);
        let s = gpu.submit(e.finish());
        gpu.wait_sub(&s, &|| false, Duration::from_secs(2)).ok()?;
        let sl = st.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let t0 = Instant::now();
        loop {
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            if let Ok(ok) = rx.try_recv() {
                if !ok {
                    return None;
                }
                break;
            }
            if t0.elapsed() > Duration::from_secs(2) {
                return None;
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        let v = sl.get_mapped_range().ok()?;
        let period = gpu.queue.get_timestamp_period() as f64;
        let mut sum = 0u64;
        for k in 0..self.n as usize {
            let b = u64::from_le_bytes(v[16 * k..16 * k + 8].try_into().ok()?);
            let e2 = u64::from_le_bytes(v[16 * k + 8..16 * k + 16].try_into().ok()?);
            sum += e2.saturating_sub(b);
            if profile_on() {
                let ms = e2.saturating_sub(b) as f64 * period * 1e-6;
                if let Ok(mut p) = PROFILE.lock() {
                    let l = self.labels.get(k).copied().unwrap_or("?");
                    match p.iter_mut().find(|e| e.0 == l) {
                        Some(e) => {
                            e.1 += ms;
                            e.2 += 1;
                        }
                        None => p.push((l, ms, 1)),
                    }
                }
            }
        }
        Some(sum as f64 * period * 1e-6)
    }
}

const SHADER: &str = include_str!("blur.wgsl");

pub fn shader_source(v: Variant, seg: u32) -> String {
    let nc = v.nc();
    let ncb = v.ncb();
    // slot layout (blur.wgsl): FR: R 0, WR 1, G 2, A 3, WG 4, B 5, WB 6, EL 7 ; else R 0, G 1, B 2, A 3, W 4, EL 5
    let (s_wr, s_g, s_b, s_wb, s_el) = if v.fr { (1, 2, 5, 6, 7) } else { (4, 1, 2, 4, 5) };
    // without ROLL the EL slot does not exist; keep its (never used) index in range for validation
    let s_el = s_el.min(ncb - 1);
    let (k, ncc) = (v.k.max(1), v.ncc());
    let head = format!(
        "const SEG: i32 = {seg};\nconst NC: i32 = {nc};\nconst NCB: i32 = {ncb};\nconst NCC: i32 = {ncc};\nconst K: i32 = {k};\nconst FR: bool = {};\nconst CAT: bool = {};\nconst ROLL: bool = {};\nconst RAW: bool = {};\n\
         const S_R: u32 = 0u;\nconst S_WR: u32 = {s_wr}u;\nconst S_G: u32 = {s_g}u;\nconst S_A: u32 = 3u;\nconst S_WG: u32 = 4u;\n\
         const S_B: u32 = {s_b}u;\nconst S_WB: u32 = {s_wb}u;\nconst S_EL: u32 = {s_el}u;\n",
        v.fr, v.cat, v.roll, v.raw
    );
    // the template's own slot constants are replaced by the header's
    let body: String = SHADER
        .lines()
        .filter(|l| !l.starts_with("const S_"))
        .collect::<Vec<_>>()
        .join("\n");
    head + &body
}

pub fn backends_of(mask: u32) -> wgpu::Backends {
    if mask == 0 {
        // Windows: both; the Vulkan adapter of the same GPU is preferred (see score below): DX12 compiles
        // WGSL through FXC, measured 4-5 s per pipeline variant vs ~0.2 s on Vulkan (gpu/out/backends.md)
        return if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else if cfg!(windows) {
            wgpu::Backends::VULKAN | wgpu::Backends::DX12
        } else {
            wgpu::Backends::VULKAN
        };
    }
    let mut b = wgpu::Backends::empty();
    if mask & ffi::BACKEND_DX12 != 0 {
        b |= wgpu::Backends::DX12;
    }
    if mask & ffi::BACKEND_VULKAN != 0 {
        b |= wgpu::Backends::VULKAN;
    }
    if mask & ffi::BACKEND_METAL != 0 {
        b |= wgpu::Backends::METAL;
    }
    b
}

pub enum InitError {
    NoAdapter(String),
    Device(String),
}

impl Gpu {
    pub fn create(cfg: &ffi::obgpu_config) -> Result<Gpu, InitError> {
        if cfg.force_no_adapter != 0 {
            return Err(InitError::NoAdapter("forced: no adapter (test)".into()));
        }
        let backends = backends_of(cfg.backends);
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        desc.flags = wgpu::InstanceFlags::empty(); // no validation layers / debug labels in the product
        desc.memory_budget_thresholds = wgpu::MemoryBudgetThresholds {
            for_resource_creation: Some(90),
            for_device_loss: None,
        };
        // DX12: the system FXC (d3dcompiler_47.dll, part of Windows). "Auto" would pick any
        // dxcompiler.dll found on PATH, which makes behaviour depend on the user's machine.
        // mac port: Windows-only (the DX12 options are inert elsewhere; macOS uses Metal, see backends_of).
        #[cfg(windows)]
        {
            desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::Fxc;
            // evaluation only (gpu/PHASE2.md "DXC"): an explicit dxcompiler.dll path. Not for the product: the DLL
            // would have to ship with the plug-in.
            if let Ok(pth) = std::env::var("OBGPU_DXC_PATH") {
                desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc { dxc_path: pth };
            }
        }
        let instance = wgpu::Instance::new(desc);
        let adapters = block_on(instance.enumerate_adapters(backends)).unwrap_or_default();
        let mut best: Option<(i32, wgpu::Adapter)> = None;
        let mut seen = Vec::new();
        for a in adapters {
            let info = a.get_info();
            let dl = a.get_downlevel_capabilities();
            let lim = a.limits();
            let mut score = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => 300,
                wgpu::DeviceType::IntegratedGpu => 200,
                wgpu::DeviceType::VirtualGpu => 100,
                wgpu::DeviceType::Other => 50,
                wgpu::DeviceType::Cpu => {
                    if cfg.allow_software != 0 {
                        10
                    } else {
                        -1
                    }
                }
            };
            // WARP reports as Cpu on DX12; llvmpipe/SwiftShader as Cpu on Vulkan. Double check by name.
            let lname = info.name.to_lowercase();
            if (lname.contains("microsoft basic render") || lname.contains("llvmpipe") || lname.contains("swiftshader"))
                && cfg.allow_software == 0
            {
                score = -1;
            }
            // Windows: prefer the Vulkan adapter of the same GPU over its DX12 one (no effect on macOS: Metal only)
            if score >= 0 && info.backend == wgpu::Backend::Vulkan {
                score += 5;
            }
            if !dl.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
                || lim.max_compute_workgroup_storage_size < 16384
                || lim.max_compute_invocations_per_workgroup < SEG_MAX
                || lim.max_compute_workgroup_size_x < SEG_MAX
                || lim.max_storage_buffers_per_shader_stage < 8
            {
                score = -1;
            }
            seen.push(format!("{} ({:?}, {:?}) score {}", info.name, info.backend, info.device_type, score));
            if score >= 0 && best.as_ref().map(|b| score > b.0).unwrap_or(true) {
                best = Some((score, a));
            }
        }
        let Some((_, adapter)) = best else {
            return Err(InitError::NoAdapter(format!("no suitable adapter; seen: [{}]", seen.join("; "))));
        };
        let info = adapter.get_info();
        let al = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size.min(2u64 << 30);
        limits.max_buffer_size = al.max_buffer_size.min(4u64 << 30);
        limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size.min(32768);
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_uniform_buffer_binding_size = limits.max_uniform_buffer_binding_size.min(al.max_uniform_buffer_binding_size);
        // GPU timestamps (pass begin / end) when available: GPU busy time beside wall time (shared GPUs)
        let ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let ddesc = wgpu::DeviceDescriptor {
            label: Some("obgpu"),
            required_features: if ts { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = match block_on(adapter.request_device(&ddesc)) {
            Some(Ok(dq)) => dq,
            Some(Err(e)) => return Err(InitError::Device(format!("request_device: {e}"))),
            None => return Err(InitError::Device("request_device did not complete".into())),
        };
        let lost = Arc::new(AtomicBool::new(false));
        let errors = Arc::new(Mutex::new(Vec::<String>::new()));
        {
            let lost1 = lost.clone();
            device.set_device_lost_callback(move |reason, msg| {
                lost1.store(true, Ordering::SeqCst);
                eprintln!("[obgpu] device lost: {reason:?} {msg}");
            });
            let errs = errors.clone();
            let lost2 = lost.clone();
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                let s = format!("{e}");
                if matches!(e, wgpu::Error::OutOfMemory { .. }) {
                    // reported to the job by its own error scope as well
                }
                if s.to_lowercase().contains("lost") {
                    lost2.store(true, Ordering::SeqCst);
                }
                if let Ok(mut v) = errs.lock() {
                    if v.len() < 64 {
                        v.push(s);
                    }
                }
            }));
        }
        let storage = |b: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("obgpu.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(crate::job::DISP_BYTES),
                    },
                    count: None,
                },
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, false),
                storage(6, false),
                storage(7, false),
                storage(8, false),
                storage(9, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("obgpu.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        Ok(Gpu {
            info: format!("{} ({:?}, {:?}, driver {} {})", info.name, info.backend, info.device_type, info.driver, info.driver_info),
            backend: info.backend,
            vendor: info.vendor,
            device_id: info.device,
            device,
            queue,
            limits,
            lost,
            errors,
            bgl,
            layout,
            setup: Mutex::new(None),
            accum: Mutex::new(HashMap::new()),
            verbose: cfg.verbose != 0,
            prep: Mutex::new(None),
            post: Mutex::new(None),
            pool_up: Mutex::new(Vec::new()),
            pool_down: Mutex::new(Vec::new()),
            ts,
            checks: std::env::var("OBGPU_CHECKS").unwrap_or_else(|_| "noloop".into()),
        })
    }

    pub fn seg(&self, v: Variant) -> u32 {
        seg_for(v.nc(), self.limits.max_compute_workgroup_storage_size)
    }

    /// submits and returns a completion flag set by the queue (on_submitted_work_done), so waits can be
    /// non-blocking polls: device.poll(Wait { timeout }) was measured to block until completion under external
    /// GPU load (a 670 ms abort latency), whatever the timeout.
    pub fn submit(&self, cb: wgpu::CommandBuffer) -> Sub {
        let idx = self.queue.submit(Some(cb));
        let f = Arc::new(AtomicBool::new(false));
        let f2 = f.clone();
        self.queue.on_submitted_work_done(move || f2.store(true, Ordering::SeqCst));
        (idx, f)
    }

    /// waits for a submission with non-blocking polls; checks `abort` every ~0.3 ms. Err: OBGPU status.
    pub fn wait_sub(&self, s: &Sub, abort: &dyn Fn() -> bool, timeout: Duration) -> Result<(), i32> {
        let t0 = Instant::now();
        loop {
            if s.1.load(Ordering::SeqCst) {
                return Ok(());
            }
            if let Err(_e) = self.device.poll(wgpu::PollType::Poll) {
                self.lost.store(true, Ordering::SeqCst);
                return Err(ffi::OBGPU_DEVICE_LOST);
            }
            if s.1.load(Ordering::SeqCst) {
                return Ok(());
            }
            if self.is_lost() {
                return Err(ffi::OBGPU_DEVICE_LOST);
            }
            if abort() {
                // abandon: wgpu keeps the submission's resources alive until it completes
                return Err(ffi::OBGPU_ABORTED);
            }
            if t0.elapsed() > timeout {
                // congestion (other applications saturate the GPU), not a hang: a hung GPU is reset by the OS
                // (TDR) and reported as a device loss. Abandon the job, keep the device.
                return Err(ffi::OBGPU_TIMEOUT);
            }
            std::thread::sleep(Duration::from_micros(300));
        }
    }

    /// phase 5: releases big buffers now (wgpu frees dropped buffers lazily, after a later maintain), so the
    /// next allocation of the same job does not stack on them (a UHD Film frame lost the device to "out of
    /// memory" while the blur's and the prepared crop's buffers were still waiting to be freed)
    pub fn free(&self, bufs: &[&wgpu::Buffer]) {
        if std::env::var("OBGPU_NOFREE").is_ok() {
            return;
        }
        for b in bufs {
            b.destroy();
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// a staging buffer of at least `size` bytes from the pool (or a new one, rounded up to 16 MB)
    pub fn staging(&self, size: u64, write: bool) -> wgpu::Buffer {
        let pool = if write { &self.pool_up } else { &self.pool_down };
        // pooling measured: 2-6x slower GPU work when frames of different paths alternate (unexplained driver
        // behaviour with re-mapped staging buffers): opt-in test switch only (OBGPU_POOL=1)
        if std::env::var("OBGPU_POOL").is_err() {
        } else if let Ok(mut v) = pool.lock() {
            if let Some(i) = v.iter().enumerate().filter(|(_, b)| b.size() >= size).min_by_key(|(_, b)| b.size()).map(|(i, _)| i) {
                return v.swap_remove(i);
            }
        }
        let usage = if write { wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC } else { wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST };
        let sz = size.max(16).div_ceil(16 << 20) * (16 << 20);
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(if write { "obgpu.up" } else { "obgpu.down" }), size: sz, usage, mapped_at_creation: false })
    }
    /// returns a staging buffer to the pool (at most 2 per direction and 2.5 GB in all per direction; the
    /// rest is freed). wgpu maps a pooled buffer only after the GPU finished the copy that used it.
    pub fn give_staging(&self, b: wgpu::Buffer, write: bool) {
        let pool = if write { &self.pool_up } else { &self.pool_down };
        if std::env::var("OBGPU_POOL").is_err() {
            b.destroy();
            return;
        }
        if let Ok(mut v) = pool.lock() {
            let total: u64 = v.iter().map(|x| x.size()).sum::<u64>() + b.size();
            if v.len() < 2 && total <= (5u64 << 29) {
                v.push(b);
                return;
            }
        }
        b.destroy();
    }
    /// writes `parts` (concatenated) into `dst` at offset 0 through a pooled staging buffer: the copy into the
    /// mapped staging memory runs on up to 8 threads (same bytes as queue.write_buffer)
    pub fn upload(&self, dst: &wgpu::Buffer, parts: &[&[u8]], abort: &dyn Fn() -> bool, timeout: Duration) -> Result<(), i32> {
        let total: u64 = parts.iter().map(|p| p.len() as u64).sum();
        if total == 0 {
            return Ok(());
        }
        let sb = self.staging(total, true);
        let sl = sb.slice(0..total);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Write, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let t0 = Instant::now();
        loop {
            if abort() {
                return Err(ffi::OBGPU_ABORTED);
            }
            let _ = self.device.poll(wgpu::PollType::Poll);
            if let Ok(ok) = rx.try_recv() {
                if !ok {
                    return Err(if self.is_lost() { ffi::OBGPU_DEVICE_LOST } else { ffi::OBGPU_INTERNAL });
                }
                break;
            }
            if t0.elapsed() > timeout {
                return Err(ffi::OBGPU_TIMEOUT);
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        {
            let mut view = sl.get_mapped_range_mut().map_err(|_| ffi::OBGPU_INTERNAL)?;
            // split the write-only mapping into one piece per part, then into up to 8 chunks copied in parallel
            let mut rest = view.slice(..);
            let mut jobs: Vec<(wgpu::WriteOnly<'_, [u8]>, &[u8])> = Vec::new();
            for p in parts {
                let n = p.len();
                let (mut piece, r) = rest.split_at(n);
                rest = r;
                let nth = if n >= (64 << 20) { 8 } else { 1 };
                let chunk = n.div_ceil(nth);
                let mut a = 0;
                while a < n {
                    let len = chunk.min(n - a);
                    let (c, r2) = piece.split_at(len);
                    piece = r2;
                    jobs.push((c, &p[a..a + len]));
                    a += len;
                }
            }
            // (disjoint pieces of one mapping: plain host memory, written by one thread each)
            struct Piece<'a>(wgpu::WriteOnly<'a, [u8]>, &'a [u8]);
            unsafe impl Send for Piece<'_> {}
            std::thread::scope(|s| {
                for (c, src) in jobs {
                    let mut pc = Piece(c, src);
                    s.spawn(move || {
                        let pc = &mut pc;
                        pc.0.copy_from_slice(pc.1)
                    });
                }
            });
        }
        sb.unmap();
        let mut e = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        e.copy_buffer_to_buffer(&sb, 0, dst, 0, total);
        self.submit(e.finish());
        self.give_staging(sb, true);
        Ok(())
    }

    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    fn compile(&self, v: Variant, entries: &[&str]) -> Result<Vec<wgpu::ComputePipeline>, String> {
        let t0 = Instant::now();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let iscope = self.device.push_error_scope(wgpu::ErrorFilter::Internal);  // backend compiler (FXC/DXC/driver) failures
        let desc = wgpu::ShaderModuleDescriptor { label: Some("obgpu.blur"), source: wgpu::ShaderSource::Wgsl(shader_source(v, self.seg(v)).into()) };
        // runtime checks (naga): "all" = bounds checks + forced loop bounds (wgpu default, safe for any shader),
        // "noloop" = bounds checks only (every loop in blur.wgsl has a data-independent upper bound),
        // "none" = also no index clamping (all indices are computed from validated sizes).
        let checks = match self.checks.as_str() {
            "none" => wgpu::ShaderRuntimeChecks { bounds_checks: false, force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() },
            "noloop" => wgpu::ShaderRuntimeChecks { bounds_checks: true, force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() },
            _ => wgpu::ShaderRuntimeChecks::checked(),
        };
        let module = unsafe { self.device.create_shader_module_trusted(desc, checks) };
        let mut out = Vec::new();
        for e in entries {
            out.push(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&self.layout),
                module: &module,
                entry_point: Some(e),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            }));
        }
        let ierr = block_on(iscope.pop()).flatten();
        if let Some(Some(err)) = block_on(scope.pop()) {
            return Err(format!("shader/pipeline error ({v:?}): {err}"));
        }
        if let Some(err) = ierr {
            return Err(format!("shader/pipeline error ({v:?}): {err}"));
        }
        if self.verbose {
            eprintln!("[obgpu] compiled {v:?} {entries:?} in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok(out)
    }

    pub fn prep_pipes(&self) -> Result<Arc<crate::frame::PrepPipes>, String> {
        let mut g = self.prep.lock().map_err(|_| "poisoned".to_string())?;
        if let Some(p) = g.as_ref() {
            return Ok(p.clone());
        }
        let p = Arc::new(crate::frame::build_pipes(self)?);
        *g = Some(p.clone());
        Ok(p)
    }

    pub fn post_pipes(&self) -> Result<Arc<crate::post::PostPipes>, String> {
        let mut g = self.post.lock().map_err(|_| "poisoned".to_string())?;
        if let Some(p) = g.as_ref() {
            return Ok(p.clone());
        }
        let p = Arc::new(crate::post::build_pipes(self)?);
        *g = Some(p.clone());
        Ok(p)
    }

    pub fn setup_pipes(&self) -> Result<Arc<Pipes>, String> {
        let mut g = self.setup.lock().map_err(|_| "poisoned".to_string())?;
        if let Some(p) = g.as_ref() {
            return Ok(p.clone());
        }
        let v = Variant { fr: false, cat: false, roll: false, raw: true, k: 1 };
        let mut ps = self.compile(v, &["setup_fine", "setup_coarse"])?;
        let setup_coarse = ps.pop().unwrap();
        let setup_fine = ps.pop().unwrap();
        let p = Arc::new(Pipes { setup_fine, setup_coarse });
        *g = Some(p.clone());
        Ok(p)
    }

    pub fn accum_pipe(&self, v: Variant) -> Result<Arc<wgpu::ComputePipeline>, String> {
        {
            let g = self.accum.lock().map_err(|_| "poisoned".to_string())?;
            if let Some(p) = g.get(&v) {
                return Ok(p.clone());
            }
        }
        // compile outside the lock (other variants stay available); a racing duplicate is harmless
        let p = Arc::new(self.compile(v, &["accum"])?.pop().unwrap());
        let mut g = self.accum.lock().map_err(|_| "poisoned".to_string())?;
        Ok(g.entry(v).or_insert(p).clone())
    }
}
