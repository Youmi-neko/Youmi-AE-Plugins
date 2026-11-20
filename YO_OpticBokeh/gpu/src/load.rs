//! Test-only GPU load generator: a second wgpu device (own instance, same adapter as the context) that keeps
//! the GPU busy with a heavy compute loop, submissions of ~`ms` each, two in flight. Stands in for "the owner's
//! other GPU software" in the contention tests (obgpu_test_hook 7). Never used by the product paths.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::gpu::block_on;

const BURN: &str = r#"
override ITER: u32 = 1000u;
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn burn(@builtin(global_invocation_id) g: vec3u) {
    var x = f32(g.x) * 1e-6 + 0.25;
    var y = 0.5;
    for (var i = 0u; i < ITER; i++) {
        x = fract(x * 1.6180339 + y);
        y = fract(y * 1.4142135 + x * 0.5);
    }
    o[g.x & 65535u] = x + y;
}
"#;

pub struct Load {
    stop: Arc<AtomicBool>,
    th: Option<JoinHandle<()>>,
}

static LOAD: Mutex<Option<Load>> = Mutex::new(None);

/// ms = 0 stops the generator. backends: as in the context config.
pub fn set(ms: u32, backends: wgpu::Backends) -> bool {
    let mut g = match LOAD.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if let Some(mut l) = g.take() {
        l.stop.store(true, Ordering::SeqCst);
        if let Some(t) = l.th.take() {
            let _ = t.join();
        }
    }
    if ms == 0 {
        return true;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    let s2 = stop.clone();
    let th = std::thread::spawn(move || {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = backends;
        d.flags = wgpu::InstanceFlags::empty();
        let inst = wgpu::Instance::new(d);
        let ad = block_on(inst.enumerate_adapters(backends)).unwrap_or_default();
        let Some(a) = ad.into_iter().find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu) else {
            let _ = tx.send(false);
            return;
        };
        let Some(Ok((dev, q))) = block_on(a.request_device(&wgpu::DeviceDescriptor::default())) else {
            let _ = tx.send(false);
            return;
        };
        dev.on_uncaptured_error(Arc::new(|_e: wgpu::Error| {}));
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("burn"), source: wgpu::ShaderSource::Wgsl(BURN.into()) });
        let buf = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: 65536 * 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let mk = |iter: f64| {
            let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &m,
                entry_point: Some("burn"),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &[("ITER", iter)], zero_initialize_workgroup_memory: false },
                cache: None,
            });
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            });
            (p, bg)
        };
        let run = |p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
            let mut e = dev.create_command_encoder(&Default::default());
            {
                let mut c = e.begin_compute_pass(&Default::default());
                c.set_pipeline(p);
                c.set_bind_group(0, bg, &[]);
                c.dispatch_workgroups(8192, 1, 1);
            }
            q.submit(Some(e.finish()))
        };
        // calibrate ITER so one submission takes ~ms on this GPU (measured alone)
        let mut iter = 2000.0;
        let mut pb = mk(iter);
        for _ in 0..4 {
            let t0 = Instant::now();
            let i = run(&pb.0, &pb.1);
            let _ = dev.poll(wgpu::PollType::Wait { submission_index: Some(i), timeout: Some(Duration::from_secs(5)) });
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            iter = (iter * ms as f64 / dt.max(0.1)).clamp(100.0, 5.0e7);
            pb = mk(iter);
        }
        let _ = tx.send(true);
        let mut prev: Option<wgpu::SubmissionIndex> = None;
        while !s2.load(Ordering::SeqCst) {
            let i = run(&pb.0, &pb.1);
            if let Some(p) = prev.take() {
                let _ = dev.poll(wgpu::PollType::Wait { submission_index: Some(p), timeout: Some(Duration::from_secs(5)) });
            }
            prev = Some(i);
        }
        let _ = dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(5)) });
    });
    let ok = rx.recv_timeout(Duration::from_secs(30)).unwrap_or(false);
    *g = Some(Load { stop, th: Some(th) });
    ok
}
