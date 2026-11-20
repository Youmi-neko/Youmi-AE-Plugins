//! ABI 1.3: post stages on the GPU (src/post.wgsl) on the device-resident blurred crop, in lenscore
//! postProcess's order: glow + vignetting -> halation -> luminance aberration (modes 0-2) -> rolloff.
//! Only the prefix the desc allows (obgpu_post_desc.stages); the wrapper's CPU postProcess runs the rest.

use std::time::Instant;

use crate::ffi::*;
use crate::gpu::{block_on, Gpu};
use crate::job::{self, Fail};

const POST_SHADER: &str = include_str!("post.wgsl");
const PD_STRIDE: u64 = 256;
const PD_BYTES: u64 = 128;
const SLOTS: u64 = 256;

const ENTRIES: [&str; 29] = [
    "po_lum", "po_grid1", "po_grid6", "po_boxline", "po_up", "po_gauss", "po_glowvig", "po_hal_mask", "po_hal_apply", "po_la", "po_roll",
    // ABI 1.4
    "po_copy", "po_sub", "po_mm_line", "dr_ya", "dr_mul", "dr_lg", "dr_v", "dr_mag", "dr_dir", "po_imgcopy", "dr_smear", "rim_b", "rim_cw",
    "rim_final", "fr_m", "fr_gate", "fr_f", "fr_final",
];

/// test switch: OBGPU_POST_NOREUSE=1 = the phase-4 plane layout (no reuse), to prove the reuse changes nothing
pub fn noreuse() -> bool {
    std::env::var("OBGPU_POST_NOREUSE").is_ok()
}

pub struct PostPipes {
    pub bgl: wgpu::BindGroupLayout,
    pub p: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
}

pub fn build_pipes(gpu: &Gpu) -> Result<PostPipes, String> {
    let dev = &gpu.device;
    let t0 = Instant::now();
    let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let iscope = dev.push_error_scope(wgpu::ErrorFilter::Internal);
    use wgpu::BufferBindingType as T;
    let buf = |b: u32, ty: T, dynamic: bool| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: if dynamic { wgpu::BufferSize::new(PD_BYTES) } else { None } },
        count: None,
    };
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("obgpu.post.bgl"),
        entries: &[
            buf(0, T::Uniform, true),
            buf(1, T::Storage { read_only: false }, false),
            buf(2, T::Storage { read_only: false }, false),
            buf(3, T::Storage { read_only: true }, false),
            buf(4, T::Storage { read_only: true }, false),
            buf(5, T::Storage { read_only: false }, false),
        ],
    });
    let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("obgpu.post"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let desc = wgpu::ShaderModuleDescriptor { label: Some("obgpu.post"), source: wgpu::ShaderSource::Wgsl(POST_SHADER.into()) };
    let checks = wgpu::ShaderRuntimeChecks { bounds_checks: true, force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() };
    let module = unsafe { dev.create_shader_module_trusted(desc, checks) };
    let mut p = std::collections::HashMap::new();
    for e in ENTRIES.iter() {
        let pl = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(e),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(e),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &[], zero_initialize_workgroup_memory: false },
            cache: None,
        });
        p.insert(*e, pl);
    }
    let ie = block_on(iscope.pop()).flatten();
    if let Some(Some(e)) = block_on(scope.pop()) {
        return Err(format!("post shader: {e}"));
    }
    if let Some(e) = ie {
        return Err(format!("post shader (backend compiler): {e}"));
    }
    if gpu.verbose {
        eprintln!("[obgpu] compiled post pipelines in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(PostPipes { bgl, p })
}

fn fail(status: i32, msg: impl Into<String>) -> Fail {
    Fail { status, msg: msg.into() }
}

pub fn check(d: &obgpu_post_desc, w: i32, h: i32) -> Result<(), Fail> {
    // (lib.rs widens a version-1 / ABI 1.3 desc into the full struct with la34 = 0)
    let v1 = d.struct_size as usize == POST_DESC_V1_SIZE && d.version == OBGPU_POST_VERSION;
    let v2 = d.struct_size as usize == std::mem::size_of::<obgpu_post_desc>() && d.version == OBGPU_POST_VERSION2;
    if !(v1 || v2) {
        return Err(fail(OBGPU_BAD_ARGS, "post desc size / version"));
    }
    if d.w != w || d.h != h {
        return Err(fail(OBGPU_BAD_ARGS, "post desc size != crop"));
    }
    let op_ok = |o: &obgpu_field_op| match o.kind {
        0 => true,
        1 => o.r >= 1 && o.r <= 64,
        2 => o.f >= 1 && o.gw >= 1 && o.gh >= 1 && o.rb0 >= 1 && o.rb1 >= 0 && o.gw <= w + o.f && o.gh <= h + o.f,
        _ => false,
    };
    if !op_ok(&d.hal_scene) || !op_ok(&d.hal_tight) || !op_ok(&d.hal_wide) || !op_ok(&d.la_field) {
        return Err(fail(OBGPU_BAD_ARGS, "post desc field ops"));
    }
    let ops34 = [&d.dr_soft, &d.dr_dir, &d.rim_reg, &d.rim_fb, &d.rim_fd, &d.rim_sig[0], &d.rim_sig[1], &d.rim_sig[2], &d.rim_s0];
    if !matches!(d.la34, 0 | 3 | 4) || !ops34.iter().all(|o| op_ok(o)) {
        return Err(fail(OBGPU_BAD_ARGS, "post desc (ABI 1.4) fields"));
    }
    let rad_ok = |r: i32| (0..=4096).contains(&r);
    if ![d.dr_s1, d.dr_s2, d.dr_sp1, d.dr_sp2, d.rim_s1, d.rim_s2, d.rim_rn, d.rim_rc].iter().all(|r| rad_ok(*r)) {
        return Err(fail(OBGPU_BAD_ARGS, "post desc (ABI 1.4) radii"));
    }
    if d.la34 == 4 && d.dr_soft.kind == 0 {
        return Err(fail(OBGPU_BAD_ARGS, "post desc: drift without map softness"));
    }
    if d.la34 == 3 && (d.rim_reg.kind == 0 || (d.rim_fringe == 0 && d.rim_fb.kind == 0)) {
        return Err(fail(OBGPU_BAD_ARGS, "post desc: rim field ops"));
    }
    if d.stages & OBGPU_POST_GLOWVIG != 0 && d.glow_amt > 0.0 && (d.glow_f < 1 || d.glow_gw < 1 || d.glow_gh < 1 || d.glow_rb < 1) {
        return Err(fail(OBGPU_BAD_ARGS, "post desc glow grid"));
    }
    Ok(())
}

/// workspace floats of a field op (grids, Gaussian temp + weights), the image-sized planes excluded
fn op_floats(o: &obgpu_field_op, n: u64) -> u64 {
    match o.kind {
        1 => n + (2 * o.r as u64 + 1) + 4,
        2 => (o.gw as u64) * (o.gh as u64) * 2 * if o.rb1 > 0 { if noreuse() { 5 } else { 4 } } else { 3 } + 4,
        _ => 0,
    }
}

/// image-sized planes and op scratch (floats) of the luminance-aberration stage
fn la_need(d: &obgpu_post_desc, n: u64) -> (u64, u64) {
    let of = |o: &obgpu_field_op| op_floats(o, n);
    match d.la34 {
        // P0..P6 (see the drift branch of run: planes are reused once dead; was 15)
        4 => (if noreuse() { 15 } else { 7 }, of(&d.dr_soft).max(of(&d.dr_dir))),
        3 => (if noreuse() { 20 } else if d.rim_fringe != 0 { 13 } else { 11 }, [&d.rim_reg, &d.rim_fb, &d.rim_fd, &d.rim_sig[0], &d.rim_sig[1], &d.rim_sig[2], &d.rim_s0].iter().map(|o| of(o)).max().unwrap_or(0)),
        _ => (2, of(&d.la_field)),
    }
}

/// device bytes of the post stages (fields + grids; the image / eL / prepared buffers are counted by the caller)
pub fn bytes(d: &obgpu_post_desc, stages: u32) -> u64 {
    let n = (d.w as u64) * (d.h as u64);
    let mut planes = 0u64;
    let mut ops = 0u64;
    if stages & OBGPU_POST_HALATION != 0 {
        planes = planes.max(5);
        ops = ops.max(op_floats(&d.hal_scene, n)).max(op_floats(&d.hal_tight, n)).max(op_floats(&d.hal_wide, n));
    }
    if stages & OBGPU_POST_LUMAB != 0 {
        let (p, o) = la_need(d, n);
        planes = planes.max(p);
        ops = ops.max(o);
    }
    let glow = if stages & OBGPU_POST_GLOWVIG != 0 && d.glow_amt > 0.0 { (d.glow_gw as u64) * (d.glow_gh as u64) * 6 * 2 } else { 0 };
    if std::env::var("OBGPU_TRACE").is_ok() {
        eprintln!("[obgpu.trace] post bytes: n {n} planes {planes} ops {ops} glow {glow} (glow grid {}x{} f {})", d.glow_gw, d.glow_gh, d.glow_f);
    }
    (planes * n + ops + glow + 16) * 4
}

struct Pass {
    ep: &'static str,
    rec: [u32; 32],
    groups: (u32, u32),
}

fn rec(a: [i32; 4], b: [i32; 4], c: [i32; 4], d: [f32; 4], e: [f32; 4]) -> [u32; 32] {
    let mut r = [0u32; 32];
    for k in 0..4 {
        r[k] = a[k] as u32;
        r[4 + k] = b[k] as u32;
        r[8 + k] = c[k] as u32;
        r[12 + k] = d[k].to_bits();
        r[16 + k] = e[k].to_bits();
    }
    r
}

pub struct Bufs<'a> {
    pub img: &'a wgpu::Buffer,
    pub el: &'a wgpu::Buffer,
    pub prgba: &'a wgpu::Buffer,
    pub paux: &'a wgpu::Buffer,
}

/// Runs the allowed prefix (stages & d.stages) on b.img / b.el. Returns the stages done.
#[allow(clippy::too_many_arguments)]
pub fn run(gpu: &Gpu, pipes: &PostPipes, d: &obgpu_post_desc, stages: u32, roll: bool, b: &Bufs, abort: &dyn Fn() -> bool, tun: &job::Tuning, st: &mut obgpu_stats) -> Result<u32, Fail> {
    let (w, h) = (d.w, d.h);
    let n = (w as u64) * (h as u64);
    let dev = &gpu.device;
    let q = &gpu.queue;
    let s = stages & d.stages;
    if s == 0 {
        return Ok(0);
    }
    // ---- workspace: planes P0..P4 (image-sized), then op scratch (grids / Gaussian temp + weights), then glow grids
    let hal = s & OBGPU_POST_HALATION != 0;
    let la = s & OBGPU_POST_LUMAB != 0;
    let glow = s & OBGPU_POST_GLOWVIG != 0 && d.glow_amt > 0.0;
    let nplanes = (if la { la_need(d, n).0 } else { 0 }).max(if hal { 5 } else { 0 });
    let mut ops_max = 0u64;
    if hal {
        ops_max = ops_max.max(op_floats(&d.hal_scene, n)).max(op_floats(&d.hal_tight, n)).max(op_floats(&d.hal_wide, n));
    }
    if la {
        ops_max = ops_max.max(la_need(d, n).1);
    }
    let plane = |k: u64| (k * n) as i32;
    let ops_off = nplanes * n;
    let glow_cells = if glow { (d.glow_gw as u64) * (d.glow_gh as u64) } else { 0 };
    let glow_off = ops_off + ops_max;
    let wk_len = glow_off + glow_cells * 6 * 2 + 16;
    if gpu.verbose {
        eprintln!("[obgpu] post: wk {} MB (planes {} x {} px, op scratch {} floats, glow cells {}), binding limit {} MB",
            wk_len * 4 >> 20, nplanes, n, ops_max, glow_cells, gpu.limits.max_storage_buffer_binding_size >> 20);
    }
    use wgpu::BufferUsages as U;
    let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mk = |label: &str, size: u64, usage: U| dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16).div_ceil(16) * 16, usage, mapped_at_creation: false });
    let b_wk = mk("post.wk", wk_len * 4, U::STORAGE | U::COPY_DST);
    let b_d = mk("post.pd", PD_STRIDE * SLOTS, U::UNIFORM | U::COPY_DST);
    if let Some(Some(e)) = block_on(oom.pop()) {
        return Err(fail(OBGPU_OUT_OF_VRAM, format!("post allocation: {e}")));
    }
    st.vram_bytes = st.vram_bytes.max(wk_len * 4);
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("obgpu.post.bg"),
        layout: &pipes.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &b_d, offset: 0, size: wgpu::BufferSize::new(PD_BYTES) }) },
            wgpu::BindGroupEntry { binding: 1, resource: b.img.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b.el.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b.prgba.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b.paux.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: b_wk.as_entire_binding() },
        ],
    });
    let px = ((w as u32).div_ceil(256), h as u32);
    let cells_groups = |cells: u64| -> (u32, u32) {
        let g = cells.div_ceil(64) as u32;
        if g <= 32768 { (g, 1) } else { (32768, g.div_ceil(32768)) }
    };
    let none = [0i32; 4];
    let nf = [0f32; 4];
    // field op: in plane -> out plane (lenscore blurField semantics, resolved in the desc)
    let field = |ps: &mut Vec<Pass>, weights: &mut Vec<(u64, Vec<f32>)>, o: &obgpu_field_op, inp: i32, out: i32| {
        let base = ops_off as i32;
        match o.kind {
            1 => {
                let tmp = base;
                let wts = base + n as i32;
                let sg = o.sigma as f64;
                let k: Vec<f32> = (-o.r..=o.r).map(|i| (-0.5 * (i * i) as f64 / (sg * sg)).exp() as f32).collect();
                weights.push((wts as u64, k));
                ps.push(Pass { ep: "po_gauss", rec: rec([w, h, o.r, 0], [inp, tmp, wts, 0], none, nf, nf), groups: px });
                ps.push(Pass { ep: "po_gauss", rec: rec([w, h, o.r, 1], [tmp, out, wts, 0], none, nf, nf), groups: px });
            }
            2 => {
                let gsz = o.gw * o.gh * 2;
                let (g0, ga, gb, gc, gd) = (base, base + gsz, base + 2 * gsz, base + 3 * gsz, base + 4 * gsz);
                let c4 = [o.gx, o.gy, o.gw, o.gh];
                ps.push(Pass { ep: "po_grid1", rec: rec([w, h, o.f, 0], [inp, g0, 0, 0], c4, nf, nf), groups: cells_groups((o.gw * o.gh) as u64) });
                let lines = |ps: &mut Vec<Pass>, rb: i32, x: i32, y: i32| -> i32 {
                    // 3 passes along rows, then 3 along columns (lenscore: each row's 3 passes, then each column's)
                    let mut src = g0;
                    let mut dst = x;
                    for k in 0..6 {
                        let dir = if k < 3 { 0 } else { 1 };
                        let gr = if dir == 0 { ((o.gw as u32).div_ceil(64), o.gh as u32) } else { ((o.gh as u32).div_ceil(64), o.gw as u32) };
                        ps.push(Pass { ep: "po_boxline", rec: rec([o.gw, o.gh, rb, dir], [src, dst, 0, 0], [2, 0, 0, 0], nf, nf), groups: gr });
                        src = dst;
                        dst = if dst == x { y } else { x };
                    }
                    src
                };
                let r0 = lines(ps, o.rb0, ga, gb);
                // phase 5: the second radius ping-pongs through ga (free once r0 is in gb) and gc: 4 grids, not 5
                let r1 = if o.rb1 > 0 { if noreuse() { lines(ps, o.rb1, gc, gd) } else { lines(ps, o.rb1, ga, gc) } } else { r0 };
                ps.push(Pass { ep: "po_up", rec: rec([w, h, o.f, (o.rb1 > 0) as i32], [r0, r1, out, 0], c4, [o.t, 0.0, 0.0, 0.0], nf), groups: px });
            }
            _ => {}
        }
    };
    // ---- the pass list in the CPU's order; each stage = one submission (abort checks between)
    let mut groups: Vec<(Vec<Pass>, Vec<(u64, Vec<f32>)>)> = Vec::new();
    let mut done = 0u32;
    if s & OBGPU_POST_GLOWVIG != 0 {
        let mut ps = Vec::new();
        let (gw, gh) = (d.glow_gw, d.glow_gh);
        let c4 = [d.glow_gox, d.glow_goy, gw, gh];
        let g0 = glow_off as i32;
        let g1 = g0 + gw * gh * 6;
        if glow {
            ps.push(Pass { ep: "po_grid6", rec: rec([w, h, d.glow_f, 0], [0, g0, 0, 0], c4, nf, nf), groups: cells_groups((gw * gh) as u64) });
            let (mut src, mut dst) = (g0, g1);
            for k in 0..6 {
                let dir = if k < 3 { 0 } else { 1 };
                let gr = if dir == 0 { ((gw as u32).div_ceil(64), gh as u32) } else { ((gh as u32).div_ceil(64), gw as u32) };
                ps.push(Pass { ep: "po_boxline", rec: rec([gw, gh, d.glow_rb, dir], [src, dst, 0, 0], [6, 0, 0, 0], nf, nf), groups: gr });
                std::mem::swap(&mut src, &mut dst);
            }
            // after 6 passes the result is back in g0
            ps.push(Pass { ep: "po_glowvig", rec: rec([w, h, d.glow_f, roll as i32], [0, src, 0, 0], c4, [d.glow_amt, d.vig, d.fcx, d.fcy], [d.half_diag, d.pa, 0.0, 0.0]), groups: px });
        } else {
            ps.push(Pass { ep: "po_glowvig", rec: rec([w, h, 1, roll as i32], none, [0, 0, 1, 1], [0.0, d.vig, d.fcx, d.fcy], [d.half_diag, d.pa, 0.0, 0.0]), groups: px });
        }
        groups.push((ps, Vec::new()));
        done |= OBGPU_POST_GLOWVIG;
    }
    if hal {
        let (ph, psc, pm, pt, pw) = (plane(0), plane(1), plane(2), plane(3), plane(4));
        // each field op reuses the op scratch: one submission per op (the scratch is free again after it)
        let mut a = Vec::new();
        let mut wa = Vec::new();
        a.push(Pass { ep: "po_lum", rec: rec([w, h, 0, 0], [ph, 0, 0, 0], none, nf, nf), groups: px });
        field(&mut a, &mut wa, &d.hal_scene, ph, psc);
        a.push(Pass { ep: "po_hal_mask", rec: rec([w, h, 0, 0], [ph, psc, pm, 0], none, [d.hal_floor, d.hal_relk, d.hal_soft, 0.0], nf), groups: px });
        groups.push((a, wa));
        if d.hal_rim > 0.0 {
            let (mut a, mut wa) = (Vec::new(), Vec::new());
            field(&mut a, &mut wa, &d.hal_tight, ph, pt);
            groups.push((a, wa));
        }
        if d.hal_rim < 1.0 {
            let (mut a, mut wa) = (Vec::new(), Vec::new());
            field(&mut a, &mut wa, &d.hal_wide, ph, pw);
            groups.push((a, wa));
        }
        groups.push((vec![Pass { ep: "po_hal_apply", rec: rec([w, h, 0, 0], [pt, pw, pm, 0], none, [d.hal_rim, d.hal_amt, 0.0, 0.0], [d.hal_tint[0], d.hal_tint[1], d.hal_tint[2], 0.0]), groups: px }], Vec::new()));
        done |= OBGPU_POST_HALATION;
    }
    if la && d.la34 == 4 {
        // ---- Channel Drift (applyChannelDrift): planes Y A T Ls As lg vx vy mag bx by, src copy (4 floats)
        let p = |k: u64| plane(k);
        // plane reuse (phase 5, same arithmetic): Y -> As -> bx in P0, A -> lg in P1, T in P2, Ls -> by in P3,
        // vx P4, vy P5, mag P6, and the 4-float source copy of the smear over P0..P3 (all dead by then): 7 planes
        let (py, pa, pt, pls, pas, plg, pvx, pvy, pmag, pbx, pby, psrc) = if noreuse() {
            (p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11)) // (the phase-4 layout, for the equality test)
        } else {
            (p(0), p(1), p(2), p(3), p(0), p(1), p(4), p(5), p(6), p(0), p(3), p(0))
        };
        let kmax = d.dr_k.iter().fold(0f32, |m, k| m.max(k.abs()));
        if d.dr_len * kmax >= 1e-3 {
            let mm = |ps: &mut Vec<Pass>, pl: i32, r: i32, dir: i32, mx: bool| {
                if r > 0 {
                    ps.push(Pass { ep: "po_mm_line", rec: rec([w, h, r, dir], [pl, pt, 0, 0], [mx as i32, 0, 0, 0], nf, nf), groups: px });
                    ps.push(Pass { ep: "po_copy", rec: rec([w, h, 0, 0], [pt, pl, 0, 0], none, nf, nf), groups: px });
                }
            };
            let mut a = vec![Pass { ep: "dr_ya", rec: rec([w, h, 0, 0], [py, pa, 0, 0], none, nf, nf), groups: px }];
            if d.dr_s1 > 0 {
                for mx in [true, false] {
                    for dir in 0..4 {
                        mm(&mut a, py, if dir < 2 { d.dr_s1 } else { d.dr_s2 }, dir, mx);
                    }
                }
            }
            a.push(Pass { ep: "dr_mul", rec: rec([w, h, 0, 0], [py, pa, 0, 0], none, nf, nf), groups: px });
            groups.push((a, Vec::new()));
            let (mut a, mut wa) = (Vec::new(), Vec::new());
            field(&mut a, &mut wa, &d.dr_soft, py, pls);
            groups.push((a, wa));
            let (mut a, mut wa) = (Vec::new(), Vec::new());
            field(&mut a, &mut wa, &d.dr_soft, pa, pas);
            a.push(Pass { ep: "dr_lg", rec: rec([w, h, 0, 0], [pls, pas, plg, 0], none, nf, nf), groups: px });
            a.push(Pass { ep: "dr_v", rec: rec([w, h, 0, 0], [plg, pvx, pvy, 0], none, [d.dr_thr, d.dr_curve, d.dr_pres, d.dr_kstop], [d.dr_len, d.dr_sm2, 0.0, 0.0]), groups: px });
            groups.push((a, wa));
            if d.dr_v2 != 0 && d.dr_sp1 > 0 {
                let mut a = vec![Pass { ep: "dr_mag", rec: rec([w, h, 0, 0], [pvx, pvy, pmag, 0], none, nf, nf), groups: px }];
                for dir in 0..4 {
                    mm(&mut a, pmag, if dir < 2 { d.dr_sp1 } else { d.dr_sp2 }, dir, true);
                }
                groups.push((a, Vec::new()));
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                field(&mut a, &mut wa, &d.dr_dir, pvx, pbx);
                groups.push((a, wa));
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                field(&mut a, &mut wa, &d.dr_dir, pvy, pby);
                a.push(Pass { ep: "dr_dir", rec: rec([w, h, 0, 0], [pvx, pvy, pbx, pby], [pmag, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, d.dr_lencap, 0.0]), groups: px });
                groups.push((a, wa));
            }
            groups.push((
                vec![
                    Pass { ep: "po_imgcopy", rec: rec([w, h, 0, 0], [psrc, 0, 0, 0], none, nf, nf), groups: px },
                    Pass { ep: "dr_smear", rec: rec([w, h, d.dr_v2, 0], [pvx, pvy, psrc, 0], none, [d.dr_k[0], d.dr_k[1], d.dr_k[2], 0.0], nf), groups: px },
                ],
                Vec::new(),
            ));
        }
        done |= OBGPU_POST_LUMAB | OBGPU_POST_LA34;
    } else if la && d.la34 == 3 {
        // ---- Contrast Rim v2.1 (applyContrastRim / contrastFringe)
        let p = |k: u64| plane(k);
        let (py, pyc, pyo, pbn, pb_, pd_, pbs, pds, pmb) = (p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8));
        // plane reuse (phase 5, same arithmetic): drawn rim 11 planes (was 18), fringe 13 (was 20); see below
        let pt = if noreuse() { p(17) } else { p(10) };
        let mm = |ps: &mut Vec<Pass>, from: i32, to: i32, r: i32, dir: i32, mx: bool| {
            ps.push(Pass { ep: "po_mm_line", rec: rec([w, h, r, dir], [from, to, 0, 0], [mx as i32, 0, 0, 0], nf, nf), groups: px });
        };
        let copy = |ps: &mut Vec<Pass>, from: i32, to: i32| ps.push(Pass { ep: "po_copy", rec: rec([w, h, 0, 0], [from, to, 0, 0], none, nf, nf), groups: px });
        let morph = |ps: &mut Vec<Pass>, pl: i32, max_first: bool| {
            for pass in 0..2 {
                for dir in 0..4 {
                    let r = if dir < 2 { d.rim_s1 } else { d.rim_s2 };
                    if r > 0 {
                        mm(ps, pl, pt, r, dir, (pass == 0) == max_first);
                        copy(ps, pt, pl);
                    }
                }
            }
        };
        let mut a = vec![Pass { ep: "po_lum", rec: rec([w, h, 0, 0], [py, 0, 0, 0], none, nf, nf), groups: px }];
        copy(&mut a, py, pyc);
        if d.rim_s1 > 0 {
            morph(&mut a, pyc, true);
        }
        copy(&mut a, pyc, pyo);
        if d.rim_s1 > 0 {
            morph(&mut a, pyo, false);
        }
        // filter2D = row pass then column pass
        mm(&mut a, pyo, pt, d.rim_rn, 0, true);
        mm(&mut a, pt, pbn, d.rim_rn, 1, true);
        mm(&mut a, pyo, pt, d.rim_rc, 0, true);
        mm(&mut a, pt, pb_, d.rim_rc, 1, true);
        mm(&mut a, pyc, pt, d.rim_rc, 0, false);
        mm(&mut a, pt, pd_, d.rim_rc, 1, false);
        groups.push((a, Vec::new()));
        let (mut a, mut wa) = (Vec::new(), Vec::new());
        field(&mut a, &mut wa, &d.rim_reg, pb_, pbs);
        groups.push((a, wa));
        let (mut a, mut wa) = (Vec::new(), Vec::new());
        field(&mut a, &mut wa, &d.rim_reg, pd_, pds);
        let pnb = p(9);
        a.push(Pass { ep: "rim_b", rec: rec([w, h, 0, 0], [pbs, pds, pyc, pmb], [pnb, 0, 0, 0], nf, nf), groups: px });
        groups.push((a, wa));
        let hi4 = [d.rim_hi[0], d.rim_hi[1], d.rim_hi[2], 0.0];
        let fcopy = |a: &mut Vec<Pass>, wa: &mut Vec<(u64, Vec<f32>)>, o: &obgpu_field_op, from: i32, to: i32| {
            if o.kind == 0 {
                a.push(Pass { ep: "po_copy", rec: rec([w, h, 0, 0], [from, to, 0, 0], none, nf, nf), groups: px });
            } else {
                field(a, wa, o, from, to);
            }
        };
        if d.rim_fringe != 0 {
            // after fr_m: Y, Yc, Yo, B, D, T are dead -> cbs in P0..P2, wm P4, gate P5, f P10, tt over m (P9, dead
            // once wm exists), spread / base P11 / P12, and each D[ch] over its cbs[ch] (dead after fr_f of ch)
            let ((pm, pf, pwm, pgate, ptt, psp, pbase), pcbs, pdd) = if noreuse() {
                ((p(9), p(10), p(14), p(15), p(16), p(18), p(19)), [p(11), p(12), p(13)], [p(4), p(5), p(9)])
            } else {
                ((p(9), p(10), p(4), p(5), p(9), p(11), p(12)), [p(0), p(1), p(2)], [p(0), p(1), p(2)])
            };
            groups.push((vec![Pass { ep: "fr_m", rec: rec([w, h, 0, 0], [py, pyc, pmb, pm], none, nf, nf), groups: px }], Vec::new()));
            for ch in 0..3 {
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                a.push(Pass { ep: "rim_cw", rec: rec([w, h, ch as i32, 1], [pm, pf, 0, 0], none, nf, nf), groups: px });
                field(&mut a, &mut wa, &d.rim_reg, pf, pcbs[ch]);
                groups.push((a, wa));
            }
            let (mut a, mut wa) = (Vec::new(), Vec::new());
            field(&mut a, &mut wa, &d.rim_reg, pm, pwm);
            a.push(Pass { ep: "fr_gate", rec: rec([w, h, 0, 0], [pbs, pds, pbn, pgate], [ptt, 0, 0, 0], [d.rim_thr, d.rim_curve, 0.0, 0.0], nf), groups: px });
            groups.push((a, wa));
            for ch in 0..3 {
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                a.push(Pass { ep: "fr_f", rec: rec([w, h, 0, 0], [pgate, pmb, pcbs[ch], pwm], [pf, 0, 0, 0], nf, nf), groups: px });
                fcopy(&mut a, &mut wa, &d.rim_sig[ch], pf, psp);
                groups.push((a, wa));
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                fcopy(&mut a, &mut wa, &d.rim_s0, pf, pbase);
                a.push(Pass { ep: "po_sub", rec: rec([w, h, 0, 0], [psp, pbase, pdd[ch], 0], none, nf, nf), groups: px });
                groups.push((a, wa));
            }
            let mut r = rec([w, h, 0, 0], [pdd[0], pdd[1], pdd[2], ptt], none, [d.rim_amt, d.rim_hs, d.rim_mixb, d.rim_comp], nf);
            for k in 0..4 {
                r[24 + k] = hi4[k].to_bits();
            }
            groups.push((vec![Pass { ep: "fr_final", rec: r, groups: px }], Vec::new()));
        } else {
            // after rim_b: Y, Yc, Yo, B, D are dead -> Fb P0, Fd P1, f P2, cb P4 P5 and P9 (nb, dead after Fd),
            // wb over T (P10)
            let ((pfb, pfd, pf, pwb), pcb) = if noreuse() {
                ((p(10), p(11), p(12), p(16)), [p(13), p(14), p(15)])
            } else {
                ((p(0), p(1), p(2), p(10)), [p(4), p(5), p(9)])
            };
            let (mut a, mut wa) = (Vec::new(), Vec::new());
            field(&mut a, &mut wa, &d.rim_fb, pmb, pfb);
            groups.push((a, wa));
            if d.rim_bright_side != 0 {
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                field(&mut a, &mut wa, &d.rim_fd, pnb, pfd);
                groups.push((a, wa));
            }
            if d.rim_mixb_on != 0 {
                for ch in 0..3 {
                    let (mut a, mut wa) = (Vec::new(), Vec::new());
                    a.push(Pass { ep: "rim_cw", rec: rec([w, h, ch as i32, 0], [pmb, pf, 0, 0], none, nf, nf), groups: px });
                    field(&mut a, &mut wa, &d.rim_reg, pf, pcb[ch]);
                    groups.push((a, wa));
                }
                let (mut a, mut wa) = (Vec::new(), Vec::new());
                field(&mut a, &mut wa, &d.rim_reg, pmb, pwb);
                groups.push((a, wa));
            }
            let mut r = rec([w, h, d.rim_bright_side, d.rim_mixb_on], [pbs, pds, pbn, pmb], [pfb, pfd, pcb[0], pcb[1]], [d.rim_amt, d.rim_thr, d.rim_curve, d.rim_hs], [d.rim_ex, d.rim_mixb, 0.0, 0.0]);
            r[20] = pcb[2] as u32;
            r[21] = pwb as u32;
            for k in 0..4 {
                r[24 + k] = hi4[k].to_bits();
            }
            groups.push((vec![Pass { ep: "rim_final", rec: r, groups: px }], Vec::new()));
        }
        done |= OBGPU_POST_LUMAB | OBGPU_POST_LA34;
    } else if la {
        let (py, pg) = (plane(0), plane(1));
        let (mut a, mut wa) = (Vec::new(), Vec::new());
        a.push(Pass { ep: "po_lum", rec: rec([w, h, 0, 0], [py, 0, 0, 0], none, nf, nf), groups: px });
        field(&mut a, &mut wa, &d.la_field, py, pg);
        a.push(Pass { ep: "po_la", rec: rec([w, h, d.la_mode, 0], [py, pg, 0, 0], none, [d.la_amt, d.la_thr, d.la_curve, d.la_hs], [d.la_hi[0], d.la_hi[1], d.la_hi[2], 0.0]), groups: px });
        groups.push((a, wa));
        done |= OBGPU_POST_LUMAB;
    }
    if s & OBGPU_POST_ROLLOFF != 0 {
        if roll {
            groups.push((vec![Pass { ep: "po_roll", rec: rec([w, h, d.roll_v2, 0], none, none, [d.roll_knee, 0.0, 0.0, 0.0], nf), groups: px }], Vec::new()));
        }
        done |= OBGPU_POST_ROLLOFF; // (not roll-relevant: the CPU would skip it as well)
    }
    let mut ts = crate::gpu::Ts::new(gpu, 64);
    let mut slot = 0u64;
    let mut last: Option<crate::gpu::Sub> = None;
    for (ps, wts) in groups.iter() {
        if abort() {
            return Err(fail(OBGPU_ABORTED, "aborted"));
        }
        for (off, k) in wts {
            q.write_buffer(&b_wk, off * 4, unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 4) });
        }
        if slot + ps.len() as u64 > SLOTS {
            // the previous submissions read their records: wait before overwriting (rare: > 256 passes)
            let idx = gpu.submit(dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }).finish());
            gpu.wait_sub(&idx, abort, tun.timeout).map_err(|e| fail(e, "post wait"))?;
            slot = 0;
        }
        let first = slot;
        for p in ps.iter() {
            q.write_buffer(&b_d, slot * PD_STRIDE, unsafe { std::slice::from_raw_parts(p.rec.as_ptr() as *const u8, 128) });
            slot += 1;
        }
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let tw = ts.as_mut().and_then(|t| t.writes_l(ps.first().map(|p| p.ep).unwrap_or("post")));
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
            for (k, p) in ps.iter().enumerate() {
                cp.set_pipeline(&pipes.p[p.ep]);
                cp.set_bind_group(0, &bg, &[((first + k as u64) * PD_STRIDE) as u32]);
                cp.dispatch_workgroups(p.groups.0, p.groups.1, 1);
            }
        }
        // phase 5: no wait per group: the op scratch / weights rewritten by the next group are queue writes, which
        // run after the earlier submissions (same results; the GPU is not left idle). Abort: checked before
        // every group and while waiting for the last one.
        last = Some(gpu.submit(enc.finish()));
        st.submissions += 1;
    }
    if let Some(idx) = last {
        let t = tun.timeout * (groups.len().max(1) as u32);
        gpu.wait_sub(&idx, abort, t).map_err(|e| fail(e, "post stage wait"))?;
    }
    job::busy_add(match ts.as_ref() { Some(t) => t.busy_ms(gpu), None => None });
    gpu.free(&[&b_wk]);
    Ok(done)
}
