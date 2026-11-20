//! ABI 1.1 frame pipeline: prepareHighlights on the GPU (src/prepare.wgsl), its output written straight into
//! the blur stage's input buffers, then the blur stage (job::run_any). One upload (window source + map),
//! one readback (blurred crop + eL [+ prepared colour on request]) plus a 28-byte flags readback in between.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ffi::*;
use crate::gpu::{block_on, Gpu};
use crate::job::{self, Fail};

const PREP_SHADER: &str = include_str!("prepare.wgsl");
const PD_STRIDE: u64 = 256;
const PD_BYTES: u64 = 64;

pub struct PrepPipes {
    pub bgl: wgpu::BindGroupLayout,
    pub p: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
}

const ENTRIES: [&str; 11] =
    ["pp_lum", "pp_box_rows", "pp_box_cols", "pp_mm_line", "pp_E", "pp_wp_cells", "pp_wp_apply", "pp_det_acc", "pp_dmin", "pp_best", "pp_dots"];

pub fn build_pipes(gpu: &Gpu) -> Result<PrepPipes, String> {
    let dev = &gpu.device;
    let t0 = Instant::now();
    let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let iscope = dev.push_error_scope(wgpu::ErrorFilter::Internal);
    let buf = |b: u32, ty: wgpu::BufferBindingType, dynamic: bool| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: if dynamic { wgpu::BufferSize::new(PD_BYTES) } else { None } },
        count: None,
    };
    use wgpu::BufferBindingType as T;
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("obgpu.prep.bgl"),
        entries: &[
            buf(0, T::Uniform, false),
            buf(1, T::Uniform, true),
            buf(2, T::Storage { read_only: true }, false),
            buf(3, T::Storage { read_only: true }, false),
            buf(4, T::Storage { read_only: false }, false),
            buf(5, T::Storage { read_only: false }, false),
            buf(6, T::Storage { read_only: false }, false),
            buf(7, T::Storage { read_only: false }, false),
        ],
    });
    let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("obgpu.prep"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let desc = wgpu::ShaderModuleDescriptor { label: Some("obgpu.prepare"), source: wgpu::ShaderSource::Wgsl(PREP_SHADER.into()) };
    let checks = wgpu::ShaderRuntimeChecks { bounds_checks: true, force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() };
    let module = unsafe { dev.create_shader_module_trusted(desc, checks) };
    let mut p = std::collections::HashMap::new();
    for e in ENTRIES.iter().chain(["pp_rows"].iter()) {
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
        return Err(format!("prepare shader: {e}"));
    }
    if let Some(e) = ie {
        return Err(format!("prepare shader (backend compiler): {e}"));
    }
    if gpu.verbose {
        eprintln!("[obgpu] compiled prepare pipelines in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(PrepPipes { bgl, p })
}

fn fail(status: i32, msg: impl Into<String>) -> Fail {
    Fail { status, msg: msg.into() }
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub struct FrameIn<'a> {
    pub src: &'a [f32],
    pub src_stride: usize,
    pub map: Option<(&'a [f32], usize)>,
    pub crop: (i32, i32, i32, i32),
}

/// footprint of an untiled prepare stage on the device (bytes)
#[allow(dead_code)]
pub fn prepare_bytes(p: &obgpu_prepare_desc, crop_px: u64) -> u64 {
    let n = (p.w as u64) * (p.h as u64);
    let planes = if p.dmin_on != 0 { 8 } else { 7 };
    let grid = if p.wp_on != 0 { (p.wp_gw as u64) * (p.wp_gh as u64) * 11 * 4 } else { 0 };
    n * 16 + if p.has_map != 0 { n * 4 } else { 0 } + n * planes * 4 + grid + crop_px * 24
}

pub fn check_prep(p: &obgpu_prepare_desc) -> Result<(), Fail> {
    if p.struct_size as usize != std::mem::size_of::<obgpu_prepare_desc>() || p.version != 1 || p.w <= 0 || p.h <= 0 {
        return Err(fail(OBGPU_BAD_ARGS, "prepare desc size / version"));
    }
    if p.render_version < 2 {
        return Err(fail(OBGPU_UNSUPPORTED, "renderVersion 1: CPU"));
    }
    if p.det_n < 0 || p.det_n > 8 || p.dmin_n < 0 || p.dmin_n > 8 || p.knee_box_r < 0 || (p.wp_on != 0 && (p.wp_f < 1 || p.wp_gw < 1 || p.wp_gh < 1)) {
        return Err(fail(OBGPU_BAD_ARGS, "prepare desc ranges"));
    }
    Ok(())
}

/// bytes of one prepare band of `rows` window rows (planes + source + map + White Protection grid)
fn band_bytes(p: &obgpu_prepare_desc, rows: u64) -> u64 {
    let n = (p.w as u64) * rows;
    let planes = if p.dmin_on != 0 { 8 } else { 7 };
    let grid = if p.wp_on != 0 { ((p.wp_gw as u64) * (rows / p.wp_f.max(1) as u64 + 2)) * 11 * 4 } else { 0 };
    n * 16 + if p.has_map != 0 { n * 4 } else { 0 } + n * planes * 4 + grid
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    gpu: &Gpu,
    pipes: &PrepPipes,
    fin: &FrameIn,
    p: &obgpu_prepare_desc,
    bd: &obgpu_blur_desc,
    out: &job::Output,
    prepared_out: Option<*mut f32>,
    prepared_el: Option<*mut f32>,
    post: Option<(&obgpu_post_desc, &crate::post::PostPipes)>,
    abort: &dyn Fn() -> bool,
    tun: &job::Tuning,
    st: &mut obgpu_stats,
) -> Result<(bool, u32), Fail> {
    let t0 = Instant::now();
    let (w, h) = (p.w as usize, p.h as usize);
    let (cx, cy, cw, ch) = fin.crop;
    if cx < 0 || cy < 0 || cw <= 0 || ch <= 0 || cx + cw > p.w || cy + ch > p.h || bd.w != cw || bd.h != ch {
        return Err(fail(OBGPU_BAD_ARGS, "crop rect / blur desc size"));
    }
    let crop_px = (cw as u64) * (ch as u64);
    let crop_bytes = crop_px * 24;
    // ---- prepare bands: output (crop) rows [oa, ob) computed on the window rows [oa - margin, ob + margin]
    // (lenscore::prepareMargin: every map value of an inner row depends only on rows within it -> the inner
    // rows equal the untiled result up to the summation start of the box means, ~1e-7)
    let pm = p.margin.max(0);
    let fits = |rows: i32| crop_bytes + band_bytes(p, rows as u64) <= tun.budget;
    let mut bands: Vec<(i32, i32, i32, i32)> = Vec::new();
    if fits(p.h) {
        bands.push((cy, cy + ch, 0, p.h));
    } else {
        if !fits((8 + 2 * pm).min(p.h)) {
            return Err(fail(OBGPU_OUT_OF_VRAM, format!("prepare: even an 8-row band does not fit the budget ({} MB)", tun.budget >> 20)));
        }
        let (mut lo, mut hi) = (8, ch.max(8));
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if fits((mid + 2 * pm).min(p.h)) { lo = mid } else { hi = mid }
        }
        let mut oa = cy;
        while oa < cy + ch {
            let ob = (oa + lo).min(cy + ch);
            bands.push((oa, ob, (oa - pm).max(0), (ob + pm).min(p.h)));
            oa = ob;
        }
    }
    let hmax = bands.iter().map(|b| b.3 - b.2).max().unwrap_or(1) as usize;
    let n = w * hmax;
    st.bands = bands.len() as i32;
    {
        // halo recomputation: window rows prepared in total / rows of the untiled pass (1.0 = none)
        let done: i64 = bands.iter().map(|b| (b.3 - b.2) as i64).sum();
        job::PREP.with(|c| c.set((bands.len() as i32, done as f64 / p.h.max(1) as f64)));
    }
    let dev = &gpu.device;
    let q = &gpu.queue;
    // ---- planes (floats): L T E D A B X [DM] LEV GS, sized for the tallest band
    let nn = n as u64;
    let mut off = 0u64;
    let mut take = |len: u64| {
        let o = off;
        off += len.max(4);
        o as i32
    };
    let (pl_l, pl_t, pl_e, pl_d, pl_a, pl_b, pl_x) = (take(nn), take(nn), take(nn), take(nn), take(nn), take(nn), take(nn));
    let pl_dm = if p.dmin_on != 0 { take(nn) } else { pl_x };
    let gw = p.wp_gw.max(1);
    let gh_max = if p.wp_on != 0 { (hmax as i32 + p.wp_f - 1) / p.wp_f + 1 } else { 1 };
    let g2 = (gw as u64) * (gh_max as u64);
    let pl_lev = take(10 * g2);
    let pl_gs = take(g2);
    let wk_len = off;
    let pl = [pl_l, pl_t, pl_e, pl_d, pl_a, pl_b, pl_a, pl_b, pl_t, pl_dm, pl_lev, pl_lev, pl_gs, 0, 0, 0];

    let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    use wgpu::BufferUsages as U;
    let mk = |label: &str, size: u64, usage: U| dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16).div_ceil(16) * 16, usage, mapped_at_creation: false });
    let b_g = mk("pg", 288, U::UNIFORM | U::COPY_DST);
    let b_d = mk("pd", PD_STRIDE * 512, U::UNIFORM | U::COPY_DST);
    let b_src = mk("src", (n * 16) as u64, U::STORAGE | U::COPY_DST);
    let b_map = mk("map", if fin.map.is_some() { (n * 4) as u64 } else { 16 }, U::STORAGE | U::COPY_DST);
    let b_wk = mk("wk", wk_len * 4, U::STORAGE | U::COPY_DST);
    let b_prgba = mk("prgba", crop_px * 16, U::STORAGE | U::COPY_SRC | U::COPY_DST);
    let b_paux = mk("paux", crop_px * 8, U::STORAGE | U::COPY_SRC | U::COPY_DST);
    let b_flags = mk("flags", 64, U::STORAGE | U::COPY_SRC | U::COPY_DST);
    let b_fl_rd = mk("flags.rd", 64, U::MAP_READ | U::COPY_DST);
    if let Some(Some(e)) = block_on(oom.pop()) {
        return Err(fail(OBGPU_OUT_OF_VRAM, format!("prepare allocation: {e}")));
    }
    st.vram_bytes = st.vram_bytes.max(crop_bytes + band_bytes(p, hmax as u64));
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("obgpu.prep.bg"),
        layout: &pipes.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_g.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &b_d, offset: 0, size: wgpu::BufferSize::new(PD_BYTES) }) },
            wgpu::BindGroupEntry { binding: 2, resource: b_src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_map.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b_wk.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: b_prgba.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: b_paux.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: b_flags.as_entire_binding() },
        ],
    });
    {
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&b_flags, 0, None);
        q.submit(Some(enc.finish()));
    }
    let tg = Instant::now();
    let mut slot = 0u64;
    let mut n_groups = 0usize;
    let mut ts = crate::gpu::Ts::new(gpu, 512);
    for &(oa, ob, ya, yb) in &bands {
        let hb = yb - ya;
        let nb = (w as u64) * (hb as u64);
        // ---- upload the band's window rows (the single upload when untiled)
        let tu = Instant::now();
        // (phase 5: rows that are already contiguous go straight to write_buffer, no packing copy; same bytes)
        let (ya_u, yb_u) = (ya as usize, yb as usize);
        if fin.src_stride == w * 4 {
            q.write_buffer(&b_src, 0, as_bytes(&fin.src[ya_u * w * 4..yb_u * w * 4]));
        } else {
            let mut packed = vec![0f32; nb as usize * 4];
            for y in 0..hb as usize {
                let sy = ya as usize + y;
                packed[y * w * 4..(y + 1) * w * 4].copy_from_slice(&fin.src[sy * fin.src_stride..sy * fin.src_stride + w * 4]);
            }
            q.write_buffer(&b_src, 0, as_bytes(&packed));
        }
        if let Some((m, ms)) = fin.map {
            if ms == w {
                q.write_buffer(&b_map, 0, as_bytes(&m[ya_u * w..yb_u * w]));
            } else {
                let mut packed = vec![0f32; nb as usize];
                for y in 0..hb as usize {
                    let sy = ya as usize + y;
                    packed[y * w..(y + 1) * w].copy_from_slice(&m[sy * ms..sy * ms + w]);
                }
                q.write_buffer(&b_map, 0, as_bytes(&packed));
            }
        }
        st.upload_ms += tu.elapsed().as_secs_f64() * 1e3;
        // White Protection grid of this band, aligned to the window's grid
        let wgy = if p.wp_on != 0 { (p.wp_gy + ya).rem_euclid(p.wp_f) } else { 0 };
        let wgh = if p.wp_on != 0 { (hb + wgy + p.wp_f - 1) / p.wp_f } else { 1 };
        let mut g = [0u32; 72];
        let iv = |g: &mut [u32; 72], o: usize, v: &[i32]| {
            for (k, x) in v.iter().enumerate() {
                g[o + k] = *x as u32;
            }
        };
        let fv = |g: &mut [u32; 72], o: usize, v: &[f32]| {
            for (k, x) in v.iter().enumerate() {
                g[o + k] = x.to_bits();
            }
        };
        iv(&mut g, 0, &[p.w, hb, (fin.map.is_some() && p.has_map != 0) as i32, p.invert_map]);
        iv(&mut g, 4, &[p.hl_on, p.knee_box_r, p.wp_on, p.det_on]);
        iv(&mut g, 8, &[p.wp_f, p.wp_gx, wgy, p.wp_gw]);
        iv(&mut g, 12, &[wgh, p.wp_rb, p.dmin_on, p.dots_on]);
        iv(&mut g, 16, &[cx, oa - ya, cw, ob - oa]);
        iv(&mut g, 20, &[(w * 4) as i32, w as i32, oa - cy, 0]);
        fv(&mut g, 24, &[p.focus, p.map_scale, p.map_gamma, p.max_radius]);
        fv(&mut g, 28, &[p.hl_gain, p.hl_threshold, p.boost, p.knee]);
        fv(&mut g, 32, &[p.curve, p.sel, p.hsat, p.ps]);
        fv(&mut g, 36, &[p.adapt_gain, p.wp_amt, p.wp_rel, p.dmin_ramp]);
        fv(&mut g, 40, &[p.dots, 0.0, 0.0, 0.0]);
        fv(&mut g, 44, &p.wp_tau);
        iv(&mut g, 56, &pl);
        q.write_buffer(&b_g, 0, as_bytes(&g));

        // ---- the pass list (CPU order); groups are separate submissions (abort checks between)
        struct Pass {
            ep: &'static str,
            rec: [u32; 16],
            groups: (u32, u32),
        }
        let mut passes: Vec<Vec<Pass>> = vec![Vec::new()];
        let (wu, hu) = (w as u32, hb as u32);
        let px = (wu.div_ceil(256), hu);
        let rec = |dims: (i32, i32, i32, i32), planes: (i32, i32, i32, i32), mx: i32, t: f32, wgt: f32| -> [u32; 16] {
            let mut r = [0u32; 16];
            for (k, v) in [dims.0, dims.1, dims.2, dims.3, planes.0, planes.1, planes.2, planes.3, mx, 0, 0, 0].iter().enumerate() {
                r[k] = *v as u32;
            }
            r[12] = t.to_bits();
            r[13] = wgt.to_bits();
            r
        };
        let full = (p.w, hb, 0, 0);
        let push = |ps: &mut Vec<Vec<Pass>>, ep: &'static str, r: [u32; 16], gr: (u32, u32)| ps.last_mut().unwrap().push(Pass { ep, rec: r, groups: gr });
        let mm = |ps: &mut Vec<Vec<Pass>>, from: i32, to: i32, r: i32, dir: i32, is_max: bool| {
            push(ps, "pp_mm_line", rec((p.w, hb, r, dir), (from, to, 0, 0), is_max as i32, 0.0, 0.0), px);
        };
        push(&mut passes, "pp_lum", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), px);
        if p.hl_on != 0 {
            for _ in 0..2 {
                push(&mut passes, "pp_box_rows", rec((p.w, hb, p.knee_box_r, 0), (pl_t, pl_x, 0, 0), 0, 0.0, 0.0), px);
                push(&mut passes, "pp_box_cols", rec((p.w, hb, p.knee_box_r, 0), (pl_x, pl_t, 0, 0), 0, 0.0, 0.0), px);
            }
            mm(&mut passes, pl_l, pl_x, 1, 0, true);
            mm(&mut passes, pl_x, pl_a, 1, 1, true);
            mm(&mut passes, pl_l, pl_x, 2, 0, false);
            mm(&mut passes, pl_x, pl_b, 2, 1, false);
            push(&mut passes, "pp_E", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), px);
            passes.push(Vec::new());
            if p.wp_on != 0 {
                let cells = (gw * wgh) as u32;
                push(&mut passes, "pp_wp_cells", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), (cells.div_ceil(64), 1));
                for k in 0..10 {
                    let lk = pl_lev + k * gw * wgh;
                    for _ in 0..2 {
                        let gpx = ((gw as u32).div_ceil(256), wgh as u32);
                        push(&mut passes, "pp_box_rows", rec((gw, wgh, p.wp_rb, 0), (lk, pl_gs, 0, 0), 0, 0.0, 0.0), gpx);
                        push(&mut passes, "pp_box_cols", rec((gw, wgh, p.wp_rb, 0), (pl_gs, lk, 0, 0), 0, 0.0, 0.0), gpx);
                    }
                }
                push(&mut passes, "pp_wp_apply", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), px);
                passes.push(Vec::new());
            }
            let size_window = |passes: &mut Vec<Vec<Pass>>, rf: &[f32], wv: &[f32], cnt: i32, outp: i32| {
                let mut cache: [(i32, i32); 2] = [(-1, pl_a), (-1, pl_b)];
                for s in 0..cnt as usize {
                    let r0 = rf[s].floor() as i32;
                    let t = rf[s] - r0 as f32;
                    let mut opening = |passes: &mut Vec<Vec<Pass>>, r: i32, keep: i32| -> i32 {
                        if let Some(c) = cache.iter().find(|c| c.0 == r) {
                            return c.1;
                        }
                        let sl = if cache[0].1 == keep { 1 } else { 0 };
                        let dst = cache[sl].1;
                        push(passes, "pp_mm_line", rec((p.w, hb, r, 0), (pl_l, pl_x, 0, 0), 0, 0.0, 0.0), px);
                        push(passes, "pp_mm_line", rec((p.w, hb, r, 1), (pl_x, dst, 0, 0), 0, 0.0, 0.0), px);
                        push(passes, "pp_mm_line", rec((p.w, hb, r, 0), (dst, pl_x, 0, 0), 1, 0.0, 0.0), px);
                        push(passes, "pp_mm_line", rec((p.w, hb, r, 1), (pl_x, dst, 0, 0), 1, 0.0, 0.0), px);
                        cache[sl] = (r, dst);
                        dst
                    };
                    let o0 = opening(passes, r0, -1);
                    let o1 = if t > 0.0 { opening(passes, r0 + 1, o0) } else { o0 };
                    push(passes, "pp_det_acc", rec(full, (o0, outp, o1, 0), 0, t, wv[s]), px);
                    passes.push(Vec::new());
                }
            };
            if p.det_on != 0 {
                size_window(&mut passes, &p.det_rf, &p.det_w, p.det_n, pl_d);
                if p.dmin_on != 0 {
                    size_window(&mut passes, &p.dmin_rf, &p.dmin_w, p.dmin_n, pl_dm);
                    push(&mut passes, "pp_dmin", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), px);
                }
                if p.dots_on != 0 {
                    for dir in 0..4 {
                        let r = p.dots_r[dir as usize];
                        mm(&mut passes, pl_l, pl_x, r, dir, false);
                        mm(&mut passes, pl_x, pl_b, r, dir, true);
                        push(&mut passes, "pp_best", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), px);
                        passes.push(Vec::new());
                    }
                    push(&mut passes, "pp_dots", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), px);
                }
            }
        }
        push(&mut passes, "pp_rows", rec(full, (0, 0, 0, 0), 0, 0.0, 0.0), ((cw as u32).div_ceil(256), (ob - oa) as u32));
        {
            // D (and DM) start at 0 in every band
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.clear_buffer(&b_wk, pl_d as u64 * 4, Some(nb * 4));
            if p.dmin_on != 0 {
                enc.clear_buffer(&b_wk, pl_dm as u64 * 4, Some(nb * 4));
            }
            q.submit(Some(enc.finish()));
        }
        let mut cleared_c = false;
        for group in passes.iter().filter(|g| !g.is_empty()) {
            if abort() {
                return Err(fail(OBGPU_ABORTED, "aborted"));
            }
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            // C (best of the 4 line openings) lives in T, free once E exists: zero it before the dots
            if !cleared_c && group.iter().any(|ps| ps.ep == "pp_best") {
                enc.clear_buffer(&b_wk, pl_t as u64 * 4, Some(nb * 4));
                cleared_c = true;
            }
            if slot + group.len() as u64 > 512 {
                slot = 0;
            }
            let first = slot;
            for ps in group.iter() {
                q.write_buffer(&b_d, slot * PD_STRIDE, as_bytes(&ps.rec));
                slot += 1;
            }
            {
                let tw = ts.as_mut().and_then(|t| t.writes_l(group.first().map(|p| p.ep).unwrap_or("prep")));
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
                for (k, ps) in group.iter().enumerate() {
                    cp.set_pipeline(&pipes.p[ps.ep]);
                    cp.set_bind_group(0, &bg, &[((first + k as u64) * PD_STRIDE) as u32]);
                    cp.dispatch_workgroups(ps.groups.0, ps.groups.1, 1);
                }
            }
            // phase 5: no wait per group (the queue runs them in order; record-slot and band-buffer writes are
            // queue-ordered too): the GPU is not left idle between groups. Abort is still checked before every
            // group and while waiting below.
            let _ = gpu.submit(enc.finish());
            st.submissions += 1;
            n_groups += 1;
        }
    }

    // flags -> maxima, rollRelevant (this wait covers every prepare submission: timeout scaled by their number)
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&b_flags, 0, &b_fl_rd, 0, 64);
    let idx = gpu.submit(enc.finish());
    let wtun = job::Tuning { timeout: tun.timeout * (n_groups.max(1) as u32), ..*tun };
    wait(gpu, idx, abort, &wtun)?;
    let flags = read_u32(gpu, &b_fl_rd, 16, tun)?;
    let prep_ms = tg.elapsed().as_secs_f64() * 1e3;
    let any_e = flags[0] != 0 && (p.wp_on == 0 || flags[1] != 0) && p.hl_on != 0;
    let roll_relevant = p.hl_gain > 0.0 || any_e;
    let maxima = [f32::from_bits(flags[2]), f32::from_bits(flags[3]), f32::from_bits(flags[4]), f32::from_bits(flags[5]), f32::from_bits(flags[6])];
    job::busy_add(match ts.as_ref() { Some(t) => t.busy_ms(gpu), None => None });
    // prepare planes are not needed any more (peak = max(prepare, blur))
    gpu.free(&[&b_src, &b_map, &b_wk]);
    drop((b_src, b_map, b_wk));
    // ABI 1.3 post on the GPU: the blurred crop (+ eL) stays on the device (crop x 20 bytes during the blur),
    // the post fields are allocated after the blur's buffers are gone; not enough budget -> the CPU does post
    let mut want = post.map(|(pd, _)| pd.stages).unwrap_or(0);
    if want != 0 {
        let pd = post.unwrap().0;
        // phase 5: the prepared crop (24 bytes / px, read only by the glow) is freed before the other stages
        // allocate their fields: peak = crop x 20 + max(prepared crop + glow grid, fields of the rest)
        let rest = want & !OBGPU_POST_GLOWVIG;
        let need = if crate::post::noreuse() {
            crop_px * 44 + crate::post::bytes(pd, want)
        } else {
            crop_px * 20 + (crop_px * 24 + crate::post::bytes(pd, want & OBGPU_POST_GLOWVIG)).max(crate::post::bytes(pd, rest))
        };
        if need > tun.budget {
            want = 0;
            if gpu.verbose {
                eprintln!("[obgpu] post: {} MB needed > budget {} MB: CPU post", need >> 20, tun.budget >> 20);
            }
        }
    }
    // the prepared crop is only needed by a CPU glow: not read back when the GPU runs it
    let (prepared_out, prepared_el) = if want & OBGPU_POST_GLOWVIG != 0 { (None, None) } else { (prepared_out, prepared_el) };
    if let Some(pp) = prepared_out {
        let stage = mk("prep.rd", crop_px * 16, U::MAP_READ | U::COPY_DST);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b_prgba, 0, &stage, 0, crop_px * 16);
        let idx = gpu.submit(enc.finish());
        wait(gpu, idx, abort, tun)?;
        map_copy(gpu, &stage, crop_px * 16, pp as *mut u8, tun)?;
    }
    if let Some(pe) = prepared_el {
        // (signed radius, eL) pairs -> eL
        let stage = mk("aux.rd", crop_px * 8, U::MAP_READ | U::COPY_DST);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b_paux, 0, &stage, 0, crop_px * 8);
        let idx = gpu.submit(enc.finish());
        wait(gpu, idx, abort, tun)?;
        let mut aux = vec![0f32; crop_px as usize * 2];
        map_copy(gpu, &stage, crop_px * 8, aux.as_mut_ptr() as *mut u8, tun)?;
        for i in 0..crop_px as usize {
            unsafe { *pe.add(i) = aux[2 * i + 1] };
        }
    }
    // ---- blur stage on the device-resident prepared crop
    let dp = job::DevPrep { prgba: &b_prgba, paux: &b_paux, maxima };
    let resident = if want != 0 { crop_px * 20 } else { 0 };
    let btun = job::Tuning { budget: tun.budget.saturating_sub(crop_px * 24 + resident).max(64 << 20), ..*tun };
    let mut bst = obgpu_stats::default();
    let mut post_done = 0u32;
    if want != 0 {
        let (pd, pp) = post.unwrap();
        let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let b_img = mk("post.img", crop_px * 16, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let b_el = mk("post.el", crop_px * 4, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        if let Some(Some(e)) = block_on(oom.pop()) {
            return Err(fail(OBGPU_OUT_OF_VRAM, format!("post buffers: {e}")));
        }
        let sink = job::Sink { rgba: &b_img, el: Some(&b_el) };
        job::run_any(gpu, None, Some(&dp), bd, out, Some(&sink), abort, &btun, &mut bst)?;
        let roll = pd.roll_candidate != 0 && roll_relevant && bd.need_el != 0;
        let tp = Instant::now();
        let mut pst = obgpu_stats::default();
        // glow (reads the prepared crop) first, then free the prepared crop, then the rest of the chain: same
        // kernels in the same order as one call, less peak memory
        if crate::post::noreuse() {
            // (equality-test switch: the phase-4 single call, prepared crop alive throughout)
            post_done = crate::post::run(gpu, pp, pd, want, roll, &crate::post::Bufs { img: &b_img, el: &b_el, prgba: &b_prgba, paux: &b_paux }, abort, tun, &mut pst)?;
        } else if want & OBGPU_POST_GLOWVIG != 0 {
            post_done |= crate::post::run(gpu, pp, pd, OBGPU_POST_GLOWVIG, roll, &crate::post::Bufs { img: &b_img, el: &b_el, prgba: &b_prgba, paux: &b_paux }, abort, tun, &mut pst)?;
        }
        st.vram_bytes = st.vram_bytes.max(pst.vram_bytes + crop_px * 44);
        drop(dp);
        gpu.free(&[&b_prgba, &b_paux]);
        drop(b_prgba);
        drop(b_paux);
        let rest = if crate::post::noreuse() { 0 } else { want & !OBGPU_POST_GLOWVIG };
        if rest != 0 {
            let dummy4 = mk("post.dummy4", 16, U::STORAGE);
            let dummy2 = mk("post.dummy2", 16, U::STORAGE);
            let mut pst2 = obgpu_stats::default();
            post_done |= crate::post::run(gpu, pp, pd, rest, roll, &crate::post::Bufs { img: &b_img, el: &b_el, prgba: &dummy4, paux: &dummy2 }, abort, tun, &mut pst2)?;
            st.vram_bytes = st.vram_bytes.max(pst2.vram_bytes + crop_px * 20);
            pst.submissions += pst2.submissions;
        }
        st.submissions += pst.submissions;
        let post_ms = tp.elapsed().as_secs_f64() * 1e3;
        // the one readback: colour (+ eL)
        let tr = Instant::now();
        let el_b = if bd.need_el != 0 { crop_px * 4 } else { 0 };
        let stage = gpu.staging(crop_px * 16 + el_b, false);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b_img, 0, &stage, 0, crop_px * 16);
        if el_b > 0 {
            enc.copy_buffer_to_buffer(&b_el, 0, &stage, crop_px * 16, el_b);
        }
        let idx = gpu.submit(enc.finish());
        wait(gpu, idx, abort, tun)?;
        // (phase 5: rows are copied straight out of the mapped staging buffer, no intermediate vector)
        let (cwu, chu) = (cw as usize, ch as usize);
        let (orgba, ostride, oel) = (out.rgba as usize, out.stride, out.el as usize);
        map_with(gpu, &stage, crop_px * 16 + el_b, tun, abort, |src: &[f32]| {
            // rows copied on up to 8 threads (same bytes)
            let sp = src.as_ptr() as usize;
            let nth = if crop_px >= (4 << 20) { 8 } else { 1 };
            let per = chu.div_ceil(nth);
            std::thread::scope(|s| {
                for k in 0..nth {
                    let (y0, y1) = (k * per, ((k + 1) * per).min(chu));
                    s.spawn(move || {
                        for y in y0..y1 {
                            unsafe { std::ptr::copy_nonoverlapping((sp as *const f32).add(y * cwu * 4), (orgba as *mut f32).add(y * ostride), 4 * cwu) };
                        }
                        if el_b > 0 && y1 > y0 {
                            unsafe { std::ptr::copy_nonoverlapping((sp as *const f32).add(cwu * chu * 4 + y0 * cwu), (oel as *mut f32).add(y0 * cwu), (y1 - y0) * cwu) };
                        }
                    });
                }
            });
        })?;
        gpu.give_staging(stage, false);
        bst.readback_ms += tr.elapsed().as_secs_f64() * 1e3;
        bst.gpu_ms += post_ms;
    } else {
        job::run_any(gpu, None, Some(&dp), bd, out, None, abort, &btun, &mut bst)?;
    }
    job::POST.with(|c| c.set(post_done));
    st.gpu_ms += prep_ms + bst.gpu_ms;
    st.convert_ms += bst.convert_ms;
    st.readback_ms += bst.readback_ms;
    st.bands = bst.bands;
    st.submissions += bst.submissions;
    st.vram_bytes = st.vram_bytes.max(bst.vram_bytes + crop_px * 24);
    st.wait_ms += prep_ms; // (prepare stage time, reported separately by the harness)
    st.total_ms += t0.elapsed().as_secs_f64() * 1e3;
    Ok((roll_relevant, OBGPU_STAGE_PREPARE | OBGPU_STAGE_BLUR | if post_done != 0 { OBGPU_STAGE_POST } else { 0 }))
}

fn wait(gpu: &Gpu, idx: crate::gpu::Sub, abort: &dyn Fn() -> bool, tun: &job::Tuning) -> Result<(), Fail> {
    gpu.wait_sub(&idx, abort, tun.timeout).map_err(|s| fail(s, "prepare stage wait"))
}

fn map_copy(gpu: &Gpu, b: &wgpu::Buffer, bytes: u64, dst: *mut u8, tun: &job::Tuning) -> Result<(), Fail> {
    map_copy_ab(gpu, b, bytes, dst, tun, &|| false)
}

fn map_copy_ab(gpu: &Gpu, b: &wgpu::Buffer, bytes: u64, dst: *mut u8, tun: &job::Tuning, abort: &dyn Fn() -> bool) -> Result<(), Fail> {
    let slice = b.slice(0..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r.is_ok());
    });
    let t0 = Instant::now();
    loop {
        if abort() {
            return Err(fail(OBGPU_ABORTED, "aborted (readback abandoned)"));
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        if let Ok(ok) = rx.try_recv() {
            if !ok {
                return Err(fail(OBGPU_DEVICE_LOST, "map failed"));
            }
            break;
        }
        if t0.elapsed() > tun.timeout {
            return Err(fail(OBGPU_TIMEOUT, "map timeout"));
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    {
        let v = slice.get_mapped_range().map_err(|e| fail(OBGPU_INTERNAL, format!("{e:?}")))?;
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), dst, bytes as usize) };
    }
    b.unmap();
    Ok(())
}

/// maps `bytes` of a staging buffer and hands the floats to `f` (abort checked while waiting)
fn map_with(gpu: &Gpu, b: &wgpu::Buffer, bytes: u64, tun: &job::Tuning, abort: &dyn Fn() -> bool, f: impl FnOnce(&[f32])) -> Result<(), Fail> {
    let slice = b.slice(0..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r.is_ok());
    });
    let t0 = Instant::now();
    loop {
        if abort() {
            return Err(fail(OBGPU_ABORTED, "aborted (readback abandoned)"));
        }
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        if let Ok(ok) = rx.try_recv() {
            if !ok {
                return Err(fail(OBGPU_DEVICE_LOST, "map failed"));
            }
            break;
        }
        if t0.elapsed() > tun.timeout {
            return Err(fail(OBGPU_TIMEOUT, "map timeout"));
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    {
        let v = slice.get_mapped_range().map_err(|e| fail(OBGPU_INTERNAL, format!("{e:?}")))?;
        let fl = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) };
        f(fl);
    }
    b.unmap();
    Ok(())
}

fn read_u32(gpu: &Gpu, b: &wgpu::Buffer, n: usize, tun: &job::Tuning) -> Result<Vec<u32>, Fail> {
    let mut v = vec![0u32; n];
    map_copy(gpu, b, (n * 4) as u64, v.as_mut_ptr() as *mut u8, tun)?;
    Ok(v)
}

pub type Shared = Arc<PrepPipes>;
