//! Adapter self-test: the embedded reference cases (src/selftest.bin, written by `obgpu_harness mkselftest`
//! from the CPU engine) rendered on this adapter and compared with the stored CPU result.

use crate::ffi::*;
use crate::gpu::Gpu;
use crate::job;

static DATA: &[u8] = include_bytes!("selftest.bin");

pub const TOL_REL: f64 = 5e-4;
pub const TOL_MEAN: f64 = 2e-5;

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.o..self.o + n)?;
        self.o += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f64s(&mut self, n: usize) -> Option<Vec<f64>> {
        Some(self.take(n * 8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn f32s(&mut self, n: usize) -> Option<Vec<f32>> {
        Some(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

pub struct Case {
    pub desc: obgpu_blur_desc,
    _pool: std::sync::Arc<Vec<Vec<f64>>>, // keeps the desc's table pointers alive
    pub rgba: Vec<f32>,
    pub radius: Vec<f64>,
    pub side: Vec<i8>,
    pub el: Vec<f32>,
    pub cpu: Vec<f32>,
    pub cpu_el: Vec<f32>,
}

pub struct FrameCase {
    pub prep: obgpu_prepare_desc,
    pub blur: Case, // desc + tables + CPU result (prepared inputs unused)
    pub src: Vec<f32>,
    pub map: Vec<f32>,
    pub roll: bool,
}

/// FNV-1a of the generator's sources stored in selftest.bin (build.rs compares it with the current files)
pub fn source_hash() -> Option<u64> {
    let mut r = Rd { b: DATA, o: 0 };
    if r.take(4)? != b"OBST" || r.u32()? != 2 {
        return None;
    }
    Some(u64::from_le_bytes(r.take(8)?.try_into().ok()?))
}

fn tables(pool: &std::sync::Arc<Vec<Vec<f64>>>, d: &mut obgpu_blur_desc, ia: u32, ic: u32, ik: u32) -> Option<()> {
    let t = |i: u32| -> Option<*const f64> { if i == u32::MAX { Some(std::ptr::null()) } else { pool.get(i as usize).map(|v| v.as_ptr()) } };
    d.ap_table = t(ia)?;
    d.circ_table = t(ic)?;
    d.cat_a = t(ik)?;
    if d.ap_table.is_null() || d.circ_table.is_null() {
        return None;
    }
    // sizes must match the desc (a corrupt file must not make the job read out of bounds)
    let len = |i: u32| if i == u32::MAX { 0 } else { pool[i as usize].len() };
    if len(ia) != 2 * (d.ap_n as usize + 1) || len(ic) != 2 * (d.circ_n as usize + 1) || (d.cat_n > 0 && len(ik) != (d.cat_n * d.cat_n) as usize) {
        return None;
    }
    Some(())
}

fn read_desc(r: &mut Rd) -> Option<obgpu_blur_desc> {
    let db = r.u32()? as usize;
    if db != std::mem::size_of::<obgpu_blur_desc>() {
        return None;
    }
    Some(unsafe { std::ptr::read_unaligned(r.take(db)?.as_ptr() as *const obgpu_blur_desc) })
}

/// (blur cases, frame cases) of selftest.bin v2
pub type PostCase = (FrameCase, obgpu_post_desc);
pub fn all_cases() -> Option<(Vec<Case>, Vec<FrameCase>, Vec<PostCase>)> {
    let mut r = Rd { b: DATA, o: 0 };
    if r.take(4)? != b"OBST" || r.u32()? != 2 {
        return None;
    }
    r.take(8)?;
    let nt = r.u32()? as usize;
    let mut pv = Vec::with_capacity(nt);
    for _ in 0..nt {
        let n = r.u32()? as usize;
        pv.push(r.f64s(n)?);
    }
    let pool = std::sync::Arc::new(pv);
    let mut blur = Vec::new();
    for _ in 0..r.u32()? as usize {
        let (w, h) = (r.u32()? as usize, r.u32()? as usize);
        let mut desc = read_desc(&mut r)?;
        let (ia, ic, ik) = (r.u32()?, r.u32()?, r.u32()?);
        tables(&pool, &mut desc, ia, ic, ik)?;
        let npx = w * h;
        let rgba = r.f32s(npx * 4)?;
        let radius: Vec<f64> = r.f32s(npx)?.into_iter().map(|v| v as f64).collect();
        let side: Vec<i8> = r.take(npx)?.iter().map(|b| *b as i8).collect();
        let ne = r.u32()? as usize;
        let el = r.f32s(ne)?;
        let cpu = r.f32s(npx * 4)?;
        let cpu_el = r.f32s(ne)?;
        blur.push(Case { desc, _pool: pool.clone(), rgba, radius, side, el, cpu, cpu_el });
    }
    let mut frames = Vec::new();
    for _ in 0..r.u32()? as usize {
        let (w, h) = (r.u32()? as usize, r.u32()? as usize);
        let pb = r.u32()? as usize;
        if pb != std::mem::size_of::<obgpu_prepare_desc>() {
            return None;
        }
        let prep: obgpu_prepare_desc = unsafe { std::ptr::read_unaligned(r.take(pb)?.as_ptr() as *const obgpu_prepare_desc) };
        let mut desc = read_desc(&mut r)?;
        let (ia, ic, ik) = (r.u32()?, r.u32()?, r.u32()?);
        tables(&pool, &mut desc, ia, ic, ik)?;
        let src = r.f32s(w * h * 4)?;
        let map = r.f32s(w * h)?;
        let roll = r.u32()? != 0;
        let ne = r.u32()? as usize;
        let cpu = r.f32s(w * h * 4)?;
        let cpu_el = r.f32s(ne)?;
        let blur = Case { desc, _pool: pool.clone(), rgba: Vec::new(), radius: Vec::new(), side: Vec::new(), el: Vec::new(), cpu, cpu_el };
        frames.push(FrameCase { prep, blur, src, map, roll });
    }
    // optional trailing section (ABI 1.3): post cases
    let mut posts = Vec::new();
    if r.o < r.b.len() {
        for _ in 0..r.u32()? as usize {
            let (w, h) = (r.u32()? as usize, r.u32()? as usize);
            let pb = r.u32()? as usize;
            if pb != std::mem::size_of::<obgpu_prepare_desc>() {
                return None;
            }
            let prep: obgpu_prepare_desc = unsafe { std::ptr::read_unaligned(r.take(pb)?.as_ptr() as *const obgpu_prepare_desc) };
            let mut desc = read_desc(&mut r)?;
            let (ia, ic, ik) = (r.u32()?, r.u32()?, r.u32()?);
            tables(&pool, &mut desc, ia, ic, ik)?;
            let qb = r.u32()? as usize;
            if qb != std::mem::size_of::<obgpu_post_desc>() {
                return None;
            }
            let post: obgpu_post_desc = unsafe { std::ptr::read_unaligned(r.take(qb)?.as_ptr() as *const obgpu_post_desc) };
            let src = r.f32s(w * h * 4)?;
            let map = r.f32s(w * h)?;
            let roll = r.u32()? != 0;
            let cpu = r.f32s(w * h * 4)?;
            let blur = Case { desc, _pool: pool.clone(), rgba: Vec::new(), radius: Vec::new(), side: Vec::new(), el: Vec::new(), cpu, cpu_el: Vec::new() };
            posts.push((FrameCase { prep, blur, src, map, roll }, post));
        }
    }
    Some((blur, frames, posts))
}

/// Runs every case twice (blur cases through obgpu_blur''s path, frame cases through the frame pipeline).
pub fn run(gpu: &Gpu, tun: &job::Tuning, inject_ppm: u64) -> Result<obgpu_selftest_report, crate::job::Fail> {
    let t0 = std::time::Instant::now();
    let (cs, frames, posts) = all_cases().ok_or(crate::job::Fail { status: OBGPU_INTERNAL, msg: "self-test data missing / corrupt".into() })?;
    let mut rep = obgpu_selftest_report {
        struct_size: std::mem::size_of::<obgpu_selftest_report>() as u32,
        pass: 1,
        cases: (cs.len() + frames.len() + posts.len()) as i32,
        deterministic: 1,
        failed_case: -1,
        ..Default::default()
    };
    if cs.is_empty() {
        rep.pass = 0;
    }
    for (ci, c) in cs.iter().enumerate() {
        let d = &c.desc;
        job::check_desc(d)?;
        let npx = (d.w * d.h) as usize;
        let input = job::Input { rgba: &c.rgba, radius: &c.radius, side: &c.side, el: if d.need_el != 0 { Some(&c.el) } else { None } };
        let mut outs: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for _ in 0..2 {
            let mut o = vec![0f32; npx * 4];
            let mut e = vec![0f32; if d.need_el != 0 { npx } else { 0 }];
            let out = job::Output { rgba: o.as_mut_ptr(), stride: (d.w * 4) as usize, el: if e.is_empty() { std::ptr::null_mut() } else { e.as_mut_ptr() } };
            let mut st = obgpu_stats::default();
            job::run(gpu, &input, d, &out, &|| false, tun, &mut st)?;
            outs.push((o, e));
        }
        if outs[0] != outs[1] {
            rep.deterministic = 0;
        }
        let (o, e) = outs.swap_remove(0);
        compare(c, o, &e, ci, inject_ppm, &mut rep);
    }
    if !frames.is_empty() {
        let pipes = gpu.prep_pipes().map_err(|e| crate::job::Fail { status: OBGPU_INTERNAL, msg: e })?;
        for (k, fc) in frames.iter().enumerate() {
            let d = &fc.blur.desc;
            job::check_desc(d)?;
            crate::frame::check_prep(&fc.prep)?;
            let (w, h) = (fc.prep.w as usize, fc.prep.h as usize);
            let mut runs: Vec<(Vec<f32>, Vec<f32>, bool)> = Vec::new();
            for _ in 0..2 {
                let mut o = vec![0f32; w * h * 4];
                let mut e = vec![0f32; if d.need_el != 0 { w * h } else { 0 }];
                let out = job::Output { rgba: o.as_mut_ptr(), stride: w * 4, el: if e.is_empty() { std::ptr::null_mut() } else { e.as_mut_ptr() } };
                let fin = crate::frame::FrameIn { src: &fc.src, src_stride: w * 4, map: Some((&fc.map, w)), crop: (0, 0, w as i32, h as i32) };
                let mut st = obgpu_stats::default();
                job::BUSY.with(|b| b.set(0.0));
                let t0 = std::time::Instant::now();
                let (roll, _) = crate::frame::run(gpu, &pipes, &fin, &fc.prep, d, &out, None, None, None, &|| false, tun, &mut st)?;
                // cold-start seed for the router: GPU busy (or wall) ms per work unit of this frame (second run: warm)
                if runs.len() == 1 {
                    let busy = job::BUSY.with(|b| b.get());
                    let wall = t0.elapsed().as_secs_f64() * 1e3;
                    let ms = if busy.is_finite() && busy > 0.0 { busy } else { wall };
                    if std::env::var("OBGPU_TRACE").is_ok() { eprintln!("[obgpu.trace] selftest frame case busy {busy:.3} ms wall {wall:.3} ms"); }
                    rep.seed_wall_ms += wall;
                    rep.seed_busy_ms += if busy.is_finite() { busy } else { wall };
                    let wu = unsafe { crate::obgpu_work_units(d) };
                    if wu > 0.0 {
                        rep.seed_ms_per_wu = rep.seed_ms_per_wu.max(ms / wu);
                    }
                }
                runs.push((o, e, roll));
            }
            if runs[0] != runs[1] {
                rep.deterministic = 0;
            }
            let (o, e, roll) = runs.swap_remove(0);
            if roll != fc.roll {
                rep.pass = 0;
                if rep.failed_case < 0 {
                    rep.failed_case = (cs.len() + k) as i32;
                }
            }
            // the CPU stored eL only where its prepare made it relevant
            let e = if fc.blur.cpu_el.is_empty() { Vec::new() } else { e };
            compare(&fc.blur, o, &e, cs.len() + k, inject_ppm, &mut rep);
        }
    }
    if !posts.is_empty() {
        let pipes = gpu.prep_pipes().map_err(|e| crate::job::Fail { status: OBGPU_INTERNAL, msg: e })?;
        let ppipes = gpu.post_pipes().map_err(|e| crate::job::Fail { status: OBGPU_INTERNAL, msg: e })?;
        for (k, (fc, pd)) in posts.iter().enumerate() {
            let ci = cs.len() + frames.len() + k;
            let d = &fc.blur.desc;
            job::check_desc(d)?;
            crate::frame::check_prep(&fc.prep)?;
            let (w, h) = (fc.prep.w as usize, fc.prep.h as usize);
            crate::post::check(pd, w as i32, h as i32)?;
            let mut runs: Vec<Vec<f32>> = Vec::new();
            for _ in 0..2 {
                let mut o = vec![0f32; w * h * 4];
                let mut e = vec![0f32; if d.need_el != 0 { w * h } else { 0 }];
                let out = job::Output { rgba: o.as_mut_ptr(), stride: w * 4, el: if e.is_empty() { std::ptr::null_mut() } else { e.as_mut_ptr() } };
                let fin = crate::frame::FrameIn { src: &fc.src, src_stride: w * 4, map: Some((&fc.map, w)), crop: (0, 0, w as i32, h as i32) };
                let mut st = obgpu_stats::default();
                job::POST.with(|c| c.set(0));
                let (roll, _) = crate::frame::run(gpu, &pipes, &fin, &fc.prep, d, &out, None, None, Some((pd, &*ppipes)), &|| false, tun, &mut st)?;
                if roll != fc.roll || job::POST.with(|c| c.get()) & !OBGPU_POST_LA34 != pd.stages {
                    rep.pass = 0;
                    if rep.failed_case < 0 {
                        rep.failed_case = ci as i32;
                    }
                }
                runs.push(o);
            }
            if runs[0] != runs[1] {
                rep.deterministic = 0;
            }
            let o = runs.swap_remove(0);
            compare(&fc.blur, o, &[], ci, inject_ppm, &mut rep);
        }
    }
    if rep.deterministic == 0 {
        rep.pass = 0;
    }
    rep.ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(rep)
}

/// HDR-aware comparison with the stored CPU result: |d| / L, L = max(1, max |cpu| over the 9x9 neighbourhood).
fn compare(c: &Case, mut o: Vec<f32>, e: &[f32], ci: usize, inject_ppm: u64, rep: &mut obgpu_selftest_report) {
    if inject_ppm > 0 {
        // test hook 6: emulate an adapter whose arithmetic is off by inject_ppm (relative) on every value
        let f = 1.0 + inject_ppm as f32 * 1e-6;
        o.iter_mut().for_each(|v| *v *= f);
    }
    let d = &c.desc;
    let (w, h) = (d.w as usize, d.h as usize);
    let mut loc = vec![0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut m = 1f64;
            for yy in y.saturating_sub(4)..(y + 5).min(h) {
                for xx in x.saturating_sub(4)..(x + 5).min(w) {
                    let i = yy * w + xx;
                    for ch in 0..4 {
                        m = m.max((c.cpu[i * 4 + ch] as f64).abs());
                    }
                    if !c.cpu_el.is_empty() {
                        m = m.max((c.cpu_el[i] as f64).abs());
                    }
                }
            }
            loc[y * w + x] = m;
        }
    }
    let mut ok = true;
    let mut sum = 0f64;
    let mut worst = (0f64, 0usize);
    for i in 0..w * h {
        for ch in 0..5 {
            let (a, b) = if ch < 4 { (o[i * 4 + ch], c.cpu[i * 4 + ch]) } else if !e.is_empty() { (e[i], c.cpu_el[i]) } else { continue };
            if !a.is_finite() {
                ok = false;
                continue;
            }
            let rel = ((a - b) as f64).abs() / loc[i];
            sum += rel;
            if rel > worst.0 {
                worst = (rel, i);
            }
        }
    }
    rep.max_rel = rep.max_rel.max(worst.0);
    if worst.0 > TOL_REL {
        ok = false;
    }
    let mean = sum / (w * h * if e.is_empty() { 4 } else { 5 }) as f64;
    rep.mean_abs = rep.mean_abs.max(mean);
    if std::env::var("OBGPU_SELFTEST_VERBOSE").is_ok() {
        eprintln!("[obgpu] self-test case {ci}: mean rel {mean:.3e} max rel {:.3e} at px ({}, {})", worst.0, worst.1 % w, worst.1 / w);
    }
    if !ok || mean > TOL_MEAN {
        rep.pass = 0;
        if rep.failed_case < 0 {
            rep.failed_case = ci as i32;
        }
    }
}