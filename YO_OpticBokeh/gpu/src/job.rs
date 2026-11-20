//! One blur job: validation, band planning (VRAM budget), upload, setup, accumulation, readback.

use std::os::raw::c_int;
use std::time::{Duration, Instant};

use crate::ffi::*;
use crate::gpu::{block_on, Gpu, Variant};

pub const DISP_BYTES: u64 = 128;

thread_local! {
    /// GPU busy milliseconds (timestamp queries) accumulated by the jobs of this thread; the ABI entry points
    /// reset it before a job and read it after (diagnostics, adaptive routing). NaN = no timestamps.
    pub static BUSY: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    /// last frame job's prepare bands and halo factor (window rows prepared / rows of one untiled pass)
    pub static PREP: std::cell::Cell<(i32, f64)> = const { std::cell::Cell::new((0, 0.0)) };
    /// last frame job's post stages done on the GPU (OBGPU_POST_* bits)
    pub static POST: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}
pub fn busy_add(ms: Option<f64>) {
    BUSY.with(|b| b.set(match ms { Some(v) if !b.get().is_nan() => b.get() + v, _ => f64::NAN }));
}
const DISP_STRIDE: u64 = 256;
const DISP_SLOTS: u64 = 4096;
const SPLAT_BYTES: u64 = 64;

pub struct Fail {
    pub status: c_int,
    pub msg: String,
}
fn fail(status: c_int, msg: impl Into<String>) -> Fail {
    Fail { status, msg: msg.into() }
}

#[derive(Clone, Copy)]
pub struct Tuning {
    pub budget: u64,
    pub timeout: Duration,
    pub band_rows: u32,
    pub delay_ms: u32,
    pub target_ms: f64,
}

pub struct Input<'a> {
    pub rgba: &'a [f32],
    pub radius: &'a [f64],
    pub side: &'a [i8],
    pub el: Option<&'a [f32]>,
}

pub struct Output {
    pub rgba: *mut f32,
    pub stride: usize,
    pub el: *mut f32,
}

/// Validates the desc (layout version, ranges, tables).
pub fn check_desc(d: &obgpu_blur_desc) -> Result<(), Fail> {
    if d.struct_size as usize != std::mem::size_of::<obgpu_blur_desc>() || d.version != DESC_VERSION {
        return Err(fail(OBGPU_BAD_ARGS, "desc size / version mismatch"));
    }
    if d.w <= 0 || d.h <= 0 || (d.w as i64) * (d.h as i64) > 400_000_000 {
        return Err(fail(OBGPU_BAD_ARGS, "bad size"));
    }
    if !(d.nk == 1 || d.nk == 3) || d.n_levels < 1 || d.n_levels as usize > MAX_LEVELS || !(d.n_tiers == 1 || d.n_tiers == TIERS as i32) {
        return Err(fail(OBGPU_BAD_ARGS, "bad nk / levels / tiers"));
    }
    if d.ap_n < 2 || d.circ_n < 2 || d.ap_table.is_null() || d.circ_table.is_null() || (d.cat_n > 0 && d.cat_a.is_null()) {
        return Err(fail(OBGPU_BAD_ARGS, "missing tables"));
    }
    if d.cat_n < 0 || d.cat_n > 1024 || d.ap_n > 1 << 20 || d.circ_n > 1 << 20 {
        return Err(fail(OBGPU_BAD_ARGS, "bad table sizes"));
    }
    if d.row_begin < 0 || d.row_end > d.h || d.row_begin > d.row_end {
        return Err(fail(OBGPU_BAD_ARGS, "bad row range"));
    }
    for s in 0..2 {
        for t in 0..TIERS {
            let p = &d.prof[s][t];
            if p.nl < 0 || p.nl as usize > MAX_LAYERS || p.share_from < 0 || p.share_from > p.nl.max(1) {
                return Err(fail(OBGPU_BAD_ARGS, "bad profile"));
            }
        }
    }
    for l in 1..d.n_levels as usize {
        let v = &d.levels[l];
        if v.f != 1 << l || v.w <= 0 || v.h <= 0 || v.gox < 0 || v.goy < 0 || v.max_reach < 0 {
            return Err(fail(OBGPU_BAD_ARGS, "bad level"));
        }
    }
    let vals = [d.ax, d.ay, d.xext, d.yext, d.level_base, d.max_radius, d.smax_g, d.half_diag];
    if vals.iter().any(|v| !v.is_finite() || *v <= 0.0 && *v != d.max_radius) || d.max_radius < 0.0 {
        return Err(fail(OBGPU_BAD_ARGS, "bad geometry"));
    }
    Ok(())
}

/// per level: x reach of the segment candidates (XR) and the loop row reach (MRe)
fn level_reach(d: &obgpu_blur_desc) -> ([i32; 8], [i32; 8]) {
    let n = d.n_levels as usize;
    let mut xr = [0i32; 8];
    let mut mr = [0i32; 8];
    for l in 0..n {
        let f = (1u32 << l) as f64;
        let rmax = if n == 1 {
            d.max_radius
        } else if l < n - 1 {
            2.0 * d.level_base
        } else {
            d.max_radius / f
        } * 1.001
            + 1e-3;
        let maxr = if l == 0 { d.max_reach } else { d.levels[l].max_reach };
        xr[l] = (rmax * d.ax * d.smax_g * d.xext).ceil() as i32 + 2;
        mr[l] = maxr.min((rmax * d.ay * d.smax_g * d.yext - 0.5).ceil().max(0.0) as i32 + 1);
    }
    (xr, mr)
}

#[derive(Clone, Copy, Default, Debug)]
struct LevelWin {
    sa: i32, // splat rows [sa, sb)
    sb: i32,
    ga: i32, // grid rows [ga, gb) the fine rows interpolate from
    gb: i32,
}

#[derive(Clone, Copy, Default, Debug)]
struct Win {
    oa: i32,
    ob: i32,
    ia: i32, // prep rows [ia, ib)
    ib: i32,
    lv: [LevelWin; 8], // lv[0] = fine splat rows
}

fn coarse_row(v: &obgpu_level, y: i32, upper: bool) -> i32 {
    let mut wv = (y as f64 + 0.5 + v.goy as f64) / v.f as f64 - 0.5;
    wv = wv.max(0.0).min(v.h as f64 - 1.0);
    let y0 = wv as i32;
    if upper {
        (y0 + 1).min(v.h - 1)
    } else {
        y0
    }
}

fn window(d: &obgpu_blur_desc, mr: &[i32; 8], oa: i32, ob: i32) -> Win {
    let mut w = Win { oa, ob, ..Default::default() };
    let fa = (oa - mr[0]).max(0);
    let fb = (ob + mr[0]).min(d.h);
    w.lv[0] = LevelWin { sa: fa, sb: fb, ga: 0, gb: 0 };
    w.ia = fa;
    w.ib = fb;
    for l in 1..d.n_levels as usize {
        let v = &d.levels[l];
        let ga = coarse_row(v, oa, false);
        let gb = coarse_row(v, ob - 1, true) + 1;
        let sa = (ga - mr[l]).max(0);
        let sb = (gb + mr[l]).min(v.h);
        w.lv[l] = LevelWin { sa, sb, ga, gb };
        let ya = (sa * v.f - v.goy).max(0);
        let yb = (sb * v.f - v.goy).min(d.h);
        if yb > ya {
            w.ia = w.ia.min(ya);
            w.ib = w.ib.max(yb);
        }
    }
    w
}

#[derive(Default, Clone, Copy, Debug)]
struct Sizes {
    prgba: u64,
    paux: u64,
    spl: u64,
    reach: u64,
    grid: u64,
    outp: u64,
    outel: u64,
}
impl Sizes {
    fn max(&self, o: &Sizes) -> Sizes {
        Sizes {
            prgba: self.prgba.max(o.prgba),
            paux: self.paux.max(o.paux),
            spl: self.spl.max(o.spl),
            reach: self.reach.max(o.reach),
            grid: self.grid.max(o.grid),
            outp: self.outp.max(o.outp),
            outel: self.outel.max(o.outel),
        }
    }
    fn total(&self) -> u64 {
        // + staging for the readback (outp + outel)
        self.prgba + self.paux + self.spl + self.reach + self.grid + 2 * (self.outp + self.outel)
    }
}

fn sizes(d: &obgpu_blur_desc, w: &Win, nc: u64) -> Sizes {
    let wd = d.w as u64;
    let mut s = Sizes::default();
    let rows = (w.ib - w.ia).max(0) as u64;
    s.prgba = rows * wd * 16;
    s.paux = rows * wd * 8;
    for l in 0..d.n_levels as usize {
        let lw = if l == 0 { wd } else { d.levels[l].w as u64 };
        let r = (w.lv[l].sb - w.lv[l].sa).max(0) as u64;
        s.spl += r * lw * SPLAT_BYTES;
        s.reach += r * 4;
        if l > 0 {
            s.grid += (w.lv[l].gb - w.lv[l].ga).max(0) as u64 * lw * nc * 4;
        }
    }
    let orows = (w.ob - w.oa).max(0) as u64;
    s.outp = orows * wd * 16;
    s.outel = if d.need_el != 0 { orows * wd * 4 } else { 0 };
    s
}

/// The band list for the output rows [rb, re) under `budget` bytes.
fn plan(d: &obgpu_blur_desc, mr: &[i32; 8], nc: u64, budget: u64, force_rows: u32, max_binding: u64) -> Result<Vec<Win>, Fail> {
    let (rb, re) = (d.row_begin, d.row_end);
    let n = re - rb;
    let fits = |hb: i32| -> bool {
        // worst band: one in the middle of the range (edge bands have clipped windows)
        let oa = rb + ((n - hb) / 2).max(0);
        let s = sizes(d, &window(d, mr, oa, (oa + hb).min(re)), nc);
        s.total() <= budget && s.spl.max(s.prgba).max(s.grid).max(s.outp) <= max_binding
    };
    let hb = if force_rows > 0 {
        (force_rows as i32).min(n).max(1)
    } else if fits(n) {
        n
    } else {
        let (mut lo, mut hi) = (0i32, n); // largest hb in [8, n] that fits
        if !fits(8.min(n)) {
            return Err(fail(OBGPU_OUT_OF_VRAM, format!("even an 8-row band exceeds the VRAM budget ({} MB)", budget >> 20)));
        }
        lo = lo.max(8.min(n));
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let mut v = Vec::new();
    let mut oa = rb;
    while oa < re {
        let ob = (oa + hb).min(re);
        v.push(window(d, mr, oa, ob));
        oa = ob;
    }
    Ok(v)
}

fn put_i(w: &mut [u32], i: usize, v: i32) {
    w[i] = v as u32;
}
fn put_f(w: &mut [u32], i: usize, v: f64) {
    w[i] = (v as f32).to_bits();
}

const GLOB_WORDS: usize = 1296;

fn glob_bytes(d: &obgpu_blur_desc, sc: &Scales, xr: &[i32; 8], mr: &[i32; 8], offs: (i32, i32, i32)) -> Vec<u32> {
    let mut g = vec![0u32; GLOB_WORDS];
    let roll = d.need_el != 0;
    let ints = [
        d.w, d.h, d.nk, d.n_tiers, d.cat, d.cat_n, d.fringe_share, d.clamp_neg, roll as i32, d.n_levels, d.max_reach, d.ap_n,
        d.circ_n, offs.0, offs.1, offs.2,
    ];
    for (i, v) in ints.iter().enumerate() {
        put_i(&mut g, i, *v);
    }
    let fl = [
        d.ax, d.ay, d.xext, d.yext, d.cat_k, d.cat_lim, d.cat_inv, d.level_base, d.fcx, d.fcy, d.half_diag, d.pa, d.ap_ymin,
        d.ap_ymax, d.ap_inv_step, d.ap_area, d.circ_ymin, d.circ_ymax, d.circ_inv_step, d.circ_area,
    ];
    for (i, v) in fl.iter().enumerate() {
        put_f(&mut g, 16 + i, *v);
    }
    for s in 0..2 {
        for c in 0..3 {
            put_f(&mut g, 36 + s * 4 + c, d.ks[s][c]);
        }
    }
    for k in 0..8 {
        g[44 + k] = sc.s[k].to_bits();
        g[52 + k] = (1.0f32 / sc.s[k]).to_bits();
        g[1280 + k] = sc.s[8 + k].to_bits();
        g[1288 + k] = (1.0f32 / sc.s[8 + k]).to_bits();
    }
    for c in 0..4 {
        g[1276 + c] = sc.tl[c].to_bits();
    }
    for l in 0..8 {
        let o = 60 + l * 8;
        if l == 0 {
            let v = [1, d.w, d.h, 0, 0, d.max_reach, xr[0], mr[0]];
            for (i, x) in v.iter().enumerate() {
                put_i(&mut g, o + i, *x);
            }
        } else if l < d.n_levels as usize {
            let lv = &d.levels[l];
            let v = [lv.f, lv.w, lv.h, lv.gox, lv.goy, lv.max_reach, xr[l], mr[l]];
            for (i, x) in v.iter().enumerate() {
                put_i(&mut g, o + i, *x);
            }
        }
    }
    for s in 0..2 {
        for t in 0..TIERS {
            let p = &d.prof[s][t];
            let o = 124 + (s * 8 + t) * 72;
            put_i(&mut g, o, p.nl);
            put_i(&mut g, o + 1, p.share);
            put_i(&mut g, o + 2, p.share_from);
            put_f(&mut g, o + 4, p.s2);
            put_f(&mut g, o + 5, p.s2out);
            put_f(&mut g, o + 6, p.s2in);
            for l in 0..MAX_LAYERS {
                put_f(&mut g, o + 8 + l, p.ls[l]);
                put_f(&mut g, o + 40 + l, p.lb[l]);
            }
        }
    }
    g
}

#[derive(Default, Clone, Copy)]
struct Disp {
    w: [u32; 32],
}
impl Disp {
    fn a(&mut self, v: [i32; 4]) -> &mut Self {
        for i in 0..4 {
            self.w[i] = v[i] as u32;
        }
        self
    }
    fn b(&mut self, v: [i32; 4]) -> &mut Self {
        for i in 0..4 {
            self.w[4 + i] = v[i] as u32;
        }
        self
    }
    fn c(&mut self, v: [u32; 4]) -> &mut Self {
        self.w[8..12].copy_from_slice(&v);
        self
    }
    fn d(&mut self, v: [i32; 4]) -> &mut Self {
        for i in 0..4 {
            self.w[12 + i] = v[i] as u32;
        }
        self
    }
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// parallel for over [0, n) in contiguous chunks (scoped threads)
fn par_chunks<F: Fn(usize, usize) + Sync>(n: usize, f: F) {
    let nt = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(16).max(1);
    if n < 65536 || nt == 1 {
        f(0, n);
        return;
    }
    let step = n.div_ceil(nt);
    std::thread::scope(|s| {
        for t in 0..nt {
            let a = t * step;
            let b = ((t + 1) * step).min(n);
            if a < b {
                let f = &f;
                s.spawn(move || f(a, b));
            }
        }
    });
}

pub struct Scales {
    pub s: [f32; 16],  // per accumulator slot
    pub tl: [f32; 4],  // log2 of the R, G, B, eL maxima
    pub k: u32,        // magnitude tiers
}

/// per-slot power-of-two scales from the frame maxima (band independent => tiles are exact)
fn slot_scales(inp: &Input, d: &obgpu_blur_desc) -> Scales {
    scales_from(frame_maxima(inp, d), d, inp.el.is_some())
}

fn frame_maxima(inp: &Input, d: &obgpu_blur_desc) -> [f32; 5] {
    let npx = (d.w as usize) * (d.h as usize);
    let parts = std::sync::Mutex::new(Vec::<[f32; 5]>::new());
    par_chunks(npx, |a, b| {
        let mut m = [0f32; 5];
        for i in a..b {
            let p = &inp.rgba[i * 4..i * 4 + 4];
            for c in 0..4 {
                let v = p[c].abs();
                if v > m[c] {
                    m[c] = v;
                }
            }
        }
        if let Some(el) = inp.el {
            for i in a..b {
                let v = el[i].abs();
                if v > m[4] {
                    m[4] = v;
                }
            }
        }
        parts.lock().unwrap().push(m);
    });
    let mut m = [0f32; 5];
    for p in parts.into_inner().unwrap() {
        for c in 0..5 {
            m[c] = m[c].max(p[c]);
        }
    }
    m
}

pub fn scales_from(m: [f32; 5], d: &obgpu_blur_desc, roll: bool) -> Scales {
    // 2^(46 - ceil(log2 max)): per-delta resolution ~1.4e-14 of the channel maximum, while spans weights
    // up to ~2^8 and per-pixel sums up to ~2^8 of the maximum stay inside the signed 64-bit range
    let sc = |mx: f64| -> f32 {
        let mx = if mx.is_finite() && mx > 0.0 { mx } else { 1e-9 };
        let e = mx.log2().ceil().clamp(-60.0, 70.0);
        (2f64).powf(46.0 - e) as f32
    };
    let lg = |mx: f32| -> f32 { if mx.is_finite() && mx > 0.0 { (mx as f64).log2() as f32 } else { -100.0 } };
    // tiers: K = 1 up to 2^8 (SDR incl. boosted highlights), +1 per 20 bits of dynamic range above that
    let mc = m[0].max(m[1]).max(m[2]).max(if roll { m[4] } else { 0.0 }) as f64;
    let k = if mc.is_finite() && mc > 256.0 { (1.0 + ((mc.log2() - 8.0) / 20.0).ceil()).min(3.0) as u32 } else { 1 };
    let tier = |mx: f32, t: u32| sc(mx as f64 * (2f64).powi(-20 * t as i32));
    let w1 = sc(1.0);
    let mut s = [1.0f32; 16];
    if d.nk == 3 {
        // R 0, WR 1, G 2, A 3, WG 4, B 5, WB 6, EL 7
        s[..8].copy_from_slice(&[tier(m[0], 0), w1, tier(m[1], 0), sc(m[3] as f64), w1, tier(m[2], 0), w1, tier(m[4], 0)]);
    } else {
        // R 0, G 1, B 2, A 3, W 4, EL 5
        s[..6].copy_from_slice(&[tier(m[0], 0), tier(m[1], 0), tier(m[2], 0), sc(m[3] as f64), w1, tier(m[4], 0)]);
    }
    // extra tiers after the base slots: (R, G, B[, eL]) per tier
    let ncb = if d.nk == 3 { 7 } else { 5 } + roll as usize;
    let ncc = 3 + roll as usize;
    for t in 1..k as usize {
        let o = ncb + (t - 1) * ncc;
        s[o] = tier(m[0], t as u32);
        s[o + 1] = tier(m[1], t as u32);
        s[o + 2] = tier(m[2], t as u32);
        if roll {
            s[o + 3] = tier(m[4], t as u32);
        }
    }
    Scales { s, tl: [lg(m[0]), lg(m[1]), lg(m[2]), lg(m[4])], k }
}

struct Bufs {
    glob: wgpu::Buffer,
    disp: wgpu::Buffer,
    tabs: wgpu::Buffer,
    prgba: wgpu::Buffer,
    paux: wgpu::Buffer,
    spl: wgpu::Buffer,
    reach: wgpu::Buffer,
    grid: wgpu::Buffer,
    outp: wgpu::Buffer,
    outel: wgpu::Buffer,
    stage: wgpu::Buffer,
    bytes: u64,
}

fn mkbuf(dev: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16).div_ceil(16) * 16, usage, mapped_at_creation: false })
}

/// Prepared buffers already on the device (written by the prepare stage, src/frame.rs): the whole crop,
/// colour (vec4) and (signed radius, eL) (vec2), plus the frame maxima of R, G, B, A, eL.
pub struct DevPrep<'a> {
    pub prgba: &'a wgpu::Buffer,
    pub paux: &'a wgpu::Buffer,
    pub maxima: [f32; 5],
}

pub fn run(gpu: &Gpu, inp: &Input, d: &obgpu_blur_desc, out: &Output, abort: &dyn Fn() -> bool, tun: &Tuning, st: &mut obgpu_stats) -> Result<(), Fail> {
    run_any(gpu, Some(inp), None, d, out, None, abort, tun, st)
}

/// ABI 1.3: the blurred crop stays on the device (the post stages run on it before the one readback)
pub struct Sink<'a> {
    pub rgba: &'a wgpu::Buffer,
    pub el: Option<&'a wgpu::Buffer>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_any(gpu: &Gpu, inp: Option<&Input>, dev: Option<&DevPrep>, d: &obgpu_blur_desc, out: &Output, sink: Option<&Sink>, abort: &dyn Fn() -> bool, tun: &Tuning, st: &mut obgpu_stats) -> Result<(), Fail> {
    // validation errors raised inside a job are attached to its failure message (diagnostics)
    let val = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let r = run_inner(gpu, inp, dev, d, out, sink, abort, tun, st);
    let e = block_on(val.pop()).flatten();
    match (r, e) {
        (Ok(()), None) => Ok(()),
        // (after a device loss every later call fails validation: report the loss, so the VRAM budgets shrink)
        (Ok(()), Some(e)) => Err(fail(if gpu.is_lost() { OBGPU_DEVICE_LOST } else { OBGPU_INTERNAL }, format!("validation: {e}"))),
        (Err(mut f), Some(e)) => {
            f.msg = format!("{} [validation: {e}]", f.msg);
            Err(f)
        }
        (Err(f), None) => Err(f),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_inner(gpu: &Gpu, inp: Option<&Input>, dev: Option<&DevPrep>, d: &obgpu_blur_desc, out: &Output, sink: Option<&Sink>, abort: &dyn Fn() -> bool, tun: &Tuning, st: &mut obgpu_stats) -> Result<(), Fail> {
    let trace_t0 = Instant::now();
    let trace = std::env::var("OBGPU_TRACE").is_ok();
    let tr = |what: &str| if trace { eprintln!("[obgpu.trace] {:8.1} ms {what}", trace_t0.elapsed().as_secs_f64() * 1e3) };
    let t_start = Instant::now();
    let (w, h) = (d.w as usize, d.h as usize);
    let roll = d.need_el != 0;
    // rows nobody reads: defined zeros (blurAlpha does the same)
    for y in 0..h {
        if (y as i32) >= d.row_begin && (y as i32) < d.row_end {
            continue;
        }
        unsafe {
            std::ptr::write_bytes(out.rgba.add(y * out.stride), 0, 4 * w);
            if roll {
                std::ptr::write_bytes(out.el.add(y * w), 0, w);
            }
        }
    }
    if d.row_end <= d.row_begin {
        return Ok(());
    }
    if gpu.is_lost() {
        return Err(fail(OBGPU_DEVICE_LOST, "device lost"));
    }
    tr("start");
    let sc = match (inp, dev) {
        (Some(i), _) => slot_scales(i, d),
        (None, Some(v)) => scales_from(v.maxima, d, d.need_el != 0),
        _ => return Err(fail(OBGPU_BAD_ARGS, "no input")),
    };
    let base_v = Variant { fr: d.nk == 3, cat: d.cat != 0, roll, raw: false, k: sc.k };
    let nc = base_v.nc() as u64;
    let seg = gpu.seg(base_v);
    let (xr, mr) = level_reach(d);
    let max_binding = gpu.limits.max_storage_buffer_binding_size.min(gpu.limits.max_buffer_size);
    let bands = plan(d, &mr, nc, tun.budget, tun.band_rows, max_binding)?;
    let mut maxs = Sizes::default();
    for b in &bands {
        maxs = maxs.max(&sizes(d, b, nc));
    }
    if maxs.spl.max(maxs.prgba).max(maxs.grid).max(maxs.outp) > max_binding {
        return Err(fail(OBGPU_OUT_OF_VRAM, "a buffer exceeds the device binding limit"));
    }
    st.bands = bands.len() as i32;

    // pipelines (compiled on first use)
    tr("planned");
    let setup = gpu.setup_pipes().map_err(|e| fail(OBGPU_INTERNAL, e))?;
    let p_fine = gpu.accum_pipe(base_v).map_err(|e| fail(OBGPU_INTERNAL, e))?;
    let p_raw = if d.n_levels > 1 { Some(gpu.accum_pipe(Variant { raw: true, ..base_v }).map_err(|e| fail(OBGPU_INTERNAL, e))?) } else { None };

    tr("pipelines");
    let t_conv = Instant::now();
    // tables: ap | circ | catA (f32)
    // integral tables: per node (IL hi, IL lo, IL[i+1]-IL[i], IR hi, IR lo, IR[i+1]-IR[i]) - see blur.wgsl lookT
    let node6 = |t: &[f64], n: usize, out: &mut Vec<f32>| {
        for i in 0..=n {
            for c in 0..2 {
                let v = t[2 * i + c];
                let hi = v as f32;
                let lo = (v - hi as f64) as f32;
                let dv = if i < n { t[2 * (i + 1) + c] - v } else { 0.0 };
                out.extend_from_slice(&[hi, lo, dv as f32]);
            }
        }
    };
    let apn = 6 * (d.ap_n as usize + 1);
    let cin = 6 * (d.circ_n as usize + 1);
    let can = if d.cat_n > 0 { (d.cat_n * d.cat_n) as usize } else { 0 };
    let mut tabs = Vec::with_capacity(apn + cin + can + 4);
    unsafe {
        node6(std::slice::from_raw_parts(d.ap_table, 2 * (d.ap_n as usize + 1)), d.ap_n as usize, &mut tabs);
        node6(std::slice::from_raw_parts(d.circ_table, 2 * (d.circ_n as usize + 1)), d.circ_n as usize, &mut tabs);
        if can > 0 {
            tabs.extend(std::slice::from_raw_parts(d.cat_a, can).iter().map(|v| *v as f32));
        }
    }
    tabs.resize(tabs.len() + 4, 0.0);
    let glob = glob_bytes(d, &sc, &xr, &mr, (0, apn as i32, (apn + cin) as i32));
    st.convert_ms += t_conv.elapsed().as_secs_f64() * 1e3;

    // ---- buffers (sized for the largest band), under OOM / validation scopes
    let devp = dev;
    let dev = &gpu.device;
    let q = &gpu.queue;
    // inner out-of-memory scope around the allocations (the validation scope is in run())
    tr("tables");
    let oom = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    use wgpu::BufferUsages as U;
    let stor = U::STORAGE | U::COPY_DST;
    let b = Bufs {
        glob: mkbuf(dev, "glob", (GLOB_WORDS * 4) as u64, U::UNIFORM | U::COPY_DST),
        disp: mkbuf(dev, "disp", DISP_STRIDE * DISP_SLOTS, U::UNIFORM | U::COPY_DST),
        tabs: mkbuf(dev, "tabs", (tabs.len() * 4) as u64, stor),
        prgba: mkbuf(dev, "prgba", if devp.is_some() { 16 } else { maxs.prgba }, stor),
        paux: mkbuf(dev, "paux", if devp.is_some() { 16 } else { maxs.paux }, stor),
        spl: mkbuf(dev, "spl", maxs.spl, stor),
        reach: mkbuf(dev, "reach", maxs.reach, stor),
        grid: mkbuf(dev, "grid", maxs.grid, stor),
        outp: mkbuf(dev, "outp", maxs.outp, stor | U::COPY_SRC),
        outel: mkbuf(dev, "outel", maxs.outel, stor | U::COPY_SRC),
        stage: mkbuf(dev, "stage", maxs.outp + maxs.outel, U::MAP_READ | U::COPY_DST),
        bytes: maxs.total() + DISP_STRIDE * DISP_SLOTS,
    };
    st.vram_bytes = st.vram_bytes.max(b.bytes);
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("obgpu.bg"),
        layout: &gpu.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b.glob.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &b.disp, offset: 0, size: wgpu::BufferSize::new(DISP_BYTES) }),
            },
            wgpu::BindGroupEntry { binding: 2, resource: b.tabs.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: devp.map(|v| v.prgba).unwrap_or(&b.prgba).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: devp.map(|v| v.paux).unwrap_or(&b.paux).as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: b.spl.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: b.reach.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: b.grid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: b.outp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: b.outel.as_entire_binding() },
        ],
    });
    if let Some(Some(e)) = block_on(oom.pop()) {
        return Err(fail(OBGPU_OUT_OF_VRAM, format!("allocation: {e}")));
    }
    tr("buffers");
    q.write_buffer(&b.glob, 0, as_bytes(&glob));
    q.write_buffer(&b.tabs, 0, as_bytes(&tabs));

    let mut slot: u64 = 0;
    let mut ts = crate::gpu::Ts::new(gpu, 1024);
    // submits one compute pass; wait_now = wait for it here (setup), else the caller waits (pipelined chunks)
    let mut submit = |enc_fn: &dyn Fn(&mut wgpu::ComputePass, u32), rec: &Disp, st: &mut obgpu_stats, wait_now: bool| -> Result<Option<crate::gpu::Sub>, Fail> {
        if abort() {
            return Err(fail(OBGPU_ABORTED, "aborted"));
        }
        let off = (slot % DISP_SLOTS) * DISP_STRIDE;
        slot += 1;
        q.write_buffer(&b.disp, off, as_bytes(&rec.w));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let tw = ts.as_mut().and_then(|t| t.writes_l(if wait_now { "blur.setup" } else { "blur.accum" }));
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
            pass.set_bind_group(0, &bg, &[off as u32]);
            enc_fn(&mut pass, off as u32);
        }
        let idx = gpu.submit(enc.finish());
        st.submissions += 1;
        if tun.delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(tun.delay_ms as u64));
        }
        if wait_now {
            wait(gpu, idx, abort, tun, st)?;
            return Ok(None);
        }
        Ok(Some(idx))
    };

    let w_u = w as u32;
    for win in &bands {
        // ---- upload the prep rows of the window
        let t_up = Instant::now();
        let (ia, ib) = (win.ia as usize, win.ib as usize);
        // window row 0 of the prep buffers: the band's first input row (host upload) or 0 (device: whole crop)
        let prow0 = if devp.is_some() { 0 } else { win.ia };
        if let Some(inp) = inp {
        let npx = (ib - ia) * w;
        q.write_buffer(&b.prgba, 0, as_bytes(&inp.rgba[ia * w * 4..ib * w * 4]));
        let mut aux = vec![0f32; npx * 2];
        {
            let auxp = aux.as_mut_ptr() as usize;
            par_chunks(npx, |a, bnd| {
                let ap = auxp as *mut f32;
                for i in a..bnd {
                    let g = ia * w + i;
                    let r = inp.radius[g] as f32;
                    let r = if inp.side[g] > 0 { r } else { -r };
                    let e = inp.el.map(|el| el[g]).unwrap_or(0.0);
                    unsafe {
                        *ap.add(2 * i) = r;
                        *ap.add(2 * i + 1) = e;
                    }
                }
            });
        }
        q.write_buffer(&b.paux, 0, as_bytes(&aux));
        drop(aux);
        }
        // level splat / reach / grid bases
        let mut sbase = [0u32; 8];
        let mut rbase = [0u32; 8];
        let mut gbase = [0u32; 8];
        let (mut so, mut ro, mut go) = (0u64, 0u64, 0u64);
        for l in 0..d.n_levels as usize {
            let lw = if l == 0 { w as u64 } else { d.levels[l].w as u64 };
            let lv = &win.lv[l];
            sbase[l] = so as u32;
            rbase[l] = ro as u32;
            gbase[l] = go as u32;
            so += (lv.sb - lv.sa).max(0) as u64 * lw;
            ro += (lv.sb - lv.sa).max(0) as u64;
            if l > 0 {
                go += (lv.gb - lv.ga).max(0) as u64 * lw * nc;
            }
        }
        {
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.clear_buffer(&b.reach, 0, None);
            q.submit(Some(enc.finish()));
        }
        st.upload_ms += t_up.elapsed().as_secs_f64() * 1e3;
        tr("uploaded");

        let t_gpu = Instant::now();
        // ---- setup: fine splats, then each coarse level's merged splats
        {
            let lv = win.lv[0];
            let mut r = Disp::default();
            r.b([lv.sa, lv.sb - lv.sa, prow0, win.ib - win.ia]).c([sbase[0], rbase[0], 0, 0]);
            let rows = (lv.sb - lv.sa) as u32;
            if rows > 0 {
                submit(&|p, _| {
                    p.set_pipeline(&setup.setup_fine);
                    p.dispatch_workgroups(w_u.div_ceil(256), rows, 1);
                }, &r, st, true)?;
            }
        }
        for l in 1..d.n_levels as usize {
            let lv = win.lv[l];
            let rows = (lv.sb - lv.sa).max(0) as u32;
            if rows == 0 {
                continue;
            }
            let mut r = Disp::default();
            r.a([l as i32, 0, 0, 0]).b([lv.sa, lv.sb - lv.sa, prow0, win.ib - win.ia]).c([sbase[l], rbase[l], 0, 0]);
            let lw = d.levels[l].w as u32;
            submit(&|p, _| {
                p.set_pipeline(&setup.setup_coarse);
                p.dispatch_workgroups(lw.div_ceil(64), rows, 1);
            }, &r, st, true)?;
        }
        // ---- accumulation: coarse levels (RAW grids), then the fine rows (resolve)
        let mut chunk_rows = |level: usize, y0: i32, y1: i32, pipe: &wgpu::ComputePipeline, st: &mut obgpu_stats| -> Result<(), Fail> {
            let lw = if level == 0 { w as i32 } else { d.levels[level].w };
            let nseg = (lw as u32).div_ceil(seg);
            let lv = win.lv[level];
            let mut y = y0;
            let mut n: i32 = 4;
            // two submissions in flight: the GPU never waits for the CPU's submit round trip. The chunk size
            // adapts to ~target_ms per submission (abort latency <= 2 chunks, far below the 2 s TDR limit).
            let mut pending: Option<(crate::gpu::Sub, i32)> = None;
            let mut t_last = Instant::now();
            while y < y1 {
                let cnt = n.min(y1 - y);
                let mut r = Disp::default();
                r.a([level as i32, y, cnt, nseg as i32]).b([lv.sa, lv.sb - lv.sa, prow0, win.ib - win.ia]);
                r.c([sbase[level], rbase[level], gbase[level], 0]);
                r.d([lv.ga, win.oa, 0, 0]);
                for l in 1..d.n_levels as usize {
                    r.w[16 + l] = gbase[l];
                    r.w[24 + l] = win.lv[l].ga as u32;
                }
                let idx = submit(&|p, _| {
                    p.set_pipeline(pipe);
                    p.dispatch_workgroups(nseg, cnt as u32, 1);
                }, &r, st, false)?;
                y += cnt;
                if let Some((pidx, pcnt)) = pending.take() {
                    wait(gpu, pidx, abort, tun, st)?;
                    let ms = t_last.elapsed().as_secs_f64() * 1e3;
                    t_last = Instant::now();
                    let per = (ms / pcnt as f64).max(1e-3);
                    n = ((tun.target_ms / per) as i32).clamp(1, 4096).min(n.saturating_mul(4));
                } else {
                    n = n.saturating_mul(2);
                }
                pending = idx.map(|i| (i, cnt));
            }
            if let Some((pidx, _)) = pending.take() {
                wait(gpu, pidx, abort, tun, st)?;
            }
            Ok(())
        };
        if let Some(pr) = p_raw.as_ref() {
            for l in 1..d.n_levels as usize {
                let lv = win.lv[l];
                chunk_rows(l, lv.ga, lv.gb, pr, st)?;
            }
        }
        chunk_rows(0, win.oa, win.ob, &p_fine, st)?;
        st.gpu_ms += t_gpu.elapsed().as_secs_f64() * 1e3;

        // ---- readback
        let t_rb = Instant::now();
        let orows = (win.ob - win.oa) as usize;
        let rgba_bytes = (orows * w * 16) as u64;
        let el_bytes = if roll { (orows * w * 4) as u64 } else { 0 };
        if let Some(sk) = sink {
            // device-resident result: rows [oa, ob) of the crop buffers (rows outside the range stay zero)
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&b.outp, 0, sk.rgba, (win.oa as u64) * (w as u64) * 16, rgba_bytes);
            if let (true, Some(e)) = (roll, sk.el) {
                enc.copy_buffer_to_buffer(&b.outel, 0, e, (win.oa as u64) * (w as u64) * 4, el_bytes);
            }
            let idx = gpu.submit(enc.finish());
            st.submissions += 1;
            wait(gpu, idx, abort, tun, st)?;
            continue;
        }
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b.outp, 0, &b.stage, 0, rgba_bytes);
        if roll {
            enc.copy_buffer_to_buffer(&b.outel, 0, &b.stage, rgba_bytes, el_bytes);
        }
        let idx = gpu.submit(enc.finish());
        st.submissions += 1;
        wait(gpu, idx, abort, tun, st)?;
        let slice = b.stage.slice(0..rgba_bytes + el_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let t_map = Instant::now();
        loop {
            if abort() {
                return Err(fail(OBGPU_ABORTED, "aborted (readback abandoned)"));
            }
            let _ = dev.poll(wgpu::PollType::Poll);
            if let Ok(ok) = rx.try_recv() {
                if !ok {
                    return Err(fail(if gpu.is_lost() { OBGPU_DEVICE_LOST } else { OBGPU_INTERNAL }, "map failed"));
                }
                break;
            }
            if t_map.elapsed() > tun.timeout {
                return Err(fail(OBGPU_TIMEOUT, "map timeout"));
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        {
            let view = slice.get_mapped_range().map_err(|e| fail(OBGPU_INTERNAL, format!("map range: {e:?}")))?;
            let src: &[u8] = &view;
            let fsrc = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, src.len() / 4) };
            for r in 0..orows {
                let y = win.oa as usize + r;
                unsafe {
                    std::ptr::copy_nonoverlapping(fsrc.as_ptr().add(r * w * 4), out.rgba.add(y * out.stride), 4 * w);
                    if roll {
                        std::ptr::copy_nonoverlapping(fsrc.as_ptr().add(orows * w * 4 + r * w), out.el.add(y * w), w);
                    }
                }
            }
        }
        b.stage.unmap();
        st.readback_ms += t_rb.elapsed().as_secs_f64() * 1e3;
        if gpu.is_lost() {
            return Err(fail(OBGPU_DEVICE_LOST, "device lost"));
        }
    }
    drop(submit);
    gpu.free(&[&b.prgba, &b.paux, &b.spl, &b.reach, &b.grid, &b.outp, &b.outel, &b.stage]);
    busy_add(match ts.as_ref() { Some(t) => t.busy_ms(gpu), None => None });
    st.total_ms += t_start.elapsed().as_secs_f64() * 1e3;
    Ok(())
}

/// Waits for a submission (non-blocking polls, abort checked every ~0.3 ms; abandons the work on abort).
fn wait(gpu: &Gpu, idx: crate::gpu::Sub, abort: &dyn Fn() -> bool, tun: &Tuning, st: &mut obgpu_stats) -> Result<(), Fail> {
    let t0 = Instant::now();
    let r = gpu.wait_sub(&idx, abort, tun.timeout);
    st.wait_ms += t0.elapsed().as_secs_f64() * 1e3;
    r.map_err(|s| fail(s, match s {
        OBGPU_ABORTED => "aborted (in-flight work abandoned)".to_string(),
        OBGPU_TIMEOUT => format!("GPU did not finish within {} ms", tun.timeout.as_millis()),
        _ => "device lost while waiting".to_string(),
    }))
}
