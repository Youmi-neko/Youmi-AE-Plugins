// prepare.wgsl - lenscore prepareHighlights (renderVersion 2, automatic highlight source) on the GPU.
// See gpu/PHASE2.md. Kernels are dispatched by src/frame.rs in the CPU's order:
//   lum -> knee box means -> 3x3 max / 5x5 min -> E -> White Protection grid -> detail openings (size
//   window) [-> min detail] -> dots (4 line openings) -> rows (prepareRow, written straight into the blur
//   stage's input buffers for the crop rect).
// Exactness: min / max filters are exact (order-free); the per-pixel float expressions that lenscore
// evaluates in float are written identically; lenscore's double expressions (pow, log, box-mean prefix
// sums) are f32 (box sums: compensated two-float prefix) -> small continuous differences.

struct PG {
    i0: vec4i,   // w, h, has_map, invert_map
    i1: vec4i,   // hl_on, knee_box_r, wp_on, det_on
    i2: vec4i,   // wp_f, wp_gx, wp_gy, wp_gw
    i3: vec4i,   // wp_gh, wp_rb, dmin_on, dots_on
    i4: vec4i,   // crop_x, crop_y, crop_w, crop_h
    i5: vec4i,   // src_stride (floats), map_stride, crop row offset of this band (rows into prgba), -
    f0: vec4f,   // focus, map_scale, map_gamma, max_radius
    f1: vec4f,   // hl_gain, hl_threshold, boost, knee
    f2: vec4f,   // curve, sel, hsat, ps
    f3: vec4f,   // adapt_gain, wp_amt, wp_rel, dmin_ramp
    f4: vec4f,   // dots, -, -, -
    tau: array<vec4f, 3>,
    pl: array<vec4i, 4>,   // plane offsets (floats): L, T, E, D, | PK, BG, A, B, | C, Dm, LEV, CNT, | GS, -, -, -
}
struct PD {
    a: vec4i,    // W, H, r, dir
    b: vec4i,    // in, out, aux, aux2 (plane offsets, floats)
    c: vec4i,    // max flag, -, -, -
    d: vec4f,    // t, w, -, -
}

@group(0) @binding(0) var<uniform> G: PG;
@group(0) @binding(1) var<uniform> D: PD;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> mapv: array<f32>;
@group(0) @binding(4) var<storage, read_write> wk: array<f32>;
@group(0) @binding(5) var<storage, read_write> prgba: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> paux: array<vec2f>;
@group(0) @binding(7) var<storage, read_write> flags: array<atomic<u32>>;   // 0 anyE, 1 anyE after WP, 2..6 max bits R G B A eL

fn comp4(v: vec4f, i: i32) -> f32 { return select(select(select(v.w, v.z, i == 2), v.y, i == 1), v.x, i == 0); }
fn comp4i(v: vec4i, i: i32) -> i32 { return select(select(select(v.w, v.z, i == 2), v.y, i == 1), v.x, i == 0); }
fn plane(k: i32) -> u32 { return u32(comp4i(G.pl[k >> 2], k & 3)); }
const P_L: i32 = 0; const P_T: i32 = 1; const P_E: i32 = 2; const P_D: i32 = 3;
const P_PK: i32 = 4; const P_BG: i32 = 5; const P_A: i32 = 6; const P_B: i32 = 7;
const P_C: i32 = 8; const P_DM: i32 = 9; const P_LEV: i32 = 10; const P_CNT: i32 = 11; const P_GS: i32 = 12;

fn Wd() -> i32 { return G.i0.x; }
fn Ht() -> i32 { return G.i0.y; }
fn smooth01(t0: f32) -> f32 { let t = clamp(t0, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }

// lenscore sanitize4 (colour NaN/-Inf -> 0, +Inf -> 65504, clamp [0, 1e6]; alpha [0, 1])
fn finite(v: f32) -> bool { return v == v && abs(v) <= 3.4028234e38; }
fn san4(s: vec4f) -> vec4f {
    var o: vec4f;
    for (var c = 0; c < 3; c++) {
        let v = s[c];
        o[c] = select(select(0.0, 65504.0, v > 0.0), min(1.0e6, max(0.0, v)), finite(v));
    }
    let a = s.w;
    o.w = select(select(0.0, 1.0, a > 0.0), min(1.0, max(0.0, a)), finite(a));
    return o;
}
fn san1(v: f32) -> f32 { return select(select(0.0, 1.0e6, v > 0.0), min(1.0e6, max(-1.0e6, v)), finite(v)); }
fn srcAt(x: i32, y: i32) -> vec4f {
    let o = u32(y) * u32(G.i5.x) + u32(x) * 4u;
    return vec4f(src[o], src[o + 1u], src[o + 2u], src[o + 3u]);
}

// ---- L (chroma-compensated brightness, all float like lenscore) and T = min(1, L)
@compute @workgroup_size(256, 1, 1)
fn pp_lum(@builtin(global_invocation_id) g: vec3u) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let s = san4(srcAt(x, y));
    let l = 0.2126 * s.x + 0.7152 * s.y + 0.0722 * s.z;
    let mx = max(s.x, max(s.y, s.z));
    let mn = min(s.x, min(s.y, s.z));
    let sat = select(0.0, max(0.0, 1.0 - mn / mx), mx > 1e-6);
    let br = l + max(0.0, mx - l) * sat * sat;
    let L = select(0.0, max(0.0, br), finite(br));
    let i = u32(y * Wd() + x);
    wk[plane(P_L) + i] = L;
    wk[plane(P_T) + i] = min(1.0, L);
}

// ---- box mean along rows / columns: o[x] = sum(in[x0..x1)) / (x1 - x0), x0 = max(0, x-r), x1 = min(n, x+r+1)
// (lenscore: double prefix sums). Here one thread per pixel sums its own clipped window in a fixed order with a
// compensated two-float sum: the result depends only on the window, never on where a band / tile starts, so
// banded prepare == untiled prepare exactly. O(r) per pixel (r ~ 4 % of the frame diagonal): a few ms.
fn twoSum(a: vec2f, v: f32) -> vec2f {  // (hi, lo) + v
    let s = a.x + v;
    let bb = s - a.x;
    let e = (a.x - (s - bb)) + (v - bb);
    return vec2f(s, a.y + e);
}
fn boxAt(base: u32, step: u32, x: i32, n: i32, r: i32, ib: u32) -> f32 {
    let x0 = max(0, x - r);
    let x1 = min(n, x + r + 1);
    var acc = vec2f(0.0);
    for (var i = x0; i < x1; i++) { acc = twoSum(acc, wk[ib + base + u32(i) * step]); }
    return (acc.x + acc.y) / f32(x1 - x0);
}
@compute @workgroup_size(256, 1, 1)
fn pp_box_rows(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    wk[u32(D.b.y) + u32(y * W + x)] = boxAt(u32(y * W), 1u, x, W, D.a.z, u32(D.b.x));
}
@compute @workgroup_size(256, 1, 1)
fn pp_box_cols(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    wk[u32(D.b.y) + u32(y * W + x)] = boxAt(u32(x), u32(W), y, H, D.a.z, u32(D.b.x));
}
// ---- min / max over the window [-r, r] along a direction, clipped to the image (= lenscore runFilter /
// filterLines / filter2D: the van Herk result is exactly the clipped window extremum)
@compute @workgroup_size(256, 1, 1)
fn pp_mm_line(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let r = D.a.z; let dir = D.a.w;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let dx = select(1, 0, dir == 1);
    let dy = select(select(1, -1, dir == 3), 0, dir == 0);
    let ib = u32(D.b.x);
    let isMax = D.c.x != 0;
    var m = wk[ib + u32(y * W + x)];
    for (var k = 1; k <= r; k++) {
        let xa = x + k * dx; let ya = y + k * dy;
        if (xa >= 0 && xa < W && ya >= 0 && ya < H) {
            let v = wk[ib + u32(ya * W + xa)];
            m = select(min(m, v), max(m, v), isMax);
        }
        let xb = x - k * dx; let yb = y - k * dy;
        if (xb >= 0 && xb < W && yb >= 0 && yb < H) {
            let v = wk[ib + u32(yb * W + xb)];
            m = select(min(m, v), max(m, v), isMax);
        }
    }
    wk[u32(D.b.y) + u32(y * W + x)] = m;
}

// ---- E (v2: curve at the 3x3 peak, coverage share against the 5x5 minimum)
@compute @workgroup_size(256, 1, 1)
fn pp_E(@builtin(global_invocation_id) g: vec3u) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let i = u32(y * Wd() + x);
    let L = wk[plane(P_L) + i];
    let P = wk[plane(P_PK) + i];
    let B = min(wk[plane(P_BG) + i], L);
    let knee = G.f1.w; let curve = G.f2.x;
    var fp = 0.0;
    if (P > 1.0) {
        fp = 1.0 + log(P);
    } else {
        let k = min(0.97, max(knee, G.f3.x * wk[plane(P_T) + i]));
        let t = (P - k) / (1.0 - k);
        if (t > 0.0) { fp = pow(t, curve); }
    }
    var e = 0.0;
    if (fp > 0.0) {
        let den = P - B;
        var w = 1.0;
        if (den > 1e-4 * max(1.0, P)) { w = clamp((L - B) / den, 0.0, 1.0); }
        var lin = 0.0;
        if (L > 1e-12) { lin = min(1.0, w * P / L); }
        e = fp * lin;
    }
    wk[plane(P_E) + i] = e;
    if (e > 0.0) { atomicOr(&flags[0], 1u); }
}

// ---- White Protection: soft brightness levels averaged per coarse cell (one thread per cell, the cell's
// pixels in lenscore's order: rows ascending, then x)
@compute @workgroup_size(64, 1, 1)
fn pp_wp_cells(@builtin(global_invocation_id) g: vec3u) {
    let f = G.i2.x; let gx = G.i2.y; let gy = G.i2.z; let gw = G.i2.w; let gh = G.i3.x;
    let c = i32(g.x);
    if (c >= gw * gh) { return; }
    let cy = c / gw; let cx = c % gw;
    let ya = max(0, cy * f - gy); let yb = min(Ht(), (cy + 1) * f - gy);
    let xa = max(0, cx * f - gx); let xb = min(Wd(), (cx + 1) * f - gx);
    var lev: array<f32, 10>;
    for (var k = 0; k < 10; k++) { lev[k] = 0.0; }
    var cnt = 0.0;
    for (var y = ya; y < yb; y++) {
        for (var x = xa; x < xb; x++) {
            let v = min(1.2, wk[plane(P_L) + u32(y * Wd() + x)]);
            cnt += 1.0;
            for (var k = 0; k < 10; k++) {
                let tau = comp4(G.tau[k >> 2], k & 3);
                var prev = 0.0;
                if (k > 0) { prev = comp4(G.tau[(k - 1) >> 2], (k - 1) & 3); }
                let wdt = 0.6 * select(0.15, tau - prev, k > 0);
                lev[k] += smooth01((v - (tau - 0.5 * wdt)) / wdt);
            }
        }
    }
    for (var k = 0; k < 10; k++) {
        wk[plane(P_LEV) + u32(k * gw * gh + c)] = select(0.0, lev[k] / cnt, cnt > 0.0);
    }
}
fn levAt(k: i32, x: i32, y: i32) -> f32 { return wk[plane(P_LEV) + u32(k * G.i2.w * G.i3.x + y * G.i2.w + x)]; }
@compute @workgroup_size(256, 1, 1)
fn pp_wp_apply(@builtin(global_invocation_id) g: vec3u) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let i = u32(y * Wd() + x);
    var e = wk[plane(P_E) + i];
    if (e > 0.0) {
        let f = f32(G.i2.x); let gw = G.i2.w; let gh = G.i3.x;
        let v = (f32(y) + f32(G.i2.z) + 0.5) / f - 0.5;
        let vc = min(f32(gh) - 1.0, max(0.0, v));
        let y0 = i32(vc); let y1 = min(y0 + 1, gh - 1); let fy = vc - f32(y0);
        let u = (f32(x) + f32(G.i2.y) + 0.5) / f - 0.5;
        let uc = min(f32(gw) - 1.0, max(0.0, u));
        let x0 = i32(uc); let x1 = min(x0 + 1, gw - 1); let fx = uc - f32(x0);
        let tv = min(1.0, G.f3.z * wk[plane(P_L) + i]);
        var k0 = 0;
        while (k0 + 1 < 10 && comp4(G.tau[(k0 + 1) >> 2], (k0 + 1) & 3) <= tv) { k0++; }
        let k1 = min(9, k0 + 1);
        let t0 = comp4(G.tau[k0 >> 2], k0 & 3); let t1 = comp4(G.tau[k1 >> 2], k1 & 3);
        var ft = 0.0;
        if (k1 != k0) { ft = min(1.0, max(0.0, (tv - t0) / (t1 - t0))); }
        var bl: array<f32, 2>;
        for (var j = 0; j < 2; j++) {
            let k = select(k0, k1, j == 1);
            let a = levAt(k, x0, y0) + (levAt(k, x1, y0) - levAt(k, x0, y0)) * fx;
            let b = levAt(k, x0, y1) + (levAt(k, x1, y1) - levAt(k, x0, y1)) * fx;
            bl[j] = a + (b - a) * fy;
        }
        let F = bl[0] + (bl[1] - bl[0]) * ft;
        e *= 1.0 - G.f3.y * smooth01((F - 0.01) / 0.09);
        wk[plane(P_E) + i] = e;
    }
    if (e > 0.0) { atomicOr(&flags[1], 1u); }
}

// ---- size window: D += w (q0^2 + t (q1^2 - q0^2)), q = max(0, L - opening) / max(L, 1e-6) (float, as lenscore)
@compute @workgroup_size(256, 1, 1)
fn pp_det_acc(@builtin(global_invocation_id) g: vec3u) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let i = u32(y * Wd() + x);
    if (!(wk[plane(P_E) + i] > 0.0)) { return; }
    let L = wk[plane(P_L) + i];
    let il = 1.0 / max(L, 1e-6);
    let q0 = max(0.0, L - wk[u32(D.b.x) + i]) * il;
    let q1 = max(0.0, L - wk[u32(D.b.z) + i]) * il;
    let o = u32(D.b.y) + i;
    wk[o] = wk[o] + D.d.y * (q0 * q0 + D.d.x * (q1 * q1 - q0 * q0));
}
@compute @workgroup_size(256, 1, 1)
fn pp_dmin(@builtin(global_invocation_id) g: vec3u) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let i = u32(y * Wd() + x);
    wk[plane(P_D) + i] = max(0.0, wk[plane(P_D) + i] - G.f3.w * wk[plane(P_DM) + i]);
}
@compute @workgroup_size(256, 1, 1)
fn pp_best(@builtin(global_invocation_id) g: vec3u) {   // C = max(C, B)
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let i = u32(y * Wd() + x);
    wk[plane(P_C) + i] = max(wk[plane(P_C) + i], wk[plane(P_B) + i]);
}
@compute @workgroup_size(256, 1, 1)
fn pp_dots(@builtin(global_invocation_id) g: vec3u) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= Wd() || y >= Ht()) { return; }
    let i = u32(y * Wd() + x);
    let d = wk[plane(P_D) + i];
    if (!(d > 0.0)) { return; }
    let L = wk[plane(P_L) + i];
    let q = max(0.0, L - wk[plane(P_C) + i]) / max(L, 1e-6);
    wk[plane(P_D) + i] = d * (1.0 - G.f4.x * (1.0 - q * q));
}

// ---- prepareRow for the crop rect -> the blur stage's inputs (colour, signed radius, eL) + maxima
fn maxBits(slot: u32, v: f32) { atomicMax(&flags[slot], bitcast<u32>(max(v, 0.0))); }
@compute @workgroup_size(256, 1, 1)
fn pp_rows(@builtin(global_invocation_id) g: vec3u) {
    let cx = i32(g.x); let cy = i32(g.y);
    let cw = G.i4.z; let ch = G.i4.w;
    if (cx >= cw || cy >= ch) { return; }
    let x = cx + G.i4.x; let y = cy + G.i4.y;
    let i = u32(y * Wd() + x);
    let s = san4(srcAt(x, y));
    var r = s.x; var gg = s.y; var b = s.z;
    var amt = 1.0; var sgn = 1.0;
    if (G.i0.z != 0) {
        var m = san1(mapv[u32(y) * u32(G.i5.y) + u32(x)]);
        if (G.i0.w != 0) { m = 1.0 - m; }
        let dd = m - G.f0.x;
        sgn = select(1.0, -1.0, dd < 0.0);
        amt = clamp(abs(dd) * G.f0.y, 0.0, 1.0);
        if (G.f0.z != 1.0 && amt > 0.0) { amt = pow(amt, G.f0.z); }
    }
    var eL = 0.0;
    let gain = G.f1.x; let thr = G.f1.y;
    if (gain > 0.0 && amt > 0.0) {
        let L = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
        if (L > thr && L > 0.0) {
            let f = 1.0 + gain * amt * (L - thr) / L;
            eL += (f - 1.0) * L;
            r *= f; gg *= f; b *= f;
        }
    }
    let R = amt * G.f0.w;
    let anyE = atomicLoad(&flags[0]) != 0u && (G.i1.z == 0 || atomicLoad(&flags[1]) != 0u);
    if (G.i1.x != 0 && anyE && amt > 0.0) {
        let E = wk[plane(P_E) + i];
        if (E > 0.0) {
            let ps = G.f2.w;
            let sel = G.f2.y;
            let hasD = G.i1.w != 0;
            let area = 3.14159265358979 * max(0.0, R * R / ps - 1.0) / 16.0;
            var dw = 0.0;
            if (hasD) { dw = wk[plane(P_D) + i]; }
            let k = min(1.0e6, G.f1.z * E * ((1.0 - sel) * amt + select(0.0, sel * dw * area, hasD)));
            if (k > 0.0) {
                var er = k * s.x; var eg = k * s.y; var eb = k * s.z;
                let Le = 0.2126 * er + 0.7152 * eg + 0.0722 * eb;
                let hs = G.f2.z;
                if (hs != 1.0) {
                    er = max(0.0, Le + (er - Le) * hs);
                    eg = max(0.0, Le + (eg - Le) * hs);
                    eb = max(0.0, Le + (eb - Le) * hs);
                    let Ln = 0.2126 * er + 0.7152 * eg + 0.0722 * eb;
                    if (Ln > 1e-12) { let q = Le / Ln; er *= q; eg *= q; eb *= q; }
                }
                r += er; gg += eg; b += eb;
                eL += Le;
            }
        }
    }
    let o = u32((cy + G.i5.z) * cw + cx);
    let c = vec4f(r, gg, b, s.w);
    prgba[o] = c;
    paux[o] = vec2f(select(-R, R, sgn > 0.0), eL);
    maxBits(2u, c.x); maxBits(3u, c.y); maxBits(4u, c.z); maxBits(5u, c.w); maxBits(6u, eL);
}
