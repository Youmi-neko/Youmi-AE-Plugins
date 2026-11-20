// post.wgsl - lenscore postProcess stages on the GPU (ABI 1.3, src/post.rs): glow + vignetting (postRowA),
// halation, luminance aberration modes 0-2, highlight rolloff. Run on the device-resident blurred crop in the
// CPU's order; grain and laMode 3 / 4 stay on the CPU (see include/obgpu.h).
// Exactness: grid cell sums are sequential f32 additions in the CPU's order (y, then x) -> identical; the CPU's
// double expressions (box-line prefix sums, Gaussian sums, bilinear sampling, the per-pixel effect formulas) are
// f32 here (sums compensated) -> small continuous differences (~1e-7 relative).

struct PD {
    a: vec4i,    // W, H, kernel-specific ints
    b: vec4i,    // plane offsets (floats into wk)
    c: vec4i,    // more ints
    d: vec4f,    // floats
    e: vec4f,
    f: vec4f,
    g: vec4f,
    h: vec4f,
}

@group(0) @binding(0) var<uniform> D: PD;
@group(0) @binding(1) var<storage, read_write> img: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> el: array<f32>;
@group(0) @binding(3) var<storage, read> prgba: array<vec4f>;
@group(0) @binding(4) var<storage, read> paux: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> wk: array<f32>;

fn smooth01(t0: f32) -> f32 { let t = clamp(t0, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
fn lumOf(o: vec4f) -> f32 { return 0.2126 * o.x + 0.7152 * o.y + 0.0722 * o.z; }
fn twoSum(a: vec2f, v: f32) -> vec2f {  // (hi, lo) + v
    let s = a.x + v;
    let bb = s - a.x;
    let e = (a.x - (s - bb)) + (v - bb);
    return vec2f(s, a.y + e);
}

// ---- field = max(0, luminance of the image)   a: W, H | b.x: out plane
@compute @workgroup_size(256, 1, 1)
fn po_lum(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let i = u32(y * W + x);
    wk[u32(D.b.x) + i] = max(0.0, lumOf(img[i]));
}

// ---- blurFieldBox grid: one thread per cell, the cell's pixels summed in the CPU's order (y, then x)
// a: W, H, f, - | b.x: in plane, b.y: grid plane (2 floats per cell) | c: gx, gy, gw, gh
@compute @workgroup_size(64, 1, 1)
fn po_grid1(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let f = D.a.z;
    let gw = D.c.z; let gh = D.c.w;
    let c = i32(g.x + g.y * 2097152u);  // 2D dispatch: 32768 groups of 64 per row
    if (c >= gw * gh) { return; }
    let cx = c % gw; let cy = c / gw;
    let ya = max(0, cy * f - D.c.y); let yb = min(H, (cy + 1) * f - D.c.y);
    let xa = max(0, cx * f - D.c.x); let xb = min(W, (cx + 1) * f - D.c.x);
    var s = 0.0;
    var n = 0.0;
    for (var y = ya; y < yb; y++) {
        for (var x = xa; x < xb; x++) {
            s += wk[u32(D.b.x) + u32(y * W + x)];
            n += 1.0;
        }
    }
    let o = u32(D.b.y) + u32(c) * 2u;
    wk[o] = s;
    wk[o + 1u] = n;
}

// ---- buildGlow grid (r g b a eL weight) from the prepared crop   a: W, H, f | b.y: grid plane | c: gox, goy, gw, gh
@compute @workgroup_size(64, 1, 1)
fn po_grid6(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let f = D.a.z;
    let gw = D.c.z; let gh = D.c.w;
    let c = i32(g.x + g.y * 2097152u);  // 2D dispatch: 32768 groups of 64 per row
    if (c >= gw * gh) { return; }
    let cx = c % gw; let cy = c / gw;
    let ya = max(0, cy * f - D.c.y); let yb = min(H, (cy + 1) * f - D.c.y);
    let xa = max(0, cx * f - D.c.x); let xb = min(W, (cx + 1) * f - D.c.x);
    var q = vec4f(0.0);
    var e = 0.0;
    var n = 0.0;
    for (var y = ya; y < yb; y++) {
        for (var x = xa; x < xb; x++) {
            let i = u32(y * W + x);
            q += prgba[i];
            e += paux[i].y;
            n += 1.0;
        }
    }
    let o = u32(D.b.y) + u32(c) * 6u;
    wk[o] = q.x; wk[o + 1u] = q.y; wk[o + 2u] = q.z; wk[o + 3u] = q.w; wk[o + 4u] = e; wk[o + 5u] = n;
}

// ---- one box pass along the grid's rows (dir 0) or columns (dir 1), zero padding, / (2 rb + 1)
// (lenscore boxLine: double prefix sums; here a compensated window sum per element)
// a: gw, gh, rb, dir | b.x: in, b.y: out | c.x: channels
@compute @workgroup_size(64, 1, 1)
fn po_boxline(@builtin(global_invocation_id) g: vec3u) {
    let gw = D.a.x; let gh = D.a.y; let rb = D.a.z; let dir = D.a.w;
    let nc = D.c.x;
    let n = select(gw, gh, dir == 1);
    let lines = select(gh, gw, dir == 1);
    let i = i32(g.x); let ln = i32(g.y);
    if (i >= n || ln >= lines) { return; }
    // element k of line ln: cell index
    let a = max(0, i - rb); let b = min(n, i + rb + 1);
    let inv = f32(2 * rb + 1);
    for (var ch = 0; ch < nc; ch++) {
        var acc = vec2f(0.0);
        for (var k = a; k < b; k++) {
            let cell = select(ln * gw + k, k * gw + ln, dir == 1);
            acc = twoSum(acc, wk[u32(D.b.x) + u32(cell * nc + ch)]);
        }
        let cell = select(ln * gw + i, i * gw + ln, dir == 1);
        wk[u32(D.b.y) + u32(cell * nc + ch)] = (acc.x + acc.y) / inv;
    }
}

// ---- bilinear upsampling of a (value, coverage) grid, weight-normalised; with a second grid (rb1) the v2.2
// variance interpolation out = o0 + t (o1 - o0)
// a: W, H, f, two | b.x: grid0, b.y: grid1, b.z: out | c: gx, gy, gw, gh | d.x: t
fn upAt(gp: u32, x: i32, y: i32) -> f32 {
    let f = f32(D.a.z);
    let gw = D.c.z; let gh = D.c.w;
    var v = (f32(y + D.c.y) + 0.5) / f - 0.5;
    v = min(f32(gh - 1), max(0.0, v));
    let y0 = i32(v); let y1 = min(y0 + 1, gh - 1);
    let fy = v - f32(y0);
    var u = (f32(x + D.c.x) + 0.5) / f - 0.5;
    u = min(f32(gw - 1), max(0.0, u));
    let x0 = i32(u); let x1 = min(x0 + 1, gw - 1);
    let fx = u - f32(x0);
    var s: vec2f;
    for (var c = 0; c < 2; c++) {
        let g00 = wk[gp + u32((y0 * gw + x0) * 2 + c)];
        let g01 = wk[gp + u32((y0 * gw + x1) * 2 + c)];
        let g10 = wk[gp + u32((y1 * gw + x0) * 2 + c)];
        let g11 = wk[gp + u32((y1 * gw + x1) * 2 + c)];
        let a = g00 + (g01 - g00) * fx;
        let b = g10 + (g11 - g10) * fx;
        s[c] = a + (b - a) * fy;
    }
    return select(0.0, s.x / s.y, s.y > 1e-9);
}
@compute @workgroup_size(256, 1, 1)
fn po_up(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    var o = upAt(u32(D.b.x), x, y);
    if (D.a.w != 0) {
        let o1 = upAt(u32(D.b.y), x, y);
        o = o + D.d.x * (o1 - o);
    }
    wk[u32(D.b.z) + u32(y * W + x)] = o;
}

// ---- gaussField: exact separable Gaussian, radius r, normalised by the in-buffer weights
// a: W, H, r, dir | b.x: in, b.y: out, b.z: weights (2r+1)
@compute @workgroup_size(256, 1, 1)
fn po_gauss(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let r = D.a.z; let dir = D.a.w;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let p = select(x, y, dir == 1);
    let n = select(W, H, dir == 1);
    var s = vec2f(0.0);
    var nn = vec2f(0.0);
    for (var i = max(-r, -p); i <= min(r, n - 1 - p); i++) {
        let k = wk[u32(D.b.z) + u32(i + r)];
        let src = select(y * W + x + i, (y + i) * W + x, dir == 1);
        s = twoSum(s, k * wk[u32(D.b.x) + u32(src)]);
        nn = twoSum(nn, k);
    }
    wk[u32(D.b.y) + u32(y * W + x)] = (s.x + s.y) / (nn.x + nn.y);
}

// ---- postRowA: glow (bilinear of the blurred 6-channel grid) and vignetting; eL follows when roll
// a: W, H, glow f, roll | b.y: glow grid | c: gox, goy, gw, gh | d: glow, vig, fcx, fcy | e: halfDiag, pa
@compute @workgroup_size(256, 1, 1)
fn po_glowvig(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let i = u32(y * W + x);
    var o = img[i];
    var l = el[i];
    let roll = D.a.w != 0;
    let ga = D.d.x;
    if (ga > 0.0) {
        let f = f32(D.a.z);
        let gw = D.c.z; let gh = D.c.w;
        var u = (f32(x + D.c.x) + 0.5) / f - 0.5;
        var v = (f32(y + D.c.y) + 0.5) / f - 0.5;
        u = min(f32(gw - 1), max(0.0, u));
        v = min(f32(gh - 1), max(0.0, v));
        let x0 = i32(u); let y0 = i32(v);
        let x1 = min(x0 + 1, gw - 1); let y1 = min(y0 + 1, gh - 1);
        let fx = u - f32(x0); let fy = v - f32(y0);
        let gp = u32(D.b.y);
        var vv: array<f32, 6>;
        for (var k = 0; k < 6; k++) {
            let a = wk[gp + u32((y0 * gw + x0) * 6 + k)];
            let b = wk[gp + u32((y0 * gw + x1) * 6 + k)];
            let c = wk[gp + u32((y1 * gw + x0) * 6 + k)];
            let d = wk[gp + u32((y1 * gw + x1) * 6 + k)];
            let t = a + (b - a) * fx;
            let s = c + (d - c) * fx;
            vv[k] = t + (s - t) * fy;
        }
        let inv = select(0.0, 1.0 / vv[5], vv[5] > 1e-9);
        o = vec4f((1.0 - ga) * o.x + ga * vv[0] * inv, (1.0 - ga) * o.y + ga * vv[1] * inv,
                  (1.0 - ga) * o.z + ga * vv[2] * inv, (1.0 - ga) * o.w + ga * vv[3] * inv);
        if (roll) { l = (1.0 - ga) * l + ga * vv[4] * inv; }
    }
    let vig = D.d.y;
    if (vig > 0.0) {
        let hd = D.e.x;
        let dy = (f32(y) + 0.5 - D.d.w) / hd;
        let dx = (f32(x) + 0.5 - D.d.z) * D.e.y / hd;
        let rn2 = min(1.0, dx * dx + dy * dy);
        let fv = 1.0 - vig * rn2;
        o = vec4f(o.x * fv, o.y * fv, o.z * fv, o.w);
        if (roll) { l *= fv; }
    }
    img[i] = o;
    if (roll) { el[i] = l; }
}

// ---- halation: source mask   a: W, H | b.x: h (lum in, source out), b.y: scene mean, b.z: m (mask out)
// d: floorT, relK, soft
@compute @workgroup_size(256, 1, 1)
fn po_hal_mask(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let i = u32(y * W + x);
    let Y = wk[u32(D.b.x) + i];
    let thr = max(D.d.x, D.d.y * wk[u32(D.b.y) + i]);
    let t0 = thr * (1.0 - 0.6 * D.d.z);
    let t1 = thr * (1.0 + 0.6 * D.d.z) + 1e-4;
    let mk = smooth01((Y - t0) / (t1 - t0));
    wk[u32(D.b.z) + i] = mk;
    wk[u32(D.b.x) + i] = min(Y, 16.0 * thr) * mk;
}
// ---- halation: add   b.x: tight blur (or -1), b.y: wide blur (or -1), b.z: m | d: rim, amt | e: tint
@compute @workgroup_size(256, 1, 1)
fn po_hal_apply(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let i = u32(y * W + x);
    var o = img[i];
    let rim = D.d.x;
    var v = 0.0;
    if (rim > 0.0) { v += rim * 1.6 * wk[u32(D.b.x) + i]; }
    if (rim < 1.0) { v += (1.0 - rim) * wk[u32(D.b.y) + i]; }
    v *= D.d.y * (1.0 - 0.9 * wk[u32(D.b.z) + i]) * max(0.0, o.w);
    o = vec4f(o.x + D.e.x * v, o.y + D.e.y * v, o.z + D.e.z * v, o.w);
    img[i] = o;
}

// ---- luminance aberration modes 0-2   a: W, H, mode | b.x: Y, b.y: G (blurred Y)
// d: amt, thr, curve, hs | e: hi rgb
@compute @workgroup_size(256, 1, 1)
fn po_la(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let mode = D.a.z;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let i = u32(y * W + x);
    var o = img[i];
    let eps = 1.0 / 256.0;
    let yv = wk[u32(D.b.x) + i];
    let gv = wk[u32(D.b.y) + i];
    let d = log2((gv + eps) / (yv + eps));
    let amt = D.d.x; let thr = D.d.y; let curve = D.d.z; let hs = D.d.w;
    let lo = vec3f(1.0, 0.62, 0.16);
    let hi = D.e.xyz;
    let span = 3.0;
    if (mode != 1 && d > thr) {
        let t = min(1.0, max(0.0, (d - thr) / span));
        let s = pow(t, curve);
        let tc = hs * t;
        let c = lo + (hi - lo) * (1.0 - hs + tc);
        let e = amt * s * max(0.0, gv - yv) * max(0.0, o.w);
        o = vec4f(o.x + c.x * e, o.y + c.y * e, o.z + c.z * e, o.w);
    }
    if (mode != 0 && -d > thr) {
        let t = min(1.0, max(0.0, (-d - thr) / span));
        let s = pow(t, curve);
        let tc = hs * t;
        let c = lo + (hi - lo) * (1.0 - hs + tc);
        let lc = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
        let k = min(1.0, 0.6 * amt * s);
        if (lc > 1e-9 && k > 0.0) {
            let tgt = c / lc * yv;
            o = vec4f(o.x + k * (tgt.x - o.x), o.y + k * (tgt.y - o.y), o.z + k * (tgt.z - o.z), o.w);
        }
    }
    img[i] = o;
}

// ---- highlight rolloff (rolloffPixel / rolloffPixelV1)   a: W, H, v2 | d.x: knee
fn rollV1(o: vec4f, e: f32, knee: f32) -> vec4f {
    let a = o.w;
    if (!(e > 0.0) || !(a > 1e-6)) { return o; }
    let M = max(o.x, max(o.y, o.z));
    let kA = knee * a;
    if (M <= kA) { return o; }
    let span = a - kA;
    let Mt = kA + span * (1.0 - exp(-(M - kA) / span));
    let Lt = lumOf(o);
    if (!(Lt > 1e-12)) { return o; }
    let phi = min(1.0, e / Lt);
    var gg = min(1.0, phi / 0.2);
    gg = gg * gg * (3.0 - 2.0 * gg);
    if (gg <= 0.0) { return o; }
    let f = 1.0 - gg + gg * (Mt / M);
    let Mf = M * f;
    let comp = max(0.0, 1.0 - Mt / M);
    let dsat = 0.85 * gg * comp * comp;
    let v = o.xyz * f;
    let r = v + dsat * (vec3f(Mf) - v);
    return vec4f(r, o.w);
}
@compute @workgroup_size(256, 1, 1)
fn po_roll(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y;
    let x = i32(g.x); let y = i32(g.y);
    if (x >= W || y >= H) { return; }
    let i = u32(y * W + x);
    let o = img[i];
    let e = el[i];
    let knee = D.d.x;
    let a = o.w;
    if (!(e > 0.0) || !(a > 1e-6)) { return; }
    let M = max(o.x, max(o.y, o.z));
    let Lt = lumOf(o);
    if (D.a.z == 0 || !(Lt > 1e-12) || !(M > 0.0)) { img[i] = rollV1(o, e, knee); return; }
    let phi = min(1.0, e / Lt);
    let Ms = (1.0 - phi) * M;
    let t = smooth01((Ms - a) / (0.5 * a));
    if (t <= 0.0) { img[i] = rollV1(o, e, knee); return; }
    var v1 = o.xyz;
    if (t < 1.0) { v1 = rollV1(o, e, knee).xyz; }
    let span = a * (1.0 - knee);
    let kA = max(knee * a, Ms);
    var r = 1.0;
    if (M > kA && span > 0.0 && M - Ms > 1e-12) {
        let Mt = kA + span * (1.0 - exp(-(M - kA) / span));
        r = max(0.0, (Mt - Ms) / (M - Ms));
    }
    let v2c = o.xyz - (1.0 - r) * phi * o.xyz;
    let res = select(v1 + t * (v2c - v1), v2c, t >= 1.0);
    img[i] = vec4f(res, o.w);
}

// ======================= ABI 1.4: luminance aberration modes 3 / 4 =======================
// Extra plane offsets ride in D.f / D.g / D.h as raw bits (bitcast).
fn fi(v: f32) -> u32 { return u32(bitcast<i32>(v)); }
fn pix(g: vec3u) -> vec2i { return vec2i(i32(g.x), i32(g.y)); }

// ---- copy a plane   b.x in, b.y out
@compute @workgroup_size(256, 1, 1)
fn po_copy(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    wk[u32(D.b.y) + i] = wk[u32(D.b.x) + i];
}
// ---- out = a - b   b.x a, b.y b, b.z out
@compute @workgroup_size(256, 1, 1)
fn po_sub(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    wk[u32(D.b.z) + i] = wk[u32(D.b.x) + i] - wk[u32(D.b.y) + i];
}
// ---- exact clipped min / max along a line direction (lenscore runFilter / filterLines / filter2D)
// a: W, H, r, dir | b.x in, b.y out | c.x max
@compute @workgroup_size(256, 1, 1)
fn po_mm_line(@builtin(global_invocation_id) g: vec3u) {
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

// ---- Channel Drift: straight luminance Y and coverage A   b.x Y, b.y A
@compute @workgroup_size(256, 1, 1)
fn dr_ya(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let o = img[i];
    let a = min(1.0, max(0.0, o.w));
    wk[u32(D.b.y) + i] = a;
    wk[u32(D.b.x) + i] = select(0.0, max(0.0, lumOf(o) / a), a > 1e-4);
}
// ---- Y *= A   b.x Y, b.y A
@compute @workgroup_size(256, 1, 1)
fn dr_mul(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    wk[u32(D.b.x) + i] = wk[u32(D.b.x) + i] * wk[u32(D.b.y) + i];
}
// ---- Ls = Ls / As (covered), lg = log2(Ls + eps)   b.x Ls, b.y As, b.z lg
@compute @workgroup_size(256, 1, 1)
fn dr_lg(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let as_ = wk[u32(D.b.y) + i];
    let ls = select(0.0, wk[u32(D.b.x) + i] / as_, as_ > 1e-3);
    wk[u32(D.b.x) + i] = ls;
    wk[u32(D.b.z) + i] = plog2(ls + 1.0 / 256.0);
}
// log2 accurate to ~1e-7 relative everywhere (the native log2 may be off by 2^-21 ABSOLUTE near 1 on Vulkan
// (GLSL.std.450 precision), which the drift's gradient of log luminance turns into large relative errors)
fn plog2(x: f32) -> f32 {
    let fr = frexp(x);                 // x = m * 2^e, m in [0.5, 1)
    var m = fr.fract;
    var e = f32(fr.exp);
    if (m < 0.70710678) { m = m * 2.0; e = e - 1.0; }
    let r = (m - 1.0) / (m + 1.0);     // |r| <= 0.1716
    let r2 = r * r;
    let s = r * (2.0 + r2 * (0.6666667 + r2 * (0.4 + r2 * (0.2857143 + r2 * (0.2222222 + r2 * 0.1818182)))));
    return e + s * 1.4426950408889634;
}
fn lround(v: f32) -> i32 { return i32(select(-floor(-v + 0.5), floor(v + 0.5), v >= 0.0)); }
// ---- displacement field   b.x lg, b.y vx, b.z vy | d: thr, curve, pres, kstop | e: len, 2 sm
@compute @workgroup_size(256, 1, 1)
fn dr_v(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let x = p.x; let y = p.y;
    let i = u32(y * W + x);
    let lp = u32(D.b.x);
    let y0 = max(0, y - 1); let y1 = min(H - 1, y + 1);
    let x0 = max(0, x - 1); let x1 = min(W - 1, x + 1);
    let gx = (wk[lp + u32(y * W + x1)] - wk[lp + u32(y * W + x0)]) / f32(max(1, x1 - x0));
    let gy = (wk[lp + u32(y1 * W + x)] - wk[lp + u32(y0 * W + x)]) / f32(max(1, y1 - y0));
    let gm = sqrt(gx * gx + gy * gy);
    let c = gm * D.d.w;
    var vx = 0.0; var vy = 0.0;
    if (!(c <= D.d.x || gm <= 0.0)) {
        var s = pow(min(1.0, (c - D.d.x) / 2.0), D.d.y);
        let pres = D.d.z;
        if (pres > 0.0) {
            let ux = gx / gm; let uy = gy / gm;
            let bx = min(W - 1, max(0, lround(f32(x) + ux * D.e.y)));
            let by = min(H - 1, max(0, lround(f32(y) + uy * D.e.y)));
            let q = smooth01((wk[lp + u32(by * W + bx)] + 3.5) / 1.5);
            s *= (1.0 - pres) + pres * q;
        }
        let L = D.e.x * s / gm;
        vx = gx * L; vy = gy * L;
    }
    wk[u32(D.b.y) + i] = vx;
    wk[u32(D.b.z) + i] = vy;
}
// ---- |v|   b.x vx, b.y vy, b.z mag
@compute @workgroup_size(256, 1, 1)
fn dr_mag(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let vx = wk[u32(D.b.x) + i]; let vy = wk[u32(D.b.y) + i];
    wk[u32(D.b.z) + i] = sqrt(vx * vx + vy * vy);
}
// ---- v2.2 band spreading: direction from the blurred field, length = dilated |v| (capped)
// b.x vx, b.y vy, b.z bx, b.w by | c.x mag | e.z lenCap
@compute @workgroup_size(256, 1, 1)
fn dr_dir(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let bx = wk[u32(D.b.z) + i]; let by = wk[u32(D.b.w) + i];
    let mag = wk[u32(D.c.x) + i];
    let bm = sqrt(bx * bx + by * by);
    var vx = 0.0; var vy = 0.0;
    if (!(mag <= 0.0 || bm < 1e-9)) {
        let l = min(mag, D.e.z) / bm;
        vx = bx * l; vy = by * l;
    }
    wk[u32(D.b.x) + i] = vx;
    wk[u32(D.b.y) + i] = vy;
}
// ---- copy of the image into wk (4 floats per pixel)   b.x dst
@compute @workgroup_size(256, 1, 1)
fn po_imgcopy(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let o = img[i];
    let d = u32(D.b.x) + 4u * i;
    wk[d] = o.x; wk[d + 1u] = o.y; wk[d + 2u] = o.z; wk[d + 3u] = o.w;
}
fn srcS(sp: u32, fx0: f32, fy0: f32, c: u32) -> f32 {
    let W = D.a.x; let H = D.a.y;
    let fx = min(f32(W - 1), max(0.0, fx0));
    let fy = min(f32(H - 1), max(0.0, fy0));
    let x0 = i32(fx); let y0 = i32(fy);
    let x1 = min(x0 + 1, W - 1); let y1 = min(y0 + 1, H - 1);
    let ax = fx - f32(x0); let ay = fy - f32(y0);
    let s00 = wk[sp + 4u * u32(y0 * W + x0) + c]; let s01 = wk[sp + 4u * u32(y0 * W + x1) + c];
    let s10 = wk[sp + 4u * u32(y1 * W + x0) + c]; let s11 = wk[sp + 4u * u32(y1 * W + x1) + c];
    let a = s00 + (s01 - s00) * ax;
    let b = s10 + (s11 - s10) * ax;
    return a + (b - a) * ay;
}
// ---- the directional smear (vector blur of the straight colour, alpha-weighted)
// a: W, H, v2 | b.x vx, b.y vy, b.z src copy | d.xyz: k per channel
@compute @workgroup_size(256, 1, 1)
fn dr_smear(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let vx = wk[u32(D.b.x) + i]; let vy = wk[u32(D.b.y) + i];
    if (vx == 0.0 && vy == 0.0) { return; }
    var o = img[i];
    let ad = o.w;
    if (!(ad > 1e-6)) { return; }
    let sp = u32(D.b.z);
    o.x = smearCh(o.x, sp, p, vx, vy, D.d.x, 0u, ad);
    o.y = smearCh(o.y, sp, p, vx, vy, D.d.y, 1u, ad);
    o.z = smearCh(o.z, sp, p, vx, vy, D.d.z, 2u, ad);
    img[i] = o;
}
// (one channel; explicit per-channel calls: no dynamic vector indexing, FXC-friendly)
fn smearCh(cur: f32, sp: u32, p: vec2i, vx: f32, vy: f32, k: f32, c: u32, ad: f32) -> f32 {
    if (k == 0.0) { return cur; }
    let v2 = D.a.z != 0;
    let dx = k * vx; let dy = k * vy;
    let dl = sqrt(dx * dx + dy * dy);
    let n = select(min(16, max(1, i32(ceil(dl * 2.0)))), min(128, max(1, i32(ceil(dl)))), v2);
    var accC = 0.0; var accA = 0.0;
    for (var j = 0; j < n; j++) {
        let t = (f32(j) + 0.5) / f32(n);
        let sx = f32(p.x) + dx * t; let sy = f32(p.y) + dy * t;
        accC += srcS(sp, sx, sy, c);
        accA += srcS(sp, sx, sy, 3u);
    }
    if (accA > 1e-9 * f32(n)) { return max(0.0, accC / accA * ad); }
    return cur;
}

// ---- Contrast Rim: membership b = soft(closed level above the geometric mid level), nb = 1 - b
// b.x Bs, b.y Ds, b.z Yc, b.w b out | c.x nb out
@compute @workgroup_size(256, 1, 1)
fn rim_b(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let eps = 1.0 / 256.0;
    let mid = 0.5 * (log2(wk[u32(D.b.x) + i] + eps) + log2(wk[u32(D.b.y) + i] + eps));
    let v = smooth01((log2(wk[u32(D.b.z) + i] + eps) - mid) + 0.5);
    wk[u32(D.b.w) + i] = v;
    wk[u32(D.c.x) + i] = 1.0 - v;
}
// ---- channel x weight   a.z channel (0-2: image channel) | b.x weight plane, b.y out
@compute @workgroup_size(256, 1, 1)
fn rim_cw(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let o = img[i];
    let ch = D.a.z;
    let v = select(select(o.z, o.y, ch == 1), o.x, ch == 0);
    let w = wk[u32(D.b.x) + i];
    // a.w: 1 = max(0, channel) (fringe source), 0 = channel as is (bright-side colour)
    wk[u32(D.b.y) + i] = select(v, max(0.0, v), D.a.w != 0) * w;
}
fn qGate(bs: f32, bn: f32) -> f32 {
    let eps = 1.0 / 256.0;
    return smooth01(1.0 - (log2(bs + eps) - log2(bn + eps))) * smooth01((log2(bs + eps) + 3.5) / 1.5);
}
// ---- drawn rim   b: Bs, Ds, Bn, b | c: Fb, Fd, cb0, cb1 | f.x cb2, f.y wb (bits)
// a.z brightSide, a.w mixB on | d: amt, thr, curve, hs | e: ex, mixB, -, - | g.xyz hi
@compute @workgroup_size(256, 1, 1)
fn rim_final(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let eps = 1.0 / 256.0;
    let Bs = wk[u32(D.b.x) + i]; let Ds = wk[u32(D.b.y) + i];
    let c = log2((Bs + eps) / (Ds + eps));
    let thr = D.d.y;
    if (c <= thr) { return; }
    let t = min(1.0, (c - thr) / 2.0);
    let s = pow(t, D.d.z);
    let q = qGate(Bs, wk[u32(D.b.z) + i]);
    if (q <= 0.0) { return; }
    let bb = wk[u32(D.b.w) + i];
    let F0 = 0.16;
    let ex = D.e.x;
    var w = (1.0 - bb) * pow(min(1.0, wk[u32(D.c.x) + i] / F0), ex);
    if (D.a.z != 0) { w += bb * pow(min(1.0, wk[u32(D.c.y) + i] / F0), ex); }
    w *= q;
    let a = min(1.0, D.d.x * s * min(1.0, w));
    if (a <= 0.0) { return; }
    var o = img[i];
    let lo = vec3f(0.55, 0.10, 0.035);
    let hi = D.g.xyz;
    let u = (1.0 - D.d.w) + D.d.w * t;
    var col = lo + (hi - lo) * u;
    let l = lumOf(vec4f(col, 0.0));
    if (l > 1e-9) { col = col / l; }
    let mixB = D.e.y;
    if (D.a.w != 0) {
        let wb = wk[fi(D.f.y) + i];
        if (wb > 1e-6) {
            var wc = vec3f(max(0.0, wk[u32(D.c.z) + i] / wb), max(0.0, wk[u32(D.c.w) + i] / wb), max(0.0, wk[fi(D.f.x) + i] / wb)) * hi;
            let lw = lumOf(vec4f(wc, 0.0));
            if (lw > 1e-9) {
                wc = wc / lw;
                col = col + (wc - col) * mixB;
            }
        }
    }
    let L = 0.55 * Bs * (0.35 + 0.65 * t);
    let al = max(0.0, o.w);
    let tgt = col * L * al;
    o = vec4f(o.x + a * (tgt.x - o.x), o.y + a * (tgt.y - o.y), o.z + a * (tgt.z - o.z), o.w);
    img[i] = o;
}
// ---- fringe source weight m = b * min(1, Y / (Yc + 1e-6))^2   b.x Y, b.y Yc, b.z b, b.w m out
@compute @workgroup_size(256, 1, 1)
fn fr_m(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let r = min(1.0, wk[u32(D.b.x) + i] / (wk[u32(D.b.y) + i] + 1e-6));
    wk[u32(D.b.w) + i] = wk[u32(D.b.z) + i] * r * r;
}
// ---- fringe gate and contrast term   b: Bs, Ds, Bn, gate out | c.x tt out | d: thr, curve
@compute @workgroup_size(256, 1, 1)
fn fr_gate(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let eps = 1.0 / 256.0;
    let Bs = wk[u32(D.b.x) + i]; let Ds = wk[u32(D.b.y) + i];
    let c = log2((Bs + eps) / (Ds + eps));
    let t = min(1.0, max(0.0, (c - D.d.x) / 2.0));
    wk[u32(D.c.x) + i] = t;
    var gt = 0.0;
    if (t > 0.0) { gt = pow(t, D.d.y) * qGate(Bs, wk[u32(D.b.z) + i]); }
    wk[u32(D.b.w) + i] = gt;
}
// ---- fringe light of one channel   b: gate, b, cbs(ch), wm | c.x out
@compute @workgroup_size(256, 1, 1)
fn fr_f(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let wm = wk[u32(D.b.w) + i];
    wk[u32(D.c.x) + i] = select(0.0, wk[u32(D.b.x) + i] * wk[u32(D.b.y) + i] * max(0.0, wk[u32(D.b.z) + i] / wm), wm > 1e-6);
}
fn frAdd(oc: f32, colc: f32, posc: f32, dc: f32, lc: f32, lp: f32, mixB: f32, comp: f32, K: f32, al: f32) -> f32 {
    let tinted = select(0.0, colc / lc * lp, lc > 1e-9);
    var add = (1.0 - mixB) * tinted + mixB * posc;
    add += comp * min(0.0, dc);
    return max(0.0, oc + K * add * al);
}
// ---- fringe add   b: D0, D1, D2, tt | d: amt, hs, mixB, comp | g.xyz hi
@compute @workgroup_size(256, 1, 1)
fn fr_final(@builtin(global_invocation_id) g: vec3u) {
    let W = D.a.x; let H = D.a.y; let p = pix(g);
    if (p.x >= W || p.y >= H) { return; }
    let i = u32(p.y * W + p.x);
    let d = vec3f(wk[u32(D.b.x) + i], wk[u32(D.b.y) + i], wk[u32(D.b.z) + i]);
    if (d.x == 0.0 && d.y == 0.0 && d.z == 0.0) { return; }
    var o = img[i];
    let al = max(0.0, o.w);
    let pos = max(d, vec3f(0.0));
    let lp = 0.2126 * pos.x + 0.7152 * pos.y + 0.0722 * pos.z;
    let lo = vec3f(0.55, 0.12, 0.05);
    let hi = D.g.xyz;
    let u = (1.0 - D.d.y) + D.d.y * wk[u32(D.b.w) + i];
    let col = lo + (hi - lo) * u;
    let lc = 0.2126 * col.x + 0.7152 * col.y + 0.0722 * col.z;
    let K = 2.0 * D.d.x;
    let mixB = D.d.z; let comp = D.d.w;
    o.x = frAdd(o.x, col.x, pos.x, d.x, lc, lp, mixB, comp, K, al);
    o.y = frAdd(o.y, col.y, pos.y, d.y, lc, lp, mixB, comp, K, al);
    o.z = frAdd(o.z, col.z, pos.z, d.z, lc, lp, mixB, comp, K, al);
    img[i] = o;
}