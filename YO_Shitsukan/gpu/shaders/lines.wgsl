// Shitsukan GPU probe: the line stage (sk_core.cpp "1. line analysis"), arithmetic in the CPU's order.
// 0.6.0.5: every plane is a range of one arena buffer (all bindings read_write: wgpu allows one buffer bound several
// times only with the same usage).
// Exactness rules (the masks bar / light / v are thresholds on alpha, so alpha must be bit-identical):
//  * every product is materialised (nc: an xor with a uniform zero the compiler cannot see through), so no
//    multiply-add is fused; additions / subtractions / multiplications are correctly rounded in Vulkan
//  * divisions are corrected to the correctly rounded quotient (cdiv)
//  * min / max are exact; sums of 0/1 counts are exact in any order
struct U {
    w: u32, h: u32, r: u32, mode: u32,
    f0: f32, f1: f32, f2: f32, zero: u32,
    // smoothstep ranges given by the host (a literal-only e1 - e0 would be folded at higher precision)
    k0: f32, k1: f32, k2: f32, k3: f32, k4: f32, k5: f32, k6: f32, k7: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> s0: array<f32>;
@group(0) @binding(2) var<storage, read_write> s1: array<f32>;
@group(0) @binding(3) var<storage, read_write> s2: array<f32>;
@group(0) @binding(4) var<storage, read_write> d0: array<f32>;
@group(0) @binding(5) var<storage, read_write> d1: array<f32>;
@group(0) @binding(6) var<storage, read_write> d2: array<f32>;

fn nc(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ u.zero); }
fn mul(a: f32, b: f32) -> f32 { return nc(a * b); }
fn nc4(x: vec4<f32>) -> vec4<f32> { return bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) ^ vec4<u32>(u.zero)); }

// correctly rounded a / b (b finite, non-zero; normal range): the candidate around the hardware quotient with the
// smallest exact residual a - c*b (fma: one rounding of an exact value), ties to even
fn cdiv(a: f32, b: f32) -> f32 {
    let q = a / b;
    var best = q;
    var bestR = abs(fma(-q, b, a));
    let qb = bitcast<i32>(q);
    for (var k = -2; k <= 2; k++) {
        if (k == 0) { continue; }
        let c = bitcast<f32>(qb + k);
        let rr = abs(fma(-c, b, a));
        if (rr < bestR || (rr == bestR && (bitcast<u32>(c) & 1u) == 0u)) {
            best = c;
            bestR = rr;
        }
    }
    return best;
}
fn smoothc(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp(cdiv(x - e0, e1 - e0), 0.0, 1.0);
    return mul(mul(t, t), 3.0 - mul(2.0, t));
}
fn luma(c0: f32, c1: f32, c2: f32) -> f32 {
    return (mul(0.2126, c0) + mul(0.7152, c1)) + mul(0.0722, c2);
}
fn pix(g: vec3<u32>) -> i32 {
    if (g.x >= u.w || g.y >= u.h) { return -1; }
    return i32(g.y * u.w + g.x);
}

// Y (d0) and the colour cue (d1) from C (s0, rgb) and A (s1); f0 = 0.5f / 255.f
@compute @workgroup_size(16, 16)
fn k_prep(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let c0 = s0[i * 3]; let c1 = s0[i * 3 + 1]; let c2 = s0[i * 3 + 2];
    let transp = s1[i] < u.f0;
    d0[i] = select(clamp(luma(c0, c1, c2), 0.0, 1.0), 1.0, transp);
    d1[i] = select(clamp(min(c0, min(c1, c2)), 0.0, 1.0), 1.0, transp);
}
// separable square min / max, radius r; mode 1 = max
@compute @workgroup_size(16, 16)
fn k_mm_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let row = i - x; let r = i32(u.r);
    var m = s0[i];
    for (var k = max(0, x - r); k <= min(i32(u.w) - 1, x + r); k++) {
        m = select(min(m, s0[row + k]), max(m, s0[row + k]), u.mode == 1u);
    }
    d0[i] = m;
}
@compute @workgroup_size(16, 16)
fn k_mm_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let y = i32(g.y); let x = i32(g.x); let r = i32(u.r); let w = i32(u.w);
    var m = s0[i];
    for (var k = max(0, y - r); k <= min(i32(u.h) - 1, y + r); k++) {
        m = select(min(m, s0[k * w + x]), max(m, s0[k * w + x]), u.mode == 1u);
    }
    d0[i] = m;
}
// top-hat coverage: s0 = signal, s1 = closing (dark) / opening (light), s2 = erosion (dark) / dilation (light);
// f0 = thr, mode 1 = dark
@compute @workgroup_size(16, 16)
fn k_tophat(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let s = s0[i]; let a2 = s1[i]; let a1 = s2[i];
    let dark = u.mode == 1u;
    let th = select(s - a2, a2 - s, dark);
    let K = select(a1 - a2, a2 - a1, dark);
    let gate = smoothc(mul(0.5, u.f0), u.f0, K);
    var o = 0.0;
    if (K > 1e-4 && th > 0.0) { o = mul(clamp(cdiv(th, K), 0.0, 1.0), gate); }
    d0[i] = o;
}
// alpha (d0) *= 1 - 0.45 Smooth(0.45, 0.85, core (s0))
@compute @workgroup_size(16, 16)
fn k_core(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    d0[i] = mul(d0[i], 1.0 - mul(0.45, smoothc(u.k0, u.k1, s0[i])));
}
// mode 0: colour cue  alpha = max(alpha, 0.85 aS);  mode 1: light lines  aS > alpha -> alpha = aS, light (d1) = 1
@compute @workgroup_size(16, 16)
fn k_merge(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let aS = s0[i];
    if (u.mode == 0u) {
        d0[i] = max(d0[i], mul(0.85, aS));
    } else {
        if (aS > d0[i]) { d0[i] = aS; d1[i] = 1.0; } else { d1[i] = 0.0; }
    }
}
// clutter: v (d0) = alpha (s0) > 0.3, row sums of v over +-r (d1)
@compute @workgroup_size(16, 16)
fn k_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    d0[i] = select(0.0, 1.0, s0[i] > 0.3);
}
@compute @workgroup_size(16, 16)
fn k_sum_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let row = i - x; let r = i32(u.r);
    var s = 0.0;
    for (var k = max(0, x - r); k <= min(i32(u.w) - 1, x + r); k++) { s += s0[row + k]; }
    d0[i] = s;
}
@compute @workgroup_size(16, 16)
fn k_sum_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let y = i32(g.y); let x = i32(g.x); let r = i32(u.r); let w = i32(u.w);
    var s = 0.0;
    for (var k = max(0, y - r); k <= min(i32(u.h) - 1, y + r); k++) { s += s0[k * w + x]; }
    d0[i] = s;
}
// recursive filter (RFPlain<1>), in place on d0, coefficient f0: one thread per row / per column
@compute @workgroup_size(64)
fn k_rf1_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.h) { return; }
    let base = i32(g.x * u.w); let w = i32(u.w); let c = u.f0;
    for (var x = 1; x < w; x++) { let r = d0[base + x]; d0[base + x] = r + mul(c, d0[base + x - 1] - r); }
    for (var x = w - 2; x >= 0; x--) { let r = d0[base + x]; d0[base + x] = r + mul(c, d0[base + x + 1] - r); }
}
@compute @workgroup_size(64)
fn k_rf1_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.w) { return; }
    let x = i32(g.x); let w = i32(u.w); let h = i32(u.h); let c = u.f0;
    for (var y = 1; y < h; y++) { let i = y * w + x; let r = d0[i]; d0[i] = r + mul(c, d0[i - w] - r); }
    for (var y = h - 2; y >= 0; y--) { let i = y * w + x; let r = d0[i]; d0[i] = r + mul(c, d0[i + w] - r); }
}
// alpha (d0) *= (1 - Smooth(0.30, 0.48, vb (s0))) * Smooth(0.4 side, 0.8 side, cnt (s1)); f0 = side; then the
// transparent pixels (A = s2 < f1) get no line
@compute @workgroup_size(16, 16)
fn k_clutter(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    var a = d0[i];
    if (u.mode == 1u) {
        let side = u.f0;
        a = mul(a, mul(1.0 - smoothc(u.k2, u.k3, s0[i]), smoothc(mul(0.4, side), mul(0.8, side), s1[i])));
    }
    if (s2[i] < u.f1) { a = 0.0; }
    d0[i] = a;
}
// bar (d1), beta (d2), the weighted fill for the in-paint: d0 = (C*wIn, wIn) as 4 floats; s0 = alpha, s1 = A, s2 = Y
// mode 1 = black walls
@compute @workgroup_size(16, 16)
fn k_bar(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let a = s0[i]; let A = s1[i];
    let b = a > 0.25 || (u.mode == 1u && A > 0.5 && s2[i] < 0.10);
    d1[i] = select(0.0, 1.0, b);
    let beta = smoothc(u.k4, u.k5, a);
    d2[i] = beta;
    let ob = 1.0 - beta;
    d0[i] = mul(mul(A, ob), ob);
}

@group(0) @binding(7) var<storage, read_write> q: array<vec4<f32>>;
// the weighted fill (SmoothNorm3): q = (C*w, w); s0 = C (rgb), s1 = w
@compute @workgroup_size(16, 16)
fn k_fill4(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let w = s1[i];
    q[i] = vec4<f32>(mul(s0[i * 3], w), mul(s0[i * 3 + 1], w), mul(s0[i * 3 + 2], w), w);
}
// RF4 without edges (constant coefficient f0), in place on q
@compute @workgroup_size(64)
fn k_rf4_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.h) { return; }
    let base = i32(g.x * u.w); let w = i32(u.w); let c = vec4<f32>(u.f0);
    for (var x = 1; x < w; x++) { let r = q[base + x]; q[base + x] = r + nc4(c * (q[base + x - 1] - r)); }
    for (var x = w - 2; x >= 0; x--) { let r = q[base + x]; q[base + x] = r + nc4(c * (q[base + x + 1] - r)); }
}
@compute @workgroup_size(64)
fn k_rf4_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.w) { return; }
    let x = i32(g.x); let w = i32(u.w); let h = i32(u.h); let c = vec4<f32>(u.f0);
    for (var y = 1; y < h; y++) { let i = y * w + x; let r = q[i]; q[i] = r + nc4(c * (q[i - w] - r)); }
    for (var y = h - 2; y >= 0; y--) { let i = y * w + x; let r = q[i]; q[i] = r + nc4(c * (q[i + w] - r)); }
}
// ink density and the in-painted fill: s0 = C, s1 = beta, s2 = light (0/1); q = smoothed (C*w, w)
// d0 = delta (rgb), d1 = C out (rgb); mode bit 0 = in-paint, bit 1 = light lines exist; f0 = ink eps
@compute @workgroup_size(16, 16)
fn k_delta(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let t = q[i];
    let beta = s1[i];
    let lt = (u.mode & 2u) != 0u && s2[i] > 0.5;
    let eps = u.f0;
    for (var c = 0; c < 3; c++) {
        let src = s0[i * 3 + c];
        var fh = src;
        if (t.w > 1e-5) { let inv = cdiv(1.0, t.w); fh = mul(t[c], inv); }
        var d = 0.0;
        if (beta > 0.0) {
            d = beta * log((fh + eps) / (src + eps));
            if (!lt && d < 0.0) { d = 0.0; } else if (lt && d > 0.0) { d = 0.0; }
        }
        d0[i * 3 + c] = d;
        var o = src;
        if ((u.mode & 1u) != 0u && d != 0.0) { o = (src + eps) * exp(d) - eps; }
        d1[i * 3 + c] = o;
    }
}

// self-test of the exact primitives: s0 = (a, b, x) triplets; d0 = (cdiv(a, b), smoothc(0.45, 0.85, x), luma(a, b, x), a * b + x unfused)
@compute @workgroup_size(64)
fn k_selftest(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = i32(g.x);
    if (g.x >= u.w) { return; }
    let a = s0[i * 3]; let b = s0[i * 3 + 1]; let x = s0[i * 3 + 2];
    d0[i * 4] = cdiv(a, b);
    d0[i * 4 + 1] = smoothc(u.k0, u.k1, x);
    d0[i * 4 + 2] = luma(a, b, x);
    d0[i * 4 + 3] = smoothc(u.f0, u.f1, x);
}

// the normalised in-paint fill Fh (SmoothNorm3's result, exact): s0 = C, q = smoothed (C*w, w); d0 = Fh (rgb)
@compute @workgroup_size(16, 16)
fn k_fh(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let t = q[i];
    if (t.w > 1e-5) {
        let inv = cdiv(1.0, t.w);
        d0[i * 3] = mul(t.x, inv); d0[i * 3 + 1] = mul(t.y, inv); d0[i * 3 + 2] = mul(t.z, inv);
    } else {
        d0[i * 3] = s0[i * 3]; d0[i * 3 + 1] = s0[i * 3 + 1]; d0[i * 3 + 2] = s0[i * 3 + 2];
    }
}
