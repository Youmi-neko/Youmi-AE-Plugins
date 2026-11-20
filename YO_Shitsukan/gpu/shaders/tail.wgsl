// Shitsukan 0.6.0.5: the painter's tail on the GPU (sk_paint.inc (5b) wet-in-wet, (5) density, (6) slivers, then
// sk_core.cpp's painter paper), arithmetic in the CPU's order - the same rules as lines.wgsl:
//  * every product is materialised (mul: an xor with a uniform zero), so nothing is fused into a multiply-add
//  * divisions are corrected to the correctly rounded quotient (cdiv); min / max / clamp written as the C++ ones
//    (std::max(a, b) = a < b ? b : a, ...) so even the sign of a zero is the CPU's
//  * f64 where the CPU uses double (the paper noise coordinates), products made opaque by adding a uniform zero
//  * exp: a double evaluation rounded to float, then a table of the inputs where the MSVC runtime's expf differs
//    (tools\gen_exp_table.cpp) - bit for bit the CPU's std::exp
struct TU {
    ps: vec4<f32>,    // x: the recursive filter's coefficient (per dispatch)
    n: vec4<u32>,      // W, H, S (plane stride, floats), zero
    m: vec4<i32>,      // working grid w, h, cell k, dDown
    e: vec4<i32>,      // flags, layer l, t, r
    e2: vec4<i32>,     // layer b, ox, oy, exp table entries
    f: array<vec4<f32>, 12>,
    g: array<vec4<i32>, 24>,  // 0..55 grid metadata, 56 prot, 57 eye, 58 gradient table offsets, 60..66 octave seeds, 72.. ink
};
@group(0) @binding(0) var<uniform> u: TU;
@group(0) @binding(1) var<storage, read> grids: array<f32>;
@group(0) @binding(2) var<storage, read_write> dw: array<f32>;
@group(0) @binding(3) var<storage, read_write> dm: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> aA: array<f32>;
@group(0) @binding(6) var<storage, read_write> pc: array<f32>;
@group(0) @binding(7) var<storage, read_write> aux: array<f32>;
@group(0) @binding(8) var<storage, read> etab: array<u32>;
@group(0) @binding(9) var<storage, read> d64: array<f64>;

// flags (u.e.x)
const F_WET: i32 = 1;
const F_GLAZE: i32 = 2;
const F_BLOOM: i32 = 4;
const F_DLINE: i32 = 8;
const F_MARGIN: i32 = 16;
const F_PA: i32 = 32;
const F_PTEX: i32 = 64;
const F_HDR: i32 = 128;
const F_PAPER: i32 = 256;  // gr > 0 || pp > 0: the paper height is needed
const F_NEEDH: i32 = 1024;  // the ink's pencil lines need it too
const F_INK: i32 = 2048;    // (ink pass flags in g[85])
// aux slots (planes of S floats): 0 alpha (lines), 1 -, 2 ev, 3 wet, 4 busy, 5 dl, 6 pW, 7 dLine, 8 paper tex, 9 pa
const X_ALPHA: u32 = 0u;
const X_EV: u32 = 2u;
const X_WET: u32 = 3u;
const X_BZ: u32 = 4u;
const X_DL: u32 = 5u;
const X_PW: u32 = 6u;
const X_DLINE: u32 = 7u;
const X_PTEX: u32 = 8u;
const X_PA: u32 = 9u;
// d64 layout (host: sk_core.cpp GpuTailD64)
const D_C: i32 = 0;      // exp: 1 / n!, n = 0..13
const D_P2: i32 = 14;    // 2^k, k = -160..160
const D_INVLN2: i32 = 335;
const D_LN2HI: i32 = 336;
const D_LN2LO: i32 = 337;
const D_ZERO: i32 = 338;
const D_ROTC: i32 = 339;
const D_ROTS: i32 = 340;
const D_FREQ: i32 = 341;  // 7 octave frequencies (bump 3, cell 2, und 2)
const D_KX: i32 = 348;    // 17.31 * o, o = 0..2
const D_KY: i32 = 351;    // 9.17 * o
const D_DS: i32 = 354;
const D_PAR: i32 = 355;
const D_BKFREQ: i32 = 356;  // the pencil / ink line noise fBm (3 octaves)

fn fp(k: i32) -> f32 { return u.f[k / 4][k % 4]; }
fn ip(k: i32) -> i32 { return u.g[k / 4][k % 4]; }
fn nc(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ u.n.w); }
// (the operands are made opaque too: with mul(0.06, 1.0 - y) the driver otherwise rewrites the expression)
fn mul(a: f32, b: f32) -> f32 { return nc(nc(a) * nc(b)); }
fn nc4(x: vec4<f32>) -> vec4<f32> { return bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) ^ vec4<u32>(u.n.w)); }
// (a and b made opaque first: with a literal divisor - Smooth(0.6, 1.0, x) - the driver folds and rewrites the
// division and the residual, and the result is not the correctly rounded one any more)
fn cdiv(a0: f32, b0: f32) -> f32 {
    let a = nc(a0);
    let b = nc(b0);
    let q0 = a / b;
    var best = q0;
    var bestR = abs(fma(-q0, b, a));
    let qb = bitcast<i32>(q0);
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
// the C++ std::max / std::min / Clampf, sign of zero included
fn fmax(a: f32, b: f32) -> f32 { return select(a, b, a < b); }
fn fmin(a: f32, b: f32) -> f32 { return select(a, b, b < a); }
fn clampf(v: f32, lo: f32, hi: f32) -> f32 { return select(select(v, hi, v > hi), lo, v < lo); }
// (e0 / e1 opaque: Smooth(e0, e0 + 0.012, x) - the paper margin - had (e0 + 0.012) - e0 simplified to 0.012)
fn smoothc(e0i: f32, e1i: f32, x: f32) -> f32 {
    let e0 = nc(e0i);
    let e1 = nc(e1i);
    let t = clampf(cdiv(nc(x) - e0, e1 - e0), 0.0, 1.0);
    return mul(mul(t, t), 3.0 - mul(2.0, t));
}
fn floorDiv(a: i32, b: i32) -> i32 { return select(-((-a + b - 1) / b), a / b, a >= 0); }

// ---- exp, bit for bit the CPU's
fn z64() -> f64 { return d64[D_ZERO]; }
// an f64 product the driver cannot fuse into the next addition: times a 1 it cannot see (a select on a uniform or
// an added stored zero were not enough: (a*b + z) - (c*d + z) came out fused)
fn m64(a: f64, b: f64) -> f64 { return (a * b) * d64[D_P2 + 160 + i32(u.n.w)]; }  // (x 2^0 from the table)  // (x 2^0 from the table: the rounded product cannot be fused)
fn floor64(x: f64) -> f64 {
    let t = f64(i32(x));  // (|x| < 2^31) truncation toward zero
    return select(t, t - 1.0lf, t > x);
}
fn exp_ms(x: f32) -> f32 {
    let xd = f64(x);
    let kd = floor64(m64(xd, d64[D_INVLN2]) + 0.5lf);
    let r = (xd - m64(kd, d64[D_LN2HI])) - m64(kd, d64[D_LN2LO]);
    var p = d64[D_C + 13];
    for (var n = 12; n >= 0; n--) { p = m64(p, r) + d64[D_C + n]; }
    var res = f32(m64(p, d64[D_P2 + i32(kd) + 160]));
    // the inputs where the runtime's expf is another float (sorted keys)
    let key = bitcast<u32>(x);
    var lo = 0; var hi = u.e2.w - 1;
    loop {
        if (lo > hi) { break; }
        let mid = (lo + hi) / 2;
        let k2 = etab[mid * 2];
        if (k2 == key) { res = bitcast<f32>(etab[mid * 2 + 1]); break; }
        if (k2 < key) { lo = mid + 1; } else { hi = mid - 1; }
    }
    return res;
}

// ---- sk_paint.inc NGrid::at (8 ints per grid at g[8 * gi]: gx0 gy0 gw gh step ox oy offset)
fn ngrid(gi: i32, x: i32, y: i32) -> f32 {
    let b = gi * 8;
    let gx0 = ip(b); let gy0 = ip(b + 1); let gw = ip(b + 2); let step = ip(b + 4);
    let lx = x + ip(b + 5); let ly = y + ip(b + 6); let off = ip(b + 7);
    let i = floorDiv(lx, step) - gx0; let j = floorDiv(ly, step) - gy0;
    let fx = cdiv(f32(lx - (gx0 + i) * step), f32(step));
    let fy = cdiv(f32(ly - (gy0 + j) * step), f32(step));
    let r0 = off + j * gw + i; let r1 = r0 + gw;
    let a = grids[r0] + mul(grids[r0 + 1] - grids[r0], fx);
    let bb = grids[r1] + mul(grids[r1 + 1] - grids[r1], fx);
    return a + mul(bb - a, fy);
}
// sk_util.h Bilerp, one channel, on a grid of the grids buffer
fn bilerp(off: i32, w2: i32, h2: i32, fxi: f32, fyi: f32) -> f32 {
    let fx = clampf(fxi, 0.0, f32(w2 - 1));
    let fy = clampf(fyi, 0.0, f32(h2 - 1));
    let x0 = min(w2 - 1, i32(fx)); let y0 = min(h2 - 1, i32(fy));
    let x1 = min(w2 - 1, x0 + 1); let y1 = min(h2 - 1, y0 + 1);
    let tx = fx - f32(x0); let ty = fy - f32(y0);
    let a00 = grids[off + y0 * w2 + x0]; let a10 = grids[off + y0 * w2 + x1];
    let a01 = grids[off + y1 * w2 + x0]; let a11 = grids[off + y1 * w2 + x1];
    let p = a00 + mul(a10 - a00, tx);
    let qq = a01 + mul(a11 - a01, tx);
    return p + mul(qq - p, ty);
}
// prAt: max(eye, 0.5 prot) of the working grid (g[56] prot offset, g[57] eye offset)
fn prAt(x: i32, y: i32) -> f32 {
    let kf = f32(u.m.z);
    let fx0 = cdiv(f32(x) + 0.5, kf) - 0.5;
    let fy0 = cdiv(f32(y) + 0.5, kf) - 0.5;
    let pr = bilerp(ip(56), u.m.x, u.m.y, fx0, fy0);
    let ey = bilerp(ip(57), u.m.x, u.m.y, fx0, fy0);
    return fmax(ey, mul(0.5, pr));
}
fn pix(g: vec3<u32>) -> i32 {
    if (g.x >= u.n.x || g.y >= u.n.y) { return -1; }
    return i32(g.y * u.n.x + g.x);
}
fn ax(slot: u32, i: i32) -> u32 { return slot * u.n.z + u32(i); }

// ---- (5b) wet-in-wet. q = (Dw a, a) -> blurred by RF4 (k_q_rows / k_q_cols)
@compute @workgroup_size(16, 16)
fn k_wetprep(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let a = aA[i];
    q[i] = vec4<f32>(mul(dw[i * 3], a), mul(dw[i * 3 + 1], a), mul(dw[i * 3 + 2], a), a);
}
@compute @workgroup_size(64)
fn k_q_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.n.y) { return; }
    let base = i32(g.x * u.n.x); let w = i32(u.n.x); let c = vec4<f32>(u.ps.x);
    for (var x = 1; x < w; x++) { let r = q[base + x]; q[base + x] = r + nc4(c * (q[base + x - 1] - r)); }
    for (var x = w - 2; x >= 0; x--) { let r = q[base + x]; q[base + x] = r + nc4(c * (q[base + x + 1] - r)); }
}
@compute @workgroup_size(64)
fn k_q_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.n.x) { return; }
    let x = i32(g.x); let w = i32(u.n.x); let h = i32(u.n.y); let c = vec4<f32>(u.ps.x);
    for (var y = 1; y < h; y++) { let i = y * w + x; let r = q[i]; q[i] = r + nc4(c * (q[i - w] - r)); }
    for (var y = h - 2; y >= 0; y--) { let i = y * w + x; let r = q[i]; q[i] = r + nc4(c * (q[i + w] - r)); }
}
// f: 0 wetA
@compute @workgroup_size(16, 16)
fn k_wet(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let y = i32(g.y);
    let wetA = fp(0);
    let t4 = q[i];
    var db: array<f32, 3>;
    for (var c = 0; c < 3; c++) { db[c] = select(dw[i * 3 + c], cdiv(t4[c], t4.w), t4.w > 1e-5); }
    let dmi = dm[i];
    let wi = aux[ax(X_WET, i)];
    let se = smoothc(0.004, 0.03, aux[ax(X_EV, i)]);
    let tm = smoothc(-0.05, 0.25, ngrid(3, x, y));
    let dbm = mul((db[0] + db[1]) + db[2], 1.0 / 3.0);
    let darkInv = smoothc(0.6, 1.0, fmax(dbm, dmi));
    let simV = 1.0 - mul(smoothc(0.06, 0.18, abs(dbm - dmi)), darkInv);
    let swA = smoothc(0.5, 1.0, wetA);
    let m = fmin(1.0, mul(mul(mul(wi, simV), se + mul(mul(1.0 - se, 0.6), swA)), 0.35 + mul(0.9, tm)));
    let band = mul(mul(mul(mul(mul(mul(mul(4.0, tm), 1.0 - tm), wi), simV), swA), 1.0 - smoothc(0.4, 0.8, dmi)), se);

    for (var c = 0; c < 3; c++) {
        let d0 = dw[i * 3 + c];
        var t = d0 + mul(m, db[c] - d0);
        t = t + mul(mul(0.12, band), fmax(db[c], d0));
        dw[i * 3 + c] = fmin(t, ((d0 + 0.08) + mul(0.1, band)) + mul(0.35, 1.0 - darkInv));
    }
    dm[i] = mul((dw[i * 3] + dw[i * 3 + 1]) + dw[i * 3 + 2], 1.0 / 3.0);
}

// ---- (5) density. q = (Dm a, a, 0, 0) -> RFPlain<2> (the same k_q passes, lanes z / w stay 0)
@compute @workgroup_size(16, 16)
fn k_rimprep(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let a = aA[i];
    q[i] = vec4<f32>(mul(dm[i], a), a, 0.0, 0.0);
}
fn db1(i: i32) -> f32 {
    let t = q[i];
    return select(dm[i], cdiv(t.x, t.y), t.y > 1e-5);
}
// f: 0 wetA, 1 rimA, 2 grav, 3 uneven, 4 charge, 5 glaze, 6 br, 7 gstr
@compute @workgroup_size(16, 16)
fn k_density(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let y = i32(g.y);
    let W = i32(u.n.x); let H = i32(u.n.y);
    let flags = u.e.x;
    let pr = prAt(x, y);
    let bz = aux[ax(X_BZ, i)];
    let dmn = dm[i];
    let wi = aux[ax(X_WET, i)];
    let dry = 1.0 - wi;
    let dk = 1.0 - mul(0.95, smoothc(0.5, 1.0, dmn));
    var k1 = 1.0 + mul(mul(mul(clampf(mul(1.6, ngrid(0, x, y)), -1.0, 1.0), fp(7)), 1.0 - mul(0.6, pr)), dk);
    let tide = fmax(0.0, dmn - db1(i));
    let jb = min(H - 1, y + u.m.w) * W + x;
    let below = select(dmn, dm[jb], aA[jb] > 0.5);
    let pool = mul(mul(mul(fp(2), 0.5), smoothc(0.02, 0.25, dmn - below)), dk);
    var add = mul(mul(mul(mul(fp(1), 0.9), tide), dry) + mul(mul(pool, dmn), 0.5), 1.0 - mul(0.6, pr));
    k1 = mul(k1, 1.0 + mul(mul(mul(mul(fp(3), 0.22), ngrid(1, x, y)), 1.0 - mul(0.7, pr)), dk));
    let cw = mul(mul(mul(mul(mul(fp(4), 0.35), ngrid(2, x, y)), dmn), 1.0 - mul(0.6, pr)), dk);
    var chg = array<f32, 3>(mul(-0.5, cw), mul(-0.5, cw), cw);
    var gl = 0.0;
    if ((flags & F_GLAZE) != 0) {
        let d = aux[ax(X_DL, i)];
        let inner = mul(0.22, smoothc(0.04, 0.06, d)) + mul(bz, mul(0.20, smoothc(0.10, 0.12, d)) + mul(0.18, smoothc(0.17, 0.19, d)));
        gl = mul(mul(mul(mul(fp(5), inner), 1.0 - mul(0.9, pr)), 1.0 - aux[ax(X_PW, i)]), dk);
    }
    if ((flags & F_BLOOM) != 0) {
        let nb = ngrid(4, x, y);
        let wh = mul(mul(mul(fp(6), smoothc(0.2, 0.5, cdiv(wi, fmax(0.05, fp(0))))), 1.0 - smoothc(0.3, 0.7, dmn)), 1.0 - pr);
        let inside = smoothc(0.36, 0.42, nb);
        let edge = mul(smoothc(0.32, 0.36, nb), 1.0 - smoothc(0.39, 0.43, nb));
        k1 = mul(k1, 1.0 - mul(mul(0.4, wh), inside));
        add = add + mul(mul(mul(0.3, wh), edge), dmn);
    }
    for (var c = 0; c < 3; c++) {
        let dn = dw[i * 3 + c];
        let share = select(1.0, cdiv(dn, dmn), dmn > 1e-4);
        dw[i * 3 + c] = fmax(0.0, (mul(dn, k1) + mul(add + mul(gl, fmax(0.25, dmn)), share)) + chg[c]);
    }
}

// ---- (6) unpainted slivers, the paper margin. f: 8 loose, 9 margin, 10 lw, 11 lh, 12 / 13 the dLine smooth range
@compute @workgroup_size(16, 16)
fn k_slivers(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let y = i32(g.y);
    let flags = u.e.x;
    let loose = fp(8);
    let pr = prAt(x, y);
    var cv = 1.0 - aux[ax(X_PW, i)];
    let gp = ngrid(5, x, y);
    let bz = aux[ax(X_BZ, i)];
    let g1 = mul(mul(mul(mul(loose, 1.0 - pr), 1.0 - aux[ax(X_WET, i)]), smoothc(0.30, 0.55, gp)), 1.0 - mul(0.7, bz));
    cv = mul(cv, 1.0 - mul(mul(mul(0.85, g1), smoothc(0.015, 0.05, aux[ax(X_EV, i)])), smoothc(0.0, 0.03, abs(dm[i] - db1(i)))));
    if ((flags & F_DLINE) != 0) {
        let gl2 = mul(mul(mul(loose, 1.0 - pr), smoothc(0.45, 0.7, -gp)), 1.0 - bz);
        cv = mul(cv, 1.0 - mul(gl2, 1.0 - smoothc(fp(12), fp(13), aux[ax(X_DLINE, i)] - 1.0)));
    }
    if ((flags & F_MARGIN) != 0) {
        let margin = fp(9); let lw = fp(10); let lh = fp(11);
        let ex = cdiv(fmin(f32(x - u.e.y), f32(u.e.w - 1 - x)), lw);
        let ey = cdiv(fmin(f32(y - u.e.z), f32(u.e2.x - 1 - y)), lh);
        let e = fmin(cdiv(mul(ex, lw), lh), ey) + mul(mul(0.03, margin), ngrid(6, x, y));
        let e0 = mul(0.07, margin);
        cv = mul(cv, smoothc(e0, e0 + 0.012, e));
    }
    for (var c = 0; c < 3; c++) { dw[i * 3 + c] = mul(dw[i * 3 + c], cv); }
}

// ---- sk_core.cpp painter: prot = pd Smooth(0.05, 0.2, RFPlain<1>(alpha, 4 ds)); pa in aux slot 9
@compute @workgroup_size(16, 16)
fn k_paprep(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    aux[ax(X_PA, i)] = select(0.0, aux[ax(X_ALPHA, i)], (u.e.x & F_PA) != 0);
}
@compute @workgroup_size(64)
fn k_pa_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.n.y) { return; }
    let base = X_PA * u.n.z + g.x * u.n.x; let w = u.n.x; let c = u.ps.x;
    for (var x = 1u; x < w; x++) { let r = aux[base + x]; aux[base + x] = r + mul(c, aux[base + x - 1u] - r); }
    for (var x = i32(w) - 2; x >= 0; x--) { let k = base + u32(x); let r = aux[k]; aux[k] = r + mul(c, aux[k + 1u] - r); }
}
@compute @workgroup_size(64)
fn k_pa_cols(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.n.x) { return; }
    let base = X_PA * u.n.z + g.x; let w = u.n.x; let h = u.n.y; let c = u.ps.x;
    for (var y = 1u; y < h; y++) { let k = base + y * w; let r = aux[k]; aux[k] = r + mul(c, aux[k - w] - r); }
    for (var y = i32(h) - 2; y >= 0; y--) { let k = base + u32(y) * w; let r = aux[k]; aux[k] = r + mul(c, aux[k + w] - r); }
}

// ---- sk_util.h Perlin / Fbm / PaperHeight (double coordinates, float values)
fn hash(xi: u32) -> u32 {
    var x = xi;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}
fn hash2(x: i32, y: i32, s: u32) -> u32 { return hash((u32(x) * 0x9E3779B1u) ^ hash((u32(y) * 0x85EBCA77u) ^ s)); }
fn fade(t: f32) -> f32 { return mul(mul(mul(t, t), t), mul(t, mul(t, 6.0) - 15.0) + 10.0); }
fn grad(h: u32, dx: f32, dy: f32) -> f32 {
    let go = ip(58);
    return mul(grids[go + i32(h)], dx) + mul(grids[go + 256 + i32(h)], dy);
}
fn perlin(x: f64, y: f64, s: u32) -> f32 {
    let fx0 = floor64(x); let fy0 = floor64(y);
    let ix = i32(fx0); let iy = i32(fy0);
    let fx = f32(x - fx0); let fy = f32(y - fy0);
    let n00 = grad(hash2(ix, iy, s) & 255u, fx, fy);
    let n10 = grad(hash2(ix + 1, iy, s) & 255u, fx - 1.0, fy);
    let n01 = grad(hash2(ix, iy + 1, s) & 255u, fx, fy - 1.0);
    let n11 = grad(hash2(ix + 1, iy + 1, s) & 255u, fx - 1.0, fy - 1.0);
    let uu = fade(fx); let vv = fade(fy);
    let a = n00 + mul(n10 - n00, uu);
    let b = n01 + mul(n11 - n01, uu);
    return mul(1.41, a + mul(b - a, vv));
}
// one Fbm from the host's tables: octave k's weight amp * fade in f[wb + k], frequency in d64[fb + k], seed in
// g[sb + k]; sqrt(norm) in f[sqk]
fn fbm(X: f64, Y: f64, wb: i32, fb: i32, sb: i32, cnt: i32, sqk: i32) -> f32 {
    var sum = 0.0;
    var x = X; var y = Y;
    let c = d64[D_ROTC]; let s = d64[D_ROTS];
    for (var k = 0; k < cnt; k++) {
        let wgt = fp(wb + k);
        let f = d64[fb + k];
        if (wgt > 0.0) {
            sum = sum + mul(wgt, perlin(m64(x, f) + d64[D_KX + k], m64(y, f) - d64[D_KY + k], bitcast<u32>(ip(sb + k))));
        }
        let nx = m64(c, x) - m64(s, y);
        let ny = m64(s, x) + m64(c, y);
        x = nx; y = ny;
    }
    return cdiv(sum, fp(sqk));
}
// f: 24 / 25 / 26 sqrt(norm) of bump / cell / undulation
fn paperHeight(gx: f64, gy: f64) -> f32 {
    let bump = fbm(gx, gy, 17, D_FREQ, 60, 3, 24);
    let cell = 1.0 - mul(2.0, abs(fbm(gx, gy, 20, D_FREQ + 3, 63, 2, 25)));
    let und = fbm(gx, gy, 22, D_FREQ + 5, 65, 2, 26);
    return clampf(((0.5 + mul(0.40, bump)) + mul(0.06, cell)) + mul(0.08, und), 0.0, 1.0);
}

// ---- the painter's paper: Dt = dw, P (pc) in / out; f: 14 gr, 15 pp, 16 pd
@compute @workgroup_size(16, 16)
fn k_paper(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let y = i32(g.y);
    let flags = u.e.x;
    let gr = fp(14); let pp = fp(15);
    let prot = mul(fp(16), smoothc(0.05, 0.2, aux[ax(X_PA, i)]));
    let c0 = pc[i * 3]; let c1 = pc[i * 3 + 1]; let c2 = pc[i * 3 + 2];
    // tend (part 2: from the fill before the paper)
    let mx = fmax(c0, fmax(c1, c2)); let mn = fmin(c0, fmin(c1, c2));
    let sat = cdiv(mx - mn, mx + 1e-4);
    let blue = smoothc(0.0, 0.15, c2 - fmax(c0, c1));
    let earth = mul(1.0 - smoothc(0.2, 0.6, sat), smoothc(0.0, 0.10, c0 - c2));
    let tend = clampf((0.2 + mul(0.8, blue)) + mul(0.5, earth), 0.0, 1.0);
    var d = array<f32, 3>(dw[i * 3], dw[i * 3 + 1], dw[i * 3 + 2]);
    let dmv = mul((d[0] + d[1]) + d[2], 1.0 / 3.0);
    var h = 0.5;
    if ((flags & (F_PAPER | F_NEEDH)) != 0) {
        if ((flags & F_PTEX) != 0) {
            h = clampf(aux[ax(X_PTEX, i)], 0.0, 1.0);
        } else {
            let gx = m64(f64(x + u.e2.y) / d64[D_DS], d64[D_PAR]);
            let gy = f64(y + u.e2.z) / d64[D_DS];
            h = paperHeight(gx, gy);
        }
    }
    if ((flags & F_NEEDH) != 0) { aux[ax(X_DL, i)] = h; }  // (the pencil lines read the paper height: aux slot 5)
    let valley = mul(0.5 - h, 2.0);
    let capD = mul(fmin(dmv, 0.55), 1.0 - 0.0);
    var addD = mul(capD, mul(mul(mul(mul(gr, tend), 0.24), valley), 1.0 - mul(0.85, prot)) + mul(mul(pp, 0.07), valley));
    if (pp > 0.0) {
        addD = addD - mul(mul(mul(mul(mul(dmv, pp), 0.3), smoothc(0.75, 0.92, h)), 1.0 - smoothc(0.12, 0.35, dmv)), 1.0 - prot);
    }
    let shade = 1.0 + mul(mul(pp, 0.04), -valley);
    var kp = array<f32, 3>(0.985, 0.975, 0.95);
    var cs = array<f32, 3>(c0, c1, c2);
    for (var c = 0; c < 3; c++) {
        let share = select(1.0, cdiv(d[c], dmv), dmv > 1e-4);
        var v = mul(mul(exp_ms(-fmax(0.0, d[c] + mul(addD, share))), shade), kp[c]);
        if ((flags & F_HDR) != 0) { v = v + fmax(0.0, cs[c] - 1.0); }
        v = v + mul(0.0, cs[c] - v);
        pc[i * 3 + c] = v;
    }
}

// self-test: exp_ms over s (aux slot 2 inputs) -> aux slot 3
@compute @workgroup_size(64)
fn k_exptest(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.n.x) { return; }
    aux[ax(X_WET, i32(g.x))] = exp_ms(aux[ax(X_EV, i32(g.x))]);
}

// ---- sk_core.cpp 5. ink (no thickness change): dl = delta, the pencil / ink restyle, the soft blur, back over B.
// dw = delta (rgb), pc = B in / out, q.xyz = dl. g: 72.. the line noise grid, 80..82 its fBm seeds, 83 its mode
// (0 none, 1 grid, 2 per pixel), 84 line style (0 as is, 1 pencil, 2 ink), 85 bit 0 inpainted, bit 1 far (aux 2)
// f: 32 dk, 33 wc, 34 kLi, 35..37 the line fBm weights, 38 its sqrt(norm)
@compute @workgroup_size(16, 16)
fn k_inkprep(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let x = i32(g.x); let y = i32(g.y);
    var dl = vec3<f32>(dw[i * 3], dw[i * 3 + 1], dw[i * 3 + 2]);
    let style = ip(84);
    if (style != 0) {
        let m = mul((dl.x + dl.y) + dl.z, 1.0 / 3.0);
        if (m > 0.0) {
            var bk = 0.0;
            if (ip(83) == 1) {
                bk = ngrid(9, x, y);
            } else {
                let gx = m64(f64(x + u.e2.y) / d64[D_DS], d64[D_PAR]);
                let gy = f64(y + u.e2.z) / d64[D_DS];
                bk = fbm(gx, gy, 35, D_BKFREQ, 80, 3, 38);
            }
            var k = 0.0;
            var tint = vec3<f32>(1.0, 1.0, 1.06);
            if (style == 1) {
                let keep = smoothc(-0.75, -0.3, bk);
                let hh = aux[ax(X_DL, i)];
                k = mul(mul(0.55, 0.35 + mul(0.65, keep)), 0.7 + mul(0.6, hh));
                tint = vec3<f32>(0.88, 1.0, 1.22);
            } else {
                k = mul(1.05, 0.85 + mul(0.3, smoothc(-0.5, 0.5, bk)));
            }
            let mk = mul(m, k);
            dl = vec3<f32>(mul(mk, tint.x), mul(mk, tint.y), mul(mk, tint.z));
        }
    }
    q[i] = vec4<f32>(dl, 0.0);
}
@compute @workgroup_size(16, 16)
fn k_ink(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = pix(g);
    if (i < 0) { return; }
    let fl = ip(85);
    let dk = fp(32);
    let m = 1.0 - mul(mul(0.85, fp(33)), aux[ax(X_BZ, i)]);
    var lf = 1.0;
    if ((fl & 2) != 0) { lf = fmax(0.0, 1.0 + mul(fp(34), aux[ax(X_EV, i)])); }
    let dl = q[i];
    for (var c = 0; c < 3; c++) {
        let de = dw[i * 3 + c];
        var d = mul(dl[c], dk);
        d = mul(de + mul(m, d - de), lf);
        if ((fl & 1) == 0) { d = d - de; }
        let b = pc[i * 3 + c];
        pc[i * 3 + c] = mul(b + 0.03, exp_ms(-d)) - 0.03;
    }
}