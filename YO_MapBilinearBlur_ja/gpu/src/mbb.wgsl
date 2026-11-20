// Map Bilinear Blur - GPU kernels. A line-by-line port of src/mbb_core.cpp (v0.3): same formulas, same
// clamping rules, same accumulation order; f64 wherever the CPU uses double (ladder, sample positions,
// hex lattice, cubic weights, dither coordinates). Differences come only from FMA contraction and tanh.
//
// Every image (levels, chains, lam / fr, outputs) lives in one storage buffer `mem`, described by `tab`.

struct Img { off: u32, x0: i32, y0: i32, w: i32, h: i32, extW: i32, extH: i32, ch: i32 };
struct Disp { a: vec4<i32>, b: vec4<i32>, c: vec4<i32>, d: vec4<i32> };
struct Glob {
    D: vec4<i32>, O: vec4<i32>, O1: vec4<i32>, R0: vec4<i32>,
    i0: vec4<i32>,  // interp, downSmooth, blend, antiRing
    i1: vec4<i32>,  // protect, clip, ditherSeed (bits), ditherFrame
    i2: vec4<i32>,  // sharpType, ring, hasAmount, top
    i3: vec4<i32>,  // Lp, anyFront, imgBase, imgAmount
    i4: vec4<i32>,  // imgLam, imgFr, imgOut1, imgOut
    i5: vec4<i32>,  // lamPyr built, anyFrontW, -, -
    f: array<f64, 8>,  // radius, ds, sharpAmount, sharpRadius, hex spacing h, sqrt(3), -, -
    lev: array<i32, 32>,
    vr: array<f64, 32>,
    pyrImg: array<i32, 32>,
    farImg: array<i32, 32>,
    nearImg: array<i32, 32>,
    nearHas: array<i32, 32>,
    bandBox: array<vec4<i32>, 32>,
    lamImg: array<i32, 32>,
};

@group(0) @binding(0) var<uniform> P: Disp;
@group(0) @binding(1) var<storage, read_write> mem: array<f32>;
@group(0) @binding(2) var<storage, read> tab: array<Img>;
@group(0) @binding(3) var<storage, read> G: Glob;
@group(0) @binding(4) var<storage, read_write> scan: array<atomic<i32>>;

const IN_NEAREST: i32 = 0;
const IN_BILINEAR: i32 = 1;
const IN_SMOOTHSTEP: i32 = 2;
const IN_BSPLINE: i32 = 3;
const IN_CATMULLROM: i32 = 5;
const IN_HEX: i32 = 7;

struct V5 { v: vec4<f32>, e: f32 };

fn v5zero() -> V5 { return V5(vec4<f32>(0.0), 0.0); }

// Lvl::at: clamped to the level extent, then to the stored range
fn atIdx(im: i32, cx0: i32, cy0: i32) -> u32 {
    let t = tab[im];
    var cx = min(t.extW - 1, max(0, cx0));
    var cy = min(t.extH - 1, max(0, cy0));
    cx = min(t.x0 + t.w - 1, max(t.x0, cx));
    cy = min(t.y0 + t.h - 1, max(t.y0, cy));
    return t.off + u32(((cy - t.y0) * t.w + (cx - t.x0)) * t.ch);
}
fn ld5(o: u32) -> V5 { return V5(vec4<f32>(mem[o], mem[o + 1u], mem[o + 2u], mem[o + 3u]), mem[o + 4u]); }
fn st5(o: u32, x: V5) { mem[o] = x.v.x; mem[o + 1u] = x.v.y; mem[o + 2u] = x.v.z; mem[o + 3u] = x.v.w; mem[o + 4u] = x.e; }
fn at5(im: i32, cx: i32, cy: i32) -> V5 { return ld5(atIdx(im, cx, cy)); }

// level-0 index (Engine::Idx0): base extent, then the base's stored range; lam / fr share the indexing
fn idx0(X0: i32, Y0: i32) -> i32 {
    let b = tab[G.i3.z];
    var X = min(b.extW - 1, max(0, X0));
    var Y = min(b.extH - 1, max(0, Y0));
    X = min(b.x0 + b.w - 1, max(b.x0, X));
    Y = min(b.y0 + b.h - 1, max(b.y0, Y));
    return (Y - b.y0) * b.w + (X - b.x0);
}
fn baseAt(i: i32) -> V5 { return ld5(tab[G.i3.z].off + u32(i * 5)); }
fn lamAt(i: i32) -> f32 { return mem[tab[G.i4.x].off + u32(i)]; }
fn frAt(i: i32) -> f32 { return mem[tab[G.i4.y].off + u32(i)]; }

fn gTop() -> i32 { return G.i2.w; }
fn lev(k: i32) -> i32 { return G.lev[k]; }
fn vr(k: i32) -> f64 { return G.vr[k]; }

// ------------------------------------------------------------------ ladder ---
struct KT { k: i32, t: f32 };
fn locate(v: f64) -> KT {
    let tp = gTop();
    if (!(v > 0.0lf) || tp == 0) { return KT(0, 0.0); }
    if (v >= vr(tp)) { return KT(tp, 0.0); }
    var i = 0;
    loop {
        if (!(i + 1 < tp && vr(i + 1) <= v)) { break; }
        i = i + 1;
    }
    return KT(i, f32((v - vr(i)) / (vr(i + 1) - vr(i))));
}
fn targetVar(a: f32) -> f64 {
    let sg = 0.5lf * (f64(abs(a)) * G.f[0]);
    return sg * sg;
}

// naga has no f64 math builtins: exact replacements (|x| < 2^31)
fn floor64(x: f64) -> f64 {
    let t = f64(i32(x));  // truncation toward zero
    return select(t, t - 1.0lf, x < t);
}
fn abs64(x: f64) -> f64 { return select(-x, x, x >= 0.0lf); }

fn smoothstepf(a: f32, b: f32, x: f32) -> f32 {
    var t = (x - a) / (b - a);
    t = min(1.0, max(0.0, t));
    return t * t * (3.0 - 2.0 * t);
}

fn hash(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Steps / Dither (Engine::BlendT)
fn blendT(t: f32, X: i32, Y: i32) -> f32 {
    let bl = G.i0.z;
    if (!(t > 0.0) || bl == 0) { return t; }
    if (bl == 1) { return select(0.0, 1.0, t >= 0.5); }
    let ds = G.f[1];
    let inv = select(1.0lf, 1.0lf / ds, ds > 0.0lf);
    let xf = i32(floor64(f64(X + G.D.x) * inv + 1e-6lf));
    let yf = i32(floor64(f64(Y + G.D.y) * inv + 1e-6lf));
    let seed = u32(G.i1.z);
    let fr = u32(G.i1.w);
    let h = hash((u32(xf) * 0x9E3779B1u) ^ hash(u32(yf) + 0x632BE5ABu) ^ hash(seed * 0x85EBCA77u + fr * 0xC2B2AE3Du));
    let r = f32(h >> 8u) * (1.0 / 16777216.0);
    return select(0.0, 1.0, r < t);
}

// ----------------------------------------------------------------- kernels ---
fn cubicBC(x0: f64, B: f64, C: f64) -> f64 {
    let x = abs64(x0);
    if (x < 1.0lf) {
        return ((12.0lf - 9.0lf * B - 6.0lf * C) * x * x * x + (-18.0lf + 12.0lf * B + 6.0lf * C) * x * x + (6.0lf - 2.0lf * B)) / 6.0lf;
    }
    if (x < 2.0lf) {
        return ((-B - 6.0lf * C) * x * x * x + (6.0lf * B + 30.0lf * C) * x * x + (-12.0lf * B - 48.0lf * C) * x + (8.0lf * B + 24.0lf * C)) / 6.0lf;
    }
    return 0.0lf;
}
struct W6 { w: array<f32, 6>, lo: i32, hi: i32, neg: bool };
fn weights(interp: i32, f: f32) -> W6 {
    var r: W6;
    for (var q = 0; q < 6; q = q + 1) { r.w[q] = 0.0; }
    r.neg = false;
    if (interp == IN_NEAREST) {
        r.lo = 0; r.hi = 1;
        r.w[0] = select(0.0, 1.0, f < 0.5);
        r.w[1] = 1.0 - r.w[0];
    } else if (interp == IN_BILINEAR) {
        r.lo = 0; r.hi = 1;
        r.w[0] = 1.0 - f;
        r.w[1] = f;
    } else if (interp == IN_SMOOTHSTEP) {
        r.lo = 0; r.hi = 1;
        let s = f * f * (3.0 - 2.0 * f);
        r.w[0] = 1.0 - s;
        r.w[1] = s;
    } else if (interp == IN_BSPLINE) {
        r.lo = -1; r.hi = 2;
        let x2 = f * f;
        let x3 = x2 * f;
        let a = 1.0 - f;
        r.w[0] = a * a * a * (1.0 / 6.0);
        r.w[1] = (3.0 * x3 - 6.0 * x2 + 4.0) * (1.0 / 6.0);
        r.w[2] = (-3.0 * x3 + 3.0 * x2 + 3.0 * f + 1.0) * (1.0 / 6.0);
        r.w[3] = x3 * (1.0 / 6.0);
    } else {  // Catmull-Rom (B = 0, C = 1/2)
        r.lo = -1; r.hi = 2; r.neg = true;
        for (var i = 0; i < 4; i = i + 1) { r.w[i] = f32(cubicBC(f64(i - 1) - f64(f), 0.0lf, 0.5lf)); }
    }
    return r;
}

// a tap of a sampled image. mode 0: plain image `im`; 1: level 0 with far weights; 2: level 0, near band j
fn tap(mode: i32, im: i32, j: i32, cx: i32, cy: i32) -> V5 {
    if (mode == 0) { return at5(im, cx, cy); }
    let i = idx0(cx, cy);
    let p = baseAt(i);
    if (mode == 1) {
        let w = 1.0 - frAt(i);
        return V5(w * p.v, w * p.e);
    }
    let l = lamAt(i);
    let k = i32(l);
    let t = l - f32(k);
    var b = 0.0;
    if (k == j) { b = 1.0 - t; } else if (k + 1 == j) { b = t; }
    let w = frAt(i) * b;
    return V5(w * p.v, p.e);
}

fn roundAway(x: f64) -> f64 {
    return select(-floor64(-x + 0.5lf), floor64(x + 0.5lf), x >= 0.0lf);
}

// SampleLevel
fn sampleLevel(interp: i32, antiRing: bool, L: i32, X: i32, Y: i32, mode: i32, im: i32, j: i32) -> V5 {
    let s = f64(1u << u32(L));
    let u = (f64(X) + 0.5lf) / s - 0.5lf;
    let v = (f64(Y) + 0.5lf) / s - 0.5lf;
    if (interp == IN_HEX) {
        let h = G.f[4];  // std::sqrt(2.0 / std::sqrt(3.0)) from the host (same IEEE value)
        let s3 = G.f[5];  // std::sqrt(3.0)
        let rr = v / (h * s3 / 2.0lf);
        let qq = u / h - rr / 2.0lf;
        var cq = roundAway(qq);
        var cr = roundAway(rr);
        let cs = roundAway(-qq - rr);
        let dq = abs64(cq - qq);
        let dr = abs64(cr - rr);
        let dsv = abs64(cs + qq + rr);
        if (dq > dr && dq > dsv) { cq = -cr - cs; } else if (dr > dsv) { cr = -cq - cs; }
        let uc = h * (cq + cr / 2.0lf);
        let vc = h * cr * s3 / 2.0lf;
        let iu = i32(floor64(uc));
        let iv = i32(floor64(vc));
        let fu = f32(uc - f64(iu));
        let fv = f32(vc - f64(iv));
        let wx = array<f32, 2>(1.0 - fu, fu);
        let wy = array<f32, 2>(1.0 - fv, fv);
        var o = v5zero();
        for (var jj = 0; jj < 2; jj = jj + 1) {
            for (var ii = 0; ii < 2; ii = ii + 1) {
                let q = tap(mode, im, j, iu + ii, iv + jj);
                let w = wx[ii] * wy[jj];
                o.v = o.v + w * q.v;
                o.e = o.e + w * q.e;
            }
        }
        return o;
    }
    let iu = i32(floor64(u));
    let iv = i32(floor64(v));
    let fu = f32(u - f64(iu));
    let fv = f32(v - f64(iv));
    let kx = weights(interp, fu);
    let ky = weights(interp, fv);
    let clampIt = kx.neg && antiRing;
    var mn = V5(vec4<f32>(3e38), 3e38);
    var mx = V5(vec4<f32>(-3e38), -3e38);
    var sum = v5zero();
    for (var jj = kx.lo; jj <= kx.hi; jj = jj + 1) {
        let wyj = ky.w[jj - kx.lo];
        var row = v5zero();
        for (var ii = kx.lo; ii <= kx.hi; ii = ii + 1) {
            let w = kx.w[ii - kx.lo];
            let inner = (ii == 0 || ii == 1) && (jj == 0 || jj == 1);
            if (w == 0.0 && !(clampIt && inner)) { continue; }
            let q = tap(mode, im, j, iu + ii, iv + jj);
            row.v = row.v + w * q.v;
            row.e = row.e + w * q.e;
            if (clampIt && inner) {
                mn.v = min(mn.v, q.v); mn.e = min(mn.e, q.e);
                mx.v = max(mx.v, q.v); mx.e = max(mx.e, q.e);
            }
        }
        if (wyj != 0.0) {
            sum.v = sum.v + wyj * row.v;
            sum.e = sum.e + wyj * row.e;
        }
    }
    if (clampIt) {
        sum.v = min(mx.v, max(mn.v, sum.v));
        sum.e = min(mx.e, max(mn.e, sum.e));
    }
    return sum;
}

fn gInterp() -> i32 { return G.i0.x; }
fn gAnti() -> bool { return G.i0.w != 0; }

// SampleIdx: unweighted ladder entry k
fn sampleIdx(k: i32, X: i32, Y: i32) -> V5 {
    if (k <= 0) { return baseAt(idx0(X, Y)); }
    let L = lev(k);
    return sampleLevel(gInterp(), gAnti(), L, X, Y, 0, G.pyrImg[L], 0);
}
fn farIdx(k: i32, X: i32, Y: i32) -> V5 {
    if (k <= 0) { return tap(1, 0, 0, X, Y); }
    let L = lev(k);
    if (L == 0) { return sampleLevel(gInterp(), gAnti(), 0, X, Y, 1, 0, 0); }
    return sampleLevel(gInterp(), gAnti(), L, X, Y, 0, G.farImg[k], 0);
}
fn nearIdx(j: i32, X: i32, Y: i32) -> V5 {
    if (j <= 0) { return tap(2, 0, j, X, Y); }
    let L = lev(j);
    if (L == 0) { return sampleLevel(gInterp(), gAnti(), 0, X, Y, 2, 0, j); }
    return sampleLevel(gInterp(), gAnti(), L, X, Y, 0, G.nearImg[j], 0);
}

fn plainV(v: f64, X: i32, Y: i32) -> vec4<f32> {
    let kt = locate(v);
    var k = kt.k;
    var t = blendT(kt.t, X, Y);
    if (t >= 1.0) { k = k + 1; t = 0.0; }
    if (k == 0 && t == 0.0) {
        let p = baseAt(idx0(X, Y));
        if (p.e == 1.0) { return p.v; }
    }
    var a = sampleIdx(k, X, Y);
    if (t > 0.0) {
        let b = sampleIdx(k + 1, X, Y);
        a.v = a.v + t * (b.v - a.v);
        a.e = a.e + t * (b.e - a.e);
    }
    if (a.e > 1e-12) {
        let inv = 1.0 / a.e;
        return a.v * inv;
    }
    return vec4<f32>(0.0);
}

const THETA: f32 = 0.15;

// FarVp, iterative: walk up while the weight is thin, then fold back down
fn farVp(j0: i32, X: i32, Y: i32) -> vec4<f32> {
    var Ss: array<V5, 32>;
    var thrs: array<f32, 32>;
    var n = 0;
    var j = j0;
    var res = vec4<f32>(0.0);
    var resolved = false;
    loop {
        let S = farIdx(j, X, Y);
        var Q = 1.0;
        if (G.i1.y != 0) { Q = sampleIdx(j, X, Y).e; }
        let thr = THETA * Q;
        if (S.e >= thr && S.e > 1e-12) {
            if (j == 0 && S.e == 1.0) { res = S.v; } else { let inv = 1.0 / S.e; res = S.v * inv; }
            resolved = true;
            break;
        }
        Ss[n] = S;
        thrs[n] = thr;
        n = n + 1;
        if (j >= gTop() || n >= 31) { break; }
        j = j + 1;
    }
    if (!resolved) {
        // nothing qualifies even at the top: the unweighted top level
        let u = sampleIdx(gTop(), X, Y);
        if (u.e > 1e-12) { res = u.v / u.e; } else { res = vec4<f32>(0.0); }
    }
    // fold: rgba = (S + (thr - max(0, S.q)) * next) / thr
    var m = n - 1;
    loop {
        if (m < 0) { break; }
        let S = Ss[m];
        let thr = thrs[m];
        if (!(thr > 1e-12)) {
            // rgba = next
        } else {
            let mm = thr - max(0.0, S.e);
            res = (S.v + mm * res) / thr;
        }
        m = m - 1;
    }
    return res;
}

fn farValue(v: f64, X: i32, Y: i32) -> vec4<f32> {
    let kt = locate(v);
    var k = kt.k;
    var t = blendT(kt.t, X, Y);
    if (t >= 1.0) { k = k + 1; t = 0.0; }
    var r = farVp(k, X, Y);
    if (t > 0.0) {
        let b = farVp(k + 1, X, Y);
        r = r + t * (b - r);
    }
    return r;
}

fn farLam(X: i32, Y: i32, ownLam: f32, f: f32) -> f32 {
    if (!(f > 0.0) || G.i5.x == 0) { return ownLam; }
    var bg = -1.0;
    var k = max(1, min(gTop(), i32(ownLam) + 1));
    loop {
        if (!(k <= gTop() && bg < 0.0)) { break; }
        let L = lev(k);
        if (L >= 1 && L <= G.i3.x) {
            let S = sampleLevel(IN_BSPLINE, false, L, X, Y, 0, G.lamImg[L], 0);
            if (S.e > 0.05) { bg = S.v.x / S.e; }
        }
        k = k + 1;
    }
    if (bg < 0.0) { bg = ownLam; }
    return (1.0 - f) * ownLam + f * bg;
}

struct NS { v: vec4<f32>, any: bool };
fn nearSum(X: i32, Y: i32) -> NS {
    var r = NS(vec4<f32>(0.0), false);
    if (G.i3.y == 0) { return r; }
    for (var j = 0; j <= gTop(); j = j + 1) {
        if (G.nearHas[j] == 0) { continue; }
        let bb = G.bandBox[j];
        if (X < bb.x || X >= bb.y || Y < bb.z || Y >= bb.w) { continue; }
        let S = nearIdx(j, X, Y);
        if (S.e > 1e-12 && (S.v.x != 0.0 || S.v.y != 0.0 || S.v.z != 0.0 || S.v.w != 0.0)) {
            let inv = 1.0 / S.e;
            r.v = r.v + S.v * inv;
            r.any = true;
        }
    }
    return r;
}

// ------------------------------------------------------------- entry points ---
const WG: u32 = 16u;

// lam / fr per R0 pixel + the scan (min ladder position, front presence, band presence and extent)
// scan: [0] anyFront, [1] min lam bits, [2..34) band present, [34..162) band boxes (x0, x1, y0, y1)
@compute @workgroup_size(16, 16)
fn lamfr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let R0 = G.R0;
    let w = R0.z - R0.x;
    let h = R0.w - R0.y;
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }
    let i = y * w + x;
    var a = 1.0;
    if (G.i2.z != 0) { a = mem[tab[G.i3.w].off + u32(i)]; }
    let kt = locate(targetVar(a));
    let l = f32(kt.k) + kt.t;
    mem[tab[G.i4.x].off + u32(i)] = l;
    var fr = 0.0;
    if (a < 0.0) {
        let dsInv = select(1.0lf, 1.0lf / G.f[1], G.f[1] > 0.0lf);
        let rFull = f32(f64(abs(a)) * G.f[0] * dsInv);
        fr = smoothstepf(0.5, 2.0, rFull);
    }
    mem[tab[G.i4.y].off + u32(i)] = fr;
    let q = baseAt(i).e;
    if (q > 0.0) {
        atomicMin(&scan[1], bitcast<i32>(l));
        if (fr > 0.0) { atomicMax(&scan[0], 1); }
    }
    if (fr > 0.0) {
        let k = i32(l);
        let t = l - f32(k);
        let b = tab[G.i3.z];
        let X = b.x0 + x;
        let Y = b.y0 + y;
        for (var jj = k; jj <= min(gTop(), k + 1); jj = jj + 1) {
            if (jj == k + 1 && !(t > 0.0)) { break; }
            atomicMax(&scan[2 + jj], 1);
            atomicMin(&scan[34 + jj * 4], X);
            atomicMax(&scan[35 + jj * 4], X + 1);
            atomicMin(&scan[36 + jj * 4], Y);
            atomicMax(&scan[37 + jj * 4], Y + 1);
        }
    }
}

// Down, horizontal pass. P.a = (mode, dst, src, aux), P.b = (k, j, smooth, tmp), P.c = (sy0, th, srcExtW, srcExtH)
// modes: 0 plain src image, 1 far weights from level 0 (entry k), 2 near band j from level 0, 3 lam pyramid
// from level 0, 4 coarse start (src = pyramid level, aux = agg level, entry k)
fn kb(sm: bool, m: i32) -> f32 {
    if (!sm) { return 0.5; }
    switch (m) {
        case 0, 5: { return 1.0 / 32.0; }
        case 1, 4: { return 5.0 / 32.0; }
        default: { return 10.0 / 32.0; }
    }
}
fn gk(k: i32, l: f32) -> f32 {
    let kk = f32(k);
    if (kk <= 1.0) { return 1.0; }
    return min(1.0, max(0.0, l - (kk - 2.0)));
}
fn baseClampIdx(x0: i32, y0: i32) -> i32 {
    let b = tab[G.i3.z];
    let x = min(b.x0 + b.w - 1, max(b.x0, x0));
    let y = min(b.y0 + b.h - 1, max(b.y0, y0));
    return (y - b.y0) * b.w + (x - b.x0);
}
fn srcTap(mode: i32, src: i32, aux: i32, k: i32, j: i32, x: i32, y: i32) -> V5 {
    if (mode == 0) { return at5(src, x, y); }
    if (mode == 4) {
        let p = at5(src, x, y);
        let ag = at5(aux, x, y);
        let w = (1.0 - ag.v.y) * gk(k, ag.v.x);
        return V5(w * p.v, w * p.e);
    }
    let i = baseClampIdx(x, y);
    let p = baseAt(i);
    if (mode == 1) {
        let w = (1.0 - frAt(i)) * gk(k, lamAt(i));
        return V5(w * p.v, w * p.e);
    }
    if (mode == 2) {
        let l = lamAt(i);
        let kk = i32(l);
        let t = l - f32(kk);
        var b = 0.0;
        if (kk == j) { b = 1.0 - t; } else if (kk + 1 == j) { b = t; }
        let wc = frAt(i) * b;
        return V5(wc * p.v, 1.0 * p.e);
    }
    // mode 3
    let w = (1.0 - frAt(i)) * p.e;
    return V5(vec4<f32>(w * lamAt(i), 0.0, 0.0, 0.0), w);
}

@compute @workgroup_size(16, 16)
fn down_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = tab[P.a.y];
    let tw = dst.w;
    let th = P.c.y;
    let i = i32(gid.x);
    let jr = i32(gid.y);
    if (i >= tw || jr >= th) { return; }
    let sm = P.b.z != 0;
    let taps = select(2, 6, sm);
    let off = select(0, -2, sm);
    let sy = P.c.x + jr;
    let cx = dst.x0 + i;
    var acc = v5zero();
    for (var m = 0; m < taps; m = m + 1) {
        let sx = min(P.c.z - 1, max(0, 2 * cx + off + m));
        let p = srcTap(P.a.x, P.a.z, P.a.w, P.b.x, P.b.y, sx, sy);
        let kk = kb(sm, m);
        acc.v = acc.v + kk * p.v;
        acc.e = acc.e + kk * p.e;
    }
    st5(tab[P.b.w].off + u32((jr * tw + i) * 5), acc);
}

// Down, vertical pass: P.c = (sy0, sy1, -, srcExtH)
@compute @workgroup_size(16, 16)
fn down_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = tab[P.a.y];
    let i = i32(gid.x);
    let jd = i32(gid.y);
    if (i >= dst.w || jd >= dst.h) { return; }
    let sm = P.b.z != 0;
    let taps = select(2, 6, sm);
    let off = select(0, -2, sm);
    let sy0 = P.c.x;
    let sy1 = P.c.y;
    let ext = P.c.w;
    let cy = dst.y0 + jd;
    let tw = dst.w;
    let toff = tab[P.b.w].off;
    var acc = v5zero();
    for (var m = 0; m < taps; m = m + 1) {
        let row = min(sy1 - 1, max(sy0, min(ext - 1, max(0, 2 * cy + off + m)))) - sy0;
        let p = ld5(toff + u32((row * tw + i) * 5));
        let kk = kb(sm, m);
        acc.v = acc.v + kk * p.v;
        acc.e = acc.e + kk * p.e;
    }
    st5(dst.off + u32((jd * dst.w + i) * 5), acc);
}

// agg level: min lam / max fr over each cell's downsample support. P.a = (L, dst, src, -), P.b.z smooth
@compute @workgroup_size(16, 16)
fn agg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = tab[P.a.y];
    let i = i32(gid.x);
    let jd = i32(gid.y);
    if (i >= dst.w || jd >= dst.h) { return; }
    let sm = P.b.z != 0;
    let dl = select(0, -2, sm);
    let dh = select(1, 3, sm);
    let cx = dst.x0 + i;
    let cy = dst.y0 + jd;
    var mn = 1e9;
    var mx = 0.0;
    for (var v = dl; v <= dh; v = v + 1) {
        for (var u = dl; u <= dh; u = u + 1) {
            var a: f32;
            var b: f32;
            if (P.a.x == 1) {
                let ii = idx0(2 * cx + u, 2 * cy + v);
                a = lamAt(ii);
                b = frAt(ii);
            } else {
                let p = at5(P.a.z, 2 * cx + u, 2 * cy + v);
                a = p.v.x;
                b = p.v.y;
            }
            mn = min(mn, a);
            mx = max(mx, b);
        }
    }
    let o = dst.off + u32((jd * dst.w + i) * 5);
    mem[o] = mn;
    mem[o + 1u] = mx;
    mem[o + 2u] = 0.0;
    mem[o + 3u] = 0.0;
    mem[o + 4u] = 0.0;
}

// Evaluate: one pixel of O1
@compute @workgroup_size(16, 16)
fn eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let O1 = G.O1;
    let w = O1.z - O1.x;
    let h = O1.w - O1.y;
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }
    let R0 = G.R0;
    let Y = O1.y + y - G.D.y;
    let X = O1.x + x - G.D.x;
    let ax = min(R0.z - 1, max(R0.x, O1.x + x)) - R0.x;
    let ay = min(R0.w - 1, max(R0.y, O1.y + y)) - R0.y;
    var am = 1.0;
    if (G.i2.z != 0) { am = mem[tab[G.i3.w].off + u32(ay * (R0.z - R0.x) + ax)]; }
    let v = targetVar(am);
    let unsharp = G.i2.x == 1 && G.f[2] > 0.0lf;
    let ss = 0.5lf * G.f[3];
    let sv = ss * ss;
    var res: vec4<f32>;
    var res2 = vec4<f32>(0.0);
    if (G.i1.x != 0) {
        let N = nearSum(X, Y);
        var vf = v;
        let ai = ay * (R0.z - R0.x) + ax;
        if (G.i5.y != 0 && frAt(ai) > 0.0) {
            let lf = farLam(X, Y, lamAt(ai), frAt(ai));
            let kk = min(gTop(), i32(lf));
            let tt = f64(lf - f32(kk));
            if (kk >= gTop()) { vf = vr(gTop()); } else { vf = vr(kk) + tt * (vr(kk + 1) - vr(kk)); }
        }
        res = farValue(vf, X, Y);
        if (unsharp) { res2 = farValue(vf + sv, X, Y); }
        if (N.any) {
            let occ = 1.0 - min(1.0, max(0.0, N.v.w));
            res = N.v + occ * res;
            if (unsharp) { res2 = N.v + occ * res2; }
        }
    } else {
        res = plainV(v, X, Y);
        if (unsharp) { res2 = plainV(v + sv, X, Y); }
    }
    if (unsharp) {
        let gain = f32(G.f[2]);
        let al = max(0.0, min(1.0, res.w));
        let lp = 0.08 * al + 1e-6;
        let ln = 0.25 * al + 1e-6;
        for (var c = 0; c < 3; c = c + 1) {
            let d = gain * (res[c] - res2[c]);
            let lim = select(ln, lp, d > 0.0);
            res[c] = res[c] + lim * tanh(d / lim);
        }
    }
    let o = tab[select(G.i4.w, G.i4.z, G.i2.y != 0)].off + u32((y * w + x) * 4);
    mem[o] = max(0.0, res.x);
    mem[o + 1u] = max(0.0, res.y);
    mem[o + 2u] = max(0.0, res.z);
    mem[o + 3u] = min(1.0, max(0.0, res.w));
}

// AE-style sharpen ring: O1 -> O (3x3, centre 1 + 8a, neighbours -a)
@compute @workgroup_size(16, 16)
fn sharpen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let O = G.O;
    let w = O.z - O.x;
    let h = O.w - O.y;
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (x >= w || y >= h) { return; }
    let a = f32(G.f[2]);
    let w1 = w + 2;
    let o1 = tab[G.i4.z].off;
    let c0 = o1 + u32(((y + 1) * w1 + (x + 1)) * 4);
    var sum = vec4<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let p = o1 + u32(((y + 1 + dy) * w1 + (x + 1 + dx)) * 4);
            sum = sum + vec4<f32>(mem[p], mem[p + 1u], mem[p + 2u], mem[p + 3u]);
        }
    }
    let c = vec4<f32>(mem[c0], mem[c0 + 1u], mem[c0 + 2u], mem[c0 + 3u]);
    let o = tab[G.i4.w].off + u32((y * w + x) * 4);
    var al = (1.0 + 8.0 * a) * c.w - a * sum.w;
    al = min(1.0, max(0.0, al));
    mem[o] = max(0.0, (1.0 + 8.0 * a) * c.x - a * sum.x);
    mem[o + 1u] = max(0.0, (1.0 + 8.0 * a) * c.y - a * sum.y);
    mem[o + 2u] = max(0.0, (1.0 + 8.0 * a) * c.z - a * sum.z);
    mem[o + 3u] = al;
}
