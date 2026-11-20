// blur.wgsl - lenscore blur stage (span-delta scatter) on the GPU. See gpu/SEMANTICS.md.
//
// The host prepends a constant block (variant): SEG (segment width = workgroup size), NC (accumulator
// channels), FR (three kernels / chroma fringe), CAT (cat eye), ROLL (eL channel), RAW (coarse level:
// write un-normalised prefix sums).
//
// Accumulation: one workgroup = one output row segment [x0, x0+SEG) of one resolution level. It visits
// every splat whose kernel can reach the segment, computes that splat's band (row offset k) span for
// every ps / kernel / layer exactly like lenscore's doBlock, and adds the 4 deltas of each span as
// 64-bit fixed-point integers (two u32 workgroup atomics + carry) into per-channel delta bins; deltas
// left of the segment fold into bin 0 (they are part of every prefix in the segment), deltas right of it
// are dropped. Each span's integer deltas sum to exactly 0, so spans that end left of the segment
// contribute nothing. A workgroup scan then turns deltas into values. Integer addition is associative:
// the result does not depend on the order in which threads add (deterministic).

// ------------------------------------------------------------------ data ----

struct Prof {
    a: vec4i,            // nl, share, shareFrom, -
    s: vec4f,            // s2, s2out, s2in, -
    ls: array<vec4f, 8>,
    lb: array<vec4f, 8>,
}
struct Level {
    a: vec4i,            // f, w, h, gox
    b: vec4i,            // goy, maxReach, xr, -
}
struct Glob {
    i0: vec4i,           // W, H, nk, nTiers
    i1: vec4i,           // cat, catN, fringeShare, clampNeg
    i2: vec4i,           // roll, nLevels, maxReach, apN
    i3: vec4i,           // circN, apOff, circOff, catOff   (offsets into tabs, floats)
    f0: vec4f,           // ax, ay, xext, yext
    f1: vec4f,           // catK, catLim, catInv, levelBase
    f2: vec4f,           // fcx, fcy, halfDiag, pa
    f3: vec4f,           // apYmin, apYmax, apInvStep, apArea
    f4: vec4f,           // ciYmin, ciYmax, ciInvStep, ciArea
    ks: array<vec4f, 2>, // [side] kernel scale R, G, B
    sc: array<vec4f, 2>, // per accumulator slot: scale (power of two)
    isc: array<vec4f, 2>,// 1 / scale
    lv: array<Level, 8>,
    prof: array<Prof, 16>, // side * 8 + tier
    tl: vec4f,           // log2 of the frame maximum of R, G, B, eL (magnitude tiers)
    sc2: array<vec4f, 2>,  // scales of slots 8..15 (magnitude tiers >= 1)
    isc2: array<vec4f, 2>,
}
struct Disp {
    a: vec4i,            // level, y0 (first row, level rows), nRows, nSeg
    b: vec4i,            // splatRow0 (level row of spl row 0), splatRows, prepRow0 (fine row of prep row 0), prepRows
    c: vec4u,            // splatBase (elements), reachBase, gridBase (floats, this level's RAW output), outBase (pixels)
    d: vec4i,            // gridRow0 (level row of this level's grid row 0), outRow0 (fine row of out row 0), -, -
    cgBase: array<vec4u, 2>,  // fine ps: grid base of level l (floats)
    cgRow0: array<vec4i, 2>,  // fine ps: level row of grid row 0 of level l
}
struct Splat {
    c: vec4f,            // colour (premultiplied)
    g: vec4f,            // R (level px; < 0 identity), eL, ox, oy
    i0: vec4f,           // invD * m, ps 0, kernels 0..2 ; w = wb
    i1: vec4f,           // invD * m, ps 1 ; w = bitcast md: (reach+1) | tier << 16 | side << 20
}

@group(0) @binding(0) var<uniform> G: Glob;
@group(0) @binding(1) var<uniform> D: Disp;
@group(0) @binding(2) var<storage, read> tabs: array<f32>;
@group(0) @binding(3) var<storage, read> prgba: array<vec4f>;
@group(0) @binding(4) var<storage, read> paux: array<vec2f>;      // signed radius (side), eL
@group(0) @binding(5) var<storage, read_write> spl: array<Splat>;
@group(0) @binding(6) var<storage, read_write> reach: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> grid: array<f32>;
@group(0) @binding(8) var<storage, read_write> outp: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> outel: array<f32>;

// ------------------------------------------------------------- helpers ----

fn W() -> i32 { return G.i0.x; }
fn Hh() -> i32 { return G.i0.y; }
fn NK() -> i32 { return G.i0.z; }
fn nTiers() -> i32 { return G.i0.w; }
fn catOn() -> bool { return G.i1.x != 0; }
fn catN() -> i32 { return G.i1.y; }
fn nLevels() -> i32 { return G.i2.y; }

// Dynamic vector-component access through selects: FXC (DX12) cannot index a vector component with a
// run-time value without unrolling the enclosing loop (error X3511, or minutes of compile time).
fn comp4(v: vec4f, i: i32) -> f32 { return select(select(select(v.w, v.z, i == 2), v.y, i == 1), v.x, i == 0); }
fn comp4u(v: vec4u, i: i32) -> u32 { return select(select(select(v.w, v.z, i == 2), v.y, i == 1), v.x, i == 0); }
fn comp4i(v: vec4i, i: i32) -> i32 { return select(select(select(v.w, v.z, i == 2), v.y, i == 1), v.x, i == 0); }
fn comp3(v: vec3f, i: i32) -> f32 { return select(select(v.z, v.y, i == 1), v.x, i == 0); }

fn lsOf(p: i32, l: i32) -> f32 { return comp4(G.prof[p].ls[l >> 2], l & 3); }
fn lbOf(p: i32, l: i32) -> f32 { return comp4(G.prof[p].lb[l >> 2], l & 3); }
fn ksOf(side: i32, ch: i32) -> f32 { return comp4(G.ks[side], ch); }

// ShapeTable lookups, cancellation-free. The tables hold per node i (6 floats): IL as hi + lo, the slab
// increment dIL = IL[i+1] - IL[i], then the same for IR. A band integral IL(u1) - IL(u0) of two O(1) values
// is O(1/R): evaluated naively in f32 its absolute error (~1e-7) becomes a span-endpoint error growing
// like R^2 (1e-3 px at R 60). Here a lookup works in cell coordinates F = (u - ymin) * invStep: the
// fraction F - floor(F) is exact, hi parts of neighbouring nodes subtract exactly, and the band height
// du is taken from the same F values, so the span average is exact for the interval actually looked up.
struct Look { lh: f32, ll: f32, rh: f32, rl: f32, F: f32 }
fn lookT(u: f32, ymin: f32, invStep: f32, n: i32, base: u32) -> Look {
    let F = (u - ymin) * invStep;
    var i = i32(F);
    i = clamp(i, 0, n - 1);
    let fr = F - f32(i);
    let o = base + 6u * u32(i);
    var L: Look;
    L.lh = tabs[o];
    L.ll = tabs[o + 1u] + fr * tabs[o + 2u];
    L.rh = tabs[o + 3u];
    L.rl = tabs[o + 4u] + fr * tabs[o + 5u];
    L.F = F;
    return L;
}
fn lookAp(u: f32) -> Look { return lookT(u, G.f3.x, G.f3.z, G.i2.w, u32(G.i3.y)); }
fn lookCi(u: f32) -> Look { return lookT(u, G.f4.x, G.f4.z, G.i3.x, u32(G.i3.z)); }
// (IL(b) - IL(a), IR(b) - IR(a), du) with du = the cell-coordinate difference / invStep
fn bandInt(a: Look, b: Look, invStep: f32) -> vec3f {
    return vec3f((b.lh - a.lh) + (b.ll - a.ll), (b.rh - a.rh) + (b.rl - a.rl), (b.F - a.F) / invStep);
}fn catArea(ox: f32, oy: f32) -> f32 {
    let n = catN();
    let lim = G.f1.y;
    let inv = G.f1.z;
    let u = clamp((ox + lim) * inv, 0.0, f32(n) - 1.000001);
    let v = clamp((oy + lim) * inv, 0.0, f32(n) - 1.000001);
    let i = i32(u);
    let j = i32(v);
    let fu = u - f32(i);
    let fv = v - f32(j);
    let o = u32(G.i3.w) + u32(j * n + i);
    let r00 = tabs[o]; let r01 = tabs[o + 1u];
    let r10 = tabs[o + u32(n)]; let r11 = tabs[o + u32(n) + 1u];
    let a = r00 + (r01 - r00) * fu;
    let b = r10 + (r11 - r10) * fu;
    return a + (b - a) * fv;
}

// band k of a kernel with radii (RxS, RyS): (ok, h, xl, xr) relative to the kernel centre
struct Span { ok: bool, h: f32, xl: f32, xr: f32 }

fn apSpan(RxS: f32, RyS: f32, k: i32) -> Span {
    var s: Span;
    s.ok = false;
    let inv = 1.0 / RyS;
    let u0 = (f32(k) - 0.5) * inv;
    let u1 = (f32(k) + 0.5) * inv;
    let ymin = G.f3.x;
    let ymax = G.f3.y;
    let du0 = min(u1, ymax) - max(u0, ymin);
    if (!(du0 > 1e-12)) { return s; }
    let I = bandInt(lookAp(clamp(u0, ymin, ymax)), lookAp(clamp(u1, ymin, ymax)), G.f3.z);
    let du = I.z;
    if (!(du > 1e-12)) { return s; }
    let sc = RxS / du;
    s.xl = I.x * sc;
    s.xr = I.y * sc;
    s.h = du * RyS;
    s.ok = s.xr > s.xl;
    return s;
}

fn catSpan(RxS: f32, RyS: f32, k: i32, hx: f32, hy: f32) -> Span {
    var s: Span;
    s.ok = false;
    let inv = 1.0 / RyS;
    let u0 = (f32(k) - 0.5) * inv;
    let u1 = (f32(k) + 0.5) * inv;
    let lo = max(G.f3.x - hy, hy + G.f4.x);
    let hi = min(G.f3.y - hy, hy + G.f4.y);
    let du = min(u1, hi) - max(u0, lo);
    if (!(du > 1e-12)) { return s; }
    let ua = min(max(u0, lo), hi);
    let ub = min(max(u1, lo), hi);
    let A = bandInt(lookAp(ua + hy), lookAp(ub + hy), G.f3.z);
    let C = bandInt(lookCi(ua - hy), lookCi(ub - hy), G.f4.z);
    if (!(A.z > 1e-12) || !(C.z > 1e-12)) { return s; }
    let L = max(A.x / A.z - hx, hx + C.x / C.z);
    let R = min(A.y / A.z - hx, hx + C.y / C.z);
    if (R <= L) { return s; }
    s.xl = L * RxS;
    s.xr = R * RxS;
    s.h = du * RyS;
    s.ok = true;
    return s;
}

fn smooth01(t0: f32) -> f32 {
    let t = clamp(t0, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// levelOf: (l0, w1)
fn levelOf(R: f32) -> vec2f {
    let nl = nLevels();
    let base = G.f1.w;
    if (nl <= 1 || R <= base) { return vec2f(0.0, 0.0); }
    let lam = log2(R / base);
    let l = i32(lam);
    if (l >= nl - 1) { return vec2f(f32(nl - 1), 0.0); }
    let w = clamp((lam - f32(l) - 0.7) / 0.3, 0.0, 1.0);
    var w1 = 0.0;
    if (l + 1 < nl) { w1 = w * w * (3.0 - 2.0 * w); }
    return vec2f(f32(l), w1);
}

// finishSplat. R in level px, (px, py) centre in fine buffer px. m = mass (folded into invD).
fn finishSplat(c: vec4f, eL: f32, R: f32, sgn: f32, px: f32, py: f32, maxReach: i32, m: f32) -> Splat {
    let side = select(0, 1, sgn > 0.0);
    let nk = NK();
    var smax = ksOf(side, 0);
    for (var ch = 1; ch < nk; ch++) { smax = max(smax, ksOf(side, ch)); }
    let ax = G.f0.x;
    let ay = G.f0.y;
    var tier = 0;
    var wb = 0.0;
    var Rout = R;
    var rch = 0;
    let Rxe = R * ax * smax * G.f0.z;
    let Rye = R * ay * smax * G.f0.w;
    if (Rxe <= 0.5 && Rye <= 0.5) {
        Rout = -1.0;
        rch = 0;
    } else {
        rch = min(maxReach, max(0, i32(ceil(Rye - 0.5))));
        if (nTiers() > 1) {
            let tau = log2(max(R, 2.0) / 2.0);
            var t = i32(tau);
            var f = tau - f32(t);
            if (t >= nTiers() - 1) { t = nTiers() - 1; f = 0.0; }
            tier = t;
            wb = smooth01((f - 0.7) / 0.3);
        }
    }
    var ox = 0.0;
    var oy = 0.0;
    if (catOn()) {
        let dxp = px - G.f2.x;
        let dyp = py - G.f2.y;
        let len = sqrt(dxp * dxp + dyp * dyp);
        let pa = G.f2.w;
        var lenSq = len;
        if (pa != 1.0) { lenSq = sqrt(dxp * dxp * pa * pa + dyp * dyp); }
        if (len > 1e-9) {
            let dx = dxp / len / ax;
            let dy = dyp / len / ay;
            let dl = sqrt(dx * dx + dy * dy);
            let mm = 0.5 * G.f1.x * min(1.5, lenSq / G.f2.z);
            ox = mm * dx / dl;
            oy = mm * dy / dl;
        }
    }
    var inv0 = vec3f(0.0);
    var inv1 = vec3f(0.0);
    if (Rout > 0.0) {
        var anyOk = false;
        var A = G.f3.w;
        if (catOn() && catN() > 0) { A = catArea(ox, oy); }
        for (var ps = 0; ps < 2; ps++) {
            let wgt = select(1.0 - wb, wb, ps == 1);
            if (ps == 1 && wgt <= 0.0) { break; }
            let pi = side * 8 + tier + ps;
            let share = G.prof[pi].a.y != 0 && nk == 3;
            let nl = G.prof[pi].a.x;
            for (var ch = 0; ch < nk; ch++) {
                let ks = ksOf(side, ch);
                var sum = 0.0;
                if (!catOn() || catN() > 0) {
                    // analytic (A = aperture area, or the tabulated clipped area)
                    if (share) {
                        sum = A * (R * ax) * (R * ay) * (G.prof[pi].s.y * ks * ks + G.prof[pi].s.z);
                    } else {
                        sum = A * G.prof[pi].s.x * (R * ax * ks) * (R * ay * ks);
                    }
                } else {
                    // legacy profile + cat eye (Standard): exact sum of the band spans
                    let Rx = R * ax * ks;
                    let Ry = R * ay * ks;
                    for (var k = -rch; k <= rch; k++) {
                        for (var l = 0; l < nl; l++) {
                            let s = catSpan(Rx * lsOf(pi, l), Ry * lsOf(pi, l), k, ox, oy);
                            if (s.ok) { sum += lbOf(pi, l) * s.h * (s.xr - s.xl); }
                        }
                    }
                }
                var inv = 0.0;
                if (sum > 1e-20) { inv = wgt / sum * m; anyOk = true; }
                let mk = vec3f(select(0.0, 1.0, ch == 0), select(0.0, 1.0, ch == 1), select(0.0, 1.0, ch == 2));
                if (ps == 0) { inv0 += mk * inv; } else { inv1 += mk * inv; }
            }
        }
        if (catOn() && !anyOk) { Rout = -1.0; rch = 0; }
    }
    var sp: Splat;
    sp.c = c;
    sp.g = vec4f(Rout, eL, ox, oy);
    sp.i0 = vec4f(inv0, wb);
    let md = u32(rch + 1) | (u32(tier) << 16u) | (u32(side) << 20u);
    sp.i1 = vec4f(inv1, bitcast<f32>(md));
    return sp;
}

fn emptySplat() -> Splat {
    var sp: Splat;
    sp.c = vec4f(0.0);
    sp.g = vec4f(0.0);
    sp.i0 = vec4f(0.0);
    sp.i1 = vec4f(0.0, 0.0, 0.0, bitcast<f32>(0u));   // reach -1
    return sp;
}

// ----------------------------------------------------------- setup passes ----

// fine level: one thread per source pixel of the window rows [splatRow0, splatRow0 + splatRows)
@compute @workgroup_size(256, 1, 1)
fn setup_fine(@builtin(global_invocation_id) gid: vec3u) {
    let x = i32(gid.x);
    let r = i32(gid.y);
    if (x >= W() || r >= D.b.y) { return; }
    let y = D.b.x + r;                                   // fine buffer row
    let pi = u32(y - D.b.z) * u32(W()) + u32(x);         // prep window index
    let c = prgba[pi];
    let ax = paux[pi];
    let R = abs(ax.x);
    let sgn = select(1.0, -1.0, ax.x < 0.0);
    let eL = select(0.0, ax.y, G.i2.x != 0);
    var share = 1.0;
    if (nLevels() > 1) {
        let lw = levelOf(R);
        share = select(0.0, 1.0 - lw.y, lw.x == 0.0);
    }
    var sp: Splat;
    if (share <= 0.0) {
        sp = emptySplat();
    } else {
        sp = finishSplat(c, eL, R, sgn, f32(x) + 0.5, f32(y) + 0.5, G.i2.z, share);
    }
    spl[D.c.x + u32(r) * u32(W()) + u32(x)] = sp;
    let md = bitcast<u32>(sp.i1.w) & 0xFFFFu;
    atomicMax(&reach[D.c.y + u32(r)], md);
}

// coarse level l = D.a.x: one thread per block of the level rows [splatRow0, splatRow0 + splatRows)
@compute @workgroup_size(64, 1, 1)
fn setup_coarse(@builtin(global_invocation_id) gid: vec3u) {
    let l = D.a.x;
    let Lv = G.lv[l];
    let f = Lv.a.x;
    let LW = Lv.a.y;
    let gox = Lv.a.w;
    let goy = Lv.b.x;
    let bx = i32(gid.x);
    let r = i32(gid.y);
    if (bx >= LW || r >= D.b.y) { return; }
    let by = D.b.x + r;
    let y0 = max(0, by * f - goy);
    let y1 = min(Hh(), (by + 1) * f - goy);
    let x0 = max(0, bx * f - gox);
    let x1 = min(W(), (bx + 1) * f - gox);
    let base = G.f1.w;
    let rLo = 0.99 * base * exp2(f32(l) - 0.3);
    var rHi = 3.0e38;
    if (l < nLevels() - 1) { rHi = 1.01 * base * exp2(f32(l + 1)); }
    var m = 0.0;
    var c = vec4f(0.0);
    var eL = 0.0;
    var Rs = 0.0;
    var sg = 0.0;
    let roll = G.i2.x != 0;
    for (var y = y0; y < y1; y++) {
        let row = u32(y - D.b.z) * u32(W());
        for (var x = x0; x < x1; x++) {
            let q = paux[row + u32(x)];
            let R = abs(q.x);
            if (!(R > rLo && R < rHi)) { continue; }
            let lw = levelOf(R);
            let l0 = i32(lw.x);
            var w = 0.0;
            if (l0 == l) { w += 1.0 - lw.y; }
            if (l0 + 1 == l) { w += lw.y; }
            if (w <= 0.0) { continue; }
            m += w;
            c += w * prgba[row + u32(x)];
            if (roll) { eL += w * q.y; }
            Rs += w * R;
            sg += select(w, -w, q.x < 0.0);
        }
    }
    var sp: Splat;
    if (m < 1e-9) {
        sp = emptySplat();
    } else {
        let inv = 1.0 / m;
        let sgn = select(-1.0, 1.0, sg >= 0.0);
        let cx = (f32(bx) + 0.5) * f32(f) - f32(gox);
        let cy = (f32(by) + 0.5) * f32(f) - f32(goy);
        sp = finishSplat(c * inv, eL * inv, Rs * inv / f32(f), sgn, cx, cy, Lv.b.y, m);
    }
    spl[D.c.x + u32(r) * u32(LW) + u32(bx)] = sp;
    let md = bitcast<u32>(sp.i1.w) & 0xFFFFu;
    atomicMax(&reach[D.c.y + u32(r)], md);
}

// ------------------------------------------------------------ accumulation ----

var<workgroup> blo: array<atomic<u32>, SEG * NC>;
var<workgroup> bhi: array<atomic<u32>, SEG * NC>;

// signed 64-bit integers as vec2u (lo, hi), two's complement
fn add64(a: vec2u, b: vec2u) -> vec2u {
    let lo = a.x + b.x;
    let c = select(0u, 1u, lo < a.x);
    return vec2u(lo, a.y + b.y + c);
}
fn neg64(a: vec2u) -> vec2u {
    let lo = ~a.x + 1u;
    return vec2u(lo, ~a.y + select(0u, 1u, lo == 0u));
}
// round(d) as int64 (|d| < 2^62). |r| < 2^31: plain i32 sign-extended. Otherwise r has >= 8 trailing zero
// bits, hi = floor(r / 2^32) (signed) and lo = r - hi 2^32 in [0, 2^32) are both exact in f32.
fn f2i(d: f32) -> vec2u {
    let r = round(d);
    let hi = floor(r * 2.3283064365386963e-10);
    let big = vec2u(u32(r - hi * 4294967296.0), bitcast<u32>(i32(hi)));
    let small = vec2u(bitcast<u32>(i32(r)), select(0u, 0xFFFFFFFFu, r < 0.0));
    return select(big, small, abs(r) < 2147483648.0);
}
fn i2f(v: vec2u) -> f32 {
    let neg = (v.y & 0x80000000u) != 0u;
    let a = select(v, neg64(v), neg);
    let m = f32(a.y) * 4294967296.0 + f32(a.x);
    return select(m, -m, neg);
}
fn atom(slot: u32, bin: u32, v: vec2u) {
    if (bin >= u32(SEG)) { return; }
    let i = slot * u32(SEG) + bin;
    if (v.x != 0u) {
        let old = atomicAdd(&blo[i], v.x);
        let h = v.y + select(0u, 1u, old + v.x < old);
        if (h != 0u) { atomicAdd(&bhi[i], h); }
    } else if (v.y != 0u) {
        atomicAdd(&bhi[i], v.y);
    }
}

fn scaleOf(slot: u32) -> f32 {
    if (slot < 8u) { return comp4(G.sc[slot >> 2u], i32(slot & 3u)); }
    return comp4(G.sc2[(slot - 8u) >> 2u], i32(slot & 3u));
}
fn iscaleOf(slot: u32) -> f32 {
    if (slot < 8u) { return comp4(G.isc[slot >> 2u], i32(slot & 3u)); }
    return comp4(G.isc2[(slot - 8u) >> 2u], i32(slot & 3u));
}

// Magnitude tiers (HDR). One fixed-point scale per channel would make the quantum relative to the FRAME
// maximum (highlight boost can push prepared values to ~1e12), i.e. far too coarse for dim pixels. With K
// tiers each colour value of a splat goes to the accumulator of its magnitude class (tier t holds values
// within 2^(-20t)..2^(-20(t+1)) of the channel maximum, the last tier everything below), each with its own
// scale; the resolve adds the tiers. Weights and alpha (<= 1) have one tier.
struct Sl { r: u32, g: u32, b: u32, e: u32 }
fn tierOf(v: f32, ch: i32) -> u32 {
    if (K == 1) { return 0u; }
    let t = floor((comp4(G.tl, ch) - log2(max(v, 1e-30))) / 20.0);
    return u32(clamp(i32(t), 0, K - 1));
}
fn slotOf(base: u32, idx: u32, t: u32) -> u32 {
    if (t == 0u) { return base; }
    return u32(NCB) + (t - 1u) * u32(NCC) + idx;
}
fn slotsOf(c: vec4f, ce: f32) -> Sl {
    var s: Sl;
    s.r = slotOf(S_R, 0u, tierOf(c.x, 0));
    s.g = slotOf(S_G, 1u, tierOf(c.y, 1));
    s.b = slotOf(S_B, 2u, tierOf(c.z, 2));
    s.e = slotOf(S_EL, 3u, tierOf(ce, 3));
    return s;
}

// bins of one span: il, il+1, ir, ir+1 relative to the segment, folded into 0 on the left
struct Bins { b0: u32, b1: u32, b2: u32, b3: u32, fl: f32, fr: f32 }

// one value channel of a span: deltas {w - w fl, w fl, w fr - w, -w fr} * v * scale
fn dep(slot: u32, v: f32, w: f32, B: Bins) {
    if (v == 0.0) { return; }
    let s = v * scaleOf(slot);
    let q01 = f2i(w * s);
    let q0 = f2i((w - w * B.fl) * s);
    let q1 = add64(q01, neg64(q0));
    let q2 = f2i((w * B.fr - w) * s);
    let q3 = neg64(add64(q01, q2));
    atom(slot, B.b0, q0);
    atom(slot, B.b1, q1);
    atom(slot, B.b2, q2);
    atom(slot, B.b3, q3);
}

// slot layout. FR: R 0, WR 1, G 2, A 3, WG 4, B 5, WB 6, EL 7 ; else R 0, G 1, B 2, A 3, W 4, EL 5
const S_R: u32 = 0u;
const S_WR: u32 = select(4u, 1u, FR);
const S_G: u32 = select(1u, 2u, FR);
const S_A: u32 = 3u;
const S_WG: u32 = 4u;
const S_B: u32 = select(2u, 5u, FR);
const S_WB: u32 = select(4u, 6u, FR);
const S_EL: u32 = select(5u, 7u, FR);

// deposit one span of kernel `ch` (0..2 with FR; 0 = all channels otherwise). One loop over the
// (slot, value) pairs of the kernel: a single inlined copy of dep() keeps the shader small (FXC compile
// time on DX12 grows steeply with inlined code).
fn deposit(ch: i32, c: vec4f, ce: f32, w: f32, B: Bins, sl: Sl) {
    var slots = array<u32, 6>(sl.r, sl.g, sl.b, S_A, S_WG, sl.e);
    var vals = array<f32, 6>(c.x, c.y, c.z, c.w, 1.0, ce);
    var n = select(5u, 6u, ROLL);
    if (FR) {
        if (ch == 0) {
            slots[0] = sl.r; vals[0] = c.x; slots[1] = S_WR; vals[1] = 1.0; n = 2u;
        } else if (ch == 1) {
            slots[0] = sl.g; vals[0] = c.y; slots[1] = S_A; vals[1] = c.w; slots[2] = S_WG; vals[2] = 1.0;
            slots[3] = sl.e; vals[3] = ce; n = select(3u, 4u, ROLL);
        } else {
            slots[0] = sl.b; vals[0] = c.z; slots[1] = S_WB; vals[1] = 1.0; n = 2u;
        }
    }
    for (var i = 0u; i < n; i++) { dep(slots[i], vals[i], w, B); }
}

// span endpoints as integer pixel + fraction. Absolute f32 coordinates (x ~ 2000) would carry ~1e-4 px of
// rounding; the kernel offsets alone (|xl| <= R) carry ~1e-5 px, which matters next to very bright (HDR)
// sources whose edge pixels are only partly covered. Clipped to the level buffer [0, LW] like lenscore
// (L < 0 -> 0, R > W -> W). Returns false if the span is empty or has no effect on the segment.
fn binsOf(sx: i32, xl: f32, xr: f32, LW: i32, x0: i32, B: ptr<function, Bins>) -> bool {
    let a = 0.5 + xl;
    let b = 0.5 + xr;
    let fa = floor(a);
    let fb = floor(b);
    var il = sx + i32(fa);
    var fl = a - fa;
    var ir = sx + i32(fb);
    var fr = b - fb;
    if (il < 0) { il = 0; fl = 0.0; }
    if (ir > LW || (ir == LW && fr > 0.0)) { ir = LW; fr = 0.0; }
    if (!(ir > il || (ir == il && fr > fl))) { return false; }
    if (ir + 1 < x0 || il >= x0 + SEG) { return false; }
    (*B).fl = fl;
    (*B).fr = fr;
    let p0 = il - x0;
    let p2 = ir - x0;
    (*B).b0 = u32(max(p0, 0));
    (*B).b1 = u32(max(p0 + 1, 0));
    (*B).b2 = u32(max(p2, 0));
    (*B).b3 = u32(max(p2 + 1, 0));
    return true;
}

fn spanOf(RxS: f32, RyS: f32, k: i32, hx: f32, hy: f32) -> Span {
    if (CAT) { return catSpan(RxS, RyS, k, hx, hy); }
    return apSpan(RxS, RyS, k);
}

// one splat, band k, into the segment [x0, x0 + SEG) of a level LW px wide
fn splatBand(sp: Splat, sx: i32, k: i32, x0: i32, LW: i32) {
    let md = bitcast<u32>(sp.i1.w);
    let R = sp.g.x;
    let c = sp.c;
    let ce = select(0.0, sp.g.y, ROLL);
    if (R < 0.0) {   // identity: only k == 0
        if (k != 0) { return; }
        let p = sx - x0;
        if (p < 0) { return; }   // +c at sx and -c at sx+1 both fold left: no effect
        var B: Bins;
        B.b0 = u32(p); B.b1 = u32(p + 1); B.b2 = 0xFFFFFFFFu; B.b3 = 0xFFFFFFFFu;
        // identity: +v at p, -v at p+1 (weight 1, no mass)
        let st = slotsOf(c, ce);
        let sl = array<u32, 8>(st.r, st.g, st.b, S_A, S_WR, S_WG, S_WB, st.e);
        let vals = array<f32, 8>(c.x, c.y, c.z, c.w, 1.0, 1.0, 1.0, ce);
        for (var i = 0u; i < 8u; i++) {
            var useIt = false;
            if (FR) { useIt = i < 7u || ROLL; } else { useIt = i < 4u || i == 5u || (i == 7u && ROLL); }
            if (!useIt) { continue; }
            if (vals[i] == 0.0) { continue; }
            let q = f2i(vals[i] * scaleOf(sl[i]));
            atom(sl[i], B.b0, q);
            atom(sl[i], B.b1, neg64(q));
        }
        return;
    }
    let side = i32((md >> 20u) & 1u);
    let slt = slotsOf(c, ce);
    // the whole kernel (every pass / kernel / layer, cat eye included) lies within |x| <= R ax smax xext
    let smaxS = select(G.ks[side].x, max(G.ks[side].x, max(G.ks[side].y, G.ks[side].z)), FR);
    let ext = R * G.f0.x * smaxS * G.f0.z + 1.0;
    let cxe = f32(sx) + 0.5;
    if (cxe + ext < f32(x0) || cxe - ext > f32(x0 + SEG)) { return; }
    let tier = i32((md >> 16u) & 15u);
    let wb = sp.i0.w;
    let hx = sp.g.z;
    let hy = sp.g.w;
    let ax = G.f0.x;
    let ay = G.f0.y;
    let nk = select(1, 3, FR);
    for (var ps = 0; ps < 2; ps++) {
        let wgt = select(1.0 - wb, wb, ps == 1);
        if (ps == 1 && wgt <= 0.0) { break; }
        let pi = side * 8 + tier + ps;
        let share = FR && G.prof[pi].a.y != 0;
        let nl = G.prof[pi].a.x;
        let sf = G.prof[pi].a.z;
        let invs = select(sp.i0.xyz, sp.i1.xyz, ps == 1);
        // jobs: each kernel's own layers (with fringe sharing (Fast): only the outer ones), then, with
        // sharing, the inner layers once with neutral geometry and each kernel's own normaliser
        let nJobs = nk + select(0, 1, share);
        for (var j = 0; j < nJobs; j++) {
            let inner = share && j == nk;
            let ch = select(j, 0, inner);
            let ks = select(ksOf(side, ch), 1.0, inner);
            let Rx = R * ax * ks;
            let Ry = R * ay * ks;
            let lA = select(0, sf, inner);
            let lB = select(select(nl, sf, share), nl, inner);
            let kA = select(ch, 0, inner);
            let kB = select(ch + 1, 3, inner);
            for (var l = lA; l < lB; l++) {
                let lsl = lsOf(pi, l);
                let s = spanOf(Rx * lsl, Ry * lsl, k, hx, hy);
                var B: Bins;
                let inSeg = s.ok && binsOf(sx, s.xl, s.xr, LW, x0, &B);
                if (!inSeg) {
                    // not ok, clipped to nothing, or no effect on this segment: without cat eye the inner
                    // layers are subsets of this one (same for them) - lenscore's "alive" rule
                    if (!CAT) { break; }
                    continue;
                }
                let wl = lbOf(pi, l) * s.h;
                for (var kc = kA; kc < kB; kc++) { deposit(kc, c, ce, wl * comp3(invs, kc), B, slt); }
            }
        }
    }
}

fn binLoad(slot: u32, t: u32) -> vec2u {
    let i = slot * u32(SEG) + t;
    return vec2u(atomicLoad(&blo[i]), atomicLoad(&bhi[i]));
}
fn binStore(slot: u32, t: u32, v: vec2u) {
    let i = slot * u32(SEG) + t;
    atomicStore(&blo[i], v.x);
    atomicStore(&bhi[i], v.y);
}

// bilinear density of coarse level l at fine pixel (x, y), all NC slots
fn addCoarse(x: i32, y: i32, v: ptr<function, array<f32, NC>>) {
    for (var l = 1; l < nLevels(); l++) {
        let Lv = G.lv[l];
        let f = f32(Lv.a.x);
        let LW = Lv.a.y;
        let LH = Lv.a.z;
        let u = clamp((f32(x) + 0.5 + f32(Lv.a.w)) / f - 0.5, 0.0, f32(LW - 1));
        let w = clamp((f32(y) + 0.5 + f32(Lv.b.x)) / f - 0.5, 0.0, f32(LH - 1));
        let x0 = i32(u);
        let y0 = i32(w);
        let x1 = min(x0 + 1, LW - 1);
        let y1 = min(y0 + 1, LH - 1);
        let fx = u - f32(x0);
        let fy = w - f32(y0);
        let s = 1.0 / (f * f);
        let w00 = (1.0 - fx) * (1.0 - fy) * s;
        let w10 = fx * (1.0 - fy) * s;
        let w01 = (1.0 - fx) * fy * s;
        let w11 = fx * fy * s;
        let gb = comp4u(D.cgBase[l >> 2], l & 3);
        let r0 = comp4i(D.cgRow0[l >> 2], l & 3);
        let ia = gb + (u32(y0 - r0) * u32(LW) + u32(x0)) * u32(NC);
        let ib = gb + (u32(y0 - r0) * u32(LW) + u32(x1)) * u32(NC);
        let ic = gb + (u32(y1 - r0) * u32(LW) + u32(x0)) * u32(NC);
        let id = gb + (u32(y1 - r0) * u32(LW) + u32(x1)) * u32(NC);
        for (var k = 0u; k < u32(NC); k++) {
            (*v)[k] += w00 * grid[ia + k] + w10 * grid[ib + k] + w01 * grid[ic + k] + w11 * grid[id + k];
        }
    }
}

@compute @workgroup_size(SEG, 1, 1)
fn accum(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) tl: u32) {
    let t = tl;
    let l = D.a.x;
    // lv[0] = the fine level. b.w = loop row reach (upper bound of every splat's reach on this level)
    let LW = G.lv[l].a.y;
    let LH = G.lv[l].a.z;
    let MR = G.lv[l].b.w;
    let XR = G.lv[l].b.z;
    let y = D.a.y + i32(wid.y);
    let x0 = i32(wid.x) * SEG;
    for (var i = t; i < u32(SEG * NC); i += u32(SEG)) {
        atomicStore(&blo[i], 0u);
        atomicStore(&bhi[i], 0u);
    }
    workgroupBarrier();

    let sr0 = D.b.x;
    let syA = max(max(0, y - MR), sr0);
    let syB = min(min(LH - 1, y + MR), sr0 + D.b.y - 1);
    let cA = max(0, x0 - XR);
    let cB = min(LW - 1, x0 + SEG - 1 + XR);
    for (var sy = syA; sy <= syB; sy++) {
        let k = y - sy;
        let ak = abs(k);
        let rr = i32(atomicLoad(&reach[D.c.y + u32(sy - sr0)])) - 1;
        if (rr < ak) { continue; }
        let rowBase = D.c.x + u32(sy - sr0) * u32(LW);
        for (var sx = cA + i32(t); sx <= cB; sx += SEG) {
            let sp = spl[rowBase + u32(sx)];
            let sr = i32(bitcast<u32>(sp.i1.w) & 0xFFFFu) - 1;
            if (sr < ak) { continue; }
            splatBand(sp, sx, k, x0, LW);
        }
    }
    workgroupBarrier();

    // inclusive scan of the delta bins (Hillis-Steele, all slots at once)
    var v: array<vec2u, NC>;
    for (var s = 0u; s < u32(NC); s++) { v[s] = binLoad(s, t); }
    for (var off = 1u; off < u32(SEG); off = off << 1u) {
        var u: array<vec2u, NC>;
        if (t >= off) {
            for (var s = 0u; s < u32(NC); s++) { u[s] = binLoad(s, t - off); }
        }
        workgroupBarrier();
        if (t >= off) {
            for (var s = 0u; s < u32(NC); s++) { v[s] = add64(v[s], u[s]); binStore(s, t, v[s]); }
        }
        workgroupBarrier();
    }

    let x = x0 + i32(t);
    if (x >= LW) { return; }
    var val: array<f32, NC>;
    for (var s = 0u; s < u32(NC); s++) { val[s] = i2f(v[s]) * iscaleOf(s); }
    // add the magnitude tiers into the tier-0 slots
    for (var t = 1u; t < u32(K); t++) {
        let o = u32(NCB) + (t - 1u) * u32(NCC);
        val[S_R] += val[o]; val[S_G] += val[o + 1u]; val[S_B] += val[o + 2u];
        if (ROLL) { val[S_EL] += val[o + 3u]; }
    }
    if (RAW) {
        let o = D.c.z + (u32(y - D.d.x) * u32(LW) + u32(x)) * u32(NC);
        for (var s = 0u; s < u32(NC); s++) { grid[o + s] = val[s]; }
        return;
    }
    if (nLevels() > 1) { addCoarse(x, y, &val); }
    let eps = 1e-9;
    var o = vec4f(0.0);
    let wg = val[S_WG];
    if (FR) {
        if (val[S_WR] > eps) { o.x = val[S_R] / val[S_WR]; }
        if (wg > eps) { o.y = val[S_G] / wg; o.w = val[S_A] / wg; }
        if (val[S_WB] > eps) { o.z = val[S_B] / val[S_WB]; }
    } else if (wg > eps) {
        let inv = 1.0 / wg;
        o = vec4f(val[S_R] * inv, val[S_G] * inv, val[S_B] * inv, val[S_A] * inv);
    }
    if (G.i1.w != 0) { o = max(o, vec4f(0.0)); }
    let oi = D.c.w + u32(y - D.d.y) * u32(W()) + u32(x);
    outp[oi] = o;
    if (ROLL) {
        var le = 0.0;
        if (wg > eps) { le = max(0.0, val[S_EL] / wg); }
        outel[oi] = le;
    }
}
