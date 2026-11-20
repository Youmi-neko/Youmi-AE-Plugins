// lenshq_stampb.inl - path B stamp (included inside lenshq.cpp's anonymous namespace).
//
// Coordinates: q = sg * (x - c) / (R * (ax, ay)) is the pupil position at wavelength scale 1,
// centred on the aperture (chief ray) c = -sg * R * (ax hx, ay hy) (at scale 1 the aperture sits at
// -h and the clip at +h around the source pixel, alpha's centred pair). Sub-scale i of bin k sees
// the pupil at p = q / s_i (minus the bin's lateral shift): A(p) * C(p). Channel kernels:
//     k_ch = sum_i w_k(ch) w_i integral_pixel A(p_i) C(p_i)
// The SAME sub-scale list built the spectral tables K_ch, so
//   * pixels where C == 1 for every wavelength take K_ch exactly (C == 0 everywhere -> 0);
//   * pixels touched by a clip edge use the complement  K_ch - sum_k w_k(ch) box_k (1 - C_k)
//     (box_k integrated over the bin's sub-scales), for every sub-pixel size;
//   * with lateral CA every pixel is integrated per bin (sum_k w_k box_k C_k).
// As cat eye / lateral CA go to 0 these become K_ch exactly: no pop.
void stampB(const Model& M, const Src& s, Stamp& st, Work& W) {
    const Tables& T = *M.T;
    const double R = s.R;
    const int side = s.side;
    const double amp = M.onionAmp(R);
    const double sg = (side == 0 && M.flipFront) ? -1.0 : 1.0;
    const Spectrum& sp = M.spec;
    const int N = sp.N;
    const std::vector<SubScale>& sub = M.sub[side];
    const int NS = (int)sub.size();
    const int tb = T.aSide[side];
    const Sat& A0 = T.A[tb][0];
    const Sat* A1 = (T.onion && amp > 0) ? &T.A[tb][1] : nullptr;
    const double hx = M.cat ? s.hx : 0.0, hy = M.cat ? s.hy : 0.0;
    const double h2x = 2 * hx, h2y = 2 * hy;
    const double rho = M.rho;
    const double dInR = std::min(T.cfME[side].dIn, T.cfM.dIn) * rho, dOutR = T.cfM.dOut * rho;
    const double mechR = T.cfM.dIn * rho;
    const double Rx = R * M.ax, Ry = R * M.ay;
    const double cx = -sg * Rx * hx, cy = -sg * Ry * hy;
    const double g = M.discG > 0 ? 0.5 * M.discG * s.rf : 0.0;
    double tdx = s.dirx / M.ax, tdy = s.diry / M.ay;
    { const double l = std::sqrt(tdx * tdx + tdy * tdy); if (l > 1e-12) { tdx /= l; tdy /= l; } }
    const bool lat = M.lat;
    const bool rimClip = M.clipRim > 0;
    // lateral shifts per bin (px): latK * lat_k * R * g
    std::vector<double>& ldx = W.fx;  // [0..N) dx, [N..2N) dy
    ldx.assign(2 * N, 0.0);
    for (int k = 0; k < N; ++k) {
        const double sh = lat ? M.latK * sp.lat[k] * R : 0.0;
        ldx[k] = sh * s.gx; ldx[N + k] = sh * s.gy;
    }
    double smin = 1e9, smax = 0;
    for (const SubScale& u : sub) { smin = std::min(smin, u.s); smax = std::max(smax, u.s); }
    // stamp bbox: pupil support (aperture bbox, clipped by the clip circle), all wavelengths
    double PX0 = A0.bx0, PX1 = A0.bx1, PY0 = A0.by0, PY1 = A0.by1;
    if (M.cat) {
        PX0 = std::max(PX0, h2x - dOutR); PX1 = std::min(PX1, h2x + dOutR);
        PY0 = std::max(PY0, h2y - dOutR); PY1 = std::min(PY1, h2y + dOutR);
    }
    if (!(PX1 > PX0 && PY1 > PY0)) { setIdentity(st, T.nch); return; }
    double X0 = 1e30, X1 = -1e30, Y0 = 1e30, Y1 = -1e30;
    for (int k = 0; k < N; ++k)
        for (int e = 0; e < 2; ++e) {
            const double sc = e ? smax : smin;
            const double xa = cx + sg * sc * PX0 * Rx + ldx[k], xb = cx + sg * sc * PX1 * Rx + ldx[k];
            const double ya = cy + sg * sc * PY0 * Ry + ldx[N + k], yb = cy + sg * sc * PY1 * Ry + ldx[N + k];
            X0 = std::min(X0, std::min(xa, xb)); X1 = std::max(X1, std::max(xa, xb));
            Y0 = std::min(Y0, std::min(ya, yb)); Y1 = std::max(Y1, std::max(ya, yb));
        }
    st.i0 = (int)std::floor(X0 - 0.5); const int i1 = (int)std::ceil(X1 + 0.5);
    st.j0 = (int)std::floor(Y0 - 0.5); const int j1 = (int)std::ceil(Y1 + 0.5);
    st.w = i1 - st.i0 + 1; st.h = j1 - st.j0 + 1;
    st.np = T.nch;
    const size_t pn = (size_t)st.w * st.h;
    st.pl.assign((size_t)st.np * pn, 0.f);
    W.nStampPx += (long long)pn;

    const double amax = std::max(M.ax, M.ay);
    const int ssub = R * amax < 6 ? 3 : (R * amax < 16 ? 2 : 1);
    const double bq = 0.5 * std::sqrt(1.0 / (Rx * Rx) + 1.0 / (Ry * Ry));  // pixel half-diagonal (q units)
    const int lvA = A0.levelFor(R * amax * smin, M.tpp);
    const int nA = A0.nl[lvA];
    const double* SA0 = A0.lv[lvA].data();
    const double* SA1 = A1 ? A1->lv[lvA].data() : nullptr;
    const double sclA = nA / 2.0;
    // bin -> range of sub-scales (sub is ordered by bin)
    std::vector<int>& binStart = W.ix;
    binStart.assign(N + 1, NS);
    for (int i = NS - 1; i >= 0; --i) binStart[sub[i].k] = i;
    for (int k = N - 1; k >= 0; --k) if (binStart[k] > binStart[k + 1]) binStart[k] = binStart[k + 1];
    // integral of A over the px box [x0,x1]x[y0,y1] for bin k, over the bin's sub-scales
    auto boxA = [&](int k, double x0, double x1, double y0, double y1) {
        double v = 0;
        const double ox = -cx - ldx[k], oy = -cy - ldx[N + k];
        for (int i = binStart[k]; i < binStart[k + 1]; ++i) {
            const double s_ = sub[i].s;
            const double fx = sg * sclA / (Rx * s_), fy = sg * sclA / (Ry * s_);
            const double ta = (x0 + ox) * fx + sclA, tb2 = (x1 + ox) * fx + sclA;
            const double ua = (y0 + oy) * fy + sclA, ub = (y1 + oy) * fy + sclA;
            double b = satAt(SA0, nA, tb2, ub) - satAt(SA0, nA, ta, ub) - satAt(SA0, nA, tb2, ua) + satAt(SA0, nA, ta, ua);
            if (SA1) b += amp * (satAt(SA1, nA, tb2, ub) - satAt(SA1, nA, ta, ub) - satAt(SA1, nA, tb2, ua) + satAt(SA1, nA, ta, ua));
            v += sub[i].w * b;  // sign-correct for sg = -1 (both axes reversed)
        }
        return v;
    };
    // relative scale spread inside bin k (clipFactor turns it into the gauge smear)
    auto rsOf = [&](int k) { return sp.ds[k] > 0 ? sp.ds[k] / sp.s[side][k] : 0.0; };
    // clip factor of bin k over a footprint centred at (xc, yc) px, size (ddx, ddy) px
    auto clipF = [&](int k, double xc, double yc, double ddx, double ddy) {
        const double s_ = sp.s[side][k];
        const double px = sg * (xc - cx - ldx[k]) / (Rx * s_), py = sg * (yc - cy - ldx[N + k]) / (Ry * s_);
        return clipFactor(M, side, px - h2x, py - h2y, px, py, ddx / (Rx * s_), ddy / (Ry * s_), -1.0, rsOf(k));
    };
    // f(t) = |q t - 2h| - rho g(q) t (pupil units) plus a bound of its linear variation over the
    // pixel box; <= 0 means the clip boundary lies outside the aperture boundary at scale 1/t.
    // f is convex in t, so checking the two end scales covers every wavelength.
    auto outerTest = [&](double t, double wx, double wy, double dw, double gq, double ggx, double ggy) {
        const double ux = dw > 1e-12 ? wx / dw : 0.0, uy = dw > 1e-12 ? wy / dw : 0.0;
        const double fx = t * (ux - rho * ggx), fy = t * (uy - rho * ggy);
        const double marg = 0.525 * (std::fabs(fx) / Rx + std::fabs(fy) / Ry);
        return dw - rho * gq * t + marg;
    };

    std::vector<uint8_t>& cls = W.cls;  // 0 = outside all, 1 = C == 1 for all, 2 = per-wavelength
    cls.assign(pn, 2);
    std::vector<float>& kb = W.tmp;     // spectral-table box of class-2 pixels (np planes)
    int lv = 0, nK = 0;
    double DK = 1, sclK = 1;
    if (!lat) {
        kb.assign((size_t)st.np * pn, 0.f);
        const int BS = 4;
        const double bqB = BS * bq;
        for (int bj = 0; bj < st.h; bj += BS) {
            if (cancelled()) return;
            for (int bi = 0; bi < st.w; bi += BS) {
                const int bw_ = std::min(BS, st.w - bi), bh_ = std::min(BS, st.h - bj);
                int dec = 2;
                if (!M.cat) dec = 1;
                else {
                    const double qx = sg * ((st.i0 + bi + 0.5 * (bw_ - 1)) - cx) / Rx;
                    const double qy = sg * ((st.j0 + bj + 0.5 * (bh_ - 1)) - cy) / Ry;
                    const double ax_ = qx / smin - h2x, ay_ = qy / smin - h2y;
                    const double bx_ = qx / smax - h2x, by_ = qy / smax - h2y;
                    const double da = std::sqrt(ax_ * ax_ + ay_ * ay_), db = std::sqrt(bx_ * bx_ + by_ * by_);
                    const double mg = bqB / smin;
                    const double dmax = std::max(da, db) + mg;
                    if (dmax <= dInR || (!rimClip && dmax <= mechR)) dec = 1;
                    else {
                        const double ex_ = bx_ - ax_, ey_ = by_ - ay_;
                        const double l2 = ex_ * ex_ + ey_ * ey_;
                        double t = l2 > 0 ? -(ax_ * ex_ + ay_ * ey_) / l2 : 0.0;
                        t = clampd(t, 0.0, 1.0);
                        const double mx = ax_ + t * ex_, my = ay_ + t * ey_;
                        if (std::sqrt(mx * mx + my * my) - mg >= dOutR) dec = 0;
                    }
                }
                if (dec != 2) {
                    for (int j = bj; j < bj + bh_; ++j) std::memset(&cls[(size_t)j * st.w + bi], dec, (size_t)bw_);
                    continue;
                }
                for (int j = bj; j < bj + bh_; ++j)
                    for (int i = bi; i < bi + bw_; ++i) {
                        const double qy = sg * ((st.j0 + j) - cy) / Ry;
                        const double qx = sg * ((st.i0 + i) - cx) / Rx;
                        const double ax_ = qx / smin - h2x, ay_ = qy / smin - h2y;
                        const double bx_ = qx / smax - h2x, by_ = qy / smax - h2y;
                        const double da = std::sqrt(ax_ * ax_ + ay_ * ay_), db = std::sqrt(bx_ * bx_ + by_ * by_);
                        const double mg = bq / smin;
                        const double dmax = std::max(da, db) + mg;
                        bool one = dmax <= dInR || (!rimClip && dmax <= mechR);
                        if (!one) {
                            double ggx, ggy;
                            const double gq = M.ap.gaugeGrad(qx, qy, ggx, ggy);
                            one = std::max(outerTest(1.0 / smin, ax_, ay_, da, gq, ggx, ggy),
                                           outerTest(1.0 / smax, bx_, by_, db, gq, ggx, ggy)) <= 1e-12;
                        }
                        uint8_t c = 1;
                        if (!one) {
                            const double ex_ = bx_ - ax_, ey_ = by_ - ay_;
                            const double l2 = ex_ * ex_ + ey_ * ey_;
                            double t = l2 > 0 ? -(ax_ * ex_ + ay_ * ey_) / l2 : 0.0;
                            t = clampd(t, 0.0, 1.0);
                            const double mx = ax_ + t * ex_, my = ay_ + t * ey_;
                            c = std::sqrt(mx * mx + my * my) - mg >= dOutR ? 0 : 2;
                        }
                        cls[(size_t)j * st.w + i] = c;
                    }
            }
        }
        // corner grid of the spectral tables (shared corners -> exact telescoping)
        const Sat& ref = T.kTab(side, 0, 0);
        lv = ref.levelFor(R * amax, M.tpp);
        nK = ref.nl[lv];
        DK = ref.D; sclK = nK / (2 * DK);
        std::vector<int> ix(st.w + 1);  // (W.ix holds binStart)
        std::vector<double>& fxs = W.fx2;
        fxs.resize(st.w + 1);
        for (int k = 0; k <= st.w; ++k) {
            double t = (sg * ((st.i0 - 0.5 + k) - cx) / Rx + DK) * sclK;
            t = t < 0 ? 0 : (t > nK ? nK : t);
            int ii = (int)t;
            if (ii > nK - 1) ii = nK - 1;
            ix[k] = ii; fxs[k] = t - ii;
        }
        W.rowA.resize(st.w + 1); W.rowB.resize(st.w + 1);
        for (int c = 0; c < st.np; ++c) {
            const double* S0 = T.kTab(side, c, 0).lv[lv].data();
            const double* S1 = A1 ? T.kTab(side, c, 1).lv[lv].data() : nullptr;
            float* out = st.pl.data() + (size_t)c * pn;
            float* kbo = kb.data() + (size_t)c * pn;
            double* prev = W.rowA.data();
            double* cur = W.rowB.data();
            for (int l = 0; l <= st.h; ++l) {
                if ((l & 15) == 0 && cancelled()) return;
                double t = (sg * ((st.j0 - 0.5 + l) - cy) / Ry + DK) * sclK;
                t = t < 0 ? 0 : (t > nK ? nK : t);
                int iy = (int)t;
                if (iy > nK - 1) iy = nK - 1;
                const double fy = t - iy;
                const double* r0 = S0 + (size_t)iy * (nK + 1);
                const double* r1 = r0 + (nK + 1);
                for (int k = 0; k <= st.w; ++k) {
                    const int ii = ix[k];
                    const double f = fxs[k];
                    const double a = r0[ii] + f * (r0[ii + 1] - r0[ii]);
                    const double b = r1[ii] + f * (r1[ii + 1] - r1[ii]);
                    cur[k] = a + fy * (b - a);
                }
                if (S1) {
                    const double* q0 = S1 + (size_t)iy * (nK + 1);
                    const double* q1 = q0 + (nK + 1);
                    for (int k = 0; k <= st.w; ++k) {
                        const int ii = ix[k];
                        const double f = fxs[k];
                        const double a = q0[ii] + f * (q0[ii + 1] - q0[ii]);
                        const double b = q1[ii] + f * (q1[ii + 1] - q1[ii]);
                        cur[k] += amp * (a + fy * (b - a));
                    }
                }
                if (l > 0) {
                    const size_t ro = (size_t)(l - 1) * st.w;
                    for (int k = 0; k < st.w; ++k) {
                        const uint8_t cc = cls[ro + k];
                        if (cc == 0) continue;
                        double v = (cur[k + 1] - cur[k]) - (prev[k + 1] - prev[k]);
                        v = v > 0 ? v : 0.0;
                        if (cc == 1) out[ro + k] = (float)v; else kbo[ro + k] = (float)v;
                    }
                }
                std::swap(prev, cur);
            }
        }
    }
    // K box of channel c over an arbitrary px box (sub-pixels; same table level)
    auto kBox = [&](int c, double x0, double x1, double y0, double y1) {
        const double* S0 = T.kTab(side, c, 0).lv[lv].data();
        const double* S1 = A1 ? T.kTab(side, c, 1).lv[lv].data() : nullptr;
        const double ta = (sg * (x0 - cx) / Rx + DK) * sclK, tb2 = (sg * (x1 - cx) / Rx + DK) * sclK;
        const double ua = (sg * (y0 - cy) / Ry + DK) * sclK, ub = (sg * (y1 - cy) / Ry + DK) * sclK;
        double v = satAt(S0, nK, tb2, ub) - satAt(S0, nK, ta, ub) - satAt(S0, nK, tb2, ua) + satAt(S0, nK, ta, ua);
        if (S1) v += amp * (satAt(S1, nK, tb2, ub) - satAt(S1, nK, ta, ub) - satAt(S1, nK, tb2, ua) + satAt(S1, nK, ta, ua));
        return v;
    };

    // ---- per-wavelength pixels (clip edges, or everything with lateral CA)
    const double hs = 1.0 / ssub;
    const bool fast = !lat && ssub == 1 && M.cat;
    std::vector<double> cl(N);
    std::vector<uint8_t> stt(N);
    for (int j = 0; j < st.h; ++j) {
        if (cancelled()) return;
        for (int i = 0; i < st.w; ++i) {
            const size_t idx = (size_t)j * st.w + i;
            if (cls[idx] != 2) continue;
            ++W.nLamPx;
            const double xa = st.i0 + i - 0.5, ya = st.j0 + j - 0.5;
            if (fast) {
                const double qx = sg * ((st.i0 + i) - cx) / Rx, qy = sg * ((st.j0 + j) - cy) / Ry;
                double ggx, ggy;
                const double gq = M.ap.gaugeGrad(qx, qy, ggx, ggy);
                {   // exact shortcut: every clip function constant over the whole range of clip and
                    // aperture gauges this pixel sees over all wavelengths -> one clip factor
                    const double tA = 1.0 / smax, tB = 1.0 / smin;
                    const double ax_ = qx * tB - h2x, ay_ = qy * tB - h2y, bx_ = qx * tA - h2x, by_ = qy * tA - h2y;
                    const double da = std::sqrt(ax_ * ax_ + ay_ * ay_), db = std::sqrt(bx_ * bx_ + by_ * by_);
                    const double ex_ = bx_ - ax_, ey_ = by_ - ay_, l2 = ex_ * ex_ + ey_ * ey_;
                    double tt = l2 > 0 ? -(ax_ * ex_ + ay_ * ey_) / l2 : 0.0;
                    tt = clampd(tt, 0.0, 1.0);
                    const double dmn = std::hypot(ax_ + tt * ex_, ay_ + tt * ey_);
                    const double hw = bq * tB * 1.05;
                    const double c0 = (dmn - hw) / rho, c1 = (std::max(da, db) + hw) / rho;
                    const double a0 = gq * tA - 1.3 * hw, a1 = gq * tB + 1.3 * hw;
                    double vm, vme, ve;
                    bool same = T.cfM.constOver(c0, c1, vm);
                    double Cc = vm;
                    if (same && rimClip) {
                        if (c1 <= a0) Cc = vm;
                        else if (c0 > a1 && T.cfME[side].constOver(c0, c1, vme) && T.cfE[side].constOver(a0, a1, ve)) {
                            double r = vme / std::max(ve, 1e-3);
                            r = std::min(r, T.eMax[side]);
                            if (ve < 0.25) r = std::min(r, 1.0);
                            Cc = (1.0 - M.clipRim) * vm + M.clipRim * r;
                        } else same = false;
                    }
                    if (same) {
                        for (int c = 0; c < st.np; ++c) st.pl[(size_t)c * pn + idx] = (float)(kb[(size_t)c * pn + idx] * Cc);
                        continue;
                    }
                }
                int n0 = 0, n1 = 0;
                for (int k = 0; k < N; ++k) {
                    const double t = 1.0 / sp.s[side][k];
                    const double px = qx * t, py = qy * t, wx = px - h2x, wy = py - h2y;
                    const double dw = std::sqrt(wx * wx + wy * wy);
                    const double mg = bq * t;
                    uint8_t sk;
                    if (dw - mg >= dOutR) sk = 0;
                    else if (dw + mg <= dInR || (!rimClip && dw + mg <= mechR)) sk = 1;
                    else if (outerTest(t, wx, wy, dw, gq, ggx, ggy) <= 1e-12) sk = 1;
                    else sk = 2;
                    stt[k] = sk;
                    if (sk == 0) { ++n0; cl[k] = 0.0; }
                    else if (sk == 1) { ++n1; cl[k] = 1.0; }
                    else cl[k] = clipFactor(M, side, wx, wy, px, py, t / Rx, t / Ry, gq * t, rsOf(k));
                }
                if (n1 == N) {
                    for (int c = 0; c < st.np; ++c) st.pl[(size_t)c * pn + idx] = kb[(size_t)c * pn + idx];
                    continue;
                }
                double acc[3] = {0, 0, 0};
                if (n0 < N) {
                    for (int c = 0; c < st.np; ++c) acc[c] = kb[(size_t)c * pn + idx];
                    for (int k = 0; k < N; ++k) {
                        if (stt[k] == 1) continue;
                        const double v = boxA(k, xa, xa + 1, ya, ya + 1) * (1.0 - cl[k]);
                        if (st.np == 1) acc[0] -= v;
                        else for (int c = 0; c < 3; ++c) acc[c] -= sp.w[c][k] * v;
                    }
                }
                for (int c = 0; c < st.np; ++c) st.pl[(size_t)c * pn + idx] = (float)std::max(0.0, acc[c]);
                continue;
            }
            double acc[3] = {0, 0, 0};
            for (int b = 0; b < ssub; ++b)
                for (int a = 0; a < ssub; ++a) {
                    const double x0 = xa + a * hs, y0 = ya + b * hs;
                    if (!lat) {
                        // complement on the sub-pixel (consistent with the table value)
                        double v[3];
                        for (int c = 0; c < st.np; ++c) v[c] = kBox(c, x0, x0 + hs, y0, y0 + hs);
                        if (M.cat)
                            for (int k = 0; k < N; ++k) {
                                const double cf = clipF(k, x0 + 0.5 * hs, y0 + 0.5 * hs, hs, hs);
                                if (cf >= 1.0) continue;
                                const double bx = boxA(k, x0, x0 + hs, y0, y0 + hs) * (1.0 - cf);
                                if (st.np == 1) v[0] -= bx;
                                else for (int c = 0; c < 3; ++c) v[c] -= sp.w[c][k] * bx;
                            }
                        for (int c = 0; c < st.np; ++c) acc[c] += std::max(0.0, v[c]);
                    } else {
                        for (int k = 0; k < N; ++k) {
                            // exact shortcuts (same tests as the per-bin classification above, with
                            // the bin's clip-edge smear added to the margin): C == 0 well outside the
                            // clip circle, C == 1 well inside it -> no clipFactor call
                            bool needClip = M.cat;
                            if (M.cat) {
                                const double s_ = sp.s[side][k];
                                const double px = sg * (x0 + 0.5 * hs - cx - ldx[k]) / (Rx * s_);
                                const double py = sg * (y0 + 0.5 * hs - cy - ldx[N + k]) / (Ry * s_);
                                const double wx = px - h2x, wy = py - h2y, dw = std::sqrt(wx * wx + wy * wy);
                                const double mg = bq / s_ + (sp.ds[k] > 0 ? std::sqrt(px * px + py * py) * sp.ds[k] / s_ : 0.0);
                                if (dw - mg >= dOutR) continue;
                                if (dw + mg <= dInR || (!rimClip && dw + mg <= mechR)) needClip = false;
                            }
                            double bx = boxA(k, x0, x0 + hs, y0, y0 + hs);
                            if (bx <= 0) continue;
                            if (needClip) bx *= clipF(k, x0 + 0.5 * hs, y0 + 0.5 * hs, hs, hs);
                            if (st.np == 1) acc[0] += bx;
                            else for (int c = 0; c < 3; ++c) acc[c] += sp.w[c][k] * bx;
                        }
                    }
                }
            for (int c = 0; c < st.np; ++c) st.pl[(size_t)c * pn + idx] = (float)std::max(0.0, acc[c]);
        }
    }
    // ---- disc falloff (cos^4-like tilt across the pupil, at the pixel centre, scale 1)
    if (g > 0)
        for (int j = 0; j < st.h; ++j) {
            const double qy = sg * ((st.j0 + j) - cy) / Ry;
            for (int i = 0; i < st.w; ++i) {
                const double qx = sg * ((st.i0 + i) - cx) / Rx;
                const float f = (float)std::max(0.0, 1.0 - g * (qx * tdx + qy * tdy));
                for (int c = 0; c < st.np; ++c) st.pl[(size_t)c * pn + (size_t)j * st.w + i] *= f;
            }
        }
}
