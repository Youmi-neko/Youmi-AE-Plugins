// Shitsukan render engine (round 2) - see sk_core.h and docs\watercolor_model.md for the design.
#include "sk_util.h"

#include <array>
#include <functional>

namespace sk {
int g_profile = 0;

namespace {

const float kInkEps = 0.03f;
const float kPaper[3] = {0.985f, 0.975f, 0.95f};  // the painter's paper (a warm white)  // ink model: out = (fill + eps) * exp(-delta) - eps
const float kInvGamma = 1.f / 2.2f;

inline float ToGamma(float v) { return v > 0.f ? pm::pow(v, kInvGamma) : 0.f; }
inline float ToLinear(float v) { return v > 0.f ? pm::pow(v, 2.2f) : 0.f; }

// black / white top-hat coverage of one signal (closing / opening by a square of radius r)
// dark: (closing - s) / (closing - erosion); light: (s - opening) / (dilation - opening); gated by the contrast
void TopHat(const float* s, int W, int H, int r, bool dark, float thr, int T, std::vector<float>& tmp, std::vector<float>& outA,
            std::vector<float>* core)
{
    const size_t N = (size_t)W * H;
    std::vector<float> a1(N), a2(N);
    MinMax(s, a1.data(), W, H, r, dark, T, tmp);          // dark: dilation (max) ; light: erosion (min)
    MinMax(a1.data(), a2.data(), W, H, r, !dark, T, tmp);  // closing / opening
    MinMax(s, a1.data(), W, H, r, !dark, T, tmp);          // dark: erosion (core) ; light: dilation
    outA.resize(N);
    if (core)
        core->assign(a1.begin(), a1.end());
    ForPx(W, H, T, [&](size_t i, int, int) {
        const float th = dark ? a2[i] - s[i] : s[i] - a2[i];
        const float K = dark ? a2[i] - a1[i] : a1[i] - a2[i];
        const float gate = Smooth(0.5f * thr, thr, K);
        outA[i] = (K > 1e-4f && th > 0.f) ? Clampf(th / K, 0.f, 1.f) * gate : 0.f;
    });
}

#include "sk_paint.inc"

Status RenderImpl(const Params& pin, Frame& f)
{
    const int W = f.w, H = f.h;
    const size_t N = (size_t)W * H;
    float* out = f.out;
    if (N == 0 || !f.rgba || !out)
        return ST_OK;
    const double ds = (f.ds > 0 && std::isfinite(f.ds)) ? f.ds : 1.0;
    const double par = (pin.par > 0.05 && pin.par < 20) ? pin.par : 1.0;
    const int T = std::max(1, f.threads);
    auto poll = [&] {
        if (f.abort && f.abort())
            throw AbortEx();
    };
    Timer tm;
    // full-res layer coordinates of buffer pixel x / y (x stretched by the pixel aspect: round on screen)
    auto Xf = [&](double x) { return (x + f.ox) / ds * par; };
    auto Yf = [&](double y) { return (y + f.oy) / ds; };

    // ---- derived settings (buffer px)
    const float amt = (float)Clampd(pin.amount / 100.0, 0, 1);
    float gains[3] = {(float)Clampd(pin.fine / 100.0, -1, 2), (float)Clampd(pin.mid / 100.0, -1, 2), (float)Clampd(pin.large / 100.0, -1, 2)};
    const bool texOn = gains[0] != 0.f || gains[1] != 0.f || gains[2] != 0.f;
    const double fillBlur = Clampd(pin.fillBlur, 0, 500) * ds;
    const bool blurOn = fillBlur > 0.05;
    const float wc = (float)Clampd(pin.wc / 100.0, 0, 1);
    const bool wcOn = wc > 0.f;
    const bool lineOps = pin.lineThick != 0 || pin.lineDark != 0 || pin.lineSoft > 0 || pin.lineStyle != LS_ASIS;
    const int view = (pin.view >= 0 && pin.view < V_COUNT) ? pin.view : V_RESULT;
    const bool wcView = view == V_RIM || view == V_STROKES || view == V_GLAZE;
    const bool anyProc = texOn || blurOn || wcOn;
    const bool protect = pin.protectLines;
    const bool needLines = (protect && anyProc) || lineOps || view == V_LINES || view == V_FILL || pin.blackWalls;
    const bool inpainted = protect && needLines;
    const uint32_t seed = Hash((uint32_t)pin.seed * 0x2545F491u + 0x9E37u) ^ Hash((uint32_t)pin.paperFrame + 0x7F4A7C15u);
    const float* M = f.map;
    auto mapOf = [&](int target, size_t i) -> float { return (M && pin.mapTarget == target) ? Clampf(M[i], 0.f, 1.f) : 1.f; };
    const double ps = 3.2 * Clampd(pin.paperScale / 100.0, 0.25, 8);
    auto paperAt = [&](size_t i, int x, int y) -> float {
        return f.paperTex ? Clampf(f.paperTex[i], 0.f, 1.f) : PaperHeight(Xf(x), Yf(y), ps, seed, ds);
    };

    // ---- 0. inputs (linear working space: processed in gamma 2.2)
    std::vector<float> C(N * 3), A(N);
    ForPx(W, H, T, [&](size_t i, int, int) {
        const float* s = f.rgba + i * 4;
        for (int c = 0; c < 3; ++c)
        {
            float v = std::isfinite(s[c]) ? Clampf(s[c], 0.f, 64.f) : 0.f;
            C[i * 3 + c] = pin.linear ? ToGamma(v) : v;
        }
        A[i] = std::isfinite(s[3]) ? Clampf(s[3], 0.f, 1.f) : 0.f;
    });
    std::vector<float> tmp4, tmp1;

    // ---- 1. line analysis -> alpha (coverage), bar (barrier), delta (ink density), Fl (fill layer, in place)
    std::vector<float> alpha, delta, Fl;
    std::vector<uint8_t> bar, light;
    int rc = 1;
    if (needLines)
    {
        rc = std::max(1, std::min(16, (int)std::ceil(Clampd(pin.lineWidth, 0.5, 32) * ds * 0.5 - 1e-6)));
        const float sens = (float)Clampd(pin.lineSens / 100.0, 0, 1);
        const float thr = 0.30f * pm::pow(0.03f / 0.30f, sens);
        std::vector<float> Y(N), core, aS;
        ForPx(W, H, T, [&](size_t i, int, int) { Y[i] = A[i] < (0.5f / 255.f) ? 1.f : Clampf(Luma(&C[i * 3]), 0.f, 1.f); });
        TopHat(Y.data(), W, H, rc, true, thr, T, tmp1, alpha, &core);
        // luma lines: dark cores are more line-like
        ForPx(W, H, T, [&](size_t i, int, int) { alpha[i] *= 1.f - 0.45f * Smooth(0.45f, 0.85f, core[i]); });
        if (pin.lineColor >= LC_COLOR)
        {
            // colour cue: a line darker than both sides in ANY channel (brown / red / blue outlines on light fills)
            // (one signal: the smallest channel - a red / brown line on skin drops in green and blue)
            std::vector<float> ch(N);
            ForPx(W, H, T, [&](size_t i, int, int) {
                const float* c = &C[i * 3];
                ch[i] = A[i] < (0.5f / 255.f) ? 1.f : Clampf(std::min(c[0], std::min(c[1], c[2])), 0.f, 1.f);
            });
            TopHat(ch.data(), W, H, rc, true, thr * 1.3f, T, tmp1, aS, nullptr);
            ForPx(W, H, T, [&](size_t i, int, int) { alpha[i] = std::max(alpha[i], 0.85f * aS[i]); });        }
        if (pin.lineColor >= LC_ALL)
        {
            // light lines on dark (white top-hat on luma): drawn back as negative ink
            TopHat(Y.data(), W, H, rc, false, thr * 1.2f, T, tmp1, aS, nullptr);
            light.assign(N, 0);
            ForPx(W, H, T, [&](size_t i, int, int) {
                if (aS[i] > alpha[i])
                {
                    alpha[i] = aS[i];
                    light[i] = 1;
                }
            });
        }
        tm.lap("lines tophat");
        // dense clusters of thin dark specks (foliage, brush texture) are not lines unless asked for
        if (!pin.bgDetailLines)
        {
            std::vector<float> v(N);
            ForPx(W, H, T, [&](size_t i, int, int) { v[i] = alpha[i] > 0.3f ? 1.f : 0.f; });
            // isolated specks (a dark grain, not a stroke): a line crossing a (2r+1)^2 window has >= 2r+1 pixels
            std::vector<float> cnt(N), tmpc(N);
            const int rb = std::max(1, (int)std::lround(2.0 * ds));
            ForPx(W, H, T, [&](size_t i, int x, int) {
                float s = 0.f;
                for (int k = std::max(0, x - rb); k <= std::min(W - 1, x + rb); ++k)
                    s += v[i - x + k];
                tmpc[i] = s;
            });
            ForPx(W, H, T, [&](size_t i, int x, int y) {
                float s = 0.f;
                for (int k = std::max(0, y - rb); k <= std::min(H - 1, y + rb); ++k)
                    s += tmpc[(size_t)k * W + x];
                cnt[i] = s;
            });
            RFPlain<1>(v.data(), W, H, 7.0 * ds, 3, T);
            const float side = 2.f * rb + 1.f;
            ForPx(W, H, T, [&](size_t i, int, int) {
                alpha[i] *= (1.f - Smooth(0.30f, 0.48f, v[i])) * Smooth(0.4f * side, 0.8f * side, cnt[i]);
            });
        }
        ForPx(W, H, T, [&](size_t i, int, int) {
            if (A[i] < (0.5f / 255.f))
                alpha[i] = 0.f;
        });
        bar.resize(N);
        std::vector<float> beta(N), wIn(N);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float a = alpha[i];
            bar[i] = (a > 0.25f || (pin.blackWalls && A[i] > 0.5f && Y[i] < 0.10f)) ? 1 : 0;
            beta[i] = Smooth(0.03f, 0.35f, a);
            const float ob = 1.f - beta[i];
            wIn[i] = A[i] * ob * ob;
        });
        Y.clear(); Y.shrink_to_fit();
        core.clear(); core.shrink_to_fit();
        aS.clear(); aS.shrink_to_fit();
        tm.lap("lines clutter");
        // in-painted fill under the lines
        std::vector<float> Fh(N * 3);
        SmoothNorm3(C.data(), wIn.data(), Fh.data(), W, H, rc + 1.0, Edges(), T, tmp4);
        delta.resize(N * 3);
        const bool hasLight = !light.empty();
        ForPx(W, H, T, [&](size_t i, int, int) {
            for (int c = 0; c < 3; ++c)
            {
                float d = 0.f;
                if (beta[i] > 0.f)
                {
                    d = beta[i] * pm::log((Fh[i * 3 + c] + kInkEps) / (C[i * 3 + c] + kInkEps));
                    if (!(hasLight && light[i]) && d < 0.f)
                        d = 0.f;  // dark lines only darken
                    else if (hasLight && light[i] && d > 0.f)
                        d = 0.f;  // light lines only lighten
                }
                delta[i * 3 + c] = d;
                if (inpainted && d != 0.f)  // in place: C becomes the fill layer
                    C[i * 3 + c] = (C[i * 3 + c] + kInkEps) * pm::exp(d) - kInkEps;
            }
        });
        tm.lap("lines");
        poll();
    }
    Fl.swap(C);  // the fill layer (the input itself when lines are not protected)

    // ---- 2. texture bands on the fill layer
    std::vector<float> Tx;
    std::vector<float> bandView;
    const int viewBand = view == V_FINE ? 0 : view == V_MID ? 1 : view == V_LARGE ? 2 : -1;
    if (texOn || viewBand >= 0 || view == V_FILL)
        Tx = Fl;  // the bands read the untouched fill
    else
        Tx.swap(Fl);
    if (texOn || viewBand >= 0)
    {
        std::vector<float> gh, gv;
        const float ea = (float)Clampd(pin.edgeAware / 100.0, 0, 1);
        if (ea > 0.f)
            BuildGrad(Fl.data(), W, H, gh, gv, T);
        const double sr = 0.5 * pm::pow(0.03 / 0.5, (double)ea);
        // fine | mid | large: sigma s, 4 s, and a "clarity" scale of 24 s (36 px at the default 1.5 px)
        const double s1 = Clampd(pin.detailSize, 0.3, 50) * ds, sig[3] = {s1, 4.0 * s1, 24.0 * s1};
        const float halo = (float)Clampd(pin.halo / 100.0, 0, 1);
        const float tau0 = 0.6f * pm::pow(0.03f / 0.6f, halo);
        const float tauK[3] = {1.f, 2.f, 6.f};  // bigger bands have bigger natural amplitudes
        const float cd = (float)Clampd(pin.colorDetail / 100.0, 0, 1);
        bool active[3], needS[4] = {true, false, false, false};
        for (int b = 0; b < 3; ++b)
        {
            active[b] = gains[b] != 0.f || viewBand == b;
            if (active[b])
                needS[b] = needS[b + 1] = true;
        }
        if (viewBand >= 0)
            bandView.assign(N, 0.f);
        // upper = S_b (S_0 = the fill itself: no copy), lower = S_(b+1); two scratch planes are swapped
        std::vector<float> bufA, bufB;
        const float* upper = Fl.data();
        for (int b = 0; b < 3; ++b)
        {
            if (!needS[b + 1])
                continue;
            Edges e;
            if (ea > 0.f)
            {
                e.gh = gh.data();
                e.gv = gv.data();
                // the large band tolerates bigger edges (clarity = local contrast across shapes), the lines and
                // silhouettes still stay in the base (no halos)
                const double srb = sr * (b == 0 ? 1.0 : b == 1 ? 1.6 : 3.5);
                e.k = sig[b] / srb;
            }
            std::vector<float>& lowerV = (upper == bufA.data() && !bufA.empty()) ? bufB : bufA;
            lowerV.resize(N * 3);
            float* lower = lowerV.data();
            SmoothNorm3(Fl.data(), A.data(), lower, W, H, sig[b], e, T, tmp4);
            if (active[b])
            {
                const float g = gains[b];
                const bool vb = viewBand == b;
                const float tau = tau0 * tauK[b];
                ForPx(W, H, T, [&](size_t i, int, int) {
                    float d[3] = {upper[i * 3] - lower[i * 3], upper[i * 3 + 1] - lower[i * 3 + 1], upper[i * 3 + 2] - lower[i * 3 + 2]};
                    const float dY = Luma(d);
                    if (vb)
                        bandView[i] = dY;
                    if (g == 0.f)
                        return;
                    const float gm = g * mapOf(MT_TEXTURE, i);
                    float gY, gC;
                    if (gm > 0.f)
                    {
                        const float q = dY / tau;
                        gY = gm / (1.f + q * q);  // big differences (edges) get little gain: no halos
                        gC = gY * cd;
                    }
                    else
                        gY = gC = gm;
                    for (int c = 0; c < 3; ++c)
                        Tx[i * 3 + c] += gY * dY + gC * (d[c] - dY);
                });
            }
            upper = lower;
            poll();
        }
        tm.lap("texture");
    }
    if (view != V_FILL)
    {
        Fl.clear();
        Fl.shrink_to_fit();
    }

    // ---- 3. fill blur (lines are barriers, line pixels get no weight, then are in-painted)
    std::vector<float>& B = Tx;
    const bool useBar = !bar.empty() && (protect || pin.blackWalls);
    auto inpaintBarrier = [&](std::vector<float>& R, const std::vector<float>& den) {
        // barrier pixels (and pixels nothing reached) take the smoothed value around them
        std::vector<float> w2(N), Pq(N * 3);
        ForPx(W, H, T, [&](size_t i, int, int) { w2[i] = (bar[i] || den[i] < 1e-4f) ? 0.f : A[i]; });
        SmoothNorm3(R.data(), w2.data(), Pq.data(), W, H, rc + 1.0, Edges(), T, tmp4);
        ForPx(W, H, T, [&](size_t i, int, int) {
            if (bar[i] || den[i] < 1e-4f)
                for (int c = 0; c < 3; ++c)
                    R[i * 3 + c] = Pq[i * 3 + c];
        });
    };
    if (blurOn)
    {
        std::vector<float> R(N * 3), gh, gv, w(N), den(N);
        const float ke = (float)Clampd(pin.keepEdges / 100.0, 0, 1);
        Edges e;
        if (ke > 0.f)
        {
            BuildGrad(Tx.data(), W, H, gh, gv, T);
            e.gh = gh.data();
            e.gv = gv.data();
            e.k = (fillBlur * 0.5) / (1.0 * pm::pow(0.04, (double)ke));
        }
        if (useBar)
            e.bar = bar.data();
        ForPx(W, H, T, [&](size_t i, int, int) { w[i] = A[i] * ((useBar && bar[i]) ? 0.f : 1.f); });
        SmoothNorm3(Tx.data(), w.data(), R.data(), W, H, fillBlur * 0.5, e, T, tmp4, den.data());
        if (useBar)
            inpaintBarrier(R, den);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float m = mapOf(MT_FILLBLUR, i);
            for (int c = 0; c < 3; ++c)
                B[i * 3 + c] += m * (R[i * 3 + c] - B[i * 3 + c]);
        });
        tm.lap("fill blur");
        poll();
    }

    // ---- 4. watercolour (docs\watercolor_model.md)
    std::vector<float> viewWc;
    if (wcOn || wcView)
    {
        const bool copyP = !(wc >= 1.f && (!M || pin.mapTarget != MT_WATERCOLOR));
        std::vector<float> Pown;
        if (copyP)
            Pown = B;
        float* P = copyP ? Pown.data() : B.data();
        struct WcFrame
        {
            int ox, oy;
            Rect layer;
        };
        // part 1 - the washes (flatten, strokes, glazes, wobble, bleed, blooms, unevenness, rims) -> total density.
        // Runs at the working resolution: full, or half for big frames (then it is what a Half preview computes).
        auto washes = [&](const int W, const int H, const double ds, const WcFrame& f, float* P, const float* B, const std::vector<float>& A,
                          const std::vector<uint8_t>& bar, const std::vector<float>& alpha, Buf& Dt, Buf& prot, Buf& busy, Buf& strokeTex,
                          std::vector<float>& viewWc) {
            const size_t N = (size_t)W * H;
            auto inpaintBarrier = [&](std::vector<float>& R, const std::vector<float>& den) {
                std::vector<float> w2(N), Pq(N * 3);
                ForPx(W, H, T, [&](size_t i, int, int) { w2[i] = (bar[i] || den[i] < 1e-4f) ? 0.f : A[i]; });
                SmoothNorm3(R.data(), w2.data(), Pq.data(), W, H, rc + 1.0, Edges(), T, tmp4);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    if (bar[i] || den[i] < 1e-4f)
                        for (int c = 0; c < 3; ++c)
                            R[i * 3 + c] = Pq[i * 3 + c];
                });
            };
            std::vector<float> w(N);
            ForPx(W, H, T, [&](size_t i, int, int) { w[i] = A[i] * ((useBar && bar[i]) ? 0.f : 1.f); });

            // (0) small-detail map: density of lines / strong edges nearby (eyes, mouths, fingers, small props).
            //     prot: how much the random parts (wobble, blooms, strokes, granulation) back off; busy: wash size cue
            prot.alloc(N);
            busy.alloc(N);
            {
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float* c = P + i * 3;
                    const float* r = x + 1 < W ? c + 3 : c;
                    const float* d = y + 1 < H ? c + (size_t)W * 3 : c;
                    const float g = std::fabs(Luma(r) - Luma(c)) + std::fabs(Luma(d) - Luma(c));
                    const float e = std::max(alpha.empty() ? 0.f : alpha[i], Clampf((g - 0.04f) * 8.f, 0.f, 1.f));
                    prot[i] = e * A[i];
                    busy[i] = Clampf((g - 0.02f) * 10.f, 0.f, 1.f) * A[i];
                });
                RFPlain<1>(prot.data(), W, H, 4.0 * ds, 3, T);
                RFPlain<1>(busy.data(), W, H, 10.0 * ds, 3, T);
                const float pd = (float)Clampd(pin.protectDetail / 100.0, 0, 1);
                ForPx(W, H, T, [&](size_t i, int, int) { prot[i] = pd * Smooth(0.10f, 0.32f, prot[i]); });
            }
            tm.lap("wc prep");
            // (a) flatten washes: strongly edge-aware smoothing, lines are barriers
            const float flat = (float)Clampd(pin.flatten / 100.0, 0, 1);
            if (flat > 0.f)
            {
                std::vector<float> E(N * 3), den(N);
                const double sf = Clampd(pin.flattenSize, 0.5, 200) * ds;
                if (N > kHalfResPx && sf >= 3.0)
                    SmoothNorm3Half(P, w.data(), E.data(), W, H, sf, 0.07, useBar ? bar.data() : nullptr, T, tmp4, den.data());
                else
                {
                    std::vector<float> gh, gv;
                    BuildGrad(P, W, H, gh, gv, T);
                    Edges e;
                    e.gh = gh.data();
                    e.gv = gv.data();
                    e.k = sf / 0.07;
                    if (useBar)
                        e.bar = bar.data();
                    SmoothNorm3(P, w.data(), E.data(), W, H, sf, e, T, tmp4, den.data());
                }
                if (useBar)
                    inpaintBarrier(E, den);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    // small features keep more of their own shape
                    const float k = flat * (1.f - 0.5f * prot[i]);
                    for (int c = 0; c < 3; ++c)
                        P[i * 3 + c] += k * (E[i * 3 + c] - P[i * 3 + c]);
                });
                tm.lap("wc flatten");
                poll();
            }
            // (b) brush strokes: a direction field (structure tensor along shapes, a fixed slanted hand where the
            //     picture has no direction) and a walk along it: the wash is smoothed along the stroke (stopping at
            //     colour changes) and a stroke texture (two noise images averaged along the walk) is collected
            const float stk = (float)Clampd(pin.strokes / 100.0, 0, 1);
            if (stk > 0.f || view == V_STROKES)
            {
                const double L = Clampd(pin.strokeLen, 2, 200) * ds;
                Buf dx, dy;
                dx.alloc(N);
                dy.alloc(N);
                {
                    Buf st;
                    st.alloc(N * 3);
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        const float* c = P + i * 3;
                        const float l = Luma(x > 0 ? c - 3 : c), r = Luma(x + 1 < W ? c + 3 : c);
                        const float u = Luma(y > 0 ? c - (size_t)W * 3 : c), d = Luma(y + 1 < H ? c + (size_t)W * 3 : c);
                        const float gx = 0.5f * (r - l) * A[i], gy = 0.5f * (d - u) * A[i];
                        st[i * 3] = gx * gx;
                        st[i * 3 + 1] = gx * gy;
                        st[i * 3 + 2] = gy * gy;
                    });
                    RFPlain<3>(st.data(), W, H, std::max(1.5, L * 0.3), 3, T);
                    const double a0 = Clampd(pin.strokeAngle, -360, 360) * 3.14159265358979 / 180.0;
                    std::vector<float> ang;
                    NoiseField(ang, W, H, f.ox, f.oy, ds, 320.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 320.0, 3, 0.5, seed ^ 0x57A0u, ds); });
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        const float jxx = st[i * 3], jxy = st[i * 3 + 1], jyy = st[i * 3 + 2];
                        const float tr = jxx + jyy, disc = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.f * jxy * jxy);
                        const float coh = disc / (tr + 1e-9f);
                        const float gate = Smooth(2e-5f, 3e-4f, tr) * Smooth(0.15f, 0.6f, coh);
                        // doubled angles: the stroke runs ALONG the shape (perpendicular to the gradient)
                        float c2 = disc > 1e-12f ? -(jxx - jyy) / disc : 0.f, s2 = disc > 1e-12f ? -2.f * jxy / disc : 0.f;
                        const double a = a0 + 0.55 * ang[i];  // the hand's default slant, varying slowly (+-30 deg)
                        // screen angle counter-clockwise with y down: direction (cos a, -sin a)
                        const float c0 = (float)pm::cos(2.0 * a), s0 = (float)-pm::sin(2.0 * a);
                        const float vx = gate * c2 + (1.f - gate) * c0, vy = gate * s2 + (1.f - gate) * s0;
                        const float th = 0.5f * pm::atan2(vy, vx);
                        dx[i] = pm::cos(th);
                        dy[i] = pm::sin(th);
                    });
                }
                // two noise images (anchored to the layer): faint bristle streaks (3 px) and soft stroke bodies (14 px)
                std::vector<float> n0, n1;
                NoiseField(n0, W, H, f.ox, f.oy, ds, 3.0 / 2, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 3.0, 2, 0.5, seed ^ 0xB415u, ds); });
                NoiseField(n1, W, H, f.ox, f.oy, ds, 14.0 / 2, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 14.0, 2, 0.5, seed ^ 0xB416u, ds); });
                const int n = std::max(1, std::min(12, (int)std::ceil(L)));
                const float step = (float)std::max(1.0, L / n);
                strokeTex.alloc(N);
                std::vector<float> Po(N * 3);
                const float sc2 = 1.f / (2.f * 0.07f * 0.07f);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float* c0 = P + i * 3;
                    float acc[3] = {c0[0], c0[1], c0[2]}, wsum = 1.f, ns = 0.25f * n0[i] + 0.75f * n1[i], nw = 1.f, nw2 = 1.f;
                    for (int sgn = -1; sgn <= 1; sgn += 2)
                    {
                        float px = x + 0.5f, py = y + 0.5f, ddx = dx[i] * sgn, ddy = dy[i] * sgn;
                        for (int k = 1; k <= n; ++k)
                        {
                            px += ddx * step;
                            py += ddy * step;
                            const int xi = (int)std::floor(px), yi = (int)std::floor(py);
                            if (xi < 0 || yi < 0 || xi >= W || yi >= H)
                                break;
                            const size_t j = (size_t)yi * W + xi;
                            if (useBar && bar[j])
                                break;  // strokes stop at lines
                            float ndx = dx[j], ndy = dy[j];
                            if (ndx * ddx + ndy * ddy < 0.f)
                            {
                                ndx = -ndx;
                                ndy = -ndy;
                            }
                            ddx = ndx;
                            ddy = ndy;
                            const float* cj = P + j * 3;
                            const float d0 = cj[0] - c0[0], d1 = cj[1] - c0[1], d2 = cj[2] - c0[2];
                            const float t = (float)k / n;
                            const float wk = pm::exp(-2.f * t * t) * pm::exp(-(d0 * d0 + d1 * d1 + d2 * d2) * sc2) * A[j];
                            acc[0] += wk * cj[0];
                            acc[1] += wk * cj[1];
                            acc[2] += wk * cj[2];
                            wsum += wk;
                            ns += wk * (0.25f * n0[j] + 0.75f * n1[j]);
                            nw += wk;
                            nw2 += wk * wk;
                        }
                    }
                    for (int c = 0; c < 3; ++c)
                        Po[i * 3 + c] = acc[c] / wsum;
                    // the averaged noise has sd ~ 0.3 * sqrt(sum w^2) / sum w: normalise to ~N(0,1)
                    strokeTex[i] = Clampf((ns / nw) / (0.3f * std::sqrt(nw2) / nw + 1e-3f), -2.5f, 2.5f);
                });
                const float k = 0.7f * stk;
                ForPx(W, H, T, [&](size_t i, int, int) {
                    for (int c = 0; c < 3; ++c)
                        P[i * 3 + c] += k * (Po[i * 3 + c] - P[i * 3 + c]);
                });
                if (view == V_STROKES)
                {
                    viewWc.resize(N * 3);
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        const float v = Clampf(0.5f + 0.18f * strokeTex[i], 0.f, 1.f);
                        viewWc[i * 3] = v * (0.6f + 0.4f * std::fabs(dx[i]));
                        viewWc[i * 3 + 1] = v;
                        viewWc[i * 3 + 2] = v * (0.6f + 0.4f * std::fabs(dy[i]));
                    });
                }
                tm.lap("wc strokes");
                poll();
            }

            // (c) optical density (paper white = 0); values above 1 (HDR) pass through unchanged.
            //     tend: how much this pigment granulates (ultramarine-like blues, earths: yes; vivid staining hues: no)
            //     ink: near-black areas stay as they are (lashes, black hair shadows), like ink, not a grey wash
            Buf D1, ink, Yw;
            D1.alloc(N * 3);
            ink.alloc(N);
            Yw.alloc(N);
            ForPx(W, H, T, [&](size_t i, int, int) {
                const float* c = P + i * 3;
                Yw[i] = std::min(c[0], std::min(c[1], c[2]));
                ink[i] = 1.f - Smooth(0.07f, 0.17f, Luma(&B[i * 3]));
                for (int k = 0; k < 3; ++k)
                    D1[i * 3 + k] = -pm::log(Clampf(c[k], 1e-3f, 1.f));
            });
            // (d) glazing: a light local-colour wash + a shadow glaze on top (tonal separation): base density =
            //     the lightest density nearby (min filter, smoothed), the glaze is the rest. Each layer gets its own
            //     wobble, bleed and rim; densities add (transparent layers multiply)
            const float gz = (float)Clampd(pin.glaze / 100.0, 0, 1);
            Buf D2;
            if (gz > 0.f || view == V_GLAZE)
            {
                std::vector<float> dm(N), env(N);
                ForPx(W, H, T, [&](size_t i, int, int) { dm[i] = (D1[i * 3] + D1[i * 3 + 1] + D1[i * 3 + 2]) * (1.f / 3.f); });
                const int R = std::max(1, (int)std::lround(14.0 * ds));
                MinMax(dm.data(), env.data(), W, H, R, false, T, tmp1);
                RFPlain<1>(env.data(), W, H, R * 0.6, 3, T);
                D2.alloc(N * 3);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    const float base = std::min(dm[i], env[i]);
                    const float s = dm[i] > 1e-4f ? 1.f - gz * (1.f - base / dm[i]) : 1.f;
                    for (int c = 0; c < 3; ++c)
                    {
                        const float d = D1[i * 3 + c];
                        D1[i * 3 + c] = d * s;
                        D2[i * 3 + c] = d * (1.f - s);
                    }
                });
                if (view == V_GLAZE)
                {
                    viewWc.resize(N * 3);
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        for (int c = 0; c < 3; ++c)
                            viewWc[i * 3 + c] = pm::exp(-D2[i * 3 + c] * 1.5f);
                    });
                }
            }
            Buf* layers[2] = {&D1, D2.empty() ? nullptr : &D2};
            // (e) wobble per layer (different fields: the glaze does not sit exactly on the base = gaps and overlaps)
            const double wob = Clampd(pin.wobble, 0, 50) * ds;
            if (wob > 0.01 && W >= 2 && H >= 2)
            {
                const double ws = Clampd(pin.wobbleScale, 2, 1000);
                for (int li = 0; li < 2; ++li)
                {
                    if (!layers[li])
                        continue;
                    Buf& Dl = *layers[li];
                    std::vector<float> VX, VY;
                    const uint32_t s2 = seed ^ (li ? 0x51A9Eu : 0u);
                    NoiseField(VX, W, H, f.ox, f.oy, ds, ws / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, ws, 3, 0.45, s2 ^ 0xA11CEu, ds); });
                    NoiseField(VY, W, H, f.ox, f.oy, ds, ws / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, ws, 3, 0.45, s2 ^ 0xB0B5u, ds); });
                    Buf Q;
                    Q.alloc(N * 3);
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        const float am = (float)(wob * 1.6) * (1.f - prot[i]);
                        const double sx = Clampd(x + am * VX[i], 0, W - 1), sy = Clampd(y + am * VY[i], 0, H - 1);
                        const int x0 = std::min(W - 2, (int)sx), y0 = std::min(H - 2, (int)sy);
                        const float fx = (float)(sx - x0), fy = (float)(sy - y0);
                        float acc[3] = {0, 0, 0}, wa = 0.f;
                        const float wxs[2] = {1.f - fx, fx}, wys[2] = {1.f - fy, fy};
                        for (int jj = 0; jj < 2; ++jj)
                            for (int ii = 0; ii < 2; ++ii)
                            {
                                const size_t k = (size_t)(y0 + jj) * W + (x0 + ii);
                                const float ww = wxs[ii] * wys[jj] * w[k];
                                wa += ww;
                                for (int c = 0; c < 3; ++c)
                                    acc[c] += ww * Dl[k * 3 + c];
                            }
                        const float eps = 0.05f;
                        for (int c = 0; c < 3; ++c)
                            Q[i * 3 + c] = (acc[c] + eps * Dl[i * 3 + c]) / (wa + eps);
                    });
                    std::swap(Dl.p, Q.p);
                }
                tm.lap("wc wobble");
                poll();
            }
            // (f) bleeding, lost and found: density diffuses across boundaries where the two sides are close in
            //     value (edges get lost), much less where the contrast is high (edges stay found); varied by noise
            const float bl = (float)Clampd(pin.bleed / 100.0, 0, 1);
            const double bw = Clampd(pin.bleedWidth, 0.5, 300);
            if (bl > 0.f)
            {
                std::vector<float> NB;
                NoiseField(NB, W, H, f.ox, f.oy, ds, bw * 5.0 / 8, T, [&](double gx, double gy) { return Fbm(gx * par, gy, bw * 5.0, 4, 0.55, seed ^ 0xB1EEDu, ds); });
                // one smoothing of the total density; each layer keeps its share of the result
                Buf Dt;
                Dt.alloc(N * 3);
                ForPx(W, H, T, [&](size_t i, int, int) {
                    for (int c = 0; c < 3; ++c)
                        Dt[i * 3 + c] = D1[i * 3 + c] + (layers[1] ? D2[i * 3 + c] : 0.f);
                });
                std::vector<float> Dg(N * 3);
                if (N > kHalfResPx && bw * ds >= 6.0)
                    SmoothNorm3Half(Dt.data(), w.data(), Dg.data(), W, H, bw * ds * 0.5, 0.0, useBar ? bar.data() : nullptr, T, tmp4, nullptr);
                else
                {
                    Edges e;
                    if (useBar)
                        e.bar = bar.data();
                    SmoothNorm3(Dt.data(), w.data(), Dg.data(), W, H, bw * ds * 0.5, e, T, tmp4);
                }
                ForPx(W, H, T, [&](size_t i, int, int) {
                    const float dm = (Dt[i * 3] + Dt[i * 3 + 1] + Dt[i * 3 + 2]) * (1.f / 3.f);
                    const float gm = (Dg[i * 3] + Dg[i * 3 + 1] + Dg[i * 3 + 2]) * (1.f / 3.f);
                    const float contrast = 2.f * std::fabs(dm - gm);  // ~ the density step of a nearby boundary
                    const float lost = 1.f - Smooth(0.06f, 0.40f, contrast);
                    const float k = Clampf(bl * (0.2f + 0.95f * lost) * (0.55f + 0.6f * Smooth(-0.4f, 0.5f, NB[i])), 0.f, 1.f) *
                                    (1.f - 0.6f * prot[i]);
                    for (int c = 0; c < 3; ++c)
                    {
                        const float t = Dt[i * 3 + c], t2 = t + k * (Dg[i * 3 + c] - t);
                        if (t > 1e-5f)
                        {
                            const float r = t2 / t;
                            D1[i * 3 + c] *= r;
                            if (layers[1])
                                D2[i * 3 + c] *= r;
                        }
                        else
                            D1[i * 3 + c] = t2;  // bled into bare paper: the base wash
                    }
                });
                tm.lap("wc bleed");
                poll();
            }
            // (g) backruns (blooms) grow FROM wet edges: wash boundaries inside "wet" patches of the picture seed a
            //     bloom that spreads into the drying wash with a fractal (cauliflower) front: lighter inside, a dark
            //     ring of pushed pigment at the front. Driven by the picture's own boundaries, not free noise blots.
            const float br = (float)Clampd(pin.backrun / 100.0, 0, 1);
            const float un = (float)Clampd(pin.uneven / 100.0, 0, 1);
            const float dil = (float)Clampd(pin.dilution / 100.0, 0, 1);
            const float kw = (float)Clampd(pin.keepWhite / 100.0, 0, 1);
            Buf kf;  // density factor shared by both layers
            kf.alloc(N);
            {
                std::vector<float> S, NF, NU, WET;
                if (br > 0.f)
                {
                    const double bloomR = std::max(10.0, bw * 2.5);
                    NoiseField(WET, W, H, f.ox, f.oy, ds, 300.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 300.0, 3, 0.5, seed ^ 0x3E7u, ds); });
                    NoiseField(NF, W, H, f.ox, f.oy, ds, bloomR / 4, T,
                               [&](double gx, double gy) { return Fbm(gx * par, gy, bloomR, 3, 0.6, seed ^ 0xCA11u, ds); });
                    S.resize(N);
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        auto dmAt = [&](size_t j) {
                            float s = (D1[j * 3] + D1[j * 3 + 1] + D1[j * 3 + 2]);
                            if (layers[1])
                                s += D2[j * 3] + D2[j * 3 + 1] + D2[j * 3 + 2];
                            return s * (1.f / 3.f);
                        };
                        const size_t r = x + 1 < W ? i + 1 : i, d = y + 1 < H ? i + W : i;
                        const float g = std::fabs(dmAt(r) - dmAt(i)) + std::fabs(dmAt(d) - dmAt(i));
                        // seeds: clear wash boundaries in wet patches, not the busy texture of small shapes
                        S[i] = Smooth(0.04f, 0.14f, g) * Smooth(0.05f, 0.35f, WET[i]) * (1.f - Smooth(0.03f, 0.15f, busy[i])) * A[i];
                    });
                    RFPlain<1>(S.data(), W, H, bloomR * ds * 0.6, 3, T);
                }
                if (un > 0.f)  // pigment flow: slow, smooth density variation inside washes
                    NoiseField(NU, W, H, f.ox, f.oy, ds, 190.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 190.0, 3, 0.45, seed ^ 0x0E7Eu, ds); });
                ForPx(W, H, T, [&](size_t i, int, int) {
                    float dm = (D1[i * 3] + D1[i * 3 + 1] + D1[i * 3 + 2]) * (1.f / 3.f);
                    if (layers[1])
                        dm += (D2[i * 3] + D2[i * 3 + 1] + D2[i * 3 + 2]) * (1.f / 3.f);
                    float k = 1.f;
                    if (br > 0.f)
                    {
                        // the bloom: where the edge proximity beats a fractal threshold, inside a painted wash
                        const float tau = 0.08f * (1.f + 0.8f * NF[i]);
                        // mid-value washes only (a bloom in near-black paint or next to ink reads as a stain), never on small features
                    const float m = Smooth(tau, tau + 0.035f, S[i]) * Smooth(0.12f, 0.35f, dm) * (1.f - Smooth(0.7f, 1.1f, dm)) * (1.f - ink[i]) *
                                    (1.f - prot[i]) * (1.f - Smooth(0.05f, 0.2f, busy[i]));
                        const float ring = Clampf(4.f * m * (1.f - m), 0.f, 1.f);
                        k *= (1.f - 0.45f * br * m) * (1.f + 0.35f * br * ring);  // the rim step adds the rest of the dark front
                    }
                    if (un > 0.f)
                        k *= std::max(0.f, 1.f + 0.65f * un * NU[i]);
                    k *= (1.f - 0.6f * dil) * (1.f - kw * Smooth(0.86f, 0.97f, Yw[i]));
                    kf[i] = k;
                });
            }
            for (int li = 0; li < 2; ++li)
                if (layers[li])
                {
                    Buf& Dl = *layers[li];
                    ForPx(W, H, T, [&](size_t i, int, int) {
                        for (int c = 0; c < 3; ++c)
                            Dl[i * 3 + c] *= kf[i];
                    });
                }
            kf.release();
            poll();
            // (h) pigment pools at each wash's edge: positive part of (density - its blur) = the "sharpen" signal,
            //     mapped to a relative density increase. The width follows the wash size (busy = small shapes = thin
            //     rims, big quiet washes = wide rims); "gravity" makes the bottom edges (where pigment ran) stronger.
            const float rim = (float)Clampd(pin.rim / 100.0, 0, 2);
            if (rim > 0.f || view == V_RIM)
            {
                const double rw = Clampd(pin.rimWidth, 0.3, 100) * ds;
                const bool paperOutside = pin.edgeMode == EM_PAPER;
                const float grav = (float)Clampd(pin.gravity / 100.0, 0, 1);
                const int NL = layers[1] ? 2 : 1;
                // channels: layer means (premultiplied by the "inside" weight), the weight
                auto fill3 = [&](Buf& v) {
                    ForPx(W, H, T, [&](size_t i, int x, int y) {
                        const bool inLayer = x >= f.layer.l && x < f.layer.r && y >= f.layer.t && y < f.layer.b;
                        // transparent pixels inside the layer = bare paper (rim at a cel's silhouette)
                        const float om = (inLayer || paperOutside) ? 1.f : 0.f;
                        for (int li = 0; li < 2; ++li)
                        {
                            float dmv = 0.f;
                            if (li < NL)
                            {
                                const Buf& Dl = *layers[li];
                                dmv = (Dl[i * 3] + Dl[i * 3 + 1] + Dl[i * 3 + 2]) * (1.f / 3.f) * A[i];
                            }
                            v[i * 3 + li] = dmv * om;
                        }
                        v[i * 3 + 2] = om;
                    });
                };
                Buf vt, vw;  // thin (rw / 2) and wide (rw * 2) blurs
                vt.alloc(N * 3);
                vw.alloc(N * 3);
                fill3(vt);
                fill3(vw);
                RFPlain<3>(vt.data(), W, H, std::max(0.6, rw * 0.5), 3, T);
                if (N > kHalfResPx && rw * 2.0 >= 3.0)
                    RFPlain3Half(vw.data(), W, H, rw * 2.0, T);
                else
                    RFPlain<3>(vw.data(), W, H, rw * 2.0, 3, T);
                const float rk = 2.4f * std::max(rim, view == V_RIM ? 0.7f : 0.f);
                if (view == V_RIM)
                    viewWc.assign(N * 3, 0.f);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float t = Smooth(0.04f, 0.25f, busy[i]);  // 0 = big quiet wash, 1 = busy small shapes
                    float total = 0.f;
                    for (int li = 0; li < NL; ++li)
                    {
                        Buf& Dl = *layers[li];
                        const float dm = (Dl[i * 3] + Dl[i * 3 + 1] + Dl[i * 3 + 2]) * (1.f / 3.f);
                        const float da = dm * A[i];
                        auto norm = [&](const Buf& v, size_t j) { return v[j * 3 + 2] > 1e-6f ? v[j * 3 + li] / v[j * 3 + 2] : da; };
                        const float gt = norm(vt, i), gw = norm(vw, i);
                        const float Rt = std::max(0.f, da - gt), Rw = std::max(0.f, da - gw);
                        float R = 0.45f * Rt + 0.55f * (t * Rt + (1.f - t) * Rw);
                        if (grav > 0.f)
                        {
                            // the wide blur falls off downwards at a wash's bottom edge (y grows downwards)
                            const size_t u = y > 0 ? i - W : i, d = y + 1 < H ? i + W : i;
                            const size_t l = x > 0 ? i - 1 : i, r = x + 1 < W ? i + 1 : i;
                            const float gy = 0.5f * (norm(vw, d) - norm(vw, u)), gx = 0.5f * (norm(vw, r) - norm(vw, l));
                            const float down = -gy / (std::sqrt(gx * gx + gy * gy) + 1e-5f);
                            R *= 1.f + grav * 0.8f * down;
                        }
                        const float rf = std::min(4.f, 1.f + rk * std::max(0.f, R) / (dm + 0.08f));
                        const float k = 1.f + (rf - 1.f) * (1.f - ink[i]);
                        if (rim > 0.f)
                            for (int c = 0; c < 3; ++c)
                                Dl[i * 3 + c] *= k;
                        total += rf - 1.f;
                    }
                    if (view == V_RIM)
                    {
                        const float v = Clampf(total * 0.8f, 0.f, 1.f);
                        viewWc[i * 3] = v;
                        viewWc[i * 3 + 1] = 0.6f * v;
                        viewWc[i * 3 + 2] = 0.2f * v;
                    }
                });
                tm.lap("wc rim");
                poll();
            }
            Dt.alloc(N * 3);
            ForPx(W, H, T, [&](size_t i, int, int) {
                for (int c = 0; c < 3; ++c)
                    Dt[i * 3 + c] = D1[i * 3 + c] + (layers[1] ? D2[i * 3 + c] : 0.f);
            });
        };
        Buf Dt, prot, busy, strokeTex;
        const bool painter = pin.wcMode == WCM_PAINT;
        if (painter)
        {
            // round 6: the region painter -> total density at buffer resolution
            Dt.alloc(N * 3);
            PaintIn pi{W, H, ds, par, f.ox, f.oy, f.layer, seed, T, P, A.data(), bar.empty() ? nullptr : bar.data(),
                       alpha.empty() ? nullptr : alpha.data()};
            PaintWatercolour(pin, pi, Dt.data(), wcView ? &viewWc : nullptr, poll);
            prot.alloc(N);
            busy.alloc(N);
            std::vector<float> pa(N, 0.f);
            if (!alpha.empty())
                ForPx(W, H, T, [&](size_t i, int, int) { pa[i] = alpha[i]; });
            RFPlain<1>(pa.data(), W, H, 4.0 * ds, 3, T);
            const float pd = (float)Clampd(pin.protectDetail / 100.0, 0, 1);
            ForPx(W, H, T, [&](size_t i, int, int) {
                prot[i] = pd * Smooth(0.05f, 0.2f, pa[i]);
                busy[i] = 0.f;
            });
            tm.lap("wc painter");
        }
        else if (N > kHalfResPx)
        {
            // big frame: the washes at half resolution (2x2 boxes; walls and lines max-pooled), then bilinear up
            const int w2 = (W + 1) / 2, h2 = (H + 1) / 2;
            const size_t N2 = (size_t)w2 * h2;
            std::vector<float> P2(N2 * 3), B2(N2 * 3), A2(N2), al2(alpha.empty() ? 0 : N2);
            std::vector<uint8_t> bar2(bar.empty() ? 0 : N2);
            PFor(h2, T, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < w2; ++x)
                    {
                        const size_t o = (size_t)y * w2 + x;
                        float pa[3] = {0, 0, 0}, pp[3] = {0, 0, 0}, bb[3] = {0, 0, 0}, as = 0.f, am = 0.f;
                        uint8_t br2 = 0;
                        for (int j = 0; j < 2; ++j)
                            for (int ii = 0; ii < 2; ++ii)
                            {
                                const size_t k = (size_t)std::min(H - 1, 2 * y + j) * W + std::min(W - 1, 2 * x + ii);
                                for (int c = 0; c < 3; ++c)
                                {
                                    pa[c] += A[k] * P[k * 3 + c];
                                    pp[c] += P[k * 3 + c];
                                    bb[c] += B[k * 3 + c];
                                }
                                as += A[k];
                                if (!alpha.empty())
                                    am = std::max(am, alpha[k]);
                                if (!bar.empty())
                                    br2 |= bar[k];
                            }
                        for (int c = 0; c < 3; ++c)
                        {
                            P2[o * 3 + c] = as > 1e-6f ? pa[c] / as : 0.25f * pp[c];
                            B2[o * 3 + c] = 0.25f * bb[c];
                        }
                        A2[o] = 0.25f * as;
                        if (!alpha.empty())
                            al2[o] = am;
                        if (!bar.empty())
                            bar2[o] = br2;
                    }
            });
            Buf Dt2, prot2, busy2, st2;
            std::vector<float> vw2;
            const Rect L2{FloorDiv(f.layer.l, 2), FloorDiv(f.layer.t, 2), (f.layer.r + 1) / 2, (f.layer.b + 1) / 2};
            washes(w2, h2, ds * 0.5, WcFrame{FloorDiv(f.ox, 2), FloorDiv(f.oy, 2), L2}, P2.data(), B2.data(), A2, bar2, al2, Dt2, prot2,
                   busy2, st2, vw2);
            auto up = [&](const Buf& s2, Buf& d, int nc) {
                d.alloc(N * nc);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    const float fx = (x + 0.5f) * 0.5f - 0.5f, fy = (y + 0.5f) * 0.5f - 0.5f;
                    for (int c = 0; c < nc; ++c)
                        d[i * nc + c] = Bilerp(s2.data(), w2, h2, nc, c, fx, fy);
                });
            };
            up(Dt2, Dt, 3);
            up(prot2, prot, 1);
            up(busy2, busy, 1);
            if (!st2.empty())
                up(st2, strokeTex, 1);
            if (!vw2.empty())
            {
                viewWc.resize(N * 3);
                ForPx(W, H, T, [&](size_t i, int x, int y) {
                    for (int c = 0; c < 3; ++c)
                        viewWc[i * 3 + c] = Bilerp(vw2.data(), w2, h2, 3, c, (x + 0.5f) * 0.5f - 0.5f, (y + 0.5f) * 0.5f - 0.5f);
                });
            }
        }
        else
            washes(W, H, ds, WcFrame{f.ox, f.oy, f.layer}, P, B.data(), A, bar, alpha, Dt, prot, busy, strokeTex, viewWc);
        // part 2 (full resolution): pigment tendency, ink areas, HDR excess from the full picture
        const float stk = (float)Clampd(pin.strokes / 100.0, 0, 1);
        Buf ink, tend;
        ink.alloc(N);
        tend.alloc(N);
        std::vector<float> hdr;
        bool anyHdr = false;
        for (size_t i = 0; i < N * 3 && !anyHdr; ++i)
            anyHdr = P[i] > 1.f;
        if (anyHdr)
            hdr.assign(N * 3, 0.f);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float* c = P + i * 3;
            const float mx = std::max(c[0], std::max(c[1], c[2])), mn = std::min(c[0], std::min(c[1], c[2]));
            ink[i] = painter ? 0.f : 1.f - Smooth(0.07f, 0.17f, Luma(&B[i * 3]));  // the painter lifts darks itself
            const float sat = (mx - mn) / (mx + 1e-4f);
            const float blue = Smooth(0.0f, 0.15f, c[2] - std::max(c[0], c[1]));
            const float earth = (1.f - Smooth(0.2f, 0.6f, sat)) * Smooth(0.0f, 0.10f, c[0] - c[2]);
            tend[i] = Clampf(0.2f + 0.8f * blue + 0.5f * earth, 0.f, 1.f);
            if (anyHdr)
                for (int k = 0; k < 3; ++k)
                    hdr[i * 3 + k] = std::max(0.f, c[k] - 1.f);
        });
        // (i) strokes in the pigment, granulation in the paper's valleys (pigment dependent, never on small
        //     features), the paper's relief; back to colour. Near-black ink areas keep the untouched picture.
        const float gr = (float)Clampd(pin.gran / 100.0, 0, 1), pp = (float)Clampd(pin.paper / 100.0, 0, 1);
        std::vector<float> FN;
        if (gr > 0.f)
            NoiseField(FN, W, H, f.ox, f.oy, ds, 1.7 / 2, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 1.7, 2, 0.6, seed ^ 0x6A11u, ds); });
        ForPx(W, H, T, [&](size_t i, int x, int y) {
            float d[3];
            for (int c = 0; c < 3; ++c)
                d[c] = Dt[i * 3 + c];
            const float dm = (d[0] + d[1] + d[2]) * (1.f / 3.f);
            const float h = (gr > 0.f || pp > 0.f) ? paperAt(i, x, y) : 0.5f;
            const float valley = (0.5f - h) * 2.f;
            const float capD = std::min(dm, 0.55f) * (1.f - ink[i]);
            float addD = capD * (gr * tend[i] * (0.30f * valley + 0.10f * (FN.empty() ? 0.f : FN[i])) * (1.f - 0.85f * prot[i]) + pp * 0.10f * valley);
            if (!strokeTex.empty())
                addD += std::min(dm, 0.8f) * (1.f - ink[i]) * stk * 0.10f * strokeTex[i] * (1.f - 0.6f * prot[i]) * (1.f - 0.6f * Smooth(0.04f, 0.25f, busy[i]));
            // painter: dry-brush sparkle - a light, fast wash skips the paper's peaks
            if (painter && pp > 0.f)
                addD -= dm * pp * 0.5f * Smooth(0.72f, 0.9f, h) * (1.f - Smooth(0.12f, 0.35f, dm)) * (1.f - prot[i]);
            const float shade = 1.f + pp * 0.07f * (-valley);
            for (int c = 0; c < 3; ++c)
            {
                const float share = dm > 1e-4f ? d[c] / dm : 1.f;
                float v = pm::exp(-std::max(0.f, d[c] + addD * share)) * shade * (painter ? kPaper[c] : 1.f);
                if (anyHdr)
                    v += hdr[i * 3 + c];
                v += ink[i] * (B[i * 3 + c] - v);
                P[i * 3 + c] = v;
            }
        });
        if (copyP && wcOn)
            ForPx(W, H, T, [&](size_t i, int, int) {
                const float m = wc * mapOf(MT_WATERCOLOR, i);
                for (int c = 0; c < 3; ++c)
                    B[i * 3 + c] += m * (P[i * 3 + c] - B[i * 3 + c]);
            });
        tm.lap("wc paper");
        poll();
    }

    // ---- 5. ink: line controls / line style, then the lines multiplied back over the processed fill
    const bool compose = needLines && ((inpainted && anyProc) || lineOps);
    if (compose)
    {
        std::vector<float> dl = delta;
        const double th = Clampd(pin.lineThick, -10, 10) * ds;
        if (th > 0.01)
        {
            const int ri = (int)std::ceil(th - 1e-6);
            const float frac = (float)(th - (ri - 1));
            std::vector<float> amax(N);
            MinMax(alpha.data(), amax.data(), W, H, ri, true, T, tmp1);
            ForPx(W, H, T, [&](size_t i, int x, int y) {
                if (amax[i] <= alpha[i] + 1e-4f)
                    return;
                float best = alpha[i], bw = 1.f;
                size_t bi = i;
                for (int dy = -ri; dy <= ri; ++dy)
                    for (int dx = -ri; dx <= ri; ++dx)
                    {
                        const int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                            continue;
                        const float dist = std::sqrt((float)(dx * dx + dy * dy));
                        if (dist > (float)ri + 0.5f)
                            continue;
                        const float wr = dist > (float)(ri - 1) + 0.5f ? frac : 1.f;
                        const size_t k = (size_t)yy * W + xx;
                        if (alpha[k] * wr > best * bw)
                        {
                            best = alpha[k];
                            bi = k;
                            bw = wr;
                        }
                    }
                if (bi != i)
                    for (int c = 0; c < 3; ++c)
                    {
                        const float v = delta[bi * 3 + c] * bw;
                        dl[i * 3 + c] = std::fabs(v) > std::fabs(dl[i * 3 + c]) ? v : dl[i * 3 + c];
                    }
            });
        }
        else if (th < -0.01)
        {
            const int ri = (int)std::ceil(-th - 1e-6);
            std::vector<float> amin(N);
            MinMax(alpha.data(), amin.data(), W, H, ri, false, T, tmp1);
            const float frac = (float)std::min(1.0, -th);
            ForPx(W, H, T, [&](size_t i, int, int) {
                if (alpha[i] <= 0.f)
                    return;
                const float s = Clampf(amin[i] / std::max(alpha[i], 1e-3f), 0.f, 1.f);
                const float k = 1.f + frac * (s - 1.f);
                for (int c = 0; c < 3; ++c)
                    dl[i * 3 + c] *= k;
            });
        }
        // line style: pencil = lighter warm grey graphite, broken here and there, catching the paper's peaks;
        // ink = a slightly varied, soft-edged black
        if (pin.lineStyle == LS_PENCIL || pin.lineStyle == LS_INK)
        {
            std::vector<float> BK;
            const bool pencil = pin.lineStyle == LS_PENCIL;
            NoiseField(BK, W, H, f.ox, f.oy, ds, (pencil ? 16.0 : 40.0) / 4, T,
                       [&](double gx, double gy) { return Fbm(gx * par, gy, pencil ? 16.0 : 40.0, 3, 0.55, seed ^ 0x9E7C1u, ds); });
            const float tint[3] = {pencil ? 0.88f : 1.f, pencil ? 1.f : 1.f, pencil ? 1.22f : 1.06f};
            ForPx(W, H, T, [&](size_t i, int x, int y) {
                const float m = (dl[i * 3] + dl[i * 3 + 1] + dl[i * 3 + 2]) * (1.f / 3.f);
                if (m <= 0.f)
                    return;
                float k;
                if (pencil)
                {
                    const float keep = Smooth(-0.45f, 0.0f, BK[i]);           // soft breaks in the stroke
                    const float h = paperAt(i, x, y);                          // graphite sits on the peaks
                    k = 0.55f * (0.35f + 0.65f * keep) * (0.7f + 0.6f * h);
                }
                else
                    k = 1.05f * (0.85f + 0.3f * Smooth(-0.5f, 0.5f, BK[i]));
                for (int c = 0; c < 3; ++c)
                    dl[i * 3 + c] = m * k * tint[c];
            });
        }
        const double soft = Clampd(pin.lineSoft, 0, 100) * ds * 0.5 + (pin.lineStyle == LS_INK ? 0.35 * ds : 0.0);
        if (soft > 0.05)
            RFPlain<3>(dl.data(), W, H, soft, 3, T);
        const float dk = 1.f + (float)Clampd(pin.lineDark / 100.0, -1, 2);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float m = mapOf(MT_LINES, i);
            for (int c = 0; c < 3; ++c)
            {
                float d = dl[i * 3 + c] * dk;
                d = delta[i * 3 + c] + m * (d - delta[i * 3 + c]);
                if (!inpainted)
                    d -= delta[i * 3 + c];
                B[i * 3 + c] = (B[i * 3 + c] + kInkEps) * pm::exp(-d) - kInkEps;
            }
        });
        tm.lap("ink");
    }
    // ---- 6. output (straight colour, the input's alpha) / views. out may be f.rgba (in place).
    ForPx(W, H, T, [&](size_t i, int, int) {
        const float* s = f.rgba + i * 4;
        float in[4] = {s[0], s[1], s[2], s[3]};
        float* o = out + i * 4;
        if (view == V_RESULT)
        {
            const float m = amt * mapOf(MT_ALL, i);
            for (int c = 0; c < 3; ++c)
            {
                const float iv = std::isfinite(in[c]) ? in[c] : 0.f;
                const float bv = pin.linear ? ToLinear(B[i * 3 + c]) : B[i * 3 + c];
                const float v = iv + m * (bv - iv);
                o[c] = std::isfinite(v) ? std::max(v, std::min(0.f, iv)) : 0.f;
            }
            o[3] = in[3];
            return;
        }
        float r = 0, g = 0, b = 0;
        switch (view)
        {
        case V_LINES:
        {
            const float a = alpha.empty() ? 0.f : alpha[i];
            const float yl = Luma(in);
            const float base = 0.25f * (std::isfinite(yl) ? Clampf(yl, 0.f, 1.f) : 0.f);
            const bool lt = !light.empty() && light[i];
            r = base + ((lt ? 0.4f : 1.f) - base) * a;
            g = base + ((lt ? 0.9f : 1.f) - base) * a;
            b = base + ((lt ? 1.f : 0.6f) - base) * a;
            if (!bar.empty() && bar[i])
                b = std::max(b, 0.15f);
            break;
        }
        case V_FINE:
        case V_MID:
        case V_LARGE:
        {
            const float k = view == V_FINE ? 4.f : view == V_MID ? 3.f : 2.f;
            r = g = b = Clampf(0.5f + k * (bandView.empty() ? 0.f : bandView[i]), 0.f, 1.f);
            break;
        }
        case V_RIM:
        case V_STROKES:
        case V_GLAZE:
            if (!viewWc.empty())
            {
                r = viewWc[i * 3];
                g = viewWc[i * 3 + 1];
                b = viewWc[i * 3 + 2];
            }
            break;
        case V_FILL:
            r = Fl[i * 3];
            g = Fl[i * 3 + 1];
            b = Fl[i * 3 + 2];
            break;
        default: break;
        }
        o[0] = r;
        o[1] = g;
        o[2] = b;
        o[3] = 1.f;
    });
    tm.lap("output");
    return ST_OK;
}

}  // namespace

bool IsIdentity(const Params& p)
{
    if (!(p.amount > 0))
        return true;
    if (p.view != V_RESULT)
        return false;
    const bool tex = p.fine != 0 || p.mid != 0 || p.large != 0;
    const bool blur = p.fillBlur > 0;
    const bool wc = p.wc > 0;
    const bool lineOps = p.lineThick != 0 || p.lineDark != 0 || p.lineSoft > 0 || p.lineStyle != LS_ASIS;
    return !tex && !blur && !wc && !lineOps;
}

Status Render(const Params& p, Frame& f)
{
    try
    {
        return RenderImpl(p, f);
    }
    catch (const AbortEx&)
    {
        return ST_ABORTED;
    }
    catch (const std::bad_alloc&)
    {
        return ST_OOM;
    }
    catch (...)
    {
        return ST_ERROR;
    }
}

}  // namespace sk
