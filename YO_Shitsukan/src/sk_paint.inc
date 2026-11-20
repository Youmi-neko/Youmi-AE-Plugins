// Shitsukan round 6: the region-based watercolour painter ("abstraction first, texture last").
// Included by sk_core.cpp (inside its anonymous namespace; uses the helpers of sk_util.h).
//
//   segmentation -> one wash per region (wet gradient, lifted darks, paper white) -> edge classification
//   (hard dried edge with a tide line | wet-in-wet merge) -> broken registration (overshoot / unpainted slivers)
//   -> pigment inside the washes (unevenness, charging, granulation, dry-brush sparkle) -> paper.
//
// Everything is computed on a working grid of k x k buffer px (k from the frame size, so Full and Half previews of
// a 1080p frame use the same grid), in optical density D = -log(colour), then brought back to the buffer with the
// registration offset applied while sampling. See docs\watercolor_model.md (round 6).

struct PaintIn
{
    int W, H;          // buffer
    double ds, par;    // buffer px per full px; pixel aspect
    int ox, oy;        // layer coords of buffer (0, 0)
    Rect layer;        // layer rect in buffer coords
    uint32_t seed;
    int T;
    const float* P;     // fill layer (lines removed), encoded RGB, N*3
    const float* A;     // alpha, N
    const uint8_t* bar; // line walls (may be null)
    const float* alpha; // line coverage (may be null)
    const float* depthFar = nullptr;  // depth matte: 0 at the reference depth .. 1 far (may be null)
    float kScale = 0.f, kWet = 0.f;  // depth influences (signed, already scaled by the master)
#ifdef SK_GPU
    // 0.6.0.5: non-null = the tail (5b), (5), (6) and the caller's painter paper may run on the GPU; then the colour
    // goes to gpuOut (the caller's fill P) and *gpuDone is set. gr / pp / pd / paperTex / ps / anyHdr: the paper's.
    GpuFrame* gpu = nullptr;
    float* gpuOut = nullptr;
    bool* gpuDone = nullptr;
    float gr = 0.f, pp = 0.f, pd = 0.f;
    const float* paperTex = nullptr;
    double ps = 1.0;
    bool anyHdr = false;
    const GpuInkIn* ink = nullptr;  // non-null: 5. ink as well (then *ink->done is set)
#endif
};

// two-pass 3-4 chamfer distance (in px) to the seed pixels (seed != 0); sequential, exact enough for washes
inline void Chamfer(const std::vector<uint8_t>& seedM, int W, int H, std::vector<float>& d)
{
    const float BIG = 1e9f, a = 1.f, b = 1.41421356f;
    d.assign((size_t)W * H, BIG);
    for (size_t i = 0; i < d.size(); ++i)
        if (seedM[i])
            d[i] = 0.f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            float& v = d[(size_t)y * W + x];
            if (x > 0) v = std::min(v, d[(size_t)y * W + x - 1] + a);
            if (y > 0)
            {
                const float* r = &d[(size_t)(y - 1) * W];
                v = std::min(v, r[x] + a);
                if (x > 0) v = std::min(v, r[x - 1] + b);
                if (x + 1 < W) v = std::min(v, r[x + 1] + b);
            }
        }
    for (int y = H - 1; y >= 0; --y)
        for (int x = W - 1; x >= 0; --x)
        {
            float& v = d[(size_t)y * W + x];
            if (x + 1 < W) v = std::min(v, d[(size_t)y * W + x + 1] + a);
            if (y + 1 < H)
            {
                const float* r = &d[(size_t)(y + 1) * W];
                v = std::min(v, r[x] + a);
                if (x > 0) v = std::min(v, r[x - 1] + b);
                if (x + 1 < W) v = std::min(v, r[x + 1] + b);
            }
        }
}

struct UF
{
    std::vector<int> p;
    explicit UF(size_t n) : p(n) { for (size_t i = 0; i < n; ++i) p[i] = (int)i; }
    int find(int x)
    {
        while (p[x] != x)
        {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    // the smaller index becomes the root: deterministic whatever the merge order
    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a < b) p[b] = a; else p[a] = b;
    }
};

// opponent colour used for region decisions: (2 x luma, r - g, (r + g) / 2 - b)
inline void Opp(const float* c, float* o)
{
    o[0] = 2.f * Luma(c);
    o[1] = c[0] - c[1];
    o[2] = 0.5f * (c[0] + c[1]) - c[2];
}
inline float OppDist(const float* a, const float* b) { return std::fabs(a[0] - b[0]) + std::fabs(a[1] - b[1]) + std::fabs(a[2] - b[2]); }

// a limited palette: 11 watercolour pigments (masstone, encoded RGB) and their 1:1 mixtures
// pigment sets (round 9): 0 watercolour pigments, 1 storybook soft, 2 storybook bright (11 pigments each, then all
// their 1:1 mixtures; the pure pigments first)
inline std::vector<std::array<float, 3>> MixSet(const float (*pig)[3], int n)
{
    std::vector<std::array<float, 3>> s;
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
        {
            std::array<float, 3> m;
            for (int c = 0; c < 3; ++c) m[c] = std::sqrt(pig[i][c] * pig[j][c]);  // subtractive mix: mean density
            s.push_back(m);
        }
    return s;
}
inline const std::vector<std::array<float, 3>>& PigmentSetN(int which)
{
    static const std::vector<std::array<float, 3>> soft = [] {
        const float pig[][3] = {{0.93f, 0.56f, 0.48f},   // coral
                                {0.64f, 0.73f, 0.57f},   // sage
                                {0.56f, 0.67f, 0.79f},   // dusty blue
                                {0.89f, 0.77f, 0.47f},   // mustard
                                {0.79f, 0.48f, 0.37f},   // terracotta
                                {0.72f, 0.65f, 0.80f},   // lavender
                                {0.58f, 0.44f, 0.36f},   // warm brown
                                {0.50f, 0.69f, 0.68f},   // soft teal
                                {0.90f, 0.66f, 0.70f},   // rose
                                {0.62f, 0.64f, 0.44f},   // olive
                                {0.44f, 0.46f, 0.56f}};  // slate
        return MixSet(pig, 11);
    }();
    static const std::vector<std::array<float, 3>> bright = [] {
        const float pig[][3] = {{0.90f, 0.22f, 0.20f},   // red
                                {0.98f, 0.83f, 0.16f},   // yellow
                                {0.16f, 0.42f, 0.86f},   // blue
                                {0.22f, 0.66f, 0.32f},   // green
                                {0.98f, 0.56f, 0.16f},   // orange
                                {0.86f, 0.28f, 0.56f},   // magenta
                                {0.38f, 0.72f, 0.95f},   // sky
                                {0.47f, 0.32f, 0.76f},   // violet
                                {0.58f, 0.34f, 0.20f},   // brown
                                {0.58f, 0.80f, 0.26f},   // leaf
                                {0.17f, 0.22f, 0.46f}};  // navy
        return MixSet(pig, 11);
    }();
    if (which == 1) return soft;
    if (which == 2) return bright;
    static const std::vector<std::array<float, 3>> wc = [] {
        const float pig[][3] = {{0.16f, 0.24f, 0.62f},  // ultramarine
                                {0.18f, 0.52f, 0.78f},  // cerulean
                                {0.10f, 0.45f, 0.38f},  // viridian
                                {0.40f, 0.52f, 0.20f},  // sap green
                                {0.80f, 0.62f, 0.30f},  // yellow ochre
                                {0.98f, 0.80f, 0.20f},  // cadmium yellow
                                {0.88f, 0.28f, 0.20f},  // cadmium red
                                {0.66f, 0.12f, 0.28f},  // alizarin
                                {0.62f, 0.33f, 0.18f},  // burnt sienna
                                {0.36f, 0.24f, 0.16f},  // burnt umber
                                {0.25f, 0.29f, 0.36f}}; // payne's grey
        return MixSet(pig, 11);
    }();
    return wc;
}

// out: painted colour (encoded RGB, N*3). viewOut (optional, N*3): the regions (mean colours with tide lines).
void PaintWatercolour(const Params& pin, const PaintIn& in, float* outRGB, std::vector<float>* viewOut, const std::function<void()>& poll,
                      std::vector<float>* busyOut = nullptr)
{
    const int W = in.W, H = in.H, T = in.T;
    const double ds = in.ds, par = in.par;
    // working grid: k x k buffer px per cell
    const int k = std::max(1, (int)std::lround(std::min(W, H) / 720.0));
    const int w = (W + k - 1) / k, h = (H + k - 1) / k;
    const size_t n = (size_t)w * h;
    const double dsw = ds / k;  // working px per full-res px
    const int oxw = FloorDiv(in.ox, k), oyw = FloorDiv(in.oy, k);
    auto XF = [&](double x) { return (x + oxw) / dsw * par; };  // full-res layer coords of a working px
    auto YF = [&](double y) { return (y + oyw) / dsw; };

    const float scale = (float)Clampd(pin.paintScale, 2, 400);                   // full-res px
    const float merge = (float)Clampd(pin.merge / 100.0, 0, 1);
    const float rimA = (float)Clampd(pin.rim / 100.0, 0, 2);
    const float rimWf = (float)Clampd(pin.rimWidth, 0.3, 100);
    const float wetA = (float)Clampd(pin.bleed / 100.0, 0, 1);
    const float wetWf = (float)Clampd(pin.bleedWidth, 0.5, 300);
    const float loose = (float)Clampd(pin.loose / 100.0, 0, 1);
    const float pw = (float)Clampd(pin.keepWhite / 100.0, 0, 1);
    const float lift = (float)Clampd(pin.liftDarks / 100.0, 0, 1);
    const float dil = (float)Clampd(pin.dilution / 100.0, 0, 1);
    const float grav = (float)Clampd(pin.gravity / 100.0, 0, 1);
    const float uneven = (float)Clampd(pin.uneven / 100.0, 0, 1);
    const float charge = (float)Clampd(pin.charge / 100.0, 0, 1);
    const float glaze = (float)Clampd(pin.glaze / 100.0, 0, 1);
    const float margin = (float)Clampd(pin.margin / 100.0, 0, 1);
    const float protA = (float)Clampd(pin.protectDetail / 100.0, 0, 1);
    const int palN = (int)Clampd(pin.palette, 0, 66);

    Timer ptm;
    // (6)'s distance to the lines (sequential chamfer): only alpha / A are needed - with the GPU tail it is computed
    // meanwhile on its own thread (the same function: the same values)
    auto lineDist = [&]() {
        std::vector<float> d;
        if (in.alpha && loose > 0.f)
        {
            // (not at the edge of transparency: a layer's own border can read as a "line" against a transparent surround)
            const size_t N0 = (size_t)W * H;
            std::vector<float> amin(N0), tmpm;
            MinMax(in.A, amin.data(), W, H, std::max(1, (int)std::lround(3.0 * ds)), false, T, tmpm);
            std::vector<uint8_t> lm(N0);
            for (size_t i = 0; i < N0; ++i) lm[i] = in.alpha[i] > 0.35f && amin[i] > 0.5f;
            amin.clear();
            amin.shrink_to_fit();
            Chamfer(lm, W, H, d);
        }
        return d;
    };
#ifdef SK_GPU
    std::future<std::vector<float>> dLineF;
    if (in.gpu && in.gpu->f && !viewOut)
        dLineF = std::async(std::launch::async, lineDist);
#endif
    // ---- (0) the working grid: alpha-weighted box means; walls / lines max-pooled
    std::vector<float> P(n * 3), A(n), al(n, 0.f);
    std::vector<uint8_t> wall(n, 0);
    PFor(h, T, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x)
            {
                float s[3] = {0, 0, 0}, sp[3] = {0, 0, 0}, as = 0.f, am = 0.f;
                int cnt = 0;
                uint8_t br = 0;
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < k; ++i)
                    {
                        const int xx = std::min(W - 1, x * k + i), yy = std::min(H - 1, y * k + j);
                        const size_t q = (size_t)yy * W + xx;
                        for (int c = 0; c < 3; ++c)
                        {
                            s[c] += in.A[q] * in.P[q * 3 + c];
                            sp[c] += in.P[q * 3 + c];
                        }
                        as += in.A[q];
                        ++cnt;
                        if (in.alpha) am = std::max(am, in.alpha[q]);
                        if (in.bar) br |= in.bar[q];
                    }
                const size_t o = (size_t)y * w + x;
                for (int c = 0; c < 3; ++c)
                    P[o * 3 + c] = Clampf(as > 1e-6f ? s[c] / as : sp[c] / cnt, 0.f, 1.f);
                A[o] = as / cnt;
                al[o] = am;
                wall[o] = br;
            }
    });
    poll();

    // small-feature map: density of line art nearby (faces, hands, small props): smaller regions, a tighter
    // registration, less pigment play there
    // a layer's own border against a transparent surround reads as a "line" to the line detector: not a line here
    {
        std::vector<float> amn(n), tmpa;
        MinMax(A.data(), amn.data(), w, h, 2, false, T, tmpa);
        ForPx(w, h, T, [&](size_t i, int, int) { al[i] *= Smooth(0.5f, 0.99f, amn[i]); });
    }
    std::vector<float> prot(al);
    RFPlain<1>(prot.data(), w, h, 6.0 * dsw, 3, T);
    ForPx(w, h, T, [&](size_t i, int, int) { prot[i] = protA * Smooth(0.05f, 0.18f, prot[i]); });

    ptm.lap("p0 grid");
    // ---- (1) abstraction: edge-preserving smoothing at the region scale (lines are walls)
    std::vector<float> S(n * 3), tmp4, wgt(n);
    // lines are SOFT walls: no weight on line pixels and a high edge cost across them, both continuous in the
    // line coverage (binary walls flip when a line moves by a fraction of a pixel: the washes then jump)
    std::vector<float> lineW(n);
    ForPx(w, h, T, [&](size_t i, int, int) {
        lineW[i] = Smooth(0.08f, 0.45f, al[i]);
        wgt[i] = A[i] * (1.f - lineW[i]);
    });
    auto guideGrad = [&](const std::vector<float>& src, std::vector<float>& gh, std::vector<float>& gv) {
        std::vector<float> g3(src);
        ForPx(w, h, T, [&](size_t i, int, int) {
            for (int c = 0; c < 3; ++c) g3[i * 3 + c] += 0.6f * lineW[i];
        });
        BuildGrad(g3.data(), w, h, gh, gv, T);
    };
    {
        std::vector<float> gh, gv;
        guideGrad(P, gh, gv);
        Edges e;
        e.gh = gh.data();
        e.gv = gv.data();
        e.k = (scale * dsw * 0.5) / 0.06;
        SmoothNorm3(P.data(), wgt.data(), S.data(), w, h, scale * dsw * 0.5, e, T, tmp4);
        guideGrad(S, gh, gv);
        e.gh = gh.data();
        e.gv = gv.data();
        SmoothNorm3(S.data(), wgt.data(), S.data(), w, h, scale * dsw * 0.5, e, T, tmp4);
    }
    // small features (eyes, mouths): much less abstraction there, so they stay readable as painted shapes
    ForPx(w, h, T, [&](size_t i, int, int) {
        const float m = 0.85f * prot[i];
        for (int c = 0; c < 3; ++c)
            S[i * 3 + c] += m * (P[i * 3 + c] - S[i * 3 + c]);
    });
    poll();

    // eyes / mouths: clusters of thick dark lines (hair strands and clothes folds are thinner and further apart:
    // they are painted as washes)
    std::vector<float> eye(al);
    RFPlain<1>(eye.data(), w, h, 3.0 * dsw, 3, T);
    ForPx(w, h, T, [&](size_t i, int, int) { eye[i] = protA * Smooth(0.20f, 0.42f, eye[i]); });
    RFPlain<1>(eye.data(), w, h, 2.0 * dsw, 2, T);

    ptm.lap("p1 abstraction");
    // ---- (2) busy texture (foliage, gravel): looser washes there, and the layered leaf-mass glazes below
    std::vector<float> busy(n);
    ForPx(w, h, T, [&](size_t i, int, int) { busy[i] = std::fabs(Luma(&P[i * 3]) - Luma(&S[i * 3])) * A[i]; });
    RFPlain<1>(busy.data(), w, h, scale * dsw, 3, T);
    ForPx(w, h, T, [&](size_t i, int, int) { busy[i] = Smooth(0.025f, 0.07f, busy[i]) * (1.f - prot[i]); });
    poll();

    ptm.lap("p2 busy");
    // ---- (3) soft washes from a LAYER-ANCHORED lattice of seeds (round 7). Every seed sits at a fixed place of the
    //      layer (one per "wash size", jittered by a hash of its lattice index) and takes the colour of the picture
    //      around it; every pixel belongs SOFTLY to its nine nearest seeds (similar colour, near). The wash colour is
    //      the membership-weighted seed colour: flat where the seeds agree, a crisp but continuous edge where the
    //      membership switches between seeds of different colours. Everything is a continuous, local function of
    //      the picture: a moving character only changes the seeds near it, nothing is re-split far away, and a pan
    //      moves the edges smoothly (no flips).
    // (capped: bigger seeds would reach further than ~80 px, and a moving character would change washes far away)
    const double G = std::min(32.0, scale * 1.4);  // seed spacing, full-res px (round 8: bigger washes)
    const float sigC0 = 0.03f + 0.08f * merge;  // round 8: tighter (distinct wash shapes)  // round 8: tighter (distinct wash shapes)
    const int ax0 = (int)std::floor((oxw / dsw) / G) - 2, ay0 = (int)std::floor((oyw / dsw) / G) - 2;
    const int ax1 = (int)std::ceil(((oxw + w) / dsw) / G) + 2, ay1 = (int)std::ceil(((oyw + h) / dsw) / G) + 2;
    const int sw = ax1 - ax0 + 1, sh = ay1 - ay0 + 1;
    std::vector<float> sx((size_t)sw * sh), sy((size_t)sw * sh), sc((size_t)sw * sh * 3), so((size_t)sw * sh * 3);
    // seeds with (almost) no opaque picture around them (outside the layer of an adjustment layer) take no part
    std::vector<uint8_t> sOK((size_t)sw * sh, 1), bad((size_t)sw * sh, 0);
    for (int b = 0; b < sh; ++b)
        for (int a = 0; a < sw; ++a)
        {
            const uint32_t hh = Hash2(ax0 + a, ay0 + b, in.seed ^ 0x5EEDu);
            const double jx = 0.7 * (((hh & 1023) / 1023.0) - 0.5), jy = 0.7 * ((((hh >> 10) & 1023) / 1023.0) - 0.5);
            const size_t s = (size_t)b * sw + a;
            sx[s] = (float)(((ax0 + a) + 0.5 + jx) * G);  // full-res layer coords
            sy[s] = (float)(((ay0 + b) + 0.5 + jy) * G);
            const float fx = (float)(sx[s] * dsw - oxw), fy = (float)(sy[s] * dsw - oyw);
            for (int c = 0; c < 3; ++c) sc[s * 3 + c] = Bilerp(S.data(), w, h, 3, c, fx, fy);
            Opp(&sc[s * 3], &so[s * 3]);
        }
    // memberships of one point (full-res layer coords X, Y, opponent colour o, colour scale sig) -> up to 9 seeds
    auto members = [&](double X, double Y, const float* o, float sig, int* idx, float* m, float sharp) -> int {
        const int ca = (int)std::floor(X / G) - ax0, cb = (int)std::floor(Y / G) - ay0;
        int cnt = 0;
        float lw[9], best = -1e30f;
        for (int b = cb - 1; b <= cb + 1; ++b)
            for (int a = ca - 1; a <= ca + 1; ++a)
            {
                if (a < 0 || b < 0 || a >= sw || b >= sh) continue;
                const size_t s = (size_t)b * sw + a;
                if (!sOK[s]) continue;
                const float dx = (float)((X - sx[s]) / G), dy = (float)((Y - sy[s]) / G);
                const float dc = OppDist(o, &so[s * 3]) / sig;
                const float l = sharp * (-(dc * dc) - (dx * dx + dy * dy) / 0.8f);
                idx[cnt] = (int)s;
                lw[cnt] = l;
                best = std::max(best, l);
                ++cnt;
            }
        float sum = 0.f;
        for (int q = 0; q < cnt; ++q) { m[q] = pm::exp(lw[q] - best); sum += m[q]; }
        if (cnt == 0) return 0;
        for (int q = 0; q < cnt; ++q) m[q] /= sum;
        return cnt;
    };
    // depth: the far map on the working grid (box means)
    std::vector<float> farW;
    if (in.depthFar && (in.kScale != 0.f || in.kWet != 0.f))
    {
        farW.assign(n, 0.f);
        PFor(h, T, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x)
                {
                    float s2 = 0.f;
                    int c2 = 0;
                    for (int j = 0; j < k; ++j)
                        for (int i = 0; i < k; ++i)
                        {
                            s2 += in.depthFar[(size_t)std::min(H - 1, y * k + j) * W + std::min(W - 1, x * k + i)];
                            ++c2;
                        }
                    farW[(size_t)y * w + x] = s2 / c2;
                }
        });
    }
    std::vector<float> Ow(n * 3), sigW(n);
    ForPx(w, h, T, [&](size_t i, int, int) {
        Opp(&S[i * 3], &Ow[i * 3]);
        // darks merge more readily (a dark booth or uniform: one or two soft value steps, not a cloudy pattern)
        sigW[i] = sigC0 * (1.f + 0.6f * busy[i]) * (1.f - 0.6f * eye[i]) * (1.f + 0.3f * (1.f - Smooth(0.12f, 0.35f, Luma(&S[i * 3])))) *
                  (farW.empty() ? 1.f : std::max(0.3f, 1.f + 1.5f * in.kScale * farW[i]));
    });
    // two refinement passes: each seed takes the membership-weighted mean colour of the pixels around it (a local,
    // soft k-means: a seed straddling an edge moves to one side's colour instead of a muddy mixture)
    const double R = 1.2 * G * dsw;  // gather radius, working px
    for (int it = 0; it < 1; ++it)  // one pass: a seed hears only from the picture around it (local reach)
    {
        std::vector<float> nc(sc);
        PFor(sh, T, [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b)
                for (int a = 0; a < sw; ++a)
                {
                    const size_t s = (size_t)b * sw + a;
                    const double cx = sx[s] * dsw - oxw, cy = sy[s] * dsw - oyw;
                    const int x0 = std::max(0, (int)std::floor(cx - R)), x1 = std::min(w - 1, (int)std::ceil(cx + R));
                    const int y0 = std::max(0, (int)std::floor(cy - R)), y1 = std::min(h - 1, (int)std::ceil(cy + R));
                    double acc[3] = {0, 0, 0}, ws = 0;
                    for (int y = y0; y <= y1; ++y)
                        for (int x = x0; x <= x1; ++x)
                        {
                            const size_t i = (size_t)y * w + x;
                            const float wi = wgt[i];
                            if (wi <= 0.f) continue;
                            int idx[9];
                            float m[9];
                            const int cnt = members((x + oxw) / dsw, (y + oyw) / dsw, &Ow[i * 3], sigW[i], idx, m, 1.f);
                            for (int q = 0; q < cnt; ++q)
                                if (idx[q] == (int)s)
                                {
                                    const double wq = (double)m[q] * wi;
                                    for (int c = 0; c < 3; ++c) acc[c] += wq * S[i * 3 + c];
                                    ws += wq;
                                }
                        }
                    if (ws > 1e-6)
                        for (int c = 0; c < 3; ++c) nc[s * 3 + c] = (float)(acc[c] / ws);
                    if (it == 0 && ws < 0.5)
                        bad[s] = 1;  // applied after the pass (members() reads sOK meanwhile)
                }
        });
        sc.swap(nc);
        for (size_t s = 0; s < (size_t)sw * sh; ++s)
        {
            Opp(&sc[s * 3], &so[s * 3]);
            if (bad[s]) sOK[s] = 0;
        }
        poll();
    }
    // round 8: neighbouring seeds of similar colour agree on ONE colour: ONE pass of a value-preserving average over
    // the 5 x 5 nearest seeds, only between similar seeds. Flat, distinct wash shapes as in round 6's merged regions,
    // but strictly local (a seed hears only from seeds at most two lattice steps away - repeated passes would chain
    // across the frame and let a moving character change far washes) and continuous (soft weights)
    {
        const float sm = 0.08f + 0.16f * merge;
        const int rr = G <= 20.0 ? 2 : 1;  // the reach stays under ~60 full-res px whatever the wash size
        // "Merge Colors" above 60 % repeats the pass: larger flat washes (round 6's look), at the price of a longer
        // reach - a moving character then changes washes further away (the stability numbers in the README)
        // (the total reach stays within ~60 full-res px: a moving character never changes washes further away)
        const int passes = std::min(1 + (int)std::floor(std::max(0.f, merge - 0.6f) * 10.f), std::max(1, (int)std::floor(60.0 / (rr * G))));
        for (int pass = 0; pass < passes; ++pass)
        {
        std::vector<float> nc(sc);
        PFor(sh, T, [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b)
                for (int a = 0; a < sw; ++a)
                {
                    const size_t s0 = (size_t)b * sw + a;
                    if (!sOK[s0]) continue;
                    double acc[3] = {sc[s0 * 3], sc[s0 * 3 + 1], sc[s0 * 3 + 2]}, ws = 1.0;
                    for (int db = -rr; db <= rr; ++db)
                        for (int da = -rr; da <= rr; ++da)
                        {
                            if (!da && !db) continue;
                            const int aa = a + da, bb = b + db;
                            if (aa < 0 || bb < 0 || aa >= sw || bb >= sh) continue;
                            const size_t s1 = (size_t)bb * sw + aa;
                            if (!sOK[s1]) continue;
                            // dark washes: one or two flat tones (round 9) - dark seeds merge more readily
                            const float dk2 = 1.f - Smooth(0.15f, 0.35f, std::max(Luma(&sc[s0 * 3]), Luma(&sc[s1 * 3])));
                            const float d = OppDist(&so[s0 * 3], &so[s1 * 3]) / (sm * (1.f + 1.2f * dk2));
                            const float wq = pm::exp(-d * d) * pm::exp(-0.25f * (float)(da * da + db * db));
                            for (int c = 0; c < 3; ++c) acc[c] += wq * sc[s1 * 3 + c];
                            ws += wq;
                        }
                    for (int c = 0; c < 3; ++c) nc[s0 * 3 + c] = (float)(acc[c] / ws);
                }
        });
        sc.swap(nc);
        for (size_t s2 = 0; s2 < (size_t)sw * sh; ++s2) Opp(&sc[s2 * 3], &so[s2 * 3]);
        }
    }

    ptm.lap("p3 washes");
    // ---- (4) at the buffer resolution: the registration offset (washes overshoot / stop short of the line art),
    //      then the soft wash colour, an "edge" measure (how different the competing seeds are), the original small
    //      shapes kept in line-dense areas (eyes, mouths)
    // round 9: the slow noises are kept as their coarse grids and sampled per pixel (memory: no full-size planes)
    struct NGrid
    {
        std::vector<float> g;
        int gx0 = 0, gy0 = 0, gw = 0, gh = 0, step = 1, ox = 0, oy = 0;
        float at(int x, int y) const
        {
            const int lx = x + ox, ly = y + oy;
            const int i = FloorDiv(lx, step) - gx0, j = FloorDiv(ly, step) - gy0;
            const float fx = (float)(lx - (gx0 + i) * step) / step, fy = (float)(ly - (gy0 + j) * step) / step;
            const float* r0 = &g[(size_t)j * gw + i];
            const float* r1 = r0 + gw;
            const float a = r0[0] + (r0[1] - r0[0]) * fx, b = r1[0] + (r1[1] - r1[0]) * fx;
            return a + (b - a) * fy;
        }
    };
    auto makeGrid = [&](NGrid& G2, double minLambdaFull, const std::function<float(double, double)>& fn) {
        G2.step = std::max(1, std::min(16, (int)std::floor(minLambdaFull * ds / 4.0)));
        G2.ox = in.ox;
        G2.oy = in.oy;
        G2.gx0 = FloorDiv(in.ox, G2.step);
        G2.gy0 = FloorDiv(in.oy, G2.step);
        G2.gw = FloorDiv(in.ox + W - 1, G2.step) - G2.gx0 + 2;
        G2.gh = FloorDiv(in.oy + H - 1, G2.step) - G2.gy0 + 2;
        G2.g.resize((size_t)G2.gw * G2.gh);
        PFor(G2.gh, T, [&](int j0, int j1) {
            for (int j = j0; j < j1; ++j)
                for (int i = 0; i < G2.gw; ++i)
                    G2.g[(size_t)j * G2.gw + i] = fn((double)(G2.gx0 + i) * G2.step / ds, (double)(G2.gy0 + j) * G2.step / ds);
        });
    };
    NGrid gapN, mgN, wetN, unN, chN, gN;
    makeGrid(gapN, 60.0 / 2, [&](double gx, double gy) { return Fbm(gx * par, gy, 60.0, 2, 0.45, in.seed ^ 0xA23u, ds); });
    makeGrid(wetN, 160.0 / 2, [&](double gx, double gy) { return Fbm(gx * par, gy, 160.0, 2, 0.5, in.seed ^ 0xA14u, ds); });
    makeGrid(unN, scale * 3.0 / 2, [&](double gx, double gy) { return Fbm(gx * par, gy, scale * 3.0, 2, 0.5, in.seed ^ 0xA15u, ds); });
    makeGrid(chN, 110.0 / 2, [&](double gx, double gy) { return Fbm(gx * par, gy, 110.0, 2, 0.5, in.seed ^ 0xA16u, ds); });
    makeGrid(gN, scale * 6.0, [&](double gx, double gy) { return Fbm(gx * par, gy, scale * 6.0, 1, 0.5, in.seed ^ 0xA11u, ds); });
    if (margin > 0.f)
        makeGrid(mgN, 60.0 / 8, [&](double gx, double gy) { return Fbm(gx * par, gy, 60.0, 4, 0.6, in.seed ^ 0xA24u, ds); });
    // the registration offset: a long wobble plus a ragged fine edge (round 9: stronger fine octaves)
    std::vector<float> nX, nY;
    const double lamR = 70.0;
    NoiseField(nX, W, H, in.ox, in.oy, ds, lamR / 32, T, [&](double gx, double gy) { return Fbm(gx * par, gy, lamR, 4, 0.72, in.seed ^ 0xA21u, ds); });
    NoiseField(nY, W, H, in.ox, in.oy, ds, lamR / 32, T, [&](double gx, double gy) { return Fbm(gx * par, gy, lamR, 4, 0.72, in.seed ^ 0xA22u, ds); });
    // detail layer for the glazes (less smoothed than the washes)
    std::vector<float> Sm(n * 3);
    SmoothNorm3(P.data(), wgt.data(), Sm.data(), w, h, std::max(1.0, scale * dsw * 0.25), Edges(), T, tmp4);
    // local value contrast of the washes (big light / dark steps: the registration offset is held back there, so a
    // dark wash never spills its pigment over a light neighbour)
    std::vector<float> vstep(n);
    ForPx(w, h, T, [&](size_t i, int x, int y) {
        const float* c = &S[i * 3];
        const float* l = &S[(size_t)y * w * 3 + std::max(0, x - 2) * 3];
        const float* r = &S[(size_t)y * w * 3 + std::min(w - 1, x + 2) * 3];
        const float* u = &S[((size_t)std::max(0, y - 2) * w + x) * 3];
        const float* d = &S[((size_t)std::min(h - 1, y + 2) * w + x) * 3];
        (void)c;
        vstep[i] = std::fabs(Luma(l) - Luma(r)) + std::fabs(Luma(u) - Luma(d));
    });
    const size_t N = (size_t)W * H;
    const float amp = loose * 9.0f * (float)ds;  // buffer px
    // no registration offset next to transparency (the washes must not pull in colour from outside the layer)
    std::vector<float> aminW(n), tmpw;
    MinMax(A.data(), aminW.data(), w, h, std::max(1, (int)std::ceil(amp / k + 1.f)), false, T, tmpw);
    // 0.6.1 (memory): the small-feature weight is not stored per pixel - it is read again from its two small grids
    // where it is needed (the same expression: the same value)
    auto prAt = [&](int x, int y) -> float {
        const float fx0 = (x + 0.5f) / k - 0.5f, fy0 = (y + 0.5f) / k - 0.5f;
        const float pr = Bilerp(prot.data(), w, h, 1, 0, fx0, fy0), ey = Bilerp(eye.data(), w, h, 1, 0, fx0, fy0);
        return std::max(ey, 0.5f * pr);
    };
    std::vector<float> Wc(N * 3), ev(N), bzF(N), dl(N), wet(N);
    PFor(H, T, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                const float fx0 = (x + 0.5f) / k - 0.5f, fy0 = (y + 0.5f) / k - 0.5f;
                const float pr = Bilerp(prot.data(), w, h, 1, 0, fx0, fy0), ey = Bilerp(eye.data(), w, h, 1, 0, fx0, fy0);
                const float vs0 = Bilerp(vstep.data(), w, h, 1, 0, fx0, fy0);
                const float a2 = amp * (1.f - 0.85f * pr) / k * Smooth(0.5f, 0.99f, Bilerp(aminW.data(), w, h, 1, 0, fx0, fy0)) *
                                 (1.f - 0.75f * Smooth(0.10f, 0.35f, vs0));
                const float fdep = in.depthFar ? in.depthFar[i] : 0.f;
                const float a3 = a2 * std::max(0.f, 1.f + in.kScale * fdep);  // (looser far away)
                const float fx = fx0 + a3 * nX[i], fy = fy0 + a3 * nY[i];
                float c[3], o[3];
                for (int q = 0; q < 3; ++q) c[q] = Bilerp(S.data(), w, h, 3, q, fx, fy);
                Opp(c, o);
                const float sg = Bilerp(sigW.data(), w, h, 1, 0, fx, fy);
                int idx[9];
                float m[9];
                // dry paper: a steep switch between the washes (a crisp outline), wet paper: a soft one ...
                float wetP = wetA * Smooth(-0.15f, 0.35f, 1.1f * wetN.at(x, y) + 0.25f * wetA);
                const double X = (fx + oxw) / dsw, Y = (fy + oyw) / dsw;
                int cnt = members(X, Y, o, sg, idx, m, 1.f + 3.f * (1.f - wetP));
                // ... and wherever the competing washes differ a lot in VALUE the edge is dry, whatever the wetness
                // (a dark mass keeps a hard painted outline and a flat inside)
                if (cnt > 1)
                {
                    float lw2 = 0.f;
                    for (int q = 0; q < cnt; ++q) lw2 += m[q] * Luma(&sc[(size_t)idx[q] * 3]);
                    float vsp = 0.f;
                    for (int q = 0; q < cnt; ++q) vsp += m[q] * std::fabs(Luma(&sc[(size_t)idx[q] * 3]) - lw2);
                    const float vd = Smooth(0.03f, 0.10f, vsp);
                    if (vd > 0.f)
                    {
                        wetP *= 1.f - vd * (1.f - 0.5f * wetA);
                        cnt = members(X, Y, o, sg, idx, m, (1.f + 3.f * (1.f - wetP)) * (1.f + 4.f * vd * (1.f - 0.6f * wetA)));
                    }
                }
                float wc[3] = {0, 0, 0};
                if (cnt == 0)
                    for (int cc = 0; cc < 3; ++cc) wc[cc] = c[cc];
                for (int q = 0; q < cnt; ++q)
                    for (int cc = 0; cc < 3; ++cc) wc[cc] += m[q] * sc[(size_t)idx[q] * 3 + cc];
                float wo[3];
                Opp(wc, wo);
                float e = 0.f;
                for (int q = 0; q < cnt; ++q) e += m[q] * OppDist(&so[(size_t)idx[q] * 3], wo);
                // small features: the original shapes and colours (lightly smoothed fill), not the wash
                const float* pf = in.P + i * 3;
                // "Protect Small Features" above 75 % also keeps the rest of the line-dense areas (hair, clothes
                // folds) close to the source: steadier in motion, less painted
                const float keep = 0.9f * std::max(ey, pr * Smooth(0.75f, 1.0f, protA));
                for (int cc = 0; cc < 3; ++cc) Wc[i * 3 + cc] = wc[cc] + keep * (Clampf(pf[cc], 0.f, 1.f) - wc[cc]);
                ev[i] = e * (1.f - keep);
                const float prFi = std::max(ey, 0.5f * pr);
                bzF[i] = Bilerp(busy.data(), w, h, 1, 0, fx, fy);
                wet[i] = Clampf(wetP * (1.f - 0.7f * prFi) * (1.f + in.kWet * fdep), 0.f, 1.f);
                float smc[3];
                for (int q = 0; q < 3; ++q) smc[q] = Bilerp(Sm.data(), w, h, 3, q, fx, fy);
                dl[i] = Luma(wc) - Luma(smc);  // darker detail than the wash (for the glazes)
            }
    });
    poll();
    for (auto* v : {&nX, &nY, &Sm, &aminW, &S, &Ow, &sigW, &P, &vstep})
    {
        v->clear();
        v->shrink_to_fit();
    }

    ptm.lap("p4 offset");
    // ---- (5) density per pixel: the value key (darks lifted, the hue kept), a clean transparent colour, the
    //      limited palette (bounded), paper white for light colourless washes
    const int pset = std::max(0, std::min(3, pin.pigSet));
    const auto& pigs = PigmentSetN(pset == 3 ? 0 : pset);
    const int palCount = (palN > 0 && pset != 3) ? std::max(3, palN) : 0;
    std::vector<int> palUse;  // a FIXED palette (the same in every frame): the pure pigments first, then mixtures
    if (palCount > 0)
    {
        for (int i = 0, k2 = 0; i < 11; ++i) { palUse.push_back(k2); k2 += 11 - i; }
        for (int p2 = 0; p2 < (int)pigs.size(); ++p2)
            if (std::find(palUse.begin(), palUse.end(), p2) == palUse.end()) palUse.push_back(p2);
        palUse.resize(std::min<size_t>(palCount, palUse.size()));
    }
    std::vector<float> palO(pigs.size() * 3);
    for (size_t p2 = 0; p2 < pigs.size(); ++p2) Opp(pigs[p2].data(), &palO[p2 * 3]);
    // the value key of round 6 (the confirmed look): darks lifted but still deep, no extra lightening
    const float Dmax = 2.3f - 1.4f * lift;
    float* Dw = outRGB;  // round 9: the densities are built in the output buffer (memory)
    std::vector<float> pW(N), Dm(N);
    ForPx(W, H, T, [&](size_t i, int x, int y) {
        float c[3] = {Wc[i * 3], Wc[i * 3 + 1], Wc[i * 3 + 2]};
        float o[3];
        Opp(c, o);
        const float pr = prAt(x, y), bz = bzF[i];
        if (!palUse.empty())
        {
            // limited palette: the hue of the nearest palette mixture, but only a BOUNDED step from the wash's own
            // colour (never an unrelated hue: foliage stays green, hair keeps its colour)
            // a SOFT choice of the nearest pigments (weights by hue distance): no sudden switch from one pigment to
            // the next when a colour drifts a little (a moving picture would flicker)
            float dbest = 1e9f;
            for (int p2 : palUse)
            {
                const float* po = &palO[(size_t)p2 * 3];
                dbest = std::min(dbest, std::fabs(po[1] / (po[0] + 0.1f) - o[1] / (o[0] + 0.1f)) + std::fabs(po[2] / (po[0] + 0.1f) - o[2] / (o[0] + 0.1f)));
            }
            float dr = 0.f;
            for (int q = 0; q < 3; ++q) dr += -pm::log(std::max(0.02f, c[q])) / 3.f;
            float pc[3] = {0, 0, 0}, wsum = 0.f, po2[3];
            for (int p2 : palUse)
            {
                const float* po = &palO[(size_t)p2 * 3];
                const float d = std::fabs(po[1] / (po[0] + 0.1f) - o[1] / (o[0] + 0.1f)) + std::fabs(po[2] / (po[0] + 0.1f) - o[2] / (o[0] + 0.1f));
                const float wq = pm::exp(-((d - dbest) / 0.05f) * ((d - dbest) / 0.05f));
                if (wq < 1e-4f) continue;
                float dp[3], dmp = 0.f;
                for (int q = 0; q < 3; ++q)
                {
                    dp[q] = -pm::log(std::max(0.02f, pigs[p2][q]));
                    dmp += dp[q] / 3.f;
                }
                const float scl = dmp > 1e-4f ? dr / dmp : 0.f;  // the pigment at this wash's value
                for (int q = 0; q < 3; ++q) pc[q] += wq * dp[q] * scl;
                wsum += wq;
            }
            for (int q = 0; q < 3; ++q) pc[q] = pm::exp(-pc[q] / std::max(1e-6f, wsum));
            Opp(pc, po2);
            const float step = OppDist(po2, o);
            // the storybook sets are a deliberate palette: they may move further (still bounded: no unrelated hue)
            const float cap = pset == 0 ? 0.07f : 0.22f;
            const float k2 = (pset == 0 ? 0.6f : 0.95f) * (1.f - pr) * (step > 1e-4f ? std::min(1.f, cap / step) : 1.f);
            for (int q = 0; q < 3; ++q) c[q] += k2 * (pc[q] - c[q]);
        }
        if (pset == 3 && palN > 0)
        {
            // source hues simplified: the hue snapped to 12 sectors, the value to a few soft steps, a gentle warm shift
            const float ch = std::sqrt(o[1] * o[1] + o[2] * o[2]);
            float hu = pm::atan2(o[2], o[1]);
            const float sec = 6.2831853f / 12.f;
            const float hq = std::round(hu / sec) * sec;
            hu += (hq - hu) * (1.f - pr);
            float o2[3] = {o[0], ch * pm::cos(hu), ch * pm::sin(hu)};
            const float Lq = std::round(o2[0] * 2.f) / 2.f;  // 2 x luma in steps of 0.5 (four tones)
            o2[0] += (Lq - o2[0]) * 0.5f * (1.f - pr) * (1.f - Smooth(0.3f, 0.6f, 1.f - o2[0] * 0.5f));  // (not in darks: no blotches)
            // back from opponent space (inverse of Opp): L2 = 2 luma, a = r - g, b = (r + g) / 2 - b
            const float L = 0.5f * o2[0];
            const float a = o2[1], bb = o2[2];
            // solve r, g, b for luma L: r = g + a, b = (r + g) / 2 - bb
            const float gg = (L - 0.2126f * a - 0.0722f * (a * 0.5f - bb)) / (0.2126f + 0.7152f + 0.0722f);
            float rgb[3] = {gg + a, gg, (gg + a + gg) * 0.5f - bb};
            for (int q = 0; q < 3; ++q) c[q] = Clampf(rgb[q], 0.005f, 1.f);
            c[2] *= 0.97f;  // a gentle warm shift
        }
        float D[3], dm = 0.f;
        for (int q = 0; q < 3; ++q)
        {
            D[q] = -pm::log(std::max(0.015f, c[q]));
            dm += D[q] / 3.f;
        }
        // clean transparent colour: light washes a little more saturated, darks keep their own chroma; foliage
        // (busy texture) a little calmer and greyer (round 6)
        float sat = (1.f + (0.3f - 0.3f * Smooth(0.5f, 1.2f, dm)) * (1.f - pr)) * (1.f - 0.25f * bz);
        if (palN > 0 && pset == 1) sat *= 0.75f;  // storybook (soft): desaturated, chalky
        if (palN > 0 && pset == 2) sat *= 1.3f;   // storybook (bright)
        const float Dc = Dmax * (1.f - pm::exp(-dm / Dmax)), ks = dm > 1e-4f ? Dc / dm : 1.f;
        for (int q = 0; q < 3; ++q)
            Dw[i * 3 + q] = std::max(0.f, dm + sat * (D[q] - dm)) * ks * (1.f - 0.35f * dil * (1.f - pr)) +
                            ((palN > 0 && pset == 1 && q == 2) ? 0.06f : 0.f);  // storybook (soft): cream paper
        Dm[i] = (Dw[i * 3] + Dw[i * 3 + 1] + Dw[i * 3 + 2]) * (1.f / 3.f);
        const float L = Luma(c), chroma = std::fabs(o[1]) + std::fabs(o[2]);
        pW[i] = pw * Smooth(0.86f - 0.10f * pw, 0.93f - 0.08f * pw, L) * (1.f - Smooth(0.10f, 0.25f, chroma)) * (1.f - 0.7f * pr);
    });
    if (!viewOut)
    {
        Wc.clear();
        Wc.shrink_to_fit();
    }
    ptm.lap("p5a key");
    const float rimW = rimWf * (float)ds, wetW = wetWf * (float)ds;
    // alpha-normalised blurs at the buffer size (transparent surroundings - adjustment layers - never take part)
    // 0.6.1 (memory): the blurred numerator / denominator planes are kept and divided where they are read (the same
    // expression, so the same values) - no separate result planes
    auto blurA3 = [&](const float* src, std::vector<float>& num, double sigma) {
        num.assign((size_t)N * 4, 0.f);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float a = in.A[i];
            for (int c = 0; c < 3; ++c) num[i * 4 + c] = src[i * 3 + c] * a;
            num[i * 4 + 3] = a;
        });
        RF4(num.data(), W, H, sigma, 3, Edges(), T);
    };
    auto blurA1 = [&](const std::vector<float>& src, std::vector<float>& num, double sigma) {
        num.assign((size_t)N * 2, 0.f);
        ForPx(W, H, T, [&](size_t i, int, int) {
            num[i * 2] = src[i] * in.A[i];
            num[i * 2 + 1] = in.A[i];
        });
        RFPlain<2>(num.data(), W, H, sigma, 3, T);
    };
    NGrid tongN, bloomN;
    const float br = (float)Clampd(pin.backrun / 100.0, 0, 1);
    if (wetA > 0.f)
        makeGrid(tongN, std::max(30.0, 2.5 * wetWf) / 4, [&](double gx, double gy) { return Fbm(gx * par, gy, std::max(30.0, 2.5 * wetWf), 3, 0.6, in.seed ^ 0xA31u, ds); });
    if (br > 0.f)
        makeGrid(bloomN, 110.0 / 16, [&](double gx, double gy) { return Fbm(gx * par, gy, 110.0, 4, 0.6, in.seed ^ 0xA32u, ds); });
    // (5b) on the CPU, on the given planes (the GPU check below runs it on copies)
    auto cpuWet = [&](float* DwX, float* DmX) {
        // wet-in-wet: where the paper is wet the colours flow across the edges over a wide zone - only between washes
        // of similar value (a dark wash never bleeds a dirty fog into a light one)
        std::vector<float> DbN;
        blurA3(DwX, DbN, std::max(0.5f, wetW));
        ForPx(W, H, T, [&](size_t i, int x, int y) {
            float Db3[3];  // (this pixel's blurred density, read before DwX[i] is changed)
            {
                const float d = DbN[i * 4 + 3];
                for (int c = 0; c < 3; ++c) Db3[c] = d > 1e-5f ? DbN[i * 4 + c] / d : DwX[i * 3 + c];
            }
            auto Db = [&](size_t j) { return Db3[j - i * 3]; };
            const float se = Smooth(0.004f, 0.03f, ev[i]);
            // the colour runs in irregular tongues (not an even blur) ...
            const float tm = Smooth(-0.05f, 0.25f, tongN.at(x, y));
            const float dbm = (Db(i * 3) + Db(i * 3 + 1) + Db(i * 3 + 2)) * (1.f / 3.f);
            // colours of light and mid value run freely (also into the paper); only where a DARK wash is involved does
            // the value difference stop the run (no grey fog)
            const float darkInv = Smooth(0.6f, 1.0f, std::max(dbm, DmX[i]));
            const float simV = 1.f - Smooth(0.06f, 0.18f, std::fabs(dbm - DmX[i])) * darkInv;
            // and never darker than this wash by more than a little (no grey fog around dark masses)
            const float m = std::min(1.f, wet[i] * simV * (se + (1.f - se) * 0.6f * Smooth(0.5f, 1.f, wetA)) * (0.35f + 0.9f * tm));
            // ... and the pigment pools where a run stops (a darker margin along the tongue's end)
            // (only where colour actually runs: near the edges between washes)
            const float band = 4.f * tm * (1.f - tm) * wet[i] * simV * Smooth(0.5f, 1.f, wetA) * (1.f - Smooth(0.4f, 0.8f, DmX[i])) * se;
            for (int q = 0; q < 3; ++q)
            {
                float t = DwX[i * 3 + q] + m * (Db(i * 3 + q) - DwX[i * 3 + q]);
                t += 0.12f * band * std::max(Db(i * 3 + q), DwX[i * 3 + q]);
                DwX[i * 3 + q] = std::min(t, DwX[i * 3 + q] + 0.08f + 0.1f * band + 0.35f * (1.f - darkInv));
            }
            DmX[i] = (DwX[i * 3] + DwX[i * 3 + 1] + DwX[i * 3 + 2]) * (1.f / 3.f);
        });
    };
    bool wetDone = false;  // (5b) already done (on the GPU, or on the CPU before the GPU tail)
#ifdef SK_GPU
    if (!viewOut && wetA > 0.f)
        LogElig(GU_WET);
    // ---- 0.6.0.5: (5b), (5), (6) and the caller's paper on the GPU (gpu\shaders\tail.wgsl, the same arithmetic)
    // 0.6.0.6: by the self-test's verdict - wet and rest on: one job; rest off: (5b) only, the CPU goes on from (5);
    // wet off: the CPU's (5b) first, then the rest on the GPU
    std::vector<float> dLineGpu;
    bool haveDLine = false;
    if (in.gpu && in.gpu->f && !viewOut)
    {
        if (dLineF.valid())
        {
            dLineGpu = dLineF.get();
            haveDLine = true;
        }
        char sb[4] = {0};
        const int stop = skplat::GetEnv("SKGPU_STOP", sb, sizeof(sb)) > 0 ? (sb[0] == '1' ? 1 : 3) : 3;  // (dev: 1 = (5b) only)
        // the small grids: the noise grids, prot / eye, the gradient table
        std::vector<float> grids;
        struct TailU
        {
            uint32_t n[4];
            int32_t m[4], e[4], e2[4];
            float f[48];
            int32_t g[96];
        } tu{};
        static_assert(sizeof(TailU) == 640, "tail.wgsl TU after ps");
        const NGrid* gl[7] = {&gN, &unN, &chN, &tongN, &bloomN, &gapN, &mgN};
        for (int q = 0; q < 7; ++q)
        {
            const NGrid& G2 = *gl[q];
            int32_t* gm = tu.g + q * 8;
            gm[0] = G2.gx0; gm[1] = G2.gy0; gm[2] = G2.gw; gm[3] = G2.gh; gm[4] = G2.step; gm[5] = G2.ox; gm[6] = G2.oy;
            gm[7] = (int32_t)grids.size();
            grids.insert(grids.end(), G2.g.begin(), G2.g.end());
        }
        tu.g[56] = (int32_t)grids.size();
        grids.insert(grids.end(), prot.begin(), prot.end());
        tu.g[57] = (int32_t)grids.size();
        grids.insert(grids.end(), eye.begin(), eye.end());
        tu.g[58] = (int32_t)grids.size();
        grids.insert(grids.end(), g_grad.gx, g_grad.gx + 256);
        grids.insert(grids.end(), g_grad.gy, g_grad.gy + 256);
        // the paper's three fBms (sk_util.h PaperHeight / Fbm): per octave amp * fade, the seed, the frequency
        std::vector<double> d64 = {0.8253356149, 0.5646424734};
        std::vector<double> freq;
        {
            const double lam[3] = {2.2 * in.ps, 5.0 * in.ps, 14.0 * in.ps};
            const int oct[3] = {3, 2, 2};
            const double gain[3] = {0.55, 0.5, 0.5};
            const uint32_t sd[3] = {in.seed ^ 0x9A9Eu, in.seed ^ 0x9A9Fu, in.seed ^ 0x9AA0u};
            int o7 = 0;
            for (int b = 0; b < 3; ++b)
            {
                double fq = 1.0 / std::max(0.5, lam[b]);
                float amp = 1.f, norm = 0.f;
                for (int o = 0; o < oct[b]; ++o, ++o7)
                {
                    const double wlBuf = ds / fq;
                    const float fade = (float)Clampd((wlBuf - 0.9) / 0.9, 0.0, 1.0);
                    tu.f[17 + o7] = fade > 0.f ? amp * fade : 0.f;
                    tu.g[60 + o7] = (int32_t)(sd[b] + 0x51ED27u * (uint32_t)o);
                    freq.push_back(fq);
                    norm += amp * amp;
                    amp *= (float)gain[b];
                    fq *= 2.0;
                }
                tu.f[24 + b] = std::sqrt(norm);
            }
        }
        d64.insert(d64.end(), freq.begin(), freq.end());
        for (int o = 0; o < 3; ++o) d64.push_back(17.31 * o);
        for (int o = 0; o < 3; ++o) d64.push_back(9.17 * o);
        d64.push_back(ds);
        d64.push_back(par);
        const GpuInkIn* ink = in.ink;
        if (ink && ink->style && ink->bkMode == 2)
        {
            // the line noise fBm (sk_core.cpp NoiseField BK, step 1) per pixel: its octave tables
            double fq = 1.0 / std::max(0.5, ink->bkLambda);
            float amp = 1.f, norm = 0.f;
            for (int o = 0; o < 3; ++o)
            {
                const float fade = (float)Clampd((ds / fq - 0.9) / 0.9, 0.0, 1.0);
                tu.f[35 + o] = fade > 0.f ? amp * fade : 0.f;
                tu.g[80 + o] = (int32_t)((in.seed ^ 0x9E7C1u) + 0x51ED27u * (uint32_t)o);
                d64.push_back(fq);
                norm += amp * amp;
                amp *= 0.55f;
                fq *= 2.0;
            }
            tu.f[38] = std::sqrt(norm);
        }
        if (ink)
        {
            tu.f[32] = ink->dk; tu.f[33] = ink->wc; tu.f[34] = ink->kLi;
            if (ink->bkMode == 1)
            {
                std::copy(ink->bkMeta, ink->bkMeta + 7, tu.g + 72);
                tu.g[79] = (int32_t)grids.size();
                grids.insert(grids.end(), ink->bkGrid.begin(), ink->bkGrid.end());
            }
            tu.g[83] = ink->style ? ink->bkMode : 0;
            tu.g[84] = ink->style;
            tu.g[85] = (ink->inpainted ? 1 : 0) | (ink->far_w ? 2 : 0);
        }
        const float glazeF = glaze;
        int flags = 0;
        if (wetA > 0.f) flags |= 1;
        if (glazeF > 0.f) flags |= 2;
        if (br > 0.f) flags |= 4;
        if (haveDLine && !dLineGpu.empty()) flags |= 8;
        if (margin > 0.f) flags |= 16;
        if (in.alpha) flags |= 32;
        if (in.paperTex) flags |= 64;
        if (in.anyHdr) flags |= 128;
        if (in.gr > 0.f || in.pp > 0.f) flags |= 256;
        if (ink && ink->style == 1) flags |= 1024;  // (the pencil lines read the paper height)
        tu.n[0] = (uint32_t)W; tu.n[1] = (uint32_t)H;
        tu.m[0] = w; tu.m[1] = h; tu.m[2] = k; tu.m[3] = std::max(1, (int)std::lround(3.0 * rimW));
        tu.e[0] = flags; tu.e[1] = in.layer.l; tu.e[2] = in.layer.t; tu.e[3] = in.layer.r;
        tu.e2[0] = in.layer.b; tu.e2[1] = in.ox; tu.e2[2] = in.oy;
        const float lwF = (float)std::max(1, in.layer.r - in.layer.l), lhF = (float)std::max(1, in.layer.b - in.layer.t);
        const float fv[17] = {wetA, rimA, grav, uneven, charge, glazeF, br, 0.45f * Clampf(uneven * 2.f, 0.f, 1.f), loose, margin, lwF, lhF,
                              1.0f * (float)ds, 2.5f * (float)ds, in.gr, in.pp, in.pd};
        std::copy(fv, fv + 17, tu.f);
        SkTailArgs ta{};
        ta.stop = stop;
        ta.wet_on = wetA > 0.f ? 1 : 0;
        ta.pa_blur = 4.0 * ds > 0.05 ? 1 : 0;
        for (int it = 0; it < 3; ++it)
        {
            ta.cwet[it] = Rf4Coef((double)std::max(0.5f, wetW), 3, it);
            ta.crim[it] = RfPlainCoef((double)std::max(0.6f, rimW), 3, it);
            ta.cpa[it] = RfPlainCoef(4.0 * ds, 3, it);
        }
        ta.uni = (const uint8_t*)&tu;
        ta.uni_bytes = (uint32_t)sizeof(tu);
        ta.d64 = d64.data();
        ta.d64_count = (uint32_t)d64.size();
        ta.grids = grids.data();
        ta.grids_count = (uint32_t)grids.size();
        ta.a = in.A;
        ta.alpha = in.alpha;
        ta.dw = Dw; ta.dm = Dm.data(); ta.p = in.gpuOut;
        ta.ev = ev.data(); ta.wet = wet.data(); ta.bz = bzF.data(); ta.dl = glazeF > 0.f ? dl.data() : nullptr; ta.pw = pW.data();
        ta.dline = (flags & 8) ? dLineGpu.data() : nullptr;
        ta.ptex = in.paperTex;
        ta.out_dw = Dw; ta.out_dm = Dm.data(); ta.out_p = in.gpuOut;
        if (ink)
        {
            ta.ink_on = 1;
            ta.ink_soft = ink->soft > 0.05 ? 1 : 0;
            for (int it = 0; it < 3; ++it) ta.cink[it] = RfPlainCoef(ink->soft, 3, it);
            ta.delta = ink->delta;
            ta.far_w = ink->far_w;
        }
        const unsigned gu = in.gpu->units;
        if (t_st)
        {
            // ---- the start-up self-test: every stage read back on its own, from the CPU's input of that stage
            // (wet: from the same densities; the rest: from the CPU's wet result), compared when the CPU below gets there
            std::vector<float> cw(Dw, Dw + N * 3), cm(Dm);
            if (wetA > 0.f)
            {
                cpuWet(cw.data(), cm.data());
                std::vector<float> gw(N * 3), gm(N);
                SkTailArgs t1 = ta;
                t1.stop = 1;
                t1.wet_on = 1;
                t1.out_dw = gw.data();
                t1.out_dm = gm.data();
                t1.out_p = nullptr;
                const int s1 = skgpu_frame_tail(in.gpu->f, &t1);
                gw.insert(gw.end(), gm.begin(), gm.end());
                t_st->put(ST_WET, s1, std::move(gw));
                std::vector<float> cc(cw);
                cc.insert(cc.end(), cm.begin(), cm.end());
                t_st->cmp(ST_WET, cc.data(), cc.size());
            }
            auto run = [&](int stage, int stp, size_t cnt, int inkOn) {
                std::vector<float> o(cnt);
                SkTailArgs t2 = ta;
                t2.wet_on = 0;
                t2.dw = cw.data();
                t2.dm = cm.data();
                t2.stop = stp;
                t2.ink_on = inkOn;
                t2.out_dw = (stp == 2 || stp == 4) ? o.data() : nullptr;
                t2.out_dm = stp == 5 ? o.data() : nullptr;
                t2.out_p = stp == 3 ? o.data() : nullptr;
                t_st->put(stage, skgpu_frame_tail(in.gpu->f, &t2), std::move(o));
            };
            run(ST_DENSITY, 2, N * 3, 0);
            run(ST_SLIVERS, 4, N * 3, 0);
            run(ST_PA, 5, N, 0);
            run(ST_PAPER, 3, N * 3, 0);
            if (ink)
                run(ST_INK, 3, N * 3, 1);
            // (then the CPU path, as without a GPU)
        }
        else if ((gu & GU_REST) && stop == 3)
        {
            ta.stop = 3;
            if (wetA > 0.f && !(gu & GU_WET))
            {
                // (the GPU's wet-in-wet failed the self-test: the CPU's, then the rest on the GPU)
                GpuFrame::LogOff(GU_WET);
                cpuWet(Dw, Dm.data());
                wetDone = true;
                ta.wet_on = 0;
            }
            const int st = skgpu_frame_tail(in.gpu->f, &ta);
            if (st == SKGPU_OK)
            {
                ptm.lap(ink ? "p5-6 + paper + ink gpu" : "p5-6 + paper gpu");
                LogUse(GU_REST | (ta.wet_on ? GU_WET : 0u) | (ink ? GU_INK : 0u));
                *in.gpuDone = true;
                if (ink)
                    *ink->done = true;
                if (busyOut)
                    busyOut->assign(bzF.begin(), bzF.end());
                return;
            }
            LogWhy(std::string("tail:") + StatusName(st));
            in.gpu->fail(st);
        }
        else if (wetA > 0.f && (gu & GU_WET))
        {
            // (5b) only (the rest failed the self-test, or dev SKGPU_STOP=1): the CPU goes on from (5)
            if (!(gu & GU_REST))
                GpuFrame::LogOff(GU_REST);
            ta.stop = 1;
            ta.wet_on = 1;
            ta.out_p = nullptr;
            const int st = skgpu_frame_tail(in.gpu->f, &ta);
            if (st == SKGPU_OK)
            {
                wetDone = true;
                LogUse(GU_WET);
                in.gpu->release();
            }
            else
            {
                LogWhy(std::string("wet:") + StatusName(st));
                in.gpu->fail(st);
            }
        }
        else if (!(gu & GU_REST))
            GpuFrame::LogOff(GU_REST);
    }
#endif
    if (wetA > 0.f && !wetDone)
        cpuWet(Dw, Dm.data());
    ptm.lap("p5b wet");
    // dried edges: the pigment pools into a thin darker rim on the DARKER side of an edge (density above its own
    // neighbourhood mean), and toward the lower edge of a wash (gravity)
    std::vector<float> Db1N;
    blurA1(Dm, Db1N, std::max(0.6f, rimW));
    auto Db1 = [&](size_t i) -> float {
        const float d = Db1N[i * 2 + 1];
        return d > 1e-5f ? Db1N[i * 2] / d : Dm[i];
    };
    const int dDown = std::max(1, (int)std::lround(3.0 * rimW));
    const float gstr = 0.45f * Clampf(uneven * 2.f, 0.f, 1.f);
    // Dn is written over Dw row by row: the pooling reads Dm (unchanged) only
    ForPx(W, H, T, [&](size_t i, int x, int y) {
        const float pr = prAt(x, y), bz = bzF[i];
        const float dmn = Dm[i];
        const float dry = 1.f - wet[i];
        // darks: one or two flat tones inside (no mottling)
        const float dk = 1.f - 0.95f * Smooth(0.5f, 1.0f, dmn);
        float k1 = 1.f + Clampf(1.6f * gN.at(x, y), -1.f, 1.f) * gstr * (1.f - 0.6f * pr) * dk;
        const float tide = std::max(0.f, dmn - Db1(i));
        const size_t jb = (size_t)std::min(H - 1, y + dDown) * W + x;
        const float below = in.A[jb] > 0.5f ? Dm[jb] : dmn;
        const float pool = grav * 0.5f * Smooth(0.02f, 0.25f, dmn - below) * dk;
        float add = (rimA * 0.9f * tide * dry + pool * dmn * 0.5f) * (1.f - 0.6f * pr);
        k1 *= 1.f + uneven * 0.22f * unN.at(x, y) * (1.f - 0.7f * pr) * dk;
        const float cw = charge * 0.35f * chN.at(x, y) * dmn * (1.f - 0.6f * pr) * dk;
        const float chg[3] = {-0.5f * cw, -0.5f * cw, cw};
        // layered glazes: darker detail inside the wash re-stated as one to three overlapping tones with their own
        // edges (leaf masses in foliage; one gentle step elsewhere; none on small features or in darks)
        float gl = 0.f;
        if (glaze > 0.f)
        {
            const float d = dl[i];
            gl = glaze * (0.22f * Smooth(0.04f, 0.06f, d) + bz * (0.20f * Smooth(0.10f, 0.12f, d) + 0.18f * Smooth(0.17f, 0.19f, d))) *
                 (1.f - 0.9f * pr) * (1.f - pW[i]) * dk;
        }
        // blooms (backruns): where water ran back into a drying wash the pigment is pushed out of a cauliflower
        // shape (lighter inside) and piles up along its frilled edge - only in wet, light-to-mid washes
        if (br > 0.f)
        {
            const float nb = bloomN.at(x, y);
            const float where = br * Smooth(0.2f, 0.5f, wet[i] / std::max(0.05f, wetA)) * (1.f - Smooth(0.3f, 0.7f, dmn)) * (1.f - pr);
            const float inside = Smooth(0.36f, 0.42f, nb), edge = Smooth(0.32f, 0.36f, nb) * (1.f - Smooth(0.39f, 0.43f, nb));
            k1 *= 1.f - 0.4f * where * inside;
            add += 0.3f * where * edge * dmn;
        }
        float* dn = Dw + i * 3;
        for (int q = 0; q < 3; ++q)
        {
            const float share = dmn > 1e-4f ? dn[q] / dmn : 1.f;  // in the wash's own hue
            dn[q] = std::max(0.f, dn[q] * k1 + (add + gl * std::max(0.25f, dmn)) * share + chg[q]);
        }
    });
    poll();
    dl.clear();
    dl.shrink_to_fit();  // (last read above)

#ifdef SK_GPU
    StCmp(ST_DENSITY, Dw, N * 3);
#endif
    ptm.lap("p5 density");
    // ---- (6) unpainted slivers along dried edges and lines (longer, calmer pieces), the paper margin
    std::vector<float> dLine;
#ifdef SK_GPU
    if (haveDLine)
        dLine.swap(dLineGpu);
    else if (dLineF.valid())
        dLine = dLineF.get();
    else
#endif
        dLine = lineDist();
    const float lw = (float)std::max(1, in.layer.r - in.layer.l), lh = (float)std::max(1, in.layer.b - in.layer.t);
    ForPx(W, H, T, [&](size_t i, int x, int y) {
        const float pr = prAt(x, y);
        float cv = 1.f - pW[i];
        const float gp = gapN.at(x, y);
        const float g1 = loose * (1.f - pr) * (1.f - wet[i]) * Smooth(0.30f, 0.55f, gp) * (1.f - 0.7f * bzF[i]);
        cv *= 1.f - 0.85f * g1 * Smooth(0.015f, 0.05f, ev[i]) * Smooth(0.0f, 0.03f, std::fabs(Dm[i] - Db1(i)));
        if (!dLine.empty())
        {
            const float gl2 = loose * (1.f - pr) * Smooth(0.45f, 0.7f, -gp) * (1.f - bzF[i]);
            cv *= 1.f - gl2 * (1.f - Smooth(1.0f * (float)ds, 2.5f * (float)ds, dLine[i] - 1.f));
        }
        if (margin > 0.f)
        {
            const float ex = std::min((float)(x - in.layer.l), (float)(in.layer.r - 1 - x)) / lw;
            const float ey = std::min((float)(y - in.layer.t), (float)(in.layer.b - 1 - y)) / lh;
            const float e = std::min(ex * lw / lh, ey) + 0.03f * margin * mgN.at(x, y);
            cv *= Smooth(0.07f * margin, 0.07f * margin + 0.012f, e);
        }
        for (int q = 0; q < 3; ++q) outRGB[i * 3 + q] *= cv;  // density (paper applied by the caller)
    });
#ifdef SK_GPU
    StCmp(ST_SLIVERS, outRGB, N * 3);
#endif
    ptm.lap("p6 slivers");
    if (busyOut)
        busyOut->assign(bzF.begin(), bzF.end());
    if (viewOut)
    {
        viewOut->resize(N * 3);
        ForPx(W, H, T, [&](size_t i, int, int) {
            const float e = Smooth(0.01f, 0.06f, ev[i]);
            for (int q = 0; q < 3; ++q)
                (*viewOut)[i * 3 + q] = Wc[i * 3 + q] + e * ((q == 0 ? 0.9f : 0.1f) - Wc[i * 3 + q]);
        });
    }
}
