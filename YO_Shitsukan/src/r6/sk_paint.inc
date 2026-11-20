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
inline const std::vector<std::array<float, 3>>& PigmentSet()
{
    static const std::vector<std::array<float, 3>> set = [] {
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
        std::vector<std::array<float, 3>> s;
        const int n = 11;
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j)
            {
                std::array<float, 3> m;
                for (int c = 0; c < 3; ++c)  // subtractive mix: mean density
                    m[c] = std::sqrt(pig[i][c] * pig[j][c]);
                s.push_back(m);
            }
        return s;
    }();
    return set;
}

// out: painted colour (encoded RGB, N*3). viewOut (optional, N*3): the regions (mean colours with tide lines).
void PaintWatercolour(const Params& pin, const PaintIn& in, float* outRGB, std::vector<float>* viewOut, const std::function<void()>& poll)
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
    std::vector<float> prot(al);
    RFPlain<1>(prot.data(), w, h, 6.0 * dsw, 3, T);
    ForPx(w, h, T, [&](size_t i, int, int) { prot[i] = protA * Smooth(0.05f, 0.18f, prot[i]); });

    // ---- (1) abstraction: edge-preserving smoothing at the region scale (lines are walls)
    std::vector<float> S(n * 3), tmp4, wgt(n);
    ForPx(w, h, T, [&](size_t i, int, int) { wgt[i] = A[i] * (wall[i] ? 0.f : 1.f); });
    {
        std::vector<float> gh, gv;
        BuildGrad(P.data(), w, h, gh, gv, T);
        Edges e;
        e.gh = gh.data();
        e.gv = gv.data();
        e.bar = wall.data();
        e.k = (scale * dsw * 0.5) / 0.06;
        SmoothNorm3(P.data(), wgt.data(), S.data(), w, h, scale * dsw * 0.5, e, T, tmp4);
        BuildGrad(S.data(), w, h, gh, gv, T);
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

    // ---- (2) regions: single-linkage merge of neighbours closer than tau (gradients chain into ONE wash; colour
    //      jumps and line walls separate), then small regions join their most similar neighbour
    std::vector<float> O(n * 3), busy(n);
    ForPx(w, h, T, [&](size_t i, int, int) {
        Opp(&S[i * 3], &O[i * 3]);
        busy[i] = std::fabs(Luma(&P[i * 3]) - Luma(&S[i * 3])) * A[i];
    });
    RFPlain<1>(busy.data(), w, h, scale * dsw, 3, T);
    ForPx(w, h, T, [&](size_t i, int, int) { busy[i] = Smooth(0.025f, 0.07f, busy[i]) * (1.f - prot[i]); });
    const float tau = 0.025f + 0.11f * merge;
    std::vector<uint8_t> skip(n);
    for (size_t i = 0; i < n; ++i)
        skip[i] = wall[i] || A[i] < 0.002f;
    // neighbours closer than t join, but only while the two regions' MEAN colours stay within 4 t: a slow gradient
    // still becomes one wash, a chain of small steps cannot carry a light face into purple hair
    UF uf(n);
    std::vector<float> rs(n * 4);
    for (size_t i = 0; i < n; ++i)
    {
        rs[i * 4] = O[i * 3];
        rs[i * 4 + 1] = O[i * 3 + 1];
        rs[i * 4 + 2] = O[i * 3 + 2];
        rs[i * 4 + 3] = 1.f;
    }
    auto join = [&](size_t a, size_t b, float t) {
        if (OppDist(&O[a * 3], &O[b * 3]) >= t) return;
        int ra = uf.find((int)a), rb = uf.find((int)b);
        if (ra == rb) return;
        const float* A4 = &rs[(size_t)ra * 4];
        const float* B4 = &rs[(size_t)rb * 4];
        const float ma[3] = {A4[0] / A4[3], A4[1] / A4[3], A4[2] / A4[3]}, mb[3] = {B4[0] / B4[3], B4[1] / B4[3], B4[2] / B4[3]};
        if (OppDist(ma, mb) >= 4.f * t) return;
        uf.unite(ra, rb);
        const int r = uf.find(ra), o = r == ra ? rb : ra;
        for (int c = 0; c < 4; ++c) rs[(size_t)r * 4 + c] += rs[(size_t)o * 4 + c];
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            const size_t i = (size_t)y * w + x;
            if (skip[i]) continue;
            const float t = tau * (1.f - 0.85f * prot[i]) * (1.f + 1.6f * busy[i]);
            if (x + 1 < w && !skip[i + 1]) join(i, i + 1, t);
            if (y + 1 < h && !skip[i + w]) join(i, i + w, t);
        }
    std::vector<int> lab(n, -1);
    for (size_t i = 0; i < n; ++i)
        if (!skip[i])
            lab[i] = uf.find((int)i);
    // wall pixels take the label of a neighbour (a few dilation sweeps; thick walls fill from both sides)
    for (int it = 0; it < 100000; ++it)
    {
        bool any = false;
        std::vector<int> nl(lab);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const size_t i = (size_t)y * w + x;
                if (lab[i] >= 0) continue;
                int best = -1;
                float bd = 1e9f;
                const int nx[4] = {x - 1, x + 1, x, x}, ny[4] = {y, y, y - 1, y + 1};
                for (int q = 0; q < 4; ++q)
                {
                    if (nx[q] < 0 || ny[q] < 0 || nx[q] >= w || ny[q] >= h) continue;
                    const size_t j = (size_t)ny[q] * w + nx[q];
                    if (lab[j] < 0) continue;
                    const float d = OppDist(&O[i * 3], &O[j * 3]);
                    if (d < bd || (d == bd && lab[j] < best)) { bd = d; best = lab[j]; }
                }
                if (best >= 0) { nl[i] = best; any = true; }
            }
        lab.swap(nl);
        if (!any) break;
    }
    for (size_t i = 0; i < n; ++i)
        if (lab[i] < 0) lab[i] = 0;
    poll();

    // region statistics (dense ids)
    struct Reg
    {
        double sum[3] = {0, 0, 0}, w = 0, x = 0, y = 0, prot = 0, busy = 0;
        int area = 0;
        float mean[3] = {0, 0, 0};
        float opp[3] = {0, 0, 0};
    };
    std::vector<int> id(n);
    std::vector<Reg> R;
    auto relabel = [&]() {
        std::vector<int> map(n, -1);
        R.clear();
        for (size_t i = 0; i < n; ++i)
        {
            int& m = map[lab[i]];
            if (m < 0) { m = (int)R.size(); R.emplace_back(); }
            id[i] = m;
        }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const size_t i = (size_t)y * w + x;
                Reg& r = R[id[i]];
                const double a = std::max(1e-3f, A[i]);
                for (int c = 0; c < 3; ++c) r.sum[c] += a * S[i * 3 + c];
                r.w += a;
                r.x += x;
                r.y += y;
                r.prot += prot[i];
                r.busy += busy[i];
                ++r.area;
            }
        for (Reg& r : R)
        {
            for (int c = 0; c < 3; ++c) r.mean[c] = (float)(r.sum[c] / r.w);
            r.x /= r.area;
            r.y /= r.area;
            r.prot /= r.area;
            r.busy /= r.area;
            Opp(r.mean, r.opp);
        }
    };
    relabel();
    {
        // small regions -> the most similar neighbour (three rounds)
        const double minA = pm::pow(scale * dsw * 1.6, 2.0);
        for (int round = 0; round < 3; ++round)
        {
            const size_t nr = R.size();
            std::vector<int> best(nr, -1);
            std::vector<float> bd(nr, 1e9f);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const size_t i = (size_t)y * w + x;
                    const int a = id[i];
                    const int nb[2] = {x + 1 < w ? id[i + 1] : a, y + 1 < h ? id[i + w] : a};
                    for (int b : nb)
                    {
                        if (b == a) continue;
                        const float d = OppDist(R[a].opp, R[b].opp);
                        if (d < bd[a] || (d == bd[a] && b < best[a])) { bd[a] = d; best[a] = b; }
                        if (d < bd[b] || (d == bd[b] && a < best[b])) { bd[b] = d; best[b] = a; }
                    }
                }
            bool any = false;
            UF u2(nr);
            for (size_t r = 0; r < nr; ++r)
            {
                const double need = minA * std::max(0.02, 1.0 - 1.2 * R[r].prot) * (1.0 + 3.0 * R[r].busy);
                if (R[r].area < need && best[r] >= 0)
                {
                    u2.unite((int)r, best[r]);
                    any = true;
                }
            }
            if (!any) break;
            for (size_t i = 0; i < n; ++i) lab[i] = u2.find(id[i]);
            relabel();
        }
    }
    const size_t NR = R.size();
    poll();

    // ---- (3) the local wash colour: the colour averaged over a LARGE neighbourhood that never crosses a region
    //      boundary (a recursive filter cut at the boundaries). Everything from here on is per pixel and local: a
    //      character walking through the frame changes the washes near it only (a whole-region mean would recolour a
    //      big background region the moment the character touched it).
    std::vector<uint8_t> cut(n, 0);
    ForPx(w, h, T, [&](size_t i, int x, int y) {
        uint8_t c = 0;
        if (x > 0 && id[i - 1] != id[i]) c |= 1;
        if (y > 0 && id[i - w] != id[i]) c |= 2;
        cut[i] = c;
    });
    std::vector<float> M(n * 3), MO(n * 3);
    {
        Edges e;
        e.cut = cut.data();
        SmoothNorm3(S.data(), A.data(), M.data(), w, h, scale * dsw * 3.0, e, T, tmp4);
    }
    ForPx(w, h, T, [&](size_t i, int, int) { Opp(&M[i * 3], &MO[i * 3]); });
    const auto& pigs = PigmentSet();
    const int palCount = palN > 0 ? std::max(3, palN) : 0;
    std::vector<int> palUse;  // a FIXED palette (the same in every frame): the pure pigments first, then the mixtures
    if (palCount > 0)
    {
        for (int i = 0, k2 = 0; i < 11; ++i) { palUse.push_back(k2); k2 += 11 - i; }
        for (int p2 = 0; p2 < (int)pigs.size(); ++p2)
            if (std::find(palUse.begin(), palUse.end(), p2) == palUse.end()) palUse.push_back(p2);
        palUse.resize(std::min<size_t>(palCount, palUse.size()));
    }
    std::vector<float> palO(pigs.size() * 3);
    for (size_t p2 = 0; p2 < pigs.size(); ++p2) Opp(pigs[p2].data(), &palO[p2 * 3]);
    // value key per pixel: density of the local wash colour, darks lifted (watercolour never reaches black, the hue
    // is kept), dark colours less saturated, the whole a little diluted; light colourless washes stay paper white
    std::vector<float> Dw(n * 3), pW(n);
    const float Dmax = 2.3f - 1.4f * lift;  // soft ceiling of the darkest wash
    ForPx(w, h, T, [&](size_t i, int, int) {
        float c[3] = {M[i * 3], M[i * 3 + 1], M[i * 3 + 2]};
        const float* o = &MO[i * 3];
        if (!palUse.empty())
        {
            // limited palette: keep the value, take the hue of the nearest palette mixture
            int bi = palUse[0];
            float bdd = 1e9f;
            for (int p2 : palUse)
            {
                const float* po = &palO[(size_t)p2 * 3];
                const float d = std::fabs(po[1] / (po[0] + 0.1f) - o[1] / (o[0] + 0.1f)) + std::fabs(po[2] / (po[0] + 0.1f) - o[2] / (o[0] + 0.1f));
                if (d < bdd) { bdd = d; bi = p2; }
            }
            float dp[3], dm = 0.f, dr = 0.f;
            for (int q = 0; q < 3; ++q)
            {
                dp[q] = -pm::log(std::max(0.02f, pigs[bi][q]));
                dm += dp[q] / 3.f;
                dr += -pm::log(std::max(0.02f, c[q])) / 3.f;
            }
            const float sc = dm > 1e-4f ? dr / dm : 0.f;
            for (int q = 0; q < 3; ++q) c[q] += 0.8f * (pm::exp(-dp[q] * sc) - c[q]);
        }
        float D[3], Dm = 0.f;
        for (int q = 0; q < 3; ++q)
        {
            D[q] = -pm::log(std::max(0.015f, c[q]));
            Dm += D[q] / 3.f;
        }
        const float desat = 0.35f * lift * Smooth(0.5f, 1.5f, Dm);
        const float sat = 1.4f - 0.45f * Smooth(0.6f, 1.4f, Dm) - desat;  // transparent pigment: a clean colour
        // the ceiling compresses the MEAN density and scales the channels with it (a dark brown stays brown)
        const float Dc = Dmax * (1.f - pm::exp(-Dm / Dmax)), ks = Dm > 1e-4f ? Dc / Dm : 1.f;
        for (int q = 0; q < 3; ++q)
            Dw[i * 3 + q] = std::max(0.f, Dm + sat * (D[q] - Dm)) * ks * (1.f - 0.35f * dil);
        const float L = Luma(&M[i * 3]);
        const float chroma = std::fabs(o[1]) + std::fabs(o[2]);
        pW[i] = pw * Smooth(0.86f - 0.10f * pw, 0.93f - 0.08f * pw, L) * (1.f - Smooth(0.10f, 0.25f, chroma)) * (1.f - 0.7f * prot[i]);
    });
    poll();

    // ---- (4) boundaries: hard (dried, tide line) or wet (colours flow into each other): hard where the local colours
    //      differ a lot, wet where they are close or the wetness field says so. Transparency is never a boundary.
    std::vector<uint8_t> hardB(n, 0), wetB(n, 0);
    std::vector<float> wetN;
    NoiseField(wetN, w, h, oxw, oyw, dsw, 160.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 160.0, 2, 0.5, in.seed ^ 0xA14u, dsw); });
    ForPx(w, h, T, [&](size_t i, int x, int y) {
        const int a = id[i];
        bool hb = false, wb = false;
        if (A[i] < 0.002f)
        {
            hardB[i] = wetB[i] = 0;
            return;
        }
        const int nx[4] = {x - 1, x + 1, x, x}, ny[4] = {y, y, y - 1, y + 1};
        for (int q = 0; q < 4; ++q)
        {
            if (nx[q] < 0 || ny[q] < 0 || nx[q] >= w || ny[q] >= h) continue;
            const size_t j = (size_t)ny[q] * w + nx[q];
            if (id[j] == a || A[j] < 0.002f) continue;
            const float d = OppDist(&MO[i * 3], &MO[j * 3]);
            const float wetness = 1.1f * wetN[i];
            const float bz = 0.5f * (busy[i] + busy[j]);
            const bool wet = (d < 0.18f + 0.30f * wetA + 0.4f * bz && wetness > 0.35f - 0.9f * wetA - 1.2f * bz) && !(pW[i] > 0.5f || pW[j] > 0.5f);
            if (wet) wb = true; else hb = true;
        }
        hardB[i] = hb;
        wetB[i] = wb && !hb;
    });
    std::vector<float> dHard, dWet;
    Chamfer(hardB, w, h, dHard);
    Chamfer(wetB, w, h, dWet);
    // distance to the region's lower boundary (pigment runs down and pools there); the frame's bottom and
    // transparency are not a boundary
    std::vector<float> dDown(n, 1e6f);
    PFor(w, T, [&](int x0, int x1) {
        for (int x = x0; x < x1; ++x)
            for (int y = h - 2; y >= 0; --y)
            {
                const size_t i = (size_t)y * w + x, j = i + w;
                dDown[i] = (id[j] != id[i] && A[j] >= 0.002f) ? 0.f : dDown[j] + 1.f;
            }
    });
    poll();

    // ---- (5) the washes in density: a wet gradient (lighter where the brush started, darker where it ended),
    //      pooling at the lower edge, a tide line along dried edges, unevenness and charging inside
    std::vector<float> Dn(n * 3), cov(n, 1.f);
    std::vector<float> unN, chN, gN;
    NoiseField(unN, w, h, oxw, oyw, dsw, scale * 3.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, scale * 3.0, 2, 0.5, in.seed ^ 0xA15u, dsw); });
    NoiseField(chN, w, h, oxw, oyw, dsw, 110.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 110.0, 2, 0.5, in.seed ^ 0xA16u, dsw); });
    NoiseField(gN, w, h, oxw, oyw, dsw, scale * 6.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, scale * 6.0, 1, 0.5, in.seed ^ 0xA11u, dsw); });
    const float rimW = rimWf * (float)dsw, wetW = wetWf * (float)dsw;
    const float gstr = 0.45f * Clampf(uneven * 2.f, 0.f, 1.f);
    ForPx(w, h, T, [&](size_t i, int, int) {
        const Reg& g = R[id[i]];
        const float pr = prot[i];
        const float t = Clampf(1.6f * gN[i], -1.f, 1.f) * gstr;
        float k1 = 1.f + t * (1.f - 0.6f * pr);
        k1 += grav * 0.6f * pm::exp(-dDown[i] / std::max(1.f, 3.f * rimW)) * (1.f - 0.6f * pr);
        // pooling: the pigment gathers into a darker thin rim just inside a dried edge (wider for bigger washes)
        const float rw = rimW * (0.7f + 0.6f * Smooth(50.f, 4000.f, (float)g.area));
        k1 += rimA * 0.55f * pm::exp(-dHard[i] / std::max(0.5f, rw)) * (1.f - 0.5f * pr);
        k1 *= 1.f + uneven * 0.22f * unN[i] * (1.f - 0.7f * pr);  // uneven pigment load
        // charging: a warm / cool drift inside the wash (a second pigment dropped in wet)
        const float* dw = &Dw[i * 3];
        const float dmn = (dw[0] + dw[1] + dw[2]) * (1.f / 3.f);
        const float cw = charge * 0.35f * chN[i] * dmn * (1.f - 0.6f * pr) * (1.f - 0.7f * Smooth(0.7f, 1.3f, dmn));
        const float chg[3] = {-0.5f * cw, -0.5f * cw, cw};  // + : warmer (yellow), - : cooler (blue)
        for (int q = 0; q < 3; ++q) Dn[i * 3 + q] = std::max(0.f, dw[q] * k1 + chg[q]);
        cov[i] = 1.f - pW[i];
    });
    // shadow glaze: darker details inside a wash re-stated as a second wash with its own hard edge
    if (glaze > 0.f)
    {
        std::vector<float> Sm(n * 3);
        Edges e;
        e.bar = wall.data();
        SmoothNorm3(P.data(), wgt.data(), Sm.data(), w, h, std::max(1.0, scale * dsw * 0.15), e, T, tmp4);
        ForPx(w, h, T, [&](size_t i, int, int) {
            const float dl = Luma(&M[i * 3]) - Luma(&Sm[i * 3]);  // darker than the wash
            const float m = glaze * Smooth(0.06f, 0.09f, dl) * (1.f - pW[i]) * (1.f - busy[i]);
            if (m <= 0.f) return;
            for (int q = 0; q < 3; ++q)
            {
                const float ds2 = -pm::log(std::max(0.015f, Sm[i * 3 + q])) + pm::log(std::max(0.015f, M[i * 3 + q]));
                Dn[i * 3 + q] += m * std::max(0.f, ds2) * 0.8f;
            }
        });
    }
    // wet merges: across wet boundaries the colours flow into each other over a wide zone
    if (wetA > 0.f)
    {
        std::vector<float> Db(Dn);
        Edges e;
        e.bar = hardB.data();
        SmoothNorm3(Db.data(), nullptr, Db.data(), w, h, std::max(0.5f, wetW), e, T, tmp4);
        ForPx(w, h, T, [&](size_t i, int, int) {
            const float m = wetA * pm::exp(-dWet[i] / std::max(0.5f, wetW * 1.2f)) * (1.f - 0.6f * prot[i]);
            for (int q = 0; q < 3; ++q) Dn[i * 3 + q] += m * (Db[i * 3 + q] - Dn[i * 3 + q]);
        });
    }
    poll();

    // the working grid's staircase along region boundaries: a slight blur before going back up (edges stay crisp at
    // the buffer size: about one buffer px soft)
    if (k > 1)
        RFPlain<3>(Dn.data(), w, h, 0.6, 3, T);
    // ---- (6) broken registration + unpainted slivers + paper margin, then back to the buffer
    //      the washes are sampled with an offset (they overshoot / stop short of the line art); along hard edges
    //      and lines a thin sliver of paper is left here and there; outside an irregular border nothing is painted
    std::vector<float> nX, nY, gapN, mgN;
    const double lamR = 70.0;
    NoiseField(nX, W, H, in.ox, in.oy, ds, lamR / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, lamR, 2, 0.5, in.seed ^ 0xA21u, ds); });
    NoiseField(nY, W, H, in.ox, in.oy, ds, lamR / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, lamR, 2, 0.5, in.seed ^ 0xA22u, ds); });
    NoiseField(gapN, W, H, in.ox, in.oy, ds, 30.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 30.0, 2, 0.5, in.seed ^ 0xA23u, ds); });
    if (margin > 0.f)
        NoiseField(mgN, W, H, in.ox, in.oy, ds, 12.0 / 4, T, [&](double gx, double gy) { return Fbm(gx * par, gy, 60.0, 4, 0.6, in.seed ^ 0xA24u, ds); });
    const float amp = loose * 9.0f * (float)ds;  // buffer px
    const float lw = (float)std::max(1, in.layer.r - in.layer.l), lh = (float)std::max(1, in.layer.b - in.layer.t);
    // line proximity at buffer resolution (slivers along the line art)
    std::vector<float> dLine;
    if (in.alpha && loose > 0.f)
    {
        std::vector<uint8_t> lm((size_t)W * H);
        for (size_t i = 0; i < lm.size(); ++i) lm[i] = in.alpha[i] > 0.35f;
        Chamfer(lm, W, H, dLine);
    }
    auto bil = [&](const std::vector<float>& v, int nc, int c, float fx, float fy) { return Bilerp(v.data(), w, h, nc, c, fx, fy); };
    PFor(H, T, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                // working-grid position of this pixel, displaced
                const float fx0 = (x + 0.5f) / k - 0.5f, fy0 = (y + 0.5f) / k - 0.5f;
                const float pr = Bilerp(prot.data(), w, h, 1, 0, fx0, fy0);
                const float a2 = amp * (1.f - 0.8f * pr) / k;
                const float fx = fx0 + a2 * nX[i], fy = fy0 + a2 * nY[i];
                float D[3];
                for (int q = 0; q < 3; ++q) D[q] = bil(Dn, 3, q, fx, fy);
                float cv = bil(cov, 1, 0, fx, fy);
                // slivers of paper along hard edges and along the lines (where the brush stopped short)
                const float dh = bil(dHard, 1, 0, fx, fy) * k;
                const float g1 = loose * (1.f - pr) * Smooth(0.15f, 0.55f, gapN[i]);
                cv *= 1.f - g1 * (1.f - Smooth(0.6f * (float)ds, 1.8f * (float)ds, dh)) * 0.9f;
                if (!dLine.empty())
                {
                    const float gl2 = loose * (1.f - pr) * Smooth(0.25f, 0.65f, -gapN[i]);
                    cv *= 1.f - gl2 * (1.f - Smooth(1.0f * (float)ds, 2.5f * (float)ds, dLine[i] - 1.f));
                }
                // paper margin: an irregular unpainted border (dry-brush ragged edge)
                if (margin > 0.f)
                {
                    const float ex = std::min((float)(x - in.layer.l), (float)(in.layer.r - 1 - x)) / lw;
                    const float ey = std::min((float)(y - in.layer.t), (float)(in.layer.b - 1 - y)) / lh;
                    const float e = std::min(ex * lw / lh, ey) + 0.03f * margin * mgN[i];
                    cv *= Smooth(0.07f * margin, 0.07f * margin + 0.012f, e);
                }
                for (int q = 0; q < 3; ++q) outRGB[i * 3 + q] = D[q] * cv;  // density (paper applied by the caller)
            }
    });
    if (viewOut)
    {
        viewOut->resize((size_t)W * H * 3);
        ForPx(W, H, T, [&](size_t i, int x, int y) {
            const int xx = std::min(w - 1, x / k), yy = std::min(h - 1, y / k);
            const size_t j = (size_t)yy * w + xx;
            const Reg& g = R[id[j]];
            const bool hb = hardB[j], wb = wetB[j];
            for (int q = 0; q < 3; ++q)
                (*viewOut)[i * 3 + q] = hb ? (q == 0 ? 0.9f : 0.1f) : wb ? (q == 2 ? 0.95f : 0.4f) : g.mean[q];
        });
    }
}
