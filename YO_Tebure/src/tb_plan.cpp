// Tebure - per-frame planning (see tb_plan.h).
#include "tb_plan.h"
#include "platform/msvc_sort.h"

#include <algorithm>
#include <cmath>

namespace tb {

double PoseDistancePx(const Pose& a, const Pose& b, double W, double focal)
{
    const double F = W * focal / 36.0;  // px per radian
    const double d2r = 3.14159265358979 / 180;
    const double dx = std::fabs(a.pan - b.pan) * d2r * F + std::fabs(a.tx - b.tx) * W / 1920;
    const double dy = std::fabs(a.tilt - b.tilt) * d2r * F + std::fabs(a.ty - b.ty) * W / 1920;
    const double dr = std::fabs(a.roll - b.roll) * d2r * W * 0.5;
    const double dz = std::fabs(a.zoom - b.zoom) / 100 * W * 0.5;
    return std::sqrt(dx * dx + dy * dy) + dr + dz;
}

namespace {
struct Ctxer
{
    const PlanInputs* in;
    TimeMap tm;
    double loop = 0, loopTau = 0, phaseScale = 1;
    double rate = 24;
    // shot time u (s since t0) -> context (u already wrapped / quantised by the caller)
    Ctx At(double u, int64_t grid, bool isComp, double gridRate) const
    {
        Ctx c;
        const double t = in->t0 + u;
        c.tau = tm.Tau(t);
        c.phase = tm.Phase(t) * phaseScale;
        c.rate = tm.Cadence(t) * phaseScale;
        c.grid = grid;
        c.gridRate = gridRate;
        c.gridIsComp = isComp;
        c.tauFilm = c.tau;
        return c;
    }
};
double WrapPos(double u, double L) { return L > 0 ? u - L * std::floor(u / L) : u; }
}  // namespace

PlanOutput MakePlan(const PlanInputs& in)
{
    PlanOutput out;
    const Shape& sh = in.shape;
    const double fps = in.fps > 0 ? in.fps : 24;
    const double L = in.loop > 0 ? in.loop : 0;
    Ctxer cx;
    cx.in = &in;
    cx.loop = L;
    cx.rate = ShakeRate(fps, sh);

    // triggers (layer time -> shot time)
    std::vector<Hit> hits = in.hits;
    if (sh.firstHit >= 0)
        hits.push_back(Hit{in.t0 + sh.firstHit, 1.0});
    if (sh.hitEvery > 0)
    {
        // random extra triggers: one candidate per cell of hitEvery seconds (70 %), shot-relative, seeded,
        // periodic with the loop; strength 0.6..1.2. Only the cells that can still act on t are generated.
        const double cell0 = std::max(0.2, sh.hitEvery);
        int64_t K = 0;
        double cell = cell0;
        if (in.loop > 0)
        {
            K = std::max<int64_t>(1, (int64_t)std::floor(in.loop / cell0 + 0.5));
            cell = in.loop / K;
        }
        const uint64_t hs = Mix64(0x51F7ull + in.ctl.seed);
        const double tsh = in.loop > 0 ? std::fmod(std::max(0.0, in.t - in.t0), in.loop) : in.t - in.t0;
        const double reach = in.loop > 0 ? in.loop : 30.0;
        for (int64_t k = (int64_t)std::floor((tsh - reach) / cell) - 1; k <= (int64_t)std::floor((tsh + 1.0) / cell); ++k)
        {
            if (k < 0 && in.loop <= 0)
                continue;
            if (in.loop > 0 && (k < 0 || k >= K))
                continue;  // (wrapped copies are added by the loop logic below)
            const int64_t hk = K > 0 ? ((k % K) + K) % K : k;
            if (Uni(hs, 1, hk) >= 0.7)
                continue;
            const double u = (k + 0.15 + 0.7 * Uni(hs, 2, hk)) * cell;
            hits.push_back(Hit{in.t0 + u, 0.6 + 0.6 * Uni(hs, 3, hk)});
        }
    }
    const double ts = in.t - in.t0;
    const double tsw = WrapPos(ts, L);
    out.shotTime = tsw;
    const double blurSpan = (in.shutterDeg > 0 ? in.shutterDeg / 360.0 : 0) / fps + 2.0 / fps;
    // the time map must cover every instant we evaluate
    double tEnd = (L > 0 ? L : std::max(0.0, ts)) + blurSpan + 2.0 / fps;
    if (in.graphSamples > 0 && L <= 0)
        tEnd = std::max(tEnd, ts + in.graphAhead + 2.0 / fps);
    for (const Hit& h : hits)
        if (h.t - in.t0 >= 0 && h.t - in.t0 <= tEnd)
            tEnd = std::max(tEnd, h.t - in.t0 + 1.0 / fps);
    cx.tm.Build(in.speed, in.cadence, in.t0, in.t0 + tEnd, fps);
    out.timeSamples = cx.tm.samples();

    int64_t loopGrid = 0;
    double loopPhase = 0;
    if (L > 0)
    {
        cx.loopTau = cx.tm.Tau(in.t0 + L);
        if (!(cx.loopTau > 1e-6))
            cx.loopTau = 1e-6;
        const double ph = cx.tm.Phase(in.t0 + L);
        // whole strides (2 steps) per loop so left/right alternation also closes
        loopPhase = std::max(2.0, 2.0 * std::floor(ph / 2.0 + 0.5));
        cx.phaseScale = ph > 1e-9 ? loopPhase / ph : 1;
        const bool comp = sh.frRate == FR_COMP || (sh.frRate == FR_HOLD && sh.holdN <= 1);
        loopGrid = comp ? std::max<int64_t>(1, (int64_t)std::floor(cx.loopTau * 24 + 0.5))
                        : std::max<int64_t>(1, (int64_t)std::floor(L * cx.rate + 0.5));
    }
    Model m;
    Controls ctl = in.ctl;
    if (in.renderVersion >= 2)
    {
        // band limit (0.3): motion the Period / speed pushes above the shake sampling rate's Nyquist is faded out
        // instead of aliasing into random jitter (content the preset itself has above Nyquist at 100 % is kept)
        ctl.bandSpeed = cx.tm.MaxSpeed();
        ctl.nyquist = 0.5 * cx.rate;
    }
    m.Build(sh, ctl, L > 0 ? cx.loopTau : 0, loopGrid, loopPhase);
    const double tmax = std::max(in.trigMax, (sh.firstHit >= 0 || sh.quakeEnv > 0) ? 1.0 : 0.0) * (sh.hitEvery > 0 ? 1.2 : 1.0);
    const bool multi = hits.size() > 1 || sh.hitEvery > 0;
    out.bound = m.Bound(in.amountMax, in.mulMax, std::max(tmax, sh.hitEvery > 0 ? 1.2 : 0.0), multi);
    out.boundAuto = m.Bound(in.amountMax, in.mulMax, std::max(tmax, sh.hitEvery > 0 ? 1.2 : 0.0), multi, true);
    if (in.envScale > 0 && in.envScale < 1)
        for (int i = 0; i < 6; ++i)
        {
            out.bound.set(i, out.bound.get(i) * in.envScale);
            out.boundAuto.set(i, out.boundAuto.get(i) * in.envScale);
        }
    const double respLen = m.ResponseLength();

    std::vector<Trigger> trig;
    for (const Hit& h : hits)
    {
        const double u = h.t - in.t0;
        if (u < 0 || !(h.strength > 0))
            continue;
        if (L > 0 && u >= L)
            continue;
        Trigger tg;
        tg.strength = h.strength;
        tg.id = (int64_t)std::llround(h.t * 10000.0);
        tg.tau = cx.tm.Tau(in.t0 + u);
        int64_t g;
        double gr;
        bool ic;
        QuantizeTime(u, fps, sh, &g, &gr, &ic);
        tg.grid = g;
        trig.push_back(tg);
        if (L > 0)
        {
            // wrapped copies: the end of the loop flows into its start
            const int maxM = respLen > 1e8 ? 1 : std::min(8, (int)std::ceil(respLen / std::max(1e-6, cx.loopTau)) + 1);
            const bool comp = sh.frRate == FR_COMP || (sh.frRate == FR_HOLD && sh.holdN <= 1);
            const int64_t gl = comp ? (int64_t)std::floor(L * fps + 0.5) : loopGrid;
            for (int k = 1; k <= maxM; ++k)
            {
                Trigger w = tg;
                w.tau -= k * cx.loopTau;
                w.grid -= k * gl;
                trig.push_back(w);
            }
        }
    }
    // equal tau is possible (a preset's first hit and a trigger key at the same time; hits at the same layer time also
    // share their id, so only the strength differs): the MSVC algorithm on every platform keeps their order - and so
    // the summation order in Model::Impact - the same as the Windows build (tests: the motion_tie_* golden lines detect
    // the order in the pose bits; the rendered keys_trigger_tie_* lines do not change with it)
    msvcsort::Sort(trig.begin(), trig.end(), [](const Trigger& a, const Trigger& b) { return a.tau < b.tau; });

    auto poseAt = [&](double u /* shot time, unwrapped */) {
        const double uw = WrapPos(u, L);
        int64_t g;
        double gr;
        bool ic;
        const double uq = QuantizeTime(uw, fps, sh, &g, &gr, &ic);
        const Ctx c = cx.At(uq, g, ic, gr);
        Pose p = m.Eval(c, trig, in.amountNow, in.mulNow);
        for (int i = 0; i < 6; ++i)
            p.set(i, SoftLimit(p.get(i), out.bound.get(i)));
        if (in.snap)
        {
            const double k = in.widthRef / 1920.0;
            if (k > 0)
            {
                p.tx = std::floor(p.tx * k + 0.5) / k;
                p.ty = std::floor(p.ty * k + 0.5) / k;
            }
        }
        return p;
    };

    int64_t gNow;
    double grNow;
    bool icNow;
    QuantizeTime(tsw, fps, sh, &gNow, &grNow, &icNow);
    out.grid = gNow;
    out.center = poseAt(ts);
    const int nmax = std::max(1, std::min(64, in.maxSamples));
    if (!(in.shutterDeg > 0) || nmax <= 1)
        out.poses.push_back(out.center);
    else
    {
        const double a = std::min(2.0, in.shutterDeg / 360.0);
        std::vector<Pose> coarse;
        auto build = [&](int n, std::vector<Pose>& v) {
            v.clear();
            if (icNow)
            {
                for (int i = 0; i < n; ++i)
                    v.push_back(poseAt(ts + ((i + 0.5) / n - 0.5) * a / fps));
            }
            else
            {
                // held samples: the camera "moves" from the previous held pose to this one (a smear frame);
                // the shutter angle is the fraction of the jump that is blurred
                const double uq = std::floor(tsw * grNow + 1e-4 * grNow / fps) / grNow;
                const Pose cur = poseAt(ts - tsw + uq);
                const Pose prev = poseAt(ts - tsw + uq - 1.0 / grNow);
                const double af = std::min(1.0, a);
                for (int i = 0; i < n; ++i)
                    v.push_back(Lerp(prev, cur, 1 - af + af * (i + 0.5) / n));
            }
        };
        build(nmax, coarse);
        double len = 0;
        for (size_t i = 1; i < coarse.size(); ++i)
            len += PoseDistancePx(coarse[i - 1], coarse[i], in.widthRef, sh.focal);
        // about one sample per 0.5 px of travel up to the user's count; large moves raise it (one per 1.5 px,
        // up to 64) so tens of px in one frame do not ghost (adaptive, deterministic)
        int n;
        // 0.2 projects (1) and new instances (3, 0.3.1+: no quality-reducing shortcuts) use the 0.2 density; projects saved
        // with 0.3.0 (2) keep its sparser rule so they render as they did
        if (in.renderVersion != 2)
            n = std::max(1, std::min(64, std::max(std::min(nmax, (int)std::ceil(len / 0.5)), (int)std::ceil(len / 1.5))));
        else if (len < 0.5)
            n = 1;  // sub-pixel travel: blur is invisible, one sample (0.3)
        else  // ~1 sample per px of travel up to the user's count; large moves still get one per 2.5 px (max 64)
            n = std::max(1, std::min(64, std::max(std::min(nmax, (int)std::ceil(len / 1.0)), (int)std::ceil(len / 2.5))));
        if (n == nmax)
            out.poses = coarse;
        else if (n > nmax)
            build(n, out.poses);
        else if (n == 1 && len < 0.05)
            out.poses.push_back(coarse[coarse.size() / 2]);
        else
            build(std::max(1, n), out.poses);
    }
    if (in.graphSamples > 1)
    {
        for (int i = 0; i < in.graphSamples; ++i)
        {
            const double u = ts - in.graphBack + (in.graphBack + in.graphAhead) * i / (in.graphSamples - 1);
            out.graphT.push_back(u - ts);
            out.graph.push_back(u < 0 && L <= 0 ? poseAt(0) : poseAt(u));
        }
    }
    return out;
}

}  // namespace tb
