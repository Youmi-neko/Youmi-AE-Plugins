// Tebure - preset picker thumbnails (see tb_thumbs.h).
#include "tb_thumbs.h"

#include "tb_plan.h"
#include "tb_presets.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

namespace tbthumb {

namespace {
std::mutex g_m;
std::vector<Path> g_paths;
bool g_done = false;

int FamilyOf(const char* key)
{
    if (!std::strcmp(key, "walk") || !std::strcmp(key, "run") || !std::strcmp(key, "heart"))
        return 1;
    if (!std::strcmp(key, "impact") || !std::strcmp(key, "explosion") || !std::strcmp(key, "quake") || !std::strcmp(key, "car") ||
        !std::strcmp(key, "tripod") || !std::strcmp(key, "drone") || !std::strcmp(key, "drunk"))
        return 2;
    if (!std::strcmp(key, "film") || !std::strncmp(key, "gamen", 5) || !std::strcmp(key, "anime_hand"))
        return 3;
    return 0;
}

void ComputeAll()
{
    int n = 0;
    const tb::Preset* P = tb::Presets(&n);
    g_paths.assign(n, Path());
    const double W = 1920, F = 1920 * 35.0 / 36.0 * 3.14159265358979 / 180;  // px per degree at 35 mm
    std::vector<double> rms(n, 0);
    for (int i = 0; i < n; ++i)
    {
        const tb::Shape& s = P[i].s;
        const double Fd = W * s.focal / 36.0 * 3.14159265358979 / 180;
        (void)F;
        tb::PlanInputs in;
        in.shape = s;
        in.ctl.seed = 1;
        in.speed.constant = 1;
        in.cadence.constant = s.cadence;
        in.fps = 24;
        in.widthRef = W;
        Path& p = g_paths[i];
        p.family = FamilyOf(P[i].key);
        double m = 1e-9, ss = 0;
        const int N = 100;  // 5 s at 20 Hz (24 fps comp for the frame grid)
        std::vector<double> xs, ys;
        for (int k = 0; k < N; ++k)
        {
            in.t = 5.0 * k / (N - 1);
            const tb::Pose q = tb::MakePlan(in).center;
            const double x = q.pan * Fd + q.tx, y = q.tilt * Fd + q.ty;
            xs.push_back(x);
            ys.push_back(y);
        }
        double mx = 0, my = 0;
        for (int k = 0; k < N; ++k)
            mx += xs[k] / N, my += ys[k] / N;
        for (int k = 0; k < N; ++k)
        {
            xs[k] -= mx;
            ys[k] -= my;
            m = std::max(m, std::max(std::fabs(xs[k]), std::fabs(ys[k])));
            ss += xs[k] * xs[k] + ys[k] * ys[k];
        }
        for (int k = 0; k < N; ++k)
        {
            p.x.push_back((float)(xs[k] / m));
            p.y.push_back((float)(ys[k] / m));
        }
        p.rmsPx = (float)std::sqrt(ss / N);
        rms[i] = p.rmsPx;
    }
    // bar: log scale between 0.5 px and the largest preset
    double hi = 1;
    for (double r : rms)
        hi = std::max(hi, r);
    for (int i = 0; i < n; ++i)
        g_paths[i].barFrac = (float)std::min(1.0, std::max(0.04, std::log(std::max(0.5, rms[i]) / 0.5) / std::log(hi / 0.5)));
}
}  // namespace

int Count()
{
    int n = 0;
    tb::Presets(&n);
    return n;
}

const Path* Get(int preset)
{
    std::lock_guard<std::mutex> lk(g_m);
    if (!g_done)
    {
        ComputeAll();
        g_done = true;
    }
    return preset >= 0 && preset < (int)g_paths.size() ? &g_paths[preset] : nullptr;
}

Grid Layout(int n, float w)
{
    Grid g;
    const float cw = kThumbW + kGap;
    g.cols = std::max(1, (int)std::floor((w - kGap) / cw));
    g.s = 1;
    if (w - kGap < cw)
    {
        g.cols = 1;
        g.s = std::max(0.3f, (w - kGap) / cw);
    }
    g.cw = cw * g.s;
    g.ch = (kThumbH + kLabelH + kGap) * g.s;
    g.rows = (n + g.cols - 1) / g.cols;
    g.x0 = kGap * 0.5f * g.s;
    g.y0 = kGap * 0.5f * g.s;
    return g;
}

int PreferredHeight(int n, float w)
{
    const Grid g = Layout(n, w);
    return (int)std::ceil(g.rows * g.ch + kGap);
}

void CellRect(const Grid& g, int i, float& x, float& y, float& w, float& h)
{
    const int c = i % g.cols, r = i / g.cols;
    x = g.x0 + c * g.cw + kGap * 0.5f * g.s;
    y = g.y0 + r * g.ch;
    w = kThumbW * g.s;
    h = kThumbH * g.s;
}

void LabelRect(const Grid& g, int i, float& x, float& y, float& w, float& h)
{
    const int c = i % g.cols, r = i / g.cols;
    x = g.x0 + c * g.cw;
    y = g.y0 + r * g.ch + kThumbH * g.s;
    w = g.cw;
    h = kLabelH * g.s;
}

int HitTest(const Grid& g, int n, float x, float y)
{
    if (x < g.x0 || y < g.y0)
        return -1;
    const int c = (int)((x - g.x0) / g.cw), r = (int)((y - g.y0) / g.ch);
    if (c < 0 || c >= g.cols || r < 0)
        return -1;
    const int i = r * g.cols + c;
    return i < n ? i : -1;
}

}  // namespace tbthumb
