// Tebure - projection, resampling, auto-scale, overlay (see tb_warp.h).
#include "tb_warp.h"
#include "platform/plat_math.h"  // tbm:: = the libm calls that feed the picture (forward to std::)
#include "tb_warp_px.h"
#include "platform/plat.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

namespace tb {

static const double kPi = 3.14159265358979323846;

RectI Intersect(const RectI& a, const RectI& b)
{
    RectI r{std::max(a.l, b.l), std::max(a.t, b.t), std::min(a.r, b.r), std::min(a.b, b.b)};
    if (r.empty())
        r = RectI{0, 0, 0, 0};
    return r;
}
RectI Union(const RectI& a, const RectI& b)
{
    if (a.empty())
        return b;
    if (b.empty())
        return a;
    return RectI{std::min(a.l, b.l), std::min(a.t, b.t), std::max(a.r, b.r), std::max(a.b, b.b)};
}

static Mat3 Mul(const Mat3& a, const Mat3& b)
{
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double s = 0;
            for (int k = 0; k < 3; ++k)
                s += a.m[i * 3 + k] * b.m[k * 3 + j];
            r.m[i * 3 + j] = s;
        }
    return r;
}

Mat3 Invert(const Mat3& a)
{
    const double* m = a.m;
    const double c0 = m[4] * m[8] - m[5] * m[7], c1 = m[5] * m[6] - m[3] * m[8], c2 = m[3] * m[7] - m[4] * m[6];
    const double det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    Mat3 r;
    if (!(std::fabs(det) > 1e-300))
        return r;
    const double id = 1.0 / det;
    r.m[0] = c0 * id;
    r.m[1] = (m[2] * m[7] - m[1] * m[8]) * id;
    r.m[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    r.m[3] = c1 * id;
    r.m[4] = (m[0] * m[8] - m[2] * m[6]) * id;
    r.m[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    r.m[6] = c2 * id;
    r.m[7] = (m[1] * m[6] - m[0] * m[7]) * id;
    r.m[8] = (m[0] * m[4] - m[1] * m[3]) * id;
    return r;
}

bool Apply(const Mat3& M, double x, double y, double& ox, double& oy)
{
    const double* m = M.m;
    const double w = m[6] * x + m[7] * y + m[8];
    ox = (m[0] * x + m[1] * y + m[2]) / w;
    oy = (m[3] * x + m[4] * y + m[5]) / w;
    return w > 0;
}

bool IsIdentity(const Mat3& M, double tol)
{
    const double* m = M.m;
    const double n = m[8];
    if (!(std::fabs(n) > 0))
        return false;
    const double id[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int i = 0; i < 9; ++i)
        if (std::fabs(m[i] / n - id[i]) > tol)
            return false;
    return true;
}

static Mat3 InverseMapIso(const Pose& p, const Geo& g);
Mat3 InverseMap(const Pose& p, const Geo& g)
{
    const double sx = g.ds, sy = g.dsy > 0 ? g.dsy : g.ds, par = g.par > 0 ? g.par : 1.0;
    if (sx == sy && par == 1.0)
        return InverseMapIso(p, g);
    // non-square pixels / anisotropic downsampling: build the map in full-res SQUARE pixels, then conjugate
    // with D = diag(dsx / par, dsy) (square full-res -> buffer)
    Geo q = g;
    q.ds = 1;
    q.dsy = 0;
    q.par = 1;
    q.widthRef = g.widthRef * par;
    q.cx = g.cx / sx * par;
    q.cy = g.cy / sy;
    const Mat3 M = InverseMapIso(p, q);
    Mat3 D, Di;
    D.m[0] = sx / par, D.m[4] = sy;
    Di.m[0] = par / sx, Di.m[4] = 1 / sy;
    Mat3 R = Mul(D, Mul(M, Di));
    const double n = R.m[8];
    if (std::fabs(n) > 0)
        for (double& v : R.m)
            v /= n;
    return R;
}

static Mat3 InverseMapIso(const Pose& p, const Geo& g)
{
    const double d2r = kPi / 180;
    const double F = g.widthRef * g.focal / 36.0 * g.ds;
    const double s = tbm::exp(p.zoom / 100.0) * g.autoScale;
    const double k = g.widthRef / 1920.0 * g.ds;
    const double tx = p.tx * k, ty = p.ty * k;
    // A: q = p - c - t;  Sc: q / s
    Mat3 A;
    A.m[2] = -g.cx - tx;
    A.m[5] = -g.cy - ty;
    Mat3 Sc;
    Sc.m[0] = Sc.m[4] = 1.0 / s;
    Mat3 Kin;
    Kin.m[8] = F;
    // R = Rz(roll) Ry(pan) Rx(tilt) (content rotation); inverse = R^T
    const double a = p.pan * d2r, b = p.tilt * d2r, r = p.roll * d2r;
    const double ca = tbm::cos(a), sa = tbm::sin(a), cb = tbm::cos(b), sb = tbm::sin(b), cr = tbm::cos(r), sr = tbm::sin(r);
    Mat3 Ry, Rx, Rz;
    Ry.m[0] = ca, Ry.m[2] = sa, Ry.m[6] = -sa, Ry.m[8] = ca;
    Rx.m[4] = cb, Rx.m[5] = sb, Rx.m[7] = -sb, Rx.m[8] = cb;
    Rz.m[0] = cr, Rz.m[1] = -sr, Rz.m[3] = sr, Rz.m[4] = cr;
    const Mat3 R = Mul(Rz, Mul(Ry, Rx));
    Mat3 Rt;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Rt.m[i * 3 + j] = R.m[j * 3 + i];
    Mat3 Kout;
    Kout.m[0] = Kout.m[4] = F;
    Mat3 B;
    B.m[2] = g.cx;
    B.m[5] = g.cy;
    Mat3 M = Mul(B, Mul(Kout, Mul(Rt, Mul(Kin, Mul(Sc, A)))));
    const double n = M.m[8];
    if (std::fabs(n) > 0)
        for (double& v : M.m)
            v /= n;
    return M;
}

bool CoversFrame(const Pose& p, const Geo& g, const RectI& out, const RectI& content, double margin)
{
    const Mat3 M = InverseMap(p, g);
    const double xs[2] = {out.l + 0.5, out.r - 0.5}, ys[2] = {out.t + 0.5, out.b - 0.5};
    for (double y : ys)
        for (double x : xs)
        {
            double sx, sy;
            if (!Apply(M, x, y, sx, sy))
                return false;
            if (sx < content.l + margin || sx > content.r - margin || sy < content.t + margin || sy > content.b - margin)
                return false;
        }
    return true;
}

double AutoScale(const Pose& bound, const RectI& content, double pivotX, double pivotY, double widthRef, double focal, double margin, double par)
{
    if (content.empty())
        return 1;
    bool any = false;
    for (int i = 0; i < 6; ++i)
        any |= bound.get(i) > 0;
    if (!any)
        return 1;
    Geo g;
    g.widthRef = widthRef;
    g.ds = 1;
    g.par = par > 0 ? par : 1.0;
    g.focal = focal;
    // the pivot must lie inside for the zoom to be able to hide the edges
    g.cx = std::min((double)content.r - margin - 1, std::max((double)content.l + margin + 1, pivotX));
    g.cy = std::min((double)content.b - margin - 1, std::max((double)content.t + margin + 1, pivotY));
    double worst = 1;
    const double v3[3] = {-1, 0, 1};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
                for (int d = 0; d < 3; ++d)
                    for (int e = 0; e < 3; ++e)
                    {
                        Pose p;
                        p.pan = v3[a] * bound.pan;
                        p.tilt = v3[b] * bound.tilt;
                        p.roll = v3[c] * bound.roll;
                        p.tx = v3[d] * bound.tx;
                        p.ty = v3[e] * bound.ty;
                        p.zoom = -bound.zoom;
                        g.autoScale = worst;
                        if (CoversFrame(p, g, content, content, margin))
                            continue;
                        double lo = worst, hi = 16;
                        g.autoScale = hi;
                        if (!CoversFrame(p, g, content, content, margin))
                            return 16;
                        for (int it = 0; it < 48; ++it)
                        {
                            const double mid = 0.5 * (lo + hi);
                            g.autoScale = mid;
                            if (CoversFrame(p, g, content, content, margin))
                                hi = mid;
                            else
                                lo = mid;
                        }
                        worst = hi;
                    }
    return worst;
}

// ------------------------------------------------------------------ SIMD dispatch
bool CpuHasAvx() { return plat::CpuHasAvx(); }  // (the 0.3.1.1 cpuid / xgetbv code is in platform/plat_win.cpp)
static std::atomic<int> g_simdMode{0};
void SetWarpSimd(int mode) { g_simdMode = mode; }
#if defined(TB_SIMD_AVX)
int WarpSimd() { return (g_simdMode.load() == 0 && CpuHasAvx()) ? 2 : 1; }
#else
int WarpSimd() { return 1; }  // no AVX kernel in this build (arm64, TB_NO_AVX, TB_FORCE_SCALAR): SSE2 or scalar taps
#endif

// ------------------------------------------------------------------ resampling
namespace {
inline void CubicW(double td, float w[4])
{
    const float t = (float)td, t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] = 1.5f * t3 - 2.5f * t2 + 1;
    w[2] = -1.5f * t3 + 2 * t2 + 0.5f * t;
    w[3] = 0.5f * t3 - 0.5f * t2;
}
inline double Sinc(double x)
{
    if (std::fabs(x) < 1e-9)
        return 1;
    const double px = kPi * x;
    return tbm::sin(px) / px;
}
inline void LanczosW(double t, float w[6])
{
    double s = 0, v[6];
    for (int i = 0; i < 6; ++i)
    {
        const double x = (i - 2) - t;
        v[i] = std::fabs(x) < 3 ? Sinc(x) * Sinc(x / 3) : 0;
        s += v[i];
    }
    for (int i = 0; i < 6; ++i)
        w[i] = (float)(v[i] / s);
}
inline int EdgeIndex(int i, int lo, int hi, int edge)  // returns INT_MIN if transparent
{
    if (i >= lo && i < hi)
        return i;
    if (edge == EDGE_AUTO || edge == EDGE_REPEAT)
        return i < lo ? lo : hi - 1;
    if (edge == EDGE_MIRROR)
    {
        const int n = hi - lo;
        if (n <= 0)
            return lo;
        int m = (i - lo) % (2 * n);
        if (m < 0)
            m += 2 * n;
        return lo + (m < n ? m : 2 * n - 1 - m);
    }
    return INT32_MIN;
}

template <class Fn>
void Parallel(int n, int threads, const Fn& fn)
{
    const int T = std::max(1, std::min(threads, n / 8));
    if (T <= 1)
    {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    std::exception_ptr ex;
    std::mutex exm;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t] {
            try
            {
                fn((int)((long long)n * t / T), (int)((long long)n * (t + 1) / T));
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lk(exm);
                if (!ex)
                    ex = std::current_exception();
            }
        });
    for (auto& th : pool)
        th.join();
    if (ex)
        std::rethrow_exception(ex);
}
}  // namespace

bool Warp(const WarpJob& job, Image* dst)
{
    const int W = job.out.w(), H = job.out.h();
    const bool rows = (bool)job.rowOut;
    if (dst)
    {
        dst->rect = job.out;
        if (!rows)
            dst->px.resize((size_t)std::max(0, W) * std::max(0, H) * 4);
    }
    if (W <= 0 || H <= 0 || job.maps.empty() || !job.src || job.src->rect.empty())
    {
        std::vector<float> z((size_t)std::max(0, W) * 4, 0.f);
        for (int y = 0; y < H; ++y)
            if (rows)
                job.rowOut(y, z.data());
            else if (dst)
                std::fill(dst->px.data() + (size_t)y * W * 4, dst->px.data() + (size_t)(y + 1) * W * 4, 0.f);
        return true;
    }
    // pure whole-pixel translation (one sample, e.g. snapped gamen-do): copy pixels, bit exact, no filtering
    bool intShift = false;
    int shx = 0, shy = 0;
    if (job.maps.size() == 1)
    {
        const double* m = job.maps[0].m;
        const double ex = m[2] - std::floor(m[2] + 0.5), ey = m[5] - std::floor(m[5] + 0.5);
        intShift = m[0] == 1 && m[1] == 0 && m[3] == 0 && m[4] == 1 && m[6] == 0 && m[7] == 0 && m[8] == 1 && std::fabs(ex) < 1e-9 &&
                   std::fabs(ey) < 1e-9;
        shx = (int)std::floor(m[2] + 0.5);
        shy = (int)std::floor(m[5] + 0.5);
    }
    const Image& S = *job.src;
    const RectI C = Intersect(job.content, S.rect).empty() ? S.rect : job.content;
    const int taps = job.interp == INTERP_LANCZOS3 ? 6 : 4;
    const int off = job.interp == INTERP_LANCZOS3 ? 2 : 1;
    const float inv = 1.0f / (float)job.maps.size();
    std::atomic<bool> stop{false};
    const bool useAvx = taps == 4 && WarpSimd() == 2;
    Parallel(H, job.threads, [&](int y0, int y1) {
        std::vector<float> acc((size_t)W * 4);
        for (int y = y0; y < y1; ++y)
        {
            if (((y - y0) & 15) == 0 && job.abort)
            {
                if (stop.load() || job.abort())
                {
                    stop = true;
                    return;
                }
            }
            std::fill(acc.begin(), acc.end(), 0.f);
            const double py = job.out.t + y + 0.5;
            if (intShift)
            {
                const int sy = EdgeIndex(job.out.t + y + shy, C.t, C.b, job.edge);
                for (int x = 0; x < W && sy != INT32_MIN; ++x)
                {
                    const int sx = EdgeIndex(job.out.l + x + shx, C.l, C.r, job.edge);
                    if (sx == INT32_MIN)
                        continue;
                    const int cx = std::min(S.rect.r - 1, std::max(S.rect.l, sx)), cy = std::min(S.rect.b - 1, std::max(S.rect.t, sy));
                    std::memcpy(&acc[(size_t)x * 4], S.px.data() + ((size_t)(cy - S.rect.t) * S.rect.w() + (cx - S.rect.l)) * 4, 16);
                }
            }
            for (const Mat3& M : job.maps)
            {
                if (intShift)
                    break;
                const double* m = M.m;
#if defined(TB_SIMD_AVX)
                if (useAvx)
                    WarpRowAvx(S, C, job.edge, m, (double)job.out.l, py, W, acc.data());
                else
#else
                (void)useAvx;
#endif
                    for (int x = 0; x < W; ++x)
                        px::SamplePixel(S, C, job.edge, taps, off, m, job.out.l + x + 0.5, py, &acc[(size_t)x * 4]);
            }
            if (job.maps.size() > 1)
                for (size_t i = 0; i < acc.size(); ++i)
                    acc[i] *= inv;
            if (rows)
                job.rowOut(y, acc.data());
            else
                std::memcpy(dst->px.data() + (size_t)y * W * 4, acc.data(), acc.size() * sizeof(float));
        }
    });
    return !stop.load();
}

RectI SourceNeeded(const std::vector<Mat3>& maps, const RectI& out, const RectI& content, int edge, int interp)
{
    if (content.empty() || out.empty())
        return RectI{0, 0, 0, 0};
    if (edge == EDGE_MIRROR)
        return content;
    double l = 1e18, t = 1e18, r = -1e18, b = -1e18;
    const double xs[2] = {(double)out.l, (double)out.r}, ys[2] = {(double)out.t, (double)out.b};
    for (const Mat3& M : maps)
        for (double y : ys)
            for (double x : xs)
            {
                double sx, sy;
                if (!Apply(M, x, y, sx, sy))
                    return content;
                l = std::min(l, sx);
                r = std::max(r, sx);
                t = std::min(t, sy);
                b = std::max(b, sy);
            }
    const int k = (interp == INTERP_LANCZOS3 ? 3 : 2) + 2;
    if (!(r - l < 1e7 && b - t < 1e7))
        return content;
    RectI q{(int)std::floor(l) - k, (int)std::floor(t) - k, (int)std::ceil(r) + k, (int)std::ceil(b) + k};
    if (edge == EDGE_AUTO || edge == EDGE_REPEAT)
    {
        // a region beyond the content still needs the edge rows / columns
        q.l = std::min(content.r - 1, std::max(content.l, q.l));
        q.t = std::min(content.b - 1, std::max(content.t, q.t));
        q.r = std::max(content.l + 1, std::min(content.r, q.r));
        q.b = std::max(content.t + 1, std::min(content.b, q.b));
        if (q.r <= q.l)
            q.r = q.l + 1;
        if (q.b <= q.t)
            q.b = q.t + 1;
    }
    return Intersect(q, content);
}

RectI ForwardBounds(const std::vector<Mat3>& maps, const RectI& content)
{
    RectI u{0, 0, 0, 0};
    const double xs[2] = {(double)content.l, (double)content.r}, ys[2] = {(double)content.t, (double)content.b};
    for (const Mat3& M : maps)
    {
        const Mat3 F = Invert(M);
        double l = 1e18, t = 1e18, r = -1e18, b = -1e18;
        bool ok = true;
        for (double y : ys)
            for (double x : xs)
            {
                double ox, oy;
                ok &= Apply(F, x, y, ox, oy);
                l = std::min(l, ox);
                r = std::max(r, ox);
                t = std::min(t, oy);
                b = std::max(b, oy);
            }
        if (!ok || !(r - l < 1e7 && b - t < 1e7))
            return content;
        u = Union(u, RectI{(int)std::floor(l) - 2, (int)std::floor(t) - 2, (int)std::ceil(r) + 2, (int)std::ceil(b) + 2});
    }
    return u;
}

// ------------------------------------------------------------------ graph overlay
namespace {
// 3x5 glyphs: digits, '.', '-', 'X', 'Y', 'R', 'Z', '%', ' '
const char* Glyph(char c)
{
    switch (c)
    {
    case '0': return "111101101101111";
    case '1': return "010110010010111";
    case '2': return "111001111100111";
    case '3': return "111001111001111";
    case '4': return "101101111001001";
    case '5': return "111100111001111";
    case '6': return "111100111101111";
    case '7': return "111001001001001";
    case '8': return "111101111101111";
    case '9': return "111101111001111";
    case '.': return "000000000000010";
    case '-': return "000000111000000";
    case '+': return "000010111010000";
    case 'X': return "101101010101101";
    case 'Y': return "101101010010010";
    case 'R': return "110101110101101";
    case 'Z': return "111001010100111";
    case '%': return "101001010100101";
    case 'p': return "000110101110100";
    case 'x': return "000101010101000";
    case 'd': return "001001111101111";
    default: return "000000000000000";
    }
}
struct Painter
{
    Image* img;
    RectI clip;
    void Blend(int x, int y, const float c[3], float a)
    {
        if (x < clip.l || y < clip.t || x >= clip.r || y >= clip.b)
            return;
        float* p = img->at(x, y);
        for (int i = 0; i < 3; ++i)
            p[i] = p[i] * (1 - a) + c[i] * a;
        p[3] = p[3] * (1 - a) + a;
    }
    void Dot(double x, double y, double r, const float c[3])
    {
        const int x0 = (int)std::floor(x - r), x1 = (int)std::ceil(x + r), y0 = (int)std::floor(y - r), y1 = (int)std::ceil(y + r);
        for (int yy = y0; yy <= y1; ++yy)
            for (int xx = x0; xx <= x1; ++xx)
            {
                const double d = tbm::hypot(xx + 0.5 - x, yy + 0.5 - y);
                const double a = std::min(1.0, std::max(0.0, r + 0.5 - d));
                if (a > 0)
                    Blend(xx, yy, c, (float)a);
            }
    }
    void Line(double x0, double y0, double x1, double y1, double r, const float c[3])
    {
        const double len = tbm::hypot(x1 - x0, y1 - y0);
        const int n = std::max(1, (int)std::ceil(len / std::max(0.35, r * 0.6)));
        for (int i = 0; i <= n; ++i)
            Dot(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, r, c);
    }
    void Text(double x, double y, double s, const char* t, const float c[3])
    {
        for (; *t; ++t, x += 4 * s)
        {
            const char* g = Glyph(*t);
            for (int j = 0; j < 5; ++j)
                for (int i = 0; i < 3; ++i)
                    if (g[j * 3 + i] == '1')
                        for (int yy = (int)(y + j * s); yy < (int)(y + (j + 1) * s); ++yy)
                            for (int xx = (int)(x + i * s); xx < (int)(x + (i + 1) * s); ++xx)
                                Blend(xx, yy, c, 1.f);
        }
    }
};
}  // namespace

void DrawGraph(Image* img, const RectI& frame, double ds, const std::vector<double>& ts, const std::vector<Pose>& path,
               const Pose& bound, double focal, double widthRef)
{
    if (!img || path.size() < 2 || frame.empty())
        return;
    RectI area = Intersect(frame, img->rect);
    if (area.empty())
        return;
    Painter P{img, img->rect};
    const double fh = frame.h(), fw = frame.w();
    const double ph = std::max(40.0, fh * 0.34), top = frame.b - ph - fh * 0.03;
    const double left = frame.l + fw * 0.04, right = frame.r - fw * 0.04;
    const float bg[3] = {0.f, 0.f, 0.f};
    for (int y = (int)top; y < (int)(top + ph); ++y)
        for (int x = (int)left; x < (int)right; ++x)
            P.Blend(x, y, bg, 0.6f);
    const double F = widthRef * focal / 36.0 * kPi / 180;  // full-res px per degree
    const double k = widthRef / 1920.0;
    auto val = [&](const Pose& p, int lane) {
        switch (lane)
        {
        case 0: return p.pan * F + p.tx * k;
        case 1: return p.tilt * F + p.ty * k;
        case 2: return p.roll;
        default: return p.zoom;
        }
    };
    const double fs[4] = {bound.pan * F + bound.tx * k, bound.tilt * F + bound.ty * k, bound.roll, bound.zoom};
    const float col[4][3] = {{1.f, 0.35f, 0.3f}, {0.35f, 1.f, 0.4f}, {0.4f, 0.6f, 1.f}, {1.f, 0.9f, 0.3f}};
    const char* lab[4] = {"X", "Y", "R", "Z"};
    const double lh = ph / 4, tsz = std::max(1.0, std::floor(lh * 0.16));
    const double t0 = ts.front(), t1 = ts.back();
    const double lw = std::max(0.6, 1.2 * std::max(ds, 0.5) * fw / 1920.0 / std::max(ds, 1e-3) * ds);
    for (int lane = 0; lane < 4; ++lane)
    {
        const double cy = top + lh * (lane + 0.5);
        const float grey[3] = {0.35f, 0.35f, 0.35f};
        P.Line(left + 30 * tsz, cy, right, cy, 0.4, grey);
        char buf[32];
        double fsv = fs[lane] > 0 ? fs[lane] : 1;
        // full scale shown as +-value (px for X/Y at full res, deg for R, % for Z)
        std::snprintf(buf, sizeof(buf), "%s %.*f", lab[lane], fsv < 10 ? 2 : 1, fsv);
        P.Text(left + tsz, cy - 2.5 * tsz, tsz, buf, col[lane]);
        if (!(fs[lane] > 0))
            continue;
        const double sc = (lh * 0.45) / fsv;
        for (size_t i = 1; i < path.size(); ++i)
        {
            const double xa = left + 30 * tsz + (right - left - 30 * tsz) * (ts[i - 1] - t0) / (t1 - t0);
            const double xb = left + 30 * tsz + (right - left - 30 * tsz) * (ts[i] - t0) / (t1 - t0);
            const double ya = cy - val(path[i - 1], lane) * ds * 0 - val(path[i - 1], lane) * sc;
            const double yb = cy - val(path[i], lane) * sc;
            P.Line(xa, ya, xb, yb, lw, col[lane]);
        }
    }
    // cursor at "now"
    const float wcol[3] = {1.f, 1.f, 1.f};
    const double xn = left + 30 * tsz + (right - left - 30 * tsz) * (0 - t0) / (t1 - t0);
    P.Line(xn, top + 2, xn, top + ph - 2, std::max(0.5, lw * 0.7), wcol);
}

}  // namespace tb
