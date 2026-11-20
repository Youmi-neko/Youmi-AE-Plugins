// Tebure - the per-pixel resampler, shared by tb_warp.cpp (SSE2) and tb_warp_avx.cpp (/arch:AVX). Everything here is
// `static` (internal linkage) ON PURPOSE: each translation unit gets its own copy compiled for its own instruction set,
// so the linker can never pick the AVX-compiled copy for the SSE2 path (that would crash on a CPU without AVX).
// The arithmetic order is the reference for both paths: results are bit-identical (tested).
// Port: the SSE2 fast path of Taps has a scalar twin (TB_SIMD_SSE2 undefined: arm64 / TB_FORCE_SCALAR) doing the
// same IEEE single-precision operations per channel in the same order; tests/fakehost golden proves the three
// builds (AVX dispatch, SSE2 only, scalar) give the same bytes.
#pragma once
#include "tb_warp.h"
#include "platform/plat_math.h"  // tbm:: = the libm calls that feed the picture (forward to std::)
#include "platform/plat.h"

#include <climits>
#include <cmath>
#include <cstdint>
#if defined(TB_SIMD_SSE2)
#include <emmintrin.h>
#endif

namespace tb {
namespace px {

static const double kPiPx = 3.14159265358979323846;

static inline void CubicW(double td, float w[4])
{
    const float t = (float)td, t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] = 1.5f * t3 - 2.5f * t2 + 1;
    w[2] = -1.5f * t3 + 2 * t2 + 0.5f * t;
    w[3] = 0.5f * t3 - 0.5f * t2;
}
static inline double Sinc(double x)
{
    if (std::fabs(x) < 1e-9)
        return 1;
    const double p = kPiPx * x;
    return tbm::sin(p) / p;
}
static inline void LanczosW(double t, float w[6])
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
static inline int EdgeIndex(int i, int lo, int hi, int edge)  // INT32_MIN if transparent
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

// taps once the weights and the integer position are known: adds the sample to o[0..3]
static inline void Taps(const Image& S, const RectI& C, int edge, int taps, int off, int ix, int iy, const float* wx, const float* wy, float* o)
{
    // fast path: every tap inside the content and the source buffer (contiguous 4-float pixels)
    if (ix - off >= C.l && ix - off + taps <= C.r && iy - off >= C.t && iy - off + taps <= C.b && ix - off >= S.rect.l &&
        ix - off + taps <= S.rect.r && iy - off >= S.rect.t && iy - off + taps <= S.rect.b)
    {
        const size_t sw4 = (size_t)S.rect.w() * 4;
        const float* base = S.px.data() + (size_t)(iy - off - S.rect.t) * sw4 + (size_t)(ix - off - S.rect.l) * 4;
#if defined(TB_SIMD_SSE2)
        __m128 accv = _mm_setzero_ps();
        for (int j = 0; j < taps; ++j)
        {
            const float* row = base + j * sw4;
            __m128 rv = _mm_mul_ps(_mm_set1_ps(wx[0]), _mm_loadu_ps(row));
            for (int k = 1; k < taps; ++k)
                rv = _mm_add_ps(rv, _mm_mul_ps(_mm_set1_ps(wx[k]), _mm_loadu_ps(row + 4 * k)));
            accv = _mm_add_ps(accv, _mm_mul_ps(_mm_set1_ps(wy[j]), rv));
        }
        _mm_storeu_ps(o, _mm_add_ps(_mm_loadu_ps(o), accv));
#else
        // scalar twin of the SSE2 block above, lane by lane: acc = 0; per row j: rv = wx0*p0, rv = rv + wxk*pk;
        // acc = acc + wyj*rv; o = o + acc (no zero-weight skipping here, unlike the edge path below)
        float accv[4] = {0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < taps; ++j)
        {
            const float* row = base + j * sw4;
            float rv[4];
            for (int c = 0; c < 4; ++c)
                rv[c] = wx[0] * row[c];
            for (int k = 1; k < taps; ++k)
                for (int c = 0; c < 4; ++c)
                    rv[c] = rv[c] + wx[k] * row[4 * k + c];
            for (int c = 0; c < 4; ++c)
                accv[c] = accv[c] + wy[j] * rv[c];
        }
        for (int c = 0; c < 4; ++c)
            o[c] = o[c] + accv[c];
#endif
        return;
    }
    int xi[6], yi[6];
    for (int k = 0; k < taps; ++k)
    {
        int a = EdgeIndex(ix - off + k, C.l, C.r, edge);
        if (a != INT32_MIN)
            a = std::min(S.rect.r - 1, std::max(S.rect.l, a));
        xi[k] = a;
        int b = EdgeIndex(iy - off + k, C.t, C.b, edge);
        if (b != INT32_MIN)
            b = std::min(S.rect.b - 1, std::max(S.rect.t, b));
        yi[k] = b;
    }
    float r = 0, g = 0, bl = 0, al = 0;
    for (int j = 0; j < taps; ++j)
    {
        if (yi[j] == INT32_MIN || wy[j] == 0.f)
            continue;
        const float* row = S.px.data() + (size_t)(yi[j] - S.rect.t) * S.rect.w() * 4;
        float rr = 0, gg = 0, bb = 0, aa = 0;
        for (int k = 0; k < taps; ++k)
        {
            if (xi[k] == INT32_MIN || wx[k] == 0.f)
                continue;
            const float* q = row + (size_t)(xi[k] - S.rect.l) * 4;
            rr += wx[k] * q[0];
            gg += wx[k] * q[1];
            bb += wx[k] * q[2];
            aa += wx[k] * q[3];
        }
        r += wy[j] * rr;
        g += wy[j] * gg;
        bl += wy[j] * bb;
        al += wy[j] * aa;
    }
    o[0] += r;
    o[1] += g;
    o[2] += bl;
    o[3] += al;
}

// one output pixel (centre px, py) of one map: the reference implementation
static inline void SamplePixel(const Image& S, const RectI& C, int edge, int taps, int off, const double* m, double px, double py, float* o)
{
    const double w = m[6] * px + m[7] * py + m[8];
    if (!(w > 0))
        return;
    const double sx = (m[0] * px + m[1] * py + m[2]) / w - 0.5;
    const double sy = (m[3] * px + m[4] * py + m[5]) / w - 0.5;
    if (!(std::fabs(sx) < 1e8 && std::fabs(sy) < 1e8))
        return;
    const double fx = std::floor(sx), fy = std::floor(sy);
    float wx[6], wy[6];
    if (taps == 6)
    {
        LanczosW(sx - fx, wx);
        LanczosW(sy - fy, wy);
    }
    else
    {
        CubicW(sx - fx, wx);
        CubicW(sy - fy, wy);
    }
    Taps(S, C, edge, taps, off, (int)fx, (int)fy, wx, wy, o);
}

}  // namespace px

// AVX row kernel (tb_warp_avx.cpp, bicubic only): one map over one output row; bit-identical to SamplePixel per pixel
void WarpRowAvx(const Image& S, const RectI& C, int edge, const double* m, double x0, double py, int W, float* acc);
bool CpuHasAvx();
void SetWarpSimd(int mode);  // tests: 0 = auto (AVX if available), 1 = SSE2 only
int WarpSimd();              // the path in use: 1 = SSE2, 2 = AVX

}  // namespace tb
