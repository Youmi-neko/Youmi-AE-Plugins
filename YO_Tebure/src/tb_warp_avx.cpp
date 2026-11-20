// Tebure - AVX row kernel for the bicubic resampler (compiled with /arch:AVX; only called when the CPU has AVX).
// Bit-identical to px::SamplePixel: the same IEEE operations in the same order (no FMA, no reassociation), four output
// pixels' coordinates and weights at a time, two source rows per 256-bit operation in the taps.
// Port: compiled only where TB_SIMD_AVX is defined (x86-64 builds: MSVC /arch:AVX, clang -mavx / -Xarch_x86_64 -mavx);
// on arm64 this translation unit is empty and tb_warp.cpp never calls WarpRowAvx.
#include "tb_warp_px.h"

#if defined(TB_SIMD_AVX)
#include <immintrin.h>

namespace tb {

void WarpRowAvx(const Image& S, const RectI& C, int edge, const double* m, double x0, double py, int W, float* acc)
{
    // x0 = out.l (integer valued); pixel centre = (double)(out.l + x) + 0.5 exactly as the reference
    const __m256d m0 = _mm256_set1_pd(m[0]), m1py = _mm256_set1_pd(m[1] * py), m2 = _mm256_set1_pd(m[2]);
    const __m256d m3 = _mm256_set1_pd(m[3]), m4py = _mm256_set1_pd(m[4] * py), m5 = _mm256_set1_pd(m[5]);
    const __m256d m6 = _mm256_set1_pd(m[6]), m7py = _mm256_set1_pd(m[7] * py), m8 = _mm256_set1_pd(m[8]);
    const __m256d half = _mm256_set1_pd(0.5);
    const size_t sw4 = (size_t)S.rect.w() * 4;
    const int ixLo = std::max(C.l, S.rect.l) + 1, ixHi = std::min(C.r, S.rect.r) - 3;  // interior: ix-1 >= lo, ix+3 <= hi
    const int iyLo = std::max(C.t, S.rect.t) + 1, iyHi = std::min(C.b, S.rect.b) - 3;
    int x = 0;
    for (; x + 4 <= W; x += 4)
    {
        const double b = x0 + x;
        const __m256d pxv = _mm256_add_pd(_mm256_set_pd(b + 3, b + 2, b + 1, b), half);
        // the reference: w = m6*px + m7*py + m8; sx = (m0*px + m1*py + m2) / w - 0.5 (m1*py is computed once: the same
        // product the reference computes per pixel)
        const __m256d w = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(m6, pxv), m7py), m8);
        const __m256d sxv = _mm256_sub_pd(_mm256_div_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(m0, pxv), m1py), m2), w), half);
        const __m256d syv = _mm256_sub_pd(_mm256_div_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(m3, pxv), m4py), m5), w), half);
        const __m256d fxv = _mm256_floor_pd(sxv), fyv = _mm256_floor_pd(syv);
        alignas(32) double ws[4], sxs[4], sys[4], fxs[4], fys[4];
        _mm256_store_pd(ws, w);
        _mm256_store_pd(sxs, sxv);
        _mm256_store_pd(sys, syv);
        _mm256_store_pd(fxs, fxv);
        _mm256_store_pd(fys, fyv);
        // weights for four pixels in x and in y (float, the reference's expression order)
        const __m128 tx = _mm256_cvtpd_ps(_mm256_sub_pd(sxv, fxv)), ty = _mm256_cvtpd_ps(_mm256_sub_pd(syv, fyv));
        alignas(16) float wxs[4][4], wys[4][4];
        for (int a = 0; a < 2; ++a)
        {
            const __m128 t = a ? ty : tx, t2 = _mm_mul_ps(t, t), t3 = _mm_mul_ps(t2, t);
            float(*wo)[4] = a ? wys : wxs;
            _mm_store_ps(wo[0], _mm_sub_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.5f), t3), t2), _mm_mul_ps(_mm_set1_ps(0.5f), t)));
            _mm_store_ps(wo[1], _mm_add_ps(_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(1.5f), t3), _mm_mul_ps(_mm_set1_ps(2.5f), t2)), _mm_set1_ps(1.f)));
            _mm_store_ps(wo[2], _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(-1.5f), t3), _mm_mul_ps(_mm_set1_ps(2.f), t2)), _mm_mul_ps(_mm_set1_ps(0.5f), t)));
            _mm_store_ps(wo[3], _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(0.5f), t3), _mm_mul_ps(_mm_set1_ps(0.5f), t2)));
        }
        for (int i = 0; i < 4; ++i)
        {
            float* o = acc + (size_t)(x + i) * 4;
            if (!(ws[i] > 0) || !(std::fabs(sxs[i]) < 1e8 && std::fabs(sys[i]) < 1e8))
                continue;  // the reference skips these pixels too
            const int ix = (int)fxs[i], iy = (int)fys[i];
            const float wx[4] = {wxs[0][i], wxs[1][i], wxs[2][i], wxs[3][i]};
            const float wy[4] = {wys[0][i], wys[1][i], wys[2][i], wys[3][i]};
            if (ix >= ixLo && ix <= ixHi && iy >= iyLo && iy <= iyHi)
            {
                const float* base = S.px.data() + (size_t)(iy - 1 - S.rect.t) * sw4 + (size_t)(ix - 1 - S.rect.l) * 4;
                __m128 accv = _mm_setzero_ps();
                for (int j = 0; j < 4; j += 2)
                {
                    const float* r0 = base + j * sw4;
                    const float* r1 = r0 + sw4;
                    __m256 rv = _mm256_mul_ps(_mm256_set1_ps(wx[0]), _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(r0)), _mm_loadu_ps(r1), 1));
                    for (int k = 1; k < 4; ++k)
                        rv = _mm256_add_ps(rv, _mm256_mul_ps(_mm256_set1_ps(wx[k]),
                                                             _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(r0 + 4 * k)), _mm_loadu_ps(r1 + 4 * k), 1)));
                    accv = _mm_add_ps(accv, _mm_mul_ps(_mm_set1_ps(wy[j]), _mm256_castps256_ps128(rv)));
                    accv = _mm_add_ps(accv, _mm_mul_ps(_mm_set1_ps(wy[j + 1]), _mm256_extractf128_ps(rv, 1)));
                }
                _mm_storeu_ps(o, _mm_add_ps(_mm_loadu_ps(o), accv));
            }
            else
                px::Taps(S, C, edge, 4, 1, ix, iy, wx, wy, o);
        }
    }
    for (; x < W; ++x)
        px::SamplePixel(S, C, edge, 4, 1, m, x0 + x + 0.5, py, acc + (size_t)x * 4);
    _mm256_zeroupper();
}

}  // namespace tb
#endif  // TB_SIMD_AVX
