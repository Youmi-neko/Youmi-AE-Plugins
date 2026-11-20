// Optic Bokeh platform layer: portable stand-in for the SSE2 double-precision intrinsics that
// core/lenscore.cpp uses (mac port, v2.4.2). Included through src/platform/compat/emmintrin.h on every
// target that has no SSE2 (Apple Silicon / arm64), and on x64 only when OB_SSE2_EMULATION is defined
// (the Windows bit-exactness test of this file: see PORTING_NOTES.md, "SIMD").
//
// core/lenscore.cpp is NOT edited (it must stay byte-identical: the GPU self-test data, gpu/src/selftest.bin,
// stores a hash of it). It includes <emmintrin.h>; on arm64 the compat directory shadows that header.
//
// Only the 11 intrinsics lenscore.cpp uses are provided, all on packed doubles:
//   _mm_loadu_pd _mm_storeu_pd _mm_set1_pd _mm_set_pd _mm_add_pd _mm_sub_pd _mm_mul_pd _mm_div_pd
//   _mm_cvtsd_f64 _mm_unpacklo_pd _mm_unpackhi_pd
// SSE2 addpd/subpd/mulpd/divpd are IEEE-754 binary64 operations, correctly rounded per lane, with the
// default MXCSR (round to nearest, no FTZ/DAZ). The scalar lanes below are the same operations in the same
// order, so the result is bit-identical PROVIDED the compiler does not contract a*b+c into an FMA:
// compile with -ffp-contract=off and without -ffast-math (CMakeLists.txt does). arm64 FPCR defaults to
// round-to-nearest with FZ off, matching the x64 default MXCSR.
//
// Default implementation: plain scalar structs (this is the variant verified bit-identical on Windows x64,
// MSVC, against the real SSE2 build). OB_SSE2_EMUL_NEON selects a NEON float64x2_t variant (written, NOT
// verified anywhere; vaddq/vsubq/vmulq/vdivq_f64 are also correctly rounded IEEE operations, so it should be
// identical too). Macros (not declarations of the reserved names) are used so this header can coexist with
// a compiler's own intrinsic declarations in the Windows test build.
#pragma once

#if defined(OB_SSE2_EMUL_NEON) && (defined(__aarch64__) || defined(__arm64__))
#include <arm_neon.h>
typedef float64x2_t ob_m128d;
static inline ob_m128d ob_mm_loadu_pd(const double* p) { return vld1q_f64(p); }
static inline void ob_mm_storeu_pd(double* p, ob_m128d a) { vst1q_f64(p, a); }
static inline ob_m128d ob_mm_set1_pd(double x) { return vdupq_n_f64(x); }
static inline ob_m128d ob_mm_set_pd(double e1, double e0) { const double v[2] = {e0, e1}; return vld1q_f64(v); }
static inline ob_m128d ob_mm_add_pd(ob_m128d a, ob_m128d b) { return vaddq_f64(a, b); }
static inline ob_m128d ob_mm_sub_pd(ob_m128d a, ob_m128d b) { return vsubq_f64(a, b); }
static inline ob_m128d ob_mm_mul_pd(ob_m128d a, ob_m128d b) { return vmulq_f64(a, b); }
static inline ob_m128d ob_mm_div_pd(ob_m128d a, ob_m128d b) { return vdivq_f64(a, b); }
static inline double ob_mm_cvtsd_f64(ob_m128d a) { return vgetq_lane_f64(a, 0); }
static inline ob_m128d ob_mm_unpacklo_pd(ob_m128d a, ob_m128d b) { return vzip1q_f64(a, b); }
static inline ob_m128d ob_mm_unpackhi_pd(ob_m128d a, ob_m128d b) { return vzip2q_f64(a, b); }
#else
struct ob_m128d {
    double lo, hi;  // lane 0, lane 1 (SSE2 element order)
};
static inline ob_m128d ob_mm_loadu_pd(const double* p) { return ob_m128d{p[0], p[1]}; }
static inline void ob_mm_storeu_pd(double* p, ob_m128d a) { p[0] = a.lo; p[1] = a.hi; }
static inline ob_m128d ob_mm_set1_pd(double x) { return ob_m128d{x, x}; }
// _mm_set_pd(e1, e0): the FIRST argument is the HIGH lane
static inline ob_m128d ob_mm_set_pd(double e1, double e0) { return ob_m128d{e0, e1}; }
static inline ob_m128d ob_mm_add_pd(ob_m128d a, ob_m128d b) { return ob_m128d{a.lo + b.lo, a.hi + b.hi}; }
static inline ob_m128d ob_mm_sub_pd(ob_m128d a, ob_m128d b) { return ob_m128d{a.lo - b.lo, a.hi - b.hi}; }
static inline ob_m128d ob_mm_mul_pd(ob_m128d a, ob_m128d b) { return ob_m128d{a.lo * b.lo, a.hi * b.hi}; }
static inline ob_m128d ob_mm_div_pd(ob_m128d a, ob_m128d b) { return ob_m128d{a.lo / b.lo, a.hi / b.hi}; }
static inline double ob_mm_cvtsd_f64(ob_m128d a) { return a.lo; }
static inline ob_m128d ob_mm_unpacklo_pd(ob_m128d a, ob_m128d b) { return ob_m128d{a.lo, b.lo}; }
static inline ob_m128d ob_mm_unpackhi_pd(ob_m128d a, ob_m128d b) { return ob_m128d{a.hi, b.hi}; }
#endif

#define __m128d ob_m128d
#define _mm_loadu_pd ob_mm_loadu_pd
#define _mm_storeu_pd ob_mm_storeu_pd
#define _mm_set1_pd ob_mm_set1_pd
#define _mm_set_pd ob_mm_set_pd
#define _mm_add_pd ob_mm_add_pd
#define _mm_sub_pd ob_mm_sub_pd
#define _mm_mul_pd ob_mm_mul_pd
#define _mm_div_pd ob_mm_div_pd
#define _mm_cvtsd_f64 ob_mm_cvtsd_f64
#define _mm_unpacklo_pd ob_mm_unpacklo_pd
#define _mm_unpackhi_pd ob_mm_unpackhi_pd
