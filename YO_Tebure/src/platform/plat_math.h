// Tebure - the libm calls whose results feed the picture, in one place (tbm::).
//
// sqrt / floor / ceil / fabs / fmod / llround are exact (IEEE / C standard) on every platform and stay std::.
// sin / cos / exp / log / pow / hypot (double) and powf (float) are NOT required to be correctly rounded: MSVC's
// UCRT and Apple's libm may return a different last bit for some arguments. What depends on them:
//   tb_model.cpp  sin, cos, exp, log, pow : THE SHAKE ITSELF (noise generators, gait, breathing, impacts, jolts,
//                                           film weave, spectral shaping / gains, Gaussian random numbers)
//   tb_plan.cpp   (none; exact functions only)
//   tb_presets.cpp exp                    : auto mood weights (AutoPick preset selection)
//   tb_warp.cpp   exp, sin, cos           : camera projection (zoom scale, pan / tilt / roll rotation matrix);
//                 sin (via tb_warp_px.h)  : Lanczos 3 weights; hypot: Motion Graph overlay strokes
//   Tebure.cpp    powf                    : "Blend in Linear Light" (2.2 gamma, 8/16 bpc tables and 32 bpc per pixel)
//   tb_thumbs.cpp log, sqrt               : preset picker thumbnails only (UI, not the rendered picture) - not wrapped
//
// Default (Windows and macOS): tbm:: forwards to std::, so the Windows build is unchanged (proven bit-identical).
// tests\libm_probe records every argument the golden set passes through here and lets the Mac compare Apple's
// results with the Windows ones (tests\refs\libm_win.bin) and measure the effect of 1-ulp perturbations.
// PROPOSAL for bit-identical shake across platforms (not implemented; changes Windows output for NEW instances
// only): a portable, correctly rounded implementation (e.g. CORE-MATH, MIT) selected by a new hidden Render
// Version 4 on both platforms; Render Version 1-3 projects keep the platform libm (Windows: exactly as now).
#pragma once
#include <cmath>

namespace tbm {

#if defined(TB_LIBM_HOOK)
// test builds only (tests\libm_probe): fn = 0 sin, 1 cos, 2 exp, 3 log, 4 pow, 5 hypot, 6 powf
extern double (*g_hook)(int fn, double a, double b, double r);
#define TBM_RET(fn, a, b, r) return g_hook ? g_hook(fn, a, b, r) : r
#define TBM_RETF(fn, a, b, r) return g_hook ? (float)g_hook(fn, a, b, r) : r
#else
#define TBM_RET(fn, a, b, r) return r
#define TBM_RETF(fn, a, b, r) return r
#endif

inline double sin(double x) { const double r = std::sin(x); TBM_RET(0, x, 0, r); }
inline double cos(double x) { const double r = std::cos(x); TBM_RET(1, x, 0, r); }
inline double exp(double x) { const double r = std::exp(x); TBM_RET(2, x, 0, r); }
inline double log(double x) { const double r = std::log(x); TBM_RET(3, x, 0, r); }
inline double pow(double x, double y) { const double r = std::pow(x, y); TBM_RET(4, x, y, r); }
inline double hypot(double x, double y) { const double r = std::hypot(x, y); TBM_RET(5, x, y, r); }
inline float powf(float x, float y) { const float r = std::pow(x, y); TBM_RETF(6, x, y, r); }

#undef TBM_RET
#undef TBM_RETF

}  // namespace tbm
