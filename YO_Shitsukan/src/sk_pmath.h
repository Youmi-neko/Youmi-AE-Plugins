// mac_port: the engine's float math, routed through one place.
//
//   SK_PORTABLE_MATH 0 (default on Windows): pm::f(x) IS std::f(x) - the MSVC runtime, exactly as before (the
//                      Windows build's picture is unchanged).
//   SK_PORTABLE_MATH 1 (default elsewhere): the float functions below are sk_pmath_model.h's portable models plus
//                      the exception tables written on Windows by tools/gen_pmath_tables.cpp, i.e. the MSVC
//                      runtime's results bit for bit inside the table domains (InTableDomain), on any platform
//                      and on any CPU (arm64 / x86_64). tests/pmath_check verifies it against tests/ref/pm_hashes.txt.
//                      'The MSVC runtime' = the VS 2022 x64 UCRT on an FMA3 CPU: the UCRT picks FMA3 or SSE2 code for
//                      logf / sinf / cosf / powf at run time and the two differ in rare inputs; the tables (and the
//                      shipped .aex as it runs on every FMA3 machine) are the FMA3 path.
//
// Proven equal to MSVC (every float of the domain tried): expf (all floats), logf [2^-32, 2^32], sinf / cosf
// (|x| < 16), powf(x, 2.2f) for x in [2^-57, 2^40], powf(x, 1/2.2f) for x in (0, 2^40], and the two fixed-base calls
// powf(0.03f/0.30f, y) (line threshold) and powf(0.03f/0.6f, y) (halo) for y in [0, 1].
// NOT proven (binary or double functions; the model / the platform libm, close but possibly 1 ulp off in rare
// cases): atan2f (sample: 1.5e-5 of inputs differ from MSVC), powf with any other base / exponent, powf(x, 2.2f) for
// x < 2^-57 (subnormal results, Linear mode 32 bpc only), and every double-precision call (exp, pow, sin, cos of
// double: Apple's libm). See PORTING_NOTES.md "libm".
#pragma once
#include <cmath>

#ifndef SK_PORTABLE_MATH
#if defined(_WIN32)
#define SK_PORTABLE_MATH 0
#else
#define SK_PORTABLE_MATH 1
#endif
#endif

namespace pm {
#if SK_PORTABLE_MATH
float ExpF(float x);
float LogF(float x);
float SinF(float x);
float CosF(float x);
float PowF(float x, float y);
float Atan2F(float y, float x);
inline float exp(float x) { return ExpF(x); }
inline float log(float x) { return LogF(x); }
inline float sin(float x) { return SinF(x); }
inline float cos(float x) { return CosF(x); }
inline float pow(float x, float y) { return PowF(x, y); }
inline float atan2(float y, float x) { return Atan2F(y, x); }
#else
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sin(float x) { return std::sin(x); }
inline float cos(float x) { return std::cos(x); }
inline float pow(float x, float y) { return std::pow(x, y); }
inline float atan2(float y, float x) { return std::atan2(y, x); }
#endif
// double precision: the platform libm on every platform (see the header comment)
inline double exp(double x) { return std::exp(x); }
inline double log(double x) { return std::log(x); }
inline double sin(double x) { return std::sin(x); }
inline double cos(double x) { return std::cos(x); }
inline double pow(double x, double y) { return std::pow(x, y); }
inline double atan2(double y, double x) { return std::atan2(y, x); }

// (tests) 1 when the portable functions are compiled in; their table sizes
int Portable();
int TableEntries(int fn);  // 0 exp, 1 log, 2 sin, 3 cos, 4 pow 2.2, 5 pow 1/2.2, 6 pow(0.1, y), 7 pow(0.05, y)
}  // namespace pm
