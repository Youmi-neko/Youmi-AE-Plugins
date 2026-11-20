// Optic Bokeh platform layer: <emmintrin.h> shadow (mac port). This directory is put FIRST on the include
// path of core/lenscore.cpp (and of the files that #include it) on macOS, so that the unchanged engine
// source builds for arm64. See ../ob_sse2_emul.h.
//   x86_64 (Intel Mac slice, Windows): the real SSE2 header (#include_next), unless OB_SSE2_EMULATION.
//   arm64 (Apple Silicon): the scalar stand-in.
// MSVC has no #include_next; on Windows this directory is only used for the OB_SSE2_EMULATION test build.
#ifndef OB_COMPAT_EMMINTRIN_H
#define OB_COMPAT_EMMINTRIN_H
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(OB_SSE2_EMULATION)
#if defined(__clang__) || defined(__GNUC__)
#include_next <emmintrin.h>
#else
#error "compat/emmintrin.h on an x86 MSVC build without OB_SSE2_EMULATION: remove src/platform/compat from the include path"
#endif
#else
#include "../ob_sse2_emul.h"
#endif
#endif
