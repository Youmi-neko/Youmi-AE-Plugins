// Windows-only TEST of the arm64 SSE2 stand-in (CMake -DOB_SSE2_EMULATION=ON with MSVC / clang-cl): force-included
// (/FI) into every C++ file. The real intrinsic headers are parsed FIRST (so the compiler's own headers, which
// include <emmintrin.h> themselves, stay intact and their include guards are set); then ob_sse2_emul.h maps
// __m128d / _mm_* used afterwards - i.e. in core/lenscore.cpp - to the scalar stand-in. On macOS the stand-in is
// reached through src/platform/compat/emmintrin.h instead (arm64 has no SSE headers to protect).
#pragma once
#include <intrin.h>
#include "ob_sse2_emul.h"
