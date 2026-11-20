// Tebure - platform layer. Everything that differs between Windows and macOS goes through here; the rest of src\ is
// plain C++17. Implementations: plat_win.cpp (the 0.3.1.1 Windows code, moved here unchanged) and plat_mac.cpp
// (written on Windows, NEVER COMPILED OR RUN ON A MAC - see PORTING_NOTES.md).
//
// Not in this layer because standard C++ already covers it on both platforms: threads (std::thread, std::atomic,
// thread_local, hardware_concurrency), timers (std::chrono::steady_clock, TB_PROFILE builds only), aligned storage
// (alignas on the stack only; no aligned heap allocation is used), snprintf. The SIMD kernels are selected at
// compile time in tb_warp_px.h / tb_warp_avx.cpp (TB_SIMD_SSE2 / TB_SIMD_AVX below) and at run time with
// plat::CpuHasAvx().
#pragma once
#include <string>

// ---- SIMD selection (compile time)
//   TB_FORCE_SCALAR : no intrinsics at all (the portable reference path; bit-identical to SSE2 and AVX, tested)
//   TB_NO_AVX       : SSE2 only (no AVX kernel compiled, CpuHasAvx() never consulted)
// x86-64 (MSVC _M_X64, clang/gcc __x86_64__): SSE2 always, AVX kernel with run-time dispatch.
// arm64 (Apple Silicon) and anything else: scalar (clang auto-vectorises the scalar loops without reordering any
// floating-point operation, because -ffp-contract=off and no fast-math are enforced by the build).
#if !defined(TB_FORCE_SCALAR) && (defined(_M_X64) || defined(__x86_64__))
#define TB_SIMD_SSE2 1
#if !defined(TB_NO_AVX)
#define TB_SIMD_AVX 1
#endif
#endif

namespace plat {

// true if the CPU and the OS support AVX (Windows: cpuid + xgetbv; macOS x86_64: sysctl hw.optional.avx1_0;
// arm64: false). Cached after the first call.
bool CpuHasAvx();

// Major version of the host application (AfterFX.exe 26.5 -> 26; macOS: the main bundle's
// CFBundleShortVersionString "26.5.0" -> 26), 0 if unknown.
int HostMajorVersion();

// UTF-8 -> wide string (UTF-16 on Windows, UTF-32 on macOS; only BMP characters are used by the tables).
// Empty on invalid UTF-8 (as MultiByteToWideChar with MB_ERR_INVALID_CHARS).
std::wstring Utf8ToWide(const char* u8);

// UTF-8 -> bytes in a code page, false if any character cannot be represented exactly (no best fit, round trip
// checked). codePage 0 = Windows: the process ANSI code page (CP_ACP); macOS: CP932 (iconv "CP932").
bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out);

// Name of the code page used for codePage 0, for messages / the encoding test ("CP_ACP" or "CP932").
const char* DefaultCodePageName();

}  // namespace plat
