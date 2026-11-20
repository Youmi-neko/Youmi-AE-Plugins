// Map Bilinear Blur - platform layer. Everything that differs between Windows and macOS lives behind this header:
//   plat_win.cpp : the Windows code moved here verbatim from MapBilinearBlur.cpp / mbb_strings.cpp (v0.3.1)
//   plat_mac.cpp : the macOS implementation (written on Windows, NOT compiled or run yet - see PORTING_NOTES.md)
// Nothing here touches pixels. The SIMD choice for the engine (mbb_core.cpp, Acc5) is made by the macros at the end.
#pragma once
#include <cstddef>
#include <string>

namespace plat {

#ifdef _WIN32
typedef std::wstring NativePath;  // UTF-16 (GetEnvironmentVariableW / _wfopen_s)
#else
typedef std::string NativePath;   // UTF-8 (POSIX)
#endif

// getenv_s semantics: true if `name` is set, non-empty and its value + NUL fits in n bytes (buf then holds it).
// v0.3.1 reads MAPBILINEARBLUR_GPU with n = 8: a longer value counts as unset (as getenv_s's ERANGE did).
bool GetEnvShort(const char* name, char* buf, size_t n);
// a path from the environment: true if set, non-empty and shorter than 1024 characters (as GetEnvironmentVariableW
// with a 1024-wchar buffer).
bool GetEnvPath(const char* name, NativePath& out);
// appends line + '\n' to the file (opened in binary append mode and closed again per call; errors ignored)
void AppendLine(const NativePath& path, const char* line);
unsigned long ProcessId();
unsigned long ThreadId();

// Major version of the host application (AE 26.5 -> 26), 0 if unknown.
//   Windows: the file version of the process executable (AfterFX.exe / aerender.exe), as v0.3.1.
//   macOS  : CFBundleShortVersionString of the main bundle, else "After Effects 20NN" in the executable path
//            (2026 -> 26). UNVERIFIED on a Mac.
int HostMajorVersion();

// Japanese host-string encoding for this platform and host major version: 1 = UTF-8, 0 = code page (CP932), 2 = English.
//   Windows: AE 26 -> UTF-8, every other version -> ANSI code page (measured with "MBB Encoding Test" in AE 26.5 / 27.0).
//   macOS  : ENGLISH (2) until measured (mac_port policy). UNVERIFIED: run the "MBB Encoding Test" effect in each Mac AE
//            version and set MBB_MAC_ENC_AE26 / MBB_MAC_ENC_OTHER in plat_mac.cpp (PORTING_NOTES.md, "Strings");
//            MAPBILINEARBLUR_UI_ENC=utf8|sjis|en overrides it for the measurement.
int JapaneseEncodingForHost(int major);

// UTF-8 -> UTF-16 (Windows) / UTF-32 (macOS wchar_t). Empty for empty or invalid input (MB_ERR_INVALID_CHARS).
std::wstring Utf8ToWide(const char* u8);
// UTF-8 -> code page bytes, exact (no best fit, no default char, round trip checked); false otherwise.
// codePage 0 = the process ANSI code page on Windows; on macOS 0 and 932 both mean CP932 (iconv "CP932").
bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out);

}  // namespace plat

// ---- SIMD selection for the CPU engine (mbb_core.cpp Acc5). All three paths are bit-identical: each lane is one
// IEEE single multiply followed by one IEEE single add (never fused: -ffp-contract=off / MSVC without /arch:AVX2,
// and the NEON path uses separate vmulq / vaddq). Define MBB_FORCE_SCALAR to build the scalar reference path.
#if !defined(MBB_FORCE_SCALAR) && (defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__))
#define MBB_SIMD_SSE2 1
#elif !defined(MBB_FORCE_SCALAR) && !defined(MBB_NO_NEON) && (defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64))
#define MBB_SIMD_NEON 1
#else
#define MBB_SIMD_SCALAR 1
#endif
#if defined(MBB_SIMD_SSE2)
#define MBB_SIMD_NAME "SSE2"
#elif defined(MBB_SIMD_NEON)
#define MBB_SIMD_NAME "NEON"
#else
#define MBB_SIMD_NAME "scalar"
#endif
