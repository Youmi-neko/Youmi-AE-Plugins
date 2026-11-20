// Optic Bokeh platform layer (mac port, from v2.4.2). Every Windows-only call of the plug-in, the engine
// test tools and the offline probes goes through here; the engine (core/) needs nothing from it except the
// SSE2 shim (ob_sse2_emul.h via compat/emmintrin.h).
//
//   header-only (no OS headers): file open, environment, sleep  -> Windows: the exact MSVC CRT calls the
//                                 v2.4.2 sources made (fopen_s / _dupenv_s / _putenv_s), so behaviour is
//                                 unchanged; macOS: POSIX.
//   ob_platform_win.cpp          : windows.h confined here (code pages, AE version, exe path, process stats)
//   ob_platform_mac.cpp          : the macOS implementation of the same functions (CoreFoundation, mach,
//                                 getrusage). Written on Windows, NOT compiled or run yet (see PORTING_NOTES.md).
// Link exactly one of the two .cpp files into anything that uses the non-inline functions.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <thread>

namespace obp {

// ---- files: `path` is passed through unchanged (Windows: the ANSI code page, as fopen_s did in v2.4.2;
// macOS: UTF-8). Returns nullptr on failure.
inline FILE* fopen(const char* path, const char* mode)
{
#ifdef _WIN32
    FILE* f = nullptr;
    if (fopen_s(&f, path, mode) || !f)
        return nullptr;
    return f;
#else
    return std::fopen(path, mode);
#endif
}

// ---- environment. getenv: true if the variable is set (value receives it). setenv: value "" removes the
// variable (what _putenv_s(name, "") does on Windows).
inline bool getenv(const char* name, std::string& value)
{
#ifdef _WIN32
    char* v = nullptr;
    size_t n = 0;
    if (_dupenv_s(&v, &n, name) || !v)
        return false;
    value = v;
    std::free(v);
    return true;
#else
    const char* v = std::getenv(name);
    if (!v)
        return false;
    value = v;
    return true;
#endif
}

inline void setenv(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (!value || !*value)
        ::unsetenv(name);
    else
        ::setenv(name, value, 1);
#endif
}

inline void sleep_ms(unsigned ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

inline char path_sep()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

// ---- implemented in ob_platform_win.cpp / ob_platform_mac.cpp

// Full path of the running executable (the host application for a plug-in). "" if unknown.
std::string exe_path();

// The legacy multibyte code page used for "ANSI" host strings: Windows GetACP() (932 on a Japanese system).
// macOS has no ANSI code page; the mac implementation returns 932 (CP932 / Shift_JIS, the only legacy
// encoding the Japanese table needs). UNVERIFIED in AE on macOS - see PORTING_NOTES.md "Strings".
unsigned ansi_code_page();

// Major version of the host application (After Effects 26 / 27 ...), 0 if unknown.
// Windows: VERSIONINFO of the .exe (v2.4.2 behaviour). macOS: CFBundleShortVersionString of the main bundle.
int host_major_version();

// UTF-8 -> wide string (Windows: UTF-16; macOS: UTF-32 code points, wchar_t is 32-bit there). Invalid
// UTF-8 -> empty string (Windows: MB_ERR_INVALID_CHARS).
std::wstring utf8_to_wide(const char* u8);

// UTF-8 -> bytes of `codePage` (0 = ansi_code_page()), false if any character is not representable or the
// round trip back to Unicode differs. Windows: WideCharToMultiByte(WC_NO_BEST_FIT_CHARS) as in v2.4.2.
// macOS: CoreFoundation (932 -> kCFStringEncodingDOSJapanese), lossByte 0, round-trip checked.
bool utf8_to_codepage(const char* u8, unsigned codePage, std::string& out);

// ---- process statistics for the benchmark / probe tools (not used by the plug-in)
double process_peak_commit_mb();  // Windows: PeakPagefileUsage; macOS: peak resident size (ru_maxrss)
double process_peak_working_set_mb();  // Windows: PeakWorkingSetSize; macOS: ru_maxrss
double process_cpu_ms();          // user + kernel CPU time of this process
double system_cpu_busy(int ms);   // system-wide busy fraction measured over `ms` milliseconds (0..1)

}  // namespace obp
