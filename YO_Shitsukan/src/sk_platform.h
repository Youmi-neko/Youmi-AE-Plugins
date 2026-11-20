// mac_port: the platform layer of the Shitsukan engine. Everything Windows-only in the engine goes through here;
// the Windows implementation is the original code (the Windows build's picture and behaviour are unchanged), the
// portable implementation (standard C++ / POSIX) is what macOS uses.
//
//   SK_PLAT_WIN32 1  Windows, the original code paths (thread pool = the Windows system thread pool, ...)
//   SK_PLAT_WIN32 0  everything else; also on Windows with -DSK_FORCE_PORTABLE=1, which is how the portable thread
//                    pool is tested on Windows (build_win.bat portable / clang / macx64 / neon; fakehost cmp / refcmp
//                    compare the pictures). Environment, sleep, log files stay Win32 calls whenever _WIN32 is defined:
//                    they do not touch the picture, and the fake host changes the environment with SetEnvironmentVariable.
//   SK_SIMD_SSE2 / SK_SIMD_NEON / neither: the 4-float kernel of the recursive filter (RF4). All three variants do
//                    the same IEEE single-precision operations per lane in the same order (no fused multiply-add),
//                    so they are bit-identical; -DSK_NO_SIMD=1 forces the scalar one.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>

#if defined(_WIN32) && !SK_FORCE_PORTABLE
#define SK_PLAT_WIN32 1
#else
#define SK_PLAT_WIN32 0
#endif

#if SK_NEON_SHIM  // (Windows tests only: the NEON branch on x86 with tests/neon_shim.h, build_win.bat neon)
#define SK_SIMD_NEON 1
#include "neon_shim.h"
#elif !SK_NO_SIMD && (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64))
#define SK_SIMD_SSE2 1
#include <emmintrin.h>
#elif !SK_NO_SIMD && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64))
#define SK_SIMD_NEON 1
#include <arm_neon.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#endif

namespace skplat {

// ---- environment: like GetEnvironmentVariableA - the length of the value copied to buf (NUL-terminated), 0 when
// unset, >= size when it does not fit (buf then holds nothing useful)
inline unsigned GetEnv(const char* name, char* buf, unsigned size)
{
#ifdef _WIN32  // (also with SK_FORCE_PORTABLE: the process environment that SetEnvironmentVariable changes)
    return (unsigned)GetEnvironmentVariableA(name, buf, size);
#else
    const char* v = std::getenv(name);
    if (!v)
        return 0;
    const size_t n = std::strlen(v);
    if (n + 1 > size)
        return (unsigned)(n + 1);
    std::memcpy(buf, v, n + 1);
    return (unsigned)n;
#endif
}

inline void SleepMs(unsigned ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}

// ---- the diagnostic log file (path from an environment variable; Windows: UTF-16, elsewhere UTF-8 bytes)
#ifdef _WIN32
typedef std::wstring LogPathT;
inline LogPathT LogPathFromEnv(const char* name)
{
    wchar_t b[1024] = {0};
    std::wstring wn(name, name + std::strlen(name));
    const DWORD n = GetEnvironmentVariableW(wn.c_str(), b, 1024);
    return (n > 0 && n < 1024) ? std::wstring(b, n) : std::wstring();
}
inline FILE* OpenAppend(const LogPathT& p)
{
    FILE* fp = nullptr;
    return (!_wfopen_s(&fp, p.c_str(), L"ab") && fp) ? fp : nullptr;
}
#else
typedef std::string LogPathT;
inline LogPathT LogPathFromEnv(const char* name)
{
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}
inline FILE* OpenAppend(const LogPathT& p) { return std::fopen(p.c_str(), "ab"); }
#endif
// "YYYY-MM-DD hh:mm:ss.mmm pid=.. tid=.. " (local time)
inline void LogHead(char* head, size_t n)
{
#ifdef _WIN32
    SYSTEMTIME t;
    GetLocalTime(&t);
    snprintf(head, n, "%04d-%02d-%02d %02d:%02d:%02d.%03d pid=%lu tid=%lu ", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds,
             GetCurrentProcessId(), GetCurrentThreadId());
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    struct tm lt;
    const time_t s = (time_t)tv.tv_sec;
    localtime_r(&s, &lt);
    unsigned long long tid = 0;
#if defined(__APPLE__)
    uint64_t t64 = 0;
    pthread_threadid_np(nullptr, &t64);
    tid = t64;
#else
    tid = (unsigned long long)(uintptr_t)pthread_self();
#endif
    snprintf(head, n, "%04d-%02d-%02d %02d:%02d:%02d.%03d pid=%lu tid=%llu ", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec,
             (int)(tv.tv_usec / 1000), (unsigned long)getpid(), tid);
#endif
}

// ---- a plain file (dev dumps): fopen
inline FILE* OpenFile(const std::string& path, const char* mode)
{
#ifdef _WIN32
    FILE* fp = nullptr;
    return (!fopen_s(&fp, path.c_str(), mode) && fp) ? fp : nullptr;
#else
    return std::fopen(path.c_str(), mode);
#endif
}
inline const char* PathSep()
{
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}

// ---- memory (profiling only): working set / peak in MB
inline void ProcessMemoryMB(size_t& ws, size_t& pk)
{
    ws = pk = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    {
        ws = pmc.WorkingSetSize >> 20;
        pk = pmc.PeakWorkingSetSize >> 20;
    }
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) == KERN_SUCCESS)
    {
        ws = (size_t)(info.resident_size >> 20);
        pk = (size_t)(info.resident_size_max >> 20);
    }
#endif
}

// ---- 4 floats (RF4): cur + c * (prev - cur), lane by lane, a separate multiply and add (never fused)
#if SK_SIMD_SSE2
struct F4
{
    __m128 v;
};
inline F4 F4Load(const float* p) { return F4{_mm_loadu_ps(p)}; }
inline void F4Store(float* p, F4 a) { _mm_storeu_ps(p, a.v); }
inline F4 F4Set1(float c) { return F4{_mm_set1_ps(c)}; }
inline F4 F4Lerp(F4 cur, F4 c, F4 prev) { return F4{_mm_add_ps(cur.v, _mm_mul_ps(c.v, _mm_sub_ps(prev.v, cur.v)))}; }
#elif SK_SIMD_NEON
struct F4
{
    float32x4_t v;
};
inline F4 F4Load(const float* p) { return F4{vld1q_f32(p)}; }
inline void F4Store(float* p, F4 a) { vst1q_f32(p, a.v); }
inline F4 F4Set1(float c) { return F4{vdupq_n_f32(c)}; }
// vmulq + vaddq (NOT vmlaq / vfmaq: those fuse or may be fused)
inline F4 F4Lerp(F4 cur, F4 c, F4 prev) { return F4{vaddq_f32(cur.v, vmulq_f32(c.v, vsubq_f32(prev.v, cur.v)))}; }
#else
struct F4
{
    float v[4];
};
inline F4 F4Load(const float* p) { return F4{{p[0], p[1], p[2], p[3]}}; }
inline void F4Store(float* p, F4 a) { p[0] = a.v[0]; p[1] = a.v[1]; p[2] = a.v[2]; p[3] = a.v[3]; }
inline F4 F4Set1(float c) { return F4{{c, c, c, c}}; }
inline F4 F4Lerp(F4 cur, F4 c, F4 prev)
{
    F4 r;
    for (int k = 0; k < 4; ++k)
    {
        const float d = prev.v[k] - cur.v[k];
        const float m = c.v[k] * d;
        r.v[k] = cur.v[k] + m;
    }
    return r;
}
#endif

}  // namespace skplat
