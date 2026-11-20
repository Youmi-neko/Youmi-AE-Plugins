// Optic Bokeh platform layer, Windows implementation (see ob_platform.h). The string / version code is the
// v2.4.2 plugin/ob_strings.cpp code moved here unchanged; the process statistics are the v2.4.2
// tools/perf_bench.cpp ones. windows.h is included only in this file.
#ifndef _WIN32
#error "ob_platform_win.cpp is the Windows implementation; build ob_platform_mac.cpp on macOS"
#endif
#include "ob_platform.h"

#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "version.lib")  // host_major_version: GetFileVersionInfoW
#pragma comment(lib, "psapi.lib")    // process_peak_commit_mb: GetProcessMemoryInfo

namespace obp {

std::string exe_path()
{
    char exe[MAX_PATH] = {0};
    const DWORD n = GetModuleFileNameA(nullptr, exe, MAX_PATH);
    return n ? std::string(exe, n) : std::string();
}

unsigned ansi_code_page() { return GetACP(); }

int host_major_version()
{
    static const int v = [] {
        wchar_t path[MAX_PATH * 2] = {0};
        if (!GetModuleFileNameW(nullptr, path, MAX_PATH * 2))
            return 0;
        DWORD h = 0;
        const DWORD n = GetFileVersionInfoSizeW(path, &h);
        if (!n)
            return 0;
        std::vector<unsigned char> buf(n);
        if (!GetFileVersionInfoW(path, 0, n, buf.data()))
            return 0;
        VS_FIXEDFILEINFO* fi = nullptr;
        UINT len = 0;
        if (!VerQueryValueW(buf.data(), L"\\", (void**)&fi, &len) || !fi || len < sizeof(VS_FIXEDFILEINFO))
            return 0;
        return (int)(fi->dwFileVersionMS >> 16);
    }();
    return v;
}

std::wstring utf8_to_wide(const char* u8)
{
    if (!u8 || !*u8)
        return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, u8, -1, nullptr, 0);
    if (n <= 0)
        return std::wstring();
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, u8, -1, &w[0], n);
    w.resize((size_t)n - 1);
    return w;
}

bool utf8_to_codepage(const char* u8, unsigned codePage, std::string& out)
{
    out.clear();
    const std::wstring w = utf8_to_wide(u8);
    if (w.empty())
        return u8 && !*u8;
    const UINT cp = codePage ? codePage : CP_ACP;
    BOOL usedDefault = FALSE;
    // WC_NO_BEST_FIT_CHARS: no silent look-alike substitution; usedDefault catches unmappable chars
    const int n = WideCharToMultiByte(cp, WC_NO_BEST_FIT_CHARS, w.c_str(), -1, nullptr, 0, nullptr, &usedDefault);
    if (n <= 0 || usedDefault)
        return false;
    std::string s((size_t)n, '\0');
    usedDefault = FALSE;
    WideCharToMultiByte(cp, WC_NO_BEST_FIT_CHARS, w.c_str(), -1, &s[0], n, nullptr, &usedDefault);
    if (usedDefault)
        return false;
    s.resize((size_t)n - 1);
    // round trip must give the same UTF-16 (guards code pages where WC_NO_BEST_FIT_CHARS is ignored)
    const int m = MultiByteToWideChar(cp, 0, s.c_str(), -1, nullptr, 0);
    if (m <= 0)
        return false;
    std::wstring back((size_t)m, L'\0');
    MultiByteToWideChar(cp, 0, s.c_str(), -1, &back[0], m);
    back.resize((size_t)m - 1);
    if (back != w)
        return false;
    out = s;
    return true;
}

static unsigned long long ft2u(const FILETIME& f) { return ((unsigned long long)f.dwHighDateTime << 32) | f.dwLowDateTime; }

double process_peak_commit_mb()
{
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PeakPagefileUsage / 1048576.0;
}

double process_peak_working_set_mb()
{
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PeakWorkingSetSize / 1048576.0;
}

double process_cpu_ms()
{
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    return (ft2u(k) + ft2u(u)) / 1e4;
}

double system_cpu_busy(int ms)
{
    FILETIME i0, k0, u0, i1, k1, u1;
    GetSystemTimes(&i0, &k0, &u0);
    Sleep(ms);
    GetSystemTimes(&i1, &k1, &u1);
    const double idle = (double)(ft2u(i1) - ft2u(i0));
    const double tot = (double)(ft2u(k1) - ft2u(k0) + ft2u(u1) - ft2u(u0));  // kernel includes idle
    return tot > 0 ? 1.0 - idle / tot : 0.0;
}

}  // namespace obp
