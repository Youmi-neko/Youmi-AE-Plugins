// Tebure - platform layer, Windows. The bodies are the 0.3.1.1 code from tb_strings.cpp / tb_warp.cpp, unchanged.
#ifdef _WIN32
#include "plat.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>

#include <vector>
#pragma comment(lib, "version.lib")  // HostMajorVersion: GetFileVersionInfoW

namespace plat {

bool CpuHasAvx()
{
#if defined(_M_X64)
    static const bool has = [] {
        int r[4] = {0, 0, 0, 0};
        __cpuid(r, 1);
        const bool avx = (r[2] & (1 << 28)) != 0, osx = (r[2] & (1 << 27)) != 0;
        return avx && osx && ((_xgetbv(0) & 6) == 6);
    }();
    return has;
#else
    return false;  // Windows on ARM64: no AVX
#endif
}

int HostMajorVersion()
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

std::wstring Utf8ToWide(const char* u8)
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

bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out)
{
    out.clear();
    const std::wstring w = Utf8ToWide(u8);
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

const char* DefaultCodePageName() { return "CP_ACP"; }

}  // namespace plat
#endif  // _WIN32
