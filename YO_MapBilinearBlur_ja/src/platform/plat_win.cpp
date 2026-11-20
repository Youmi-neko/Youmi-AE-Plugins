// Map Bilinear Blur - platform layer, Windows. The bodies are the v0.3.1 code moved here unchanged
// (MapBilinearBlur.cpp: GpuGlobalSetup / LogLine / LogFrame; mbb_strings.cpp: conversions, HostMajorVersion).
#ifdef _WIN32
#include "plat.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "version.lib")  // HostMajorVersion: GetFileVersionInfoW

namespace plat {

bool GetEnvShort(const char* name, char* buf, size_t n)
{
    size_t got = 0;
    return getenv_s(&got, buf, n, name) == 0 && got > 0;
}

bool GetEnvPath(const char* name, NativePath& out)
{
    wchar_t wname[128] = {0};
    for (size_t i = 0; name[i] && i < 127; ++i)
        wname[i] = (wchar_t)(unsigned char)name[i];  // ASCII variable names
    wchar_t lp[1024] = {0};
    const DWORD ln = GetEnvironmentVariableW(wname, lp, 1024);
    if (ln > 0 && ln < 1024)
    {
        out = lp;
        return true;
    }
    return false;
}

void AppendLine(const NativePath& path, const char* line)
{
    FILE* f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"ab") == 0 && f)
    {
        std::fputs(line, f);
        std::fputc('\n', f);
        std::fclose(f);
    }
}

unsigned long ProcessId() { return (unsigned long)GetCurrentProcessId(); }
unsigned long ThreadId() { return (unsigned long)GetCurrentThreadId(); }

int HostMajorVersion()
{
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
}

int JapaneseEncodingForHost(int major) { return major == 26 ? 1 : 0; }

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

}  // namespace plat
#endif
