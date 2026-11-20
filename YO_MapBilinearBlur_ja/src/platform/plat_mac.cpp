// Map Bilinear Blur - platform layer, macOS (and other POSIX for the offline tests).
// WRITTEN ON WINDOWS, NEVER COMPILED OR RUN ON A MAC. Behaviour must match plat_win.cpp (see plat.h for the contract).
#ifndef _WIN32
#include "plat.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iconv.h>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <mach-o/dyld.h>
#endif

namespace plat {

bool GetEnvShort(const char* name, char* buf, size_t n)
{
    const char* v = std::getenv(name);
    if (!v || !*v || std::strlen(v) + 1 > n)
        return false;
    std::memcpy(buf, v, std::strlen(v) + 1);
    return true;
}

bool GetEnvPath(const char* name, NativePath& out)
{
    const char* v = std::getenv(name);
    if (!v || !*v || std::strlen(v) >= 1024)
        return false;
    out = v;
    return true;
}

void AppendLine(const NativePath& path, const char* line)
{
    FILE* f = std::fopen(path.c_str(), "ab");
    if (f)
    {
        std::fputs(line, f);
        std::fputc('\n', f);
        std::fclose(f);
    }
}

unsigned long ProcessId() { return (unsigned long)getpid(); }
unsigned long ThreadId()
{
#ifdef __APPLE__
    uint64_t tid = 0;
    pthread_threadid_np(nullptr, &tid);
    return (unsigned long)tid;
#else
    return (unsigned long)(uintptr_t)pthread_self();
#endif
}

namespace {
int LeadingInt(const char* s)
{
    int v = 0;
    bool any = false;
    for (; s && *s >= '0' && *s <= '9'; ++s)
    {
        v = v * 10 + (*s - '0');
        any = true;
        if (v > 100000)
            return 0;
    }
    return any ? v : 0;
}
}  // namespace

int HostMajorVersion()
{
#ifdef __APPLE__
    // 1. the main bundle's short version ("26.5" -> 26). AE's application bundle is the main bundle in the UI.
    if (CFBundleRef b = CFBundleGetMainBundle())
    {
        CFTypeRef v = CFBundleGetValueForInfoDictionaryKey(b, CFSTR("CFBundleShortVersionString"));
        if (v && CFGetTypeID(v) == CFStringGetTypeID())
        {
            char s[64] = {0};
            if (CFStringGetCString((CFStringRef)v, s, sizeof s, kCFStringEncodingUTF8))
            {
                const int m = LeadingInt(s);
                if (m >= 10 && m < 100)
                    return m;
            }
        }
    }
    // 2. the executable path (aerender / a host without a plist): ".../Adobe After Effects 2026/..." -> 26
    char path[4096] = {0};
    uint32_t sz = sizeof path;
    if (_NSGetExecutablePath(path, &sz) == 0)
    {
        const char* k = std::strstr(path, "After Effects 20");
        if (k)
        {
            const int y = LeadingInt(k + std::strlen("After Effects "));
            if (y >= 2010 && y < 2100)
                return y - 2000;
        }
    }
#endif
    return 0;
}

// mac_port POLICY (mac_common/POLICY_macos.md, the same in all six packages): which encoding Mac After Effects wants
// for Japanese names is UNVERIFIED, so the Mac build shows ENGLISH names (2) until it has been measured in Mac AE 2026
// AND AE 2027 ("MBB Encoding Test" diag variant, PORTING_NOTES section 9). Then put the measured result into the two
// macros (0 = CP932, 1 = UTF-8, 2 = English): MBB_MAC_ENC_AE26 for AE 26.x, MBB_MAC_ENC_OTHER for every other
// version. For the measurement only, MAPBILINEARBLUR_UI_ENC (utf8 | sjis | en, set when AE starts) overrides both.
// Windows is not affected (plat_win.cpp keeps its measured rule: 26 -> UTF-8, other -> CP932).
#ifndef MBB_MAC_ENC_AE26
#define MBB_MAC_ENC_AE26 2
#endif
#ifndef MBB_MAC_ENC_OTHER
#define MBB_MAC_ENC_OTHER 2
#endif
int JapaneseEncodingForHost(int major)
{
    const char* v = std::getenv("MAPBILINEARBLUR_UI_ENC");
    if (v && !std::strcmp(v, "utf8"))
        return 1;
    if (v && !std::strcmp(v, "sjis"))
        return 0;
    if (v && !std::strcmp(v, "en"))
        return 2;
    return major == 26 ? MBB_MAC_ENC_AE26 : MBB_MAC_ENC_OTHER;
}

std::wstring Utf8ToWide(const char* u8)
{
    std::wstring w;
    if (!u8 || !*u8)
        return w;
    const unsigned char* p = (const unsigned char*)u8;
    while (*p)
    {
        unsigned c = *p++, n = 0, cp = 0;
        if (c < 0x80) { cp = c; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 3; }
        else return std::wstring();
        for (unsigned i = 0; i < n; ++i, ++p)
        {
            if ((*p & 0xC0) != 0x80)
                return std::wstring();
            cp = (cp << 6) | (*p & 0x3F);
        }
        const unsigned minv[4] = {0, 0x80, 0x800, 0x10000};
        if (cp < minv[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            return std::wstring();  // overlong / out of range / surrogate: invalid like MB_ERR_INVALID_CHARS
        w.push_back((wchar_t)cp);
    }
    return w;
}

namespace {
bool Iconv(const char* to, const char* from, const std::string& in, std::string& out)
{
    iconv_t cd = iconv_open(to, from);
    if (cd == (iconv_t)-1)
        return false;
    std::vector<char> buf(in.size() * 4 + 16);
    char* src = const_cast<char*>(in.data());
    size_t sl = in.size();
    char* dst = buf.data();
    size_t dl = buf.size();
    const size_t r = iconv(cd, &src, &sl, &dst, &dl);
    iconv_close(cd);
    if (r == (size_t)-1 || sl != 0 || r != 0)  // r > 0: irreversible (substituted) conversions
        return false;
    out.assign(buf.data(), (size_t)(dst - buf.data()));
    return true;
}
}  // namespace

bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out)
{
    out.clear();
    if (!u8)
        return false;
    if (!*u8)
        return true;
    if (Utf8ToWide(u8).empty())
        return false;  // invalid UTF-8
    if (codePage != 0 && codePage != 932)
        return false;  // only CP932 is needed (Japanese UI)
    std::string s, back;
    if (!Iconv("CP932", "UTF-8", u8, s) || !Iconv("UTF-8", "CP932", s, back) || back != u8)
        return false;
    out = s;
    return true;
}

}  // namespace plat
#endif
