// Tebure - platform layer, macOS. WRITTEN ON WINDOWS, NEVER COMPILED OR RUN: verify on the Mac (PORTING_NOTES.md).
// Links: CoreFoundation (bundle version), libiconv (part of libSystem on macOS; CMake adds -liconv to be safe).
#if defined(__APPLE__)
#include "plat.h"

#include <CoreFoundation/CoreFoundation.h>
#include <iconv.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace plat {

bool CpuHasAvx()
{
#if defined(__x86_64__)
    // hw.optional.avx1_0 is 1 only when the CPU has AVX AND the kernel saves the YMM state (the xgetbv check).
    // Under Rosetta 2 (x86_64 slice on Apple Silicon) it reports 0 on macOS 13/14 and 1 on macOS 15+ (Rosetta
    // translates AVX/AVX2 there): either way the result is bit-identical (the AVX kernel equals SSE2 and scalar).
    static const bool has = [] {
        int v = 0;
        size_t n = sizeof(v);
        return sysctlbyname("hw.optional.avx1_0", &v, &n, nullptr, 0) == 0 && v != 0;
    }();
    return has;
#else
    return false;  // arm64: no AVX (the scalar path is used)
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
    // Same detection as the Map Bilinear Blur package (so its diag build shows what Tebure detects):
    // 1. the main bundle's CFBundleShortVersionString ("26.5" -> 26; accepted 10..99). AE's application bundle is the
    //    main bundle in the UI. 2. the executable path (aerender, or a host without a plist):
    //    ".../Adobe After Effects 2026/..." -> 26. UNVERIFIED on macOS: check both AE and aerender (PORTING_NOTES 9).
    static const int v = [] {
        if (CFBundleRef b = CFBundleGetMainBundle())
        {
            CFTypeRef t = CFBundleGetValueForInfoDictionaryKey(b, CFSTR("CFBundleShortVersionString"));
            if (t && CFGetTypeID(t) == CFStringGetTypeID())
            {
                char s[64] = {0};
                if (CFStringGetCString((CFStringRef)t, s, sizeof s, kCFStringEncodingUTF8))
                {
                    const int m = LeadingInt(s);
                    if (m >= 10 && m < 100)
                        return m;
                }
            }
        }
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
        return 0;
    }();
    return v;
}

// strict UTF-8 decoder (rejects overlongs, surrogates, > U+10FFFF, truncated sequences): empty on error, like
// MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS)
std::wstring Utf8ToWide(const char* u8)
{
    std::wstring w;
    if (!u8)
        return w;
    const unsigned char* p = (const unsigned char*)u8;
    while (*p)
    {
        unsigned c = *p++;
        int extra = 0;
        unsigned minv = 0;
        if (c < 0x80) { extra = 0; }
        else if ((c & 0xE0) == 0xC0) { extra = 1; c &= 0x1F; minv = 0x80; }
        else if ((c & 0xF0) == 0xE0) { extra = 2; c &= 0x0F; minv = 0x800; }
        else if ((c & 0xF8) == 0xF0) { extra = 3; c &= 0x07; minv = 0x10000; }
        else return std::wstring();
        for (int i = 0; i < extra; ++i)
        {
            if ((*p & 0xC0) != 0x80)
                return std::wstring();
            c = (c << 6) | (*p++ & 0x3F);
        }
        if (c < minv || c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))
            return std::wstring();
        w.push_back((wchar_t)c);  // wchar_t is 32-bit on macOS: one code point each
    }
    return w;
}

namespace {
bool Iconv(const char* to, const char* from, const std::string& in, std::string& out)
{
    out.clear();
    iconv_t cd = iconv_open(to, from);
    if (cd == (iconv_t)-1)
        return false;
    std::vector<char> buf(in.size() * 4 + 16);
    char* src = const_cast<char*>(in.data());
    size_t srcLeft = in.size();
    char* dst = buf.data();
    size_t dstLeft = buf.size();
    const size_t r = iconv(cd, &src, &srcLeft, &dst, &dstLeft);
    iconv_close(cd);
    // r > 0 = number of irreversible (substituted) conversions: refuse, like WC_NO_BEST_FIT_CHARS + usedDefault
    if (r == (size_t)-1 || r != 0 || srcLeft != 0)
        return false;
    out.assign(buf.data(), buf.size() - dstLeft);
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
    const char* cp = (codePage == 0 || codePage == 932) ? "CP932" : nullptr;
    if (codePage == 65001)
    {
        out = u8;
        return true;
    }
    if (!cp)
        return false;  // other code pages are not needed on macOS
    std::string s, back;
    if (!Iconv(cp, "UTF-8", u8, s) || !Iconv("UTF-8", cp, s, back) || back != u8)
        return false;  // the round trip must give the same text (as the Windows code checks)
    out = s;
    return true;
}

const char* DefaultCodePageName() { return "CP932"; }

}  // namespace plat
#endif  // __APPLE__
