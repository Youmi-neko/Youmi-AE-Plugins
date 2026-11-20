// Shitsukan: English / Japanese UI strings (see sk_strings.h; copied from Map Bilinear Blur / Optic Bokeh v2.3).
#include "sk_strings.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "version.lib")  // HostMajorVersion: GetFileVersionInfoW
#elif defined(__APPLE__)
// mac_port: the host version from the main bundle, Shift_JIS (CP932) conversion through CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <cstdlib>
#endif

// mac_port POLICY (mac_common/POLICY_macos.md, the same in all six packages): which encoding Mac After Effects wants
// for Japanese names is UNVERIFIED (PORTING_NOTES.md "Strings"), so the Mac build shows ENGLISH names (encoding 2)
// until it has been measured in Mac AE 2026 AND AE 2027. Then put the measured result into the two macros (0 = CP932
// via CoreFoundation, 1 = UTF-8, 2 = English): SK_MAC_ENC_AE26 for AE 26.x, SK_MAC_ENC_OTHER for every other
// version. For the measurement only, the environment variable SHITSUKAN_UI_ENC (utf8 | sjis | en, set when AE starts)
// overrides both. Windows is not affected.
#if defined(__APPLE__)
#ifndef SK_MAC_ENC_AE26
#define SK_MAC_ENC_AE26 2
#endif
#ifndef SK_MAC_ENC_OTHER
#define SK_MAC_ENC_OTHER 2
#endif
namespace {
int MacEncodingForMajor(int major)
{
    return major == 26 ? SK_MAC_ENC_AE26 : SK_MAC_ENC_OTHER;  // UNVERIFIED: English until measured
}
int MacEncodingOverride()
{
    const char* v = std::getenv("SHITSUKAN_UI_ENC");
    if (!v)
        return -1;
    if (!std::strcmp(v, "utf8"))
        return 1;
    if (!std::strcmp(v, "sjis"))
        return 0;
    if (!std::strcmp(v, "en"))
        return 2;
    return -1;
}
}  // namespace
#endif

// Host string encoding for the Japanese names (build switch, MapBilinearBlur):
//   SK_STR_ENC 0 = Windows ANSI code page (CP932; Optic Bokeh v2.3 way - real AE 2026 showed BLANK names)
//   SK_STR_ENC 1 = UTF-8 bytes (default; limits checked in UTF-8 bytes)
//   SK_STR_ENC 2 = English only
//   SK_STR_ENC 3 = by host version (default): AE 26.x wants UTF-8 (CP932 names come out blank), AE 27.0 wants
//                   the Windows ANSI code page (UTF-8 names come out as mojibake) - measured with the "MBB Encoding
//                   Test" effect in both versions (LensBlur\aetest\logs\diag_names_26.log / _27.log). Mirrors
//                   Optic Bokeh's ob_strings.cpp (OB_HOST_STRING_ENCODING 3).
#ifndef SK_STR_ENC
#define SK_STR_ENC 3
#endif

namespace obs {

static const Entry kEntries[] = {
#include "sk_strings_table.inc"
};

const Entry* Table(int* count)
{
    if (count)
        *count = (int)(sizeof(kEntries) / sizeof(kEntries[0]));
    return kEntries;
}

std::wstring Utf8ToWide(const char* u8)
{
#ifdef _WIN32
    if (!u8 || !*u8)
        return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, u8, -1, nullptr, 0);
    if (n <= 0)
        return std::wstring();
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, u8, -1, &w[0], n);
    w.resize((size_t)n - 1);
    return w;
#else
    std::wstring w;
    for (const char* p = u8; p && *p; ++p)
        w.push_back((wchar_t)(unsigned char)*p);  // ASCII only off Windows (English is used there)
    return w;
#endif
}

bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out)
{
    out.clear();
#ifdef _WIN32
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
#elif defined(__APPLE__)
    // CP932 (Windows Shift_JIS) bytes, no lossy substitution, round trip checked (codePage 0 / 932 only)
    if (codePage != 0 && codePage != 932)
        return false;
    if (!u8 || !*u8)
        return u8 != nullptr;
    CFStringRef cs = CFStringCreateWithCString(kCFAllocatorDefault, u8, kCFStringEncodingUTF8);
    if (!cs)
        return false;
    const CFIndex len = CFStringGetLength(cs);
    CFIndex need = 0;
    const CFIndex conv = CFStringGetBytes(cs, CFRangeMake(0, len), kCFStringEncodingDOSJapanese, 0, false, nullptr, 0, &need);
    bool ok = conv == len && need > 0;
    std::string s;
    if (ok)
    {
        s.resize((size_t)need);
        CFStringGetBytes(cs, CFRangeMake(0, len), kCFStringEncodingDOSJapanese, 0, false, (UInt8*)&s[0], need, &need);
        CFStringRef back = CFStringCreateWithBytes(kCFAllocatorDefault, (const UInt8*)s.data(), (CFIndex)s.size(), kCFStringEncodingDOSJapanese, false);
        ok = back && CFStringCompare(back, cs, 0) == kCFCompareEqualTo;
        if (back)
            CFRelease(back);
    }
    CFRelease(cs);
    if (!ok)
        return false;
    out = s;
    return true;
#else
    (void)codePage;
    for (const char* p = u8; p && *p; ++p)
        if ((unsigned char)*p >= 0x80)
            return false;
    out = u8 ? u8 : "";
    return true;
#endif
}

bool FitsLimits(int kind, const std::string& b, const char* en, std::string* why)
{
    auto fail = [&](const char* m) {
        if (why)
            *why = std::string(m) + ": " + (en ? en : "");
        return false;
    };
    if (kind == OBS_NAME || kind == OBS_ITEM)
    {
        if (b.size() > (size_t)kMaxNameBytes)
            return fail("longer than 31 bytes");
        // a popup item must not contain a '|' byte (Shift_JIS trail byte 0x7C, e.g. in U+30DD);
        // param names are never split, so they may
        if (kind == OBS_ITEM && b.find('|') != std::string::npos)
            return fail("popup item contains a '|' byte");
    }
    else if (kind == OBS_POPUP)
    {
        size_t start = 0;
        for (;;)
        {
            const size_t bar = b.find('|', start);
            const size_t len = (bar == std::string::npos ? b.size() : bar) - start;
            if (len == 0 || len > (size_t)kMaxNameBytes)
                return fail("popup item empty or longer than 31 bytes");
            if (bar == std::string::npos)
                break;
            start = bar + 1;
        }
    }
    else if (kind == OBS_MSG)
    {
        if (b.size() > (size_t)kMaxMsgBytes)
            return fail("message longer than 200 bytes");
    }
    return true;
}

// the major version of the host executable (AfterFX.exe 26.5 -> 26), 0 if unknown
int HostMajorVersion()
{
#ifdef _WIN32
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
#elif defined(__APPLE__)
    // the host application's CFBundleShortVersionString ("26.5" -> 26), 0 if unknown
    static const int v = [] {
        CFBundleRef b = CFBundleGetMainBundle();
        if (!b)
            return 0;
        CFTypeRef s = CFBundleGetValueForInfoDictionaryKey(b, CFSTR("CFBundleShortVersionString"));
        if (!s || CFGetTypeID(s) != CFStringGetTypeID())
            return 0;
        char buf[64] = {0};
        if (!CFStringGetCString((CFStringRef)s, buf, sizeof(buf), kCFStringEncodingUTF8))
            return 0;
        return std::atoi(buf);
    }();
    return v;
#else
    return 0;
#endif
}

// a popup's Japanese bytes must split into as many items as the English (a Shift_JIS trail byte 0x7C, as in
// U+30DD, would add a split): checked in the host encoding actually used
bool SameItemCount(int kind, const std::string& b, const char* en, std::string* why)
{
    if (kind != OBS_POPUP || !en)
        return true;
    const size_t ne = std::count(en, en + std::strlen(en), '|'), nb = std::count(b.begin(), b.end(), '|');
    if (ne == nb)
        return true;
    if (why)
        *why = std::string("popup item count changes in the host encoding (a '|' trail byte?): ") + en;
    return false;
}

namespace {
std::atomic<int> g_forceMajor{-1};  // tests
std::atomic<int> g_enc{0};
std::once_flag g_once;
std::atomic<int> g_lang{LANG_EN};  // release after the maps are built, acquire in readers
std::string g_note;
std::unordered_map<std::string, std::string> g_host;  // en -> host bytes (Japanese)
std::unordered_map<std::string, std::string> g_u8;    // en -> UTF-8 (Japanese)
}

void SetHostMajorForTests(int major) { g_forceMajor.store(major); }
int CurrentEncoding() { return g_enc.load(); }

Lang Init(const char* langTag, unsigned codePage)
{
    std::call_once(g_once, [&] {
        int enc = SK_STR_ENC;
        if (enc == 3)
        {
            const int major = g_forceMajor.load() >= 0 ? g_forceMajor.load() : HostMajorVersion();
#if defined(__APPLE__)
            const int ov = MacEncodingOverride();
            enc = ov >= 0 ? ov : MacEncodingForMajor(major);  // mac_port: UNVERIFIED, see the top of this file
#else
            enc = major == 26 ? 1 : 0;
#endif
        }
        g_enc.store(enc);
        const bool ja = langTag && std::strncmp(langTag, "ja", 2) == 0;
        if (!ja || enc == 2)
            return;
        int n = 0;
        const Entry* t = Table(&n);
        std::unordered_map<std::string, std::string> host, u8;
        for (int i = 0; i < n; ++i)
        {
            std::string b;
            if (enc == 1)
                b = t[i].jaHost;  // UTF-8 as is
            else if (!Utf8ToCodePage(t[i].jaHost, codePage, b))
            {
                g_note = std::string("the Windows ANSI code page cannot represent: ") + t[i].en;
                return;  // English for everything
            }
            std::string why;
            if (!FitsLimits(t[i].kind, b, t[i].en, &why) || !SameItemCount(t[i].kind, b, t[i].en, &why))
            {
                g_note = why;
                return;
            }
            host[t[i].en] = b;
            u8[t[i].en] = t[i].ja;
        }
        g_host.swap(host);
        g_u8.swap(u8);
        g_lang.store(LANG_JA, std::memory_order_release);
    });
    return (Lang)g_lang.load(std::memory_order_acquire);
}

Lang Current()
{
    return (Lang)g_lang.load(std::memory_order_acquire);
}

const char* InitNote()
{
    return g_note.c_str();
}

const char* T(const char* en)
{
    if (Current() != LANG_JA || !en)
        return en;
    const auto it = g_host.find(en);
    return it == g_host.end() ? en : it->second.c_str();
}

std::wstring W(const char* en)
{
    if (Current() == LANG_JA && en)
    {
        const auto it = g_u8.find(en);
        if (it != g_u8.end())
            return Utf8ToWide(it->second.c_str());
    }
    return Utf8ToWide(en);
}

}  // namespace obs
