// Map Bilinear Blur: English / Japanese UI strings (see mbb_strings.h; from Optic Bokeh v2.3).
#include "mbb_strings.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "platform/plat.h"  // mac_port: the Windows-only pieces (conversions, host version) moved to plat_win.cpp

// Host string encoding for the Japanese names (build switch, MapBilinearBlur):
//   MBB_STR_ENC 0 = Windows ANSI code page (CP932; Optic Bokeh v2.3 way - real AE 2026 showed BLANK names)
//   MBB_STR_ENC 1 = UTF-8 bytes (default; limits checked in UTF-8 bytes)
//   MBB_STR_ENC 2 = English only
//   MBB_STR_ENC 3 = by host version (default): AE 26.x wants UTF-8 (CP932 names come out blank), AE 27.0 wants
//                   the Windows ANSI code page (UTF-8 names come out as mojibake) - measured with the "MBB Encoding
//                   Test" effect in both versions (LensBlur\aetest\logs\diag_names_26.log / _27.log). Mirrors
//                   Optic Bokeh's ob_strings.cpp (OB_HOST_STRING_ENCODING 3).
#ifndef MBB_STR_ENC
#define MBB_STR_ENC 3
#endif

namespace obs {

static const Entry kEntries[] = {
#include "mbb_strings_table.inc"
};

const Entry* Table(int* count)
{
    if (count)
        *count = (int)(sizeof(kEntries) / sizeof(kEntries[0]));
    return kEntries;
}

std::wstring Utf8ToWide(const char* u8) { return plat::Utf8ToWide(u8); }

bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out) { return plat::Utf8ToCodePage(u8, codePage, out); }

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

// the major version of the host (AfterFX.exe 26.5 -> 26), 0 if unknown (platform/plat_*.cpp)
int HostMajorVersion()
{
    static const int v = plat::HostMajorVersion();
    return v;
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
        int enc = MBB_STR_ENC;
        if (enc == 3)
        {
            const int major = g_forceMajor.load() >= 0 ? g_forceMajor.load() : HostMajorVersion();
            enc = plat::JapaneseEncodingForHost(major);  // Windows: 26 -> UTF-8, else code page (macOS: English until measured)
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
