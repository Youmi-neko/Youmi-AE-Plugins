// Tebure (copied from Map Bilinear Blur): English / Japanese UI strings (see tb_strings.h; from Optic Bokeh v2.3).
#include "tb_strings.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "platform/plat.h"  // Utf8ToWide / Utf8ToCodePage / HostMajorVersion (Windows: the 0.3.1.1 code, moved)

// Host string encoding for the Japanese names (build switch, Tebure):
//   TB_STR_ENC 0 = Windows ANSI code page (CP932; Optic Bokeh v2.3 way - real AE 2026 showed BLANK names)
//   TB_STR_ENC 1 = UTF-8 bytes (default; limits checked in UTF-8 bytes)
//   TB_STR_ENC 2 = English only
//   TB_STR_ENC 3 = by host version (default): AE 26.x wants UTF-8 (CP932 names come out blank), AE 27.0 wants
//                   the Windows ANSI code page (UTF-8 names come out as mojibake) - measured with the "MBB Encoding
//                   Test" effect in both versions (LensBlur\aetest\logs\diag_names_26.log / _27.log). Mirrors
//                   Optic Bokeh's ob_strings.cpp (OB_HOST_STRING_ENCODING 3).
#ifndef TB_STR_ENC
#define TB_STR_ENC 3
#endif

#if defined(__APPLE__)
// mac_port POLICY (mac_common/POLICY_macos.md, the same in all six packages): which encoding Mac After Effects wants
// for Japanese names is UNVERIFIED, so the Mac build shows ENGLISH names (encoding 2) until it has been measured in
// Mac AE 2026 AND AE 2027 (PORTING_NOTES "Strings"). Then put the measured result into the two macros (0 = CP932,
// 1 = UTF-8, 2 = English): TB_MAC_ENC_AE26 for AE 26.x, TB_MAC_ENC_OTHER for every other version. For the
// measurement only, the environment variable TEBURE_UI_ENC (utf8 | sjis | en, set when AE starts) overrides both.
// Windows is not affected: its rule is the #else branch below, unchanged.
#include <cstdlib>
#ifndef TB_MAC_ENC_AE26
#define TB_MAC_ENC_AE26 2
#endif
#ifndef TB_MAC_ENC_OTHER
#define TB_MAC_ENC_OTHER 2
#endif
namespace {
int MacEncodingForMajor(int major)
{
    const char* v = std::getenv("TEBURE_UI_ENC");
    if (v && !std::strcmp(v, "utf8"))
        return 1;
    if (v && !std::strcmp(v, "sjis"))
        return 0;
    if (v && !std::strcmp(v, "en"))
        return 2;
    return major == 26 ? TB_MAC_ENC_AE26 : TB_MAC_ENC_OTHER;
}
}  // namespace
#endif

namespace obs {

static const Entry kEntries[] = {
#include "tb_strings_table.inc"
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

// the major version of the host executable (AfterFX.exe 26.5 -> 26; macOS: the app bundle version), 0 if unknown
int HostMajorVersion() { return plat::HostMajorVersion(); }

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
        int enc = TB_STR_ENC;
        if (enc == 3)
        {
            const int major = g_forceMajor.load() >= 0 ? g_forceMajor.load() : HostMajorVersion();
#if defined(__APPLE__)
            enc = MacEncodingForMajor(major);  // mac_port: English until measured (see above)
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
