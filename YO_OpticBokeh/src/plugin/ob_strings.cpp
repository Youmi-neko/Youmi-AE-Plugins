// Optic Bokeh v2.3: English / Japanese UI strings (see ob_strings.h).
#include "ob_strings.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
// mac port: the Windows code of v2.4.2 (MultiByteToWideChar / WideCharToMultiByte / GetFileVersionInfoW) moved
// unchanged to src/platform/ob_platform_win.cpp; src/platform/ob_platform_mac.cpp is the macOS counterpart.
#include "../platform/ob_platform.h"

namespace obs {

static const Entry kEntries[] = {
#include "ob_strings_table.inc"
};

const Entry* Table(int* count)
{
    if (count)
        *count = (int)(sizeof(kEntries) / sizeof(kEntries[0]));
    return kEntries;
}

std::wstring Utf8ToWide(const char* u8)
{
    return obp::utf8_to_wide(u8);
}

bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out)
{
    return obp::utf8_to_codepage(u8, codePage, out);
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

int EncodingForHost(int hostMajor)
{
#ifdef __APPLE__
    return hostMajor == 26 ? OB_MAC_ENCODING_AE26 : OB_MAC_ENCODING_OTHER;  // ob_strings.h (UNVERIFIED on macOS)
#else
    return hostMajor == 26 ? HOST_UTF8 : HOST_ANSI;
#endif
}

int HostMajorVersion()
{
    return obp::host_major_version();
}

bool ValidUtf8(const std::string& s)
{
    size_t i = 0;
    while (i < s.size())
    {
        const unsigned char c = (unsigned char)s[i];
        int n = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : -1;
        if (n < 0 || (n == 0 && c == 0))
            return false;
        for (int k = 1; k <= n; ++k)
            if (i + k >= s.size() || ((unsigned char)s[i + k] >> 6) != 2)
                return false;
        i += n + 1;
    }
    return true;
}

namespace {
std::atomic<int> g_forceEnc{-1};
std::atomic<int> g_enc{HOST_ANSI};
std::once_flag g_once;
std::atomic<int> g_lang{LANG_EN};  // release after the maps are built, acquire in readers
std::string g_note;
std::unordered_map<std::string, std::string> g_host;  // en -> host bytes (Japanese)
std::unordered_map<std::string, std::string> g_u8;    // en -> UTF-8 (Japanese)
}

void SetEncodingForTests(int encoding)
{
    g_forceEnc.store(encoding);
}

int CurrentEncoding()
{
    return g_enc.load();
}

Lang Init(const char* langTag, unsigned codePage, int encoding)
{
    std::call_once(g_once, [&] {
        int forced = g_forceEnc.load();
#ifdef __APPLE__
        // mac port, for the encoding test in AE on macOS (PORTING_NOTES.md "Strings"): OPTICBOKEH_STRING_ENCODING
        // = 0 (English only) / 1 (CP932) / 2 (UTF-8) overrides the by-version choice. Unset: no effect.
        // Remove (or keep as a support switch) once the right encoding per AE version on macOS is measured.
        if (forced < 0)
        {
            std::string ev;
            if (obp::getenv("OPTICBOKEH_STRING_ENCODING", ev) && ev.size() == 1 && ev[0] >= '0' && ev[0] <= '2')
                forced = ev[0] - '0';
        }
#endif
        int enc = forced >= 0 ? forced : encoding;
        if (enc == HOST_BY_VERSION)
            enc = EncodingForHost(HostMajorVersion());
        g_enc.store(enc);
        const bool ja = langTag && std::strncmp(langTag, "ja", 2) == 0;
        if (!ja)
            return;
        if (enc == HOST_ENGLISH_ONLY)
        {
            g_note = "Japanese host strings are switched off in this build (OB_HOST_STRING_ENCODING)";
            return;
        }
        int n = 0;
        const Entry* t = Table(&n);
        std::unordered_map<std::string, std::string> host, u8;
        for (int i = 0; i < n; ++i)
        {
            std::string b;
            if (enc == HOST_UTF8)
            {
                b = t[i].jaUtf8;  // the table is UTF-8 already (never cut: a too-long entry fails below)
                if (!ValidUtf8(b))
                {
                    g_note = std::string("invalid UTF-8 in the table: ") + t[i].en;
                    return;
                }
            }
            else if (!Utf8ToCodePage(t[i].jaHost, codePage, b))
            {
                g_note = std::string("the Windows ANSI code page cannot represent: ") + t[i].en;
                return;  // English for everything
            }
            std::string why;
            if (!FitsLimits(t[i].kind, b, t[i].en, &why))
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
