// Optic Bokeh v2.3: English / Japanese UI strings.
// The table (ob_strings_table.inc, generated from strings_ja.tsv) maps each English string used in the
// code to its Japanese text (UTF-8). The language is chosen once per process (Init) from AE's UI
// language; Japanese is used only if EVERY Japanese string converts losslessly to the host's ANSI
// code page and fits the SDK limits - otherwise everything stays English (never half-translated).
#pragma once
#include <string>

namespace obs {

enum Kind { OBS_NAME = 0, OBS_POPUP, OBS_ITEM, OBS_LABEL, OBS_MSG };
struct Entry
{
    int kind;
    const char* en;
    const char* ja;      // UTF-8: labels (Drawbot, UTF-16) and, unless jaHost differs, host strings
    const char* jaHost;  // UTF-8: the host (ANSI) string - differs only where a Shift_JIS trail byte
                         // 0x7C ('|') would split a popup item
    const char* jaUtf8;  // UTF-8: the host string in UTF-8 mode - differs only where `ja` is longer than
                         // the 31-byte limit in UTF-8 (a shortened name)
};
const Entry* Table(int* count);

enum Lang { LANG_EN = 0, LANG_JA };

// v2.3: how Japanese reaches the host's char strings (param / group / popup names, return_msg).
// Which one AE expects is being verified in AE; the build picks one with OB_HOST_STRING_ENCODING.
//   HOST_ENGLISH_ONLY: never Japanese host strings (picker labels stay English too: one language)
//   HOST_ANSI        : the Windows ANSI code page (CP932 on Japanese Windows), as v2.3 first shipped
//   HOST_UTF8        : UTF-8 bytes (shortened names where the UTF-8 form exceeds 31 bytes)
enum HostEncoding { HOST_ENGLISH_ONLY = 0, HOST_ANSI = 1, HOST_UTF8 = 2, HOST_BY_VERSION = 3 };
#ifdef __APPLE__
// mac_port POLICY (mac_common/POLICY_macos.md, the same in all six packages): the macOS mapping of HOST_BY_VERSION is
// ENGLISH until the encoding has been measured in Mac AE 2026 AND AE 2027 (PORTING_NOTES.md section 9,
// tests/aetest_mac/strings_test.sh). After measuring, set ONLY these two macros (here or with -D):
// EncodingForHost (ob_strings.cpp) and the expectations of host_probe.cpp / kantan_probe.cpp ("host string encoding
// by AE version") both use them. For the measurement, OPTICBOKEH_STRING_ENCODING (0 / 1 / 2) overrides them.
// Windows is not affected (its rule is in EncodingForHost's #else branch).
#ifndef OB_MAC_ENCODING_AE26
#define OB_MAC_ENCODING_AE26 HOST_ENGLISH_ONLY   // AE 26.x (measured candidate on Windows: HOST_UTF8)
#endif
#ifndef OB_MAC_ENCODING_OTHER
#define OB_MAC_ENCODING_OTHER HOST_ENGLISH_ONLY  // AE 27 and every other / unknown version (Windows: HOST_ANSI = CP932)
#endif
#endif
// HOST_BY_VERSION (v2.3 release): measured in AE with ExtendScript .name (aetest\diag_names.jsx):
// AE 26.x decodes effect param names as UTF-8 (ANSI bytes are dropped -> blank names), AE 27.0 decodes
// them in the Windows ANSI code page (UTF-8 bytes show as mojibake). So: UTF-8 for AE 26, the ANSI
// code page for every other version (27 and later, and the older versions, which predate UTF-8).
int HostMajorVersion();                // major version of the host executable (26, 27, ...), 0 = unknown
int EncodingForHost(int hostMajor);    // HOST_UTF8 for 26, HOST_ANSI otherwise

// Decides the language once (later calls are ignored). langTag: AE's PF_AppGetLanguage ("ja_JP",
// "en_US", ...), nullptr = unknown (English). codePage: 0 = the process ANSI code page (CP_ACP).
// Thread-safe. Returns the language in effect.
Lang Init(const char* langTag, unsigned codePage = 0, int encoding = HOST_ANSI);
// Tests only: overrides the `encoding` argument of the first Init (-1 = no override).
void SetEncodingForTests(int encoding);
int CurrentEncoding();        // the encoding in effect (HOST_*)
Lang Current();               // LANG_EN until Init chose Japanese
const char* InitNote();       // why English was chosen (empty if Japanese or not Japanese UI)

// Host string (ANSI code page bytes) for an English key: the Japanese text if Japanese is in
// effect and the key is in the table, else the key itself. Always a valid, NUL-terminated string.
const char* T(const char* en);
// UTF-16 for Drawbot labels (same language rule as T).
std::wstring W(const char* en);

// Helpers (also used by tools\check_strings): UTF-8 -> code page bytes, false if any character
// cannot be represented exactly (no best-fit, no default char). ValidUtf8: well-formed UTF-8.
bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out);
std::wstring Utf8ToWide(const char* u8);
bool ValidUtf8(const std::string& s);

// SDK limits: param / group / checkbox names and each popup item <= 31 bytes; return_msg 255.
const int kMaxNameBytes = 31;
const int kMaxMsgBytes = 200;
// Checks one entry's converted Japanese text against the limits; *why explains a failure.
bool FitsLimits(int kind, const std::string& hostBytes, const char* en, std::string* why);

}  // namespace obs
