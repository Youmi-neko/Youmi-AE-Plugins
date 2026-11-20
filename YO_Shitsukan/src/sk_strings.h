// Shitsukan (copied from MapBilinearBlur): English / Japanese UI strings (copied from Optic Bokeh v2.3 ob_strings, namespace obs kept).
// The table (sk_strings_table.inc, generated from strings_ja.tsv by tools\gen_strings.py) maps each English string used in the
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
};
const Entry* Table(int* count);

enum Lang { LANG_EN = 0, LANG_JA };

// Decides the language once (later calls are ignored). langTag: AE's PF_AppGetLanguage ("ja_JP",
// "en_US", ...), nullptr = unknown (English). codePage: 0 = the process ANSI code page (CP_ACP).
// Thread-safe. Returns the language in effect.
Lang Init(const char* langTag, unsigned codePage = 0);
Lang Current();               // LANG_EN until Init chose Japanese
const char* InitNote();       // why English was chosen (empty if Japanese or not Japanese UI)

// Host string (ANSI code page bytes) for an English key: the Japanese text if Japanese is in
// effect and the key is in the table, else the key itself. Always a valid, NUL-terminated string.
const char* T(const char* en);
// UTF-16 for Drawbot labels (same language rule as T).
std::wstring W(const char* en);

// Helpers (also used by tools\check_strings): UTF-8 -> code page bytes, false if any character
// cannot be represented exactly (no best-fit, no default char).
bool Utf8ToCodePage(const char* u8, unsigned codePage, std::string& out);
std::wstring Utf8ToWide(const char* u8);

// SDK limits: param / group / checkbox names and each popup item <= 31 bytes; return_msg 255.
const int kMaxNameBytes = 31;
const int kMaxMsgBytes = 200;
// Checks one entry's converted Japanese text against the limits; *why explains a failure.
bool FitsLimits(int kind, const std::string& hostBytes, const char* en, std::string* why);
// a popup splits into the same number of items as its English text
bool SameItemCount(int kind, const std::string& hostBytes, const char* en, std::string* why);
// Map Bilinear Blur: host string encoding chosen in Init (0 = ANSI code page, 1 = UTF-8, 2 = English only)
int HostMajorVersion();
int CurrentEncoding();
void SetHostMajorForTests(int major);  // before Init: pretend the host is this AE major version

}  // namespace obs
