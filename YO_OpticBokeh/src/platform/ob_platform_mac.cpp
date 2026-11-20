// Optic Bokeh platform layer, macOS implementation (see ob_platform.h).
// WRITTEN ON WINDOWS, NEVER COMPILED. There is no separate self-test target. tests/run_offline_suite.py exercises
// check_strings (utf8_to_codepage, CP932 and 1252), host_probe ja ansi|utf8|off and kantan_probe ja ... (code pages,
// utf8_to_wide) and obgpu_harness selftest (exe_path). The process statistics (process_peak_commit_mb, process_cpu_ms,
// system_cpu_busy: getrusage / host_statistics) are used only by perf_bench and timing_probe, which the suite does NOT
// run: run them BY HAND once on the Mac and check the numbers are plausible (non-zero, not absurd): timing_probe
// (`./build/timing_probe 1920 1080 24 1 8 after 3`: its third number is the peak MB) and one perf_bench run on a
// synthetic raw input (CPU ms, CPU busy, peak MB) - PORTING_NOTES.md section 12.
// Links: -framework CoreFoundation.
#ifndef __APPLE__
#error "ob_platform_mac.cpp is the macOS implementation; build ob_platform_win.cpp on Windows"
#endif
#include "ob_platform.h"

#include <CoreFoundation/CoreFoundation.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace obp {

std::string exe_path()
{
    uint32_t n = 0;
    _NSGetExecutablePath(nullptr, &n);  // n = required size
    std::vector<char> buf(n + 1, 0);
    if (_NSGetExecutablePath(buf.data(), &n) != 0)
        return std::string();
    char real[PATH_MAX] = {0};
    if (realpath(buf.data(), real))
        return std::string(real);
    return std::string(buf.data());
}

// There is no process "ANSI" code page on macOS. The only legacy encoding the Japanese string table needs
// is CP932 (Shift_JIS with the Microsoft extensions). UNVERIFIED: which encoding AE on macOS expects.
unsigned ansi_code_page() { return 932; }

// After Effects' own bundle: CFBundleShortVersionString is e.g. "26.5" / "27.0" (the plug-in runs inside the
// AE process, so the main bundle is AE). Falls back to CFBundleVersion. 0 if neither parses.
int host_major_version()
{
    static const int v = [] {
        CFBundleRef b = CFBundleGetMainBundle();
        if (!b)
            return 0;
        const CFStringRef keys[2] = {CFSTR("CFBundleShortVersionString"), kCFBundleVersionKey};
        for (CFStringRef key : keys)
        {
            CFTypeRef t = CFBundleGetValueForInfoDictionaryKey(b, key);
            if (!t || CFGetTypeID(t) != CFStringGetTypeID())
                continue;
            char s[64] = {0};
            if (!CFStringGetCString((CFStringRef)t, s, sizeof(s), kCFStringEncodingUTF8))
                continue;
            const int major = std::atoi(s);
            if (major > 0)
                return major;
        }
        return 0;
    }();
    return v;
}

// Strict UTF-8 decoder (no overlongs, no surrogates, max U+10FFFF): invalid input -> empty, like
// MultiByteToWideChar(MB_ERR_INVALID_CHARS). wchar_t is 32-bit on macOS: one element per code point.
std::wstring utf8_to_wide(const char* u8)
{
    std::wstring w;
    if (!u8)
        return w;
    const unsigned char* p = (const unsigned char*)u8;
    while (*p)
    {
        unsigned c = *p++;
        int n = 0;
        unsigned minv = 0;
        if (c < 0x80)
            n = 0;
        else if ((c >> 5) == 6) { n = 1; c &= 0x1F; minv = 0x80; }
        else if ((c >> 4) == 14) { n = 2; c &= 0x0F; minv = 0x800; }
        else if ((c >> 3) == 30) { n = 3; c &= 0x07; minv = 0x10000; }
        else
            return std::wstring();
        for (int k = 0; k < n; ++k)
        {
            if ((*p >> 6) != 2)
                return std::wstring();
            c = (c << 6) | (*p++ & 0x3F);
        }
        if (n && c < minv)
            return std::wstring();
        if (c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF))
            return std::wstring();
        w.push_back((wchar_t)c);
    }
    return w;
}

static CFStringEncoding cf_encoding(unsigned codePage)
{
    switch (codePage)
    {
    case 932: return kCFStringEncodingDOSJapanese;  // CP932
    case 65001: return kCFStringEncodingUTF8;
    case 1252: return kCFStringEncodingWindowsLatin1;
    default: return CFStringConvertWindowsCodepageToEncoding(codePage);
    }
}

bool utf8_to_codepage(const char* u8, unsigned codePage, std::string& out)
{
    out.clear();
    if (!u8)
        return false;
    if (!*u8)
        return true;
    if (utf8_to_wide(u8).empty())
        return false;  // invalid UTF-8
    const unsigned cp = codePage ? codePage : ansi_code_page();
    const CFStringEncoding enc = cf_encoding(cp);
    if (enc == kCFStringEncodingInvalidId || !CFStringIsEncodingAvailable(enc))
        return false;
    CFStringRef s = CFStringCreateWithCString(kCFAllocatorDefault, u8, kCFStringEncodingUTF8);
    if (!s)
        return false;
    const CFIndex len = CFStringGetLength(s);
    CFIndex need = 0;
    // lossByte 0: an unmappable character stops the conversion (converted < len) instead of a substitute
    CFIndex conv = CFStringGetBytes(s, CFRangeMake(0, len), enc, 0, false, nullptr, 0, &need);
    bool ok = conv == len && need > 0;
    std::string bytes;
    if (ok)
    {
        bytes.resize((size_t)need);
        conv = CFStringGetBytes(s, CFRangeMake(0, len), enc, 0, false, (UInt8*)&bytes[0], need, &need);
        ok = conv == len;
    }
    if (ok)
    {
        // round trip must give the same Unicode string (as the Windows implementation checks)
        CFStringRef back = CFStringCreateWithBytes(kCFAllocatorDefault, (const UInt8*)bytes.data(), (CFIndex)bytes.size(), enc, false);
        ok = back && CFStringCompare(back, s, 0) == kCFCompareEqualTo;
        if (back)
            CFRelease(back);
    }
    CFRelease(s);
    if (!ok)
        return false;
    out = bytes;
    return true;
}

static double rusage_maxrss_mb()
{
    struct rusage ru;
    std::memset(&ru, 0, sizeof(ru));
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1048576.0;  // macOS: bytes (Linux would be KiB)
}

double process_peak_commit_mb() { return rusage_maxrss_mb(); }
double process_peak_working_set_mb() { return rusage_maxrss_mb(); }

double process_cpu_ms()
{
    struct rusage ru;
    std::memset(&ru, 0, sizeof(ru));
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

static bool cpu_ticks(unsigned long long& busy, unsigned long long& total)
{
    host_cpu_load_info_data_t info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&info, &count) != KERN_SUCCESS)
        return false;
    const unsigned long long u = info.cpu_ticks[CPU_STATE_USER], s = info.cpu_ticks[CPU_STATE_SYSTEM],
                             n = info.cpu_ticks[CPU_STATE_NICE], i = info.cpu_ticks[CPU_STATE_IDLE];
    busy = u + s + n;
    total = busy + i;
    return true;
}

double system_cpu_busy(int ms)
{
    unsigned long long b0 = 0, t0 = 0, b1 = 0, t1 = 0;
    const bool ok0 = cpu_ticks(b0, t0);
    sleep_ms((unsigned)(ms > 0 ? ms : 0));
    if (!ok0 || !cpu_ticks(b1, t1) || t1 <= t0)
        return 0.0;
    return (double)(b1 - b0) / (double)(t1 - t0);
}

}  // namespace obp
