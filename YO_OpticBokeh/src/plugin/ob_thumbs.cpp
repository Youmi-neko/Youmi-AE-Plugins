// Optic Bokeh v2.3: engine-rendered preset thumbnails (see ob_thumbs.h).
#include "ob_thumbs.h"

#include "lenscore.h"
#include "lens_presets.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cwchar>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>

namespace obthumb {

int Count(Kind k)
{
    return k == LENS ? lenscore::kLensPresetCount : k == STYLE ? lenscore::kAberrationStyleCount : lenscore::kGrainStockCount;
}

const char* NameEn(Kind k, int i)
{
    if (i < 0 || i >= Count(k))
        return "";
    return k == LENS ? lenscore::kLensPresets[i].ui : k == STYLE ? lenscore::kAberrationStyles[i].ui : lenscore::kGrainStocks[i].ui;
}

namespace {

inline unsigned char Enc(float v)
{
    // display encoding of a linear value (gamma 2.2, as the plug-in's Work in Linear Light)
    if (!(v > 0.f))
        return 0;
    const float e = std::pow(std::min(1.f, v), 1.f / 2.2f);
    return (unsigned char)std::lround(e * 255.f);
}

void ToBGRA(const std::vector<float>& rgba, int w, int h, Image& out)
{
    out.w = w;
    out.h = h;
    out.bgra.assign((size_t)w * h * 4, 255);
    for (size_t i = 0; i < (size_t)w * h; ++i)
    {
        const float a = std::max(1e-6f, rgba[i * 4 + 3]);
        out.bgra[i * 4 + 0] = Enc(rgba[i * 4 + 2] / a);
        out.bgra[i * 4 + 1] = Enc(rgba[i * 4 + 1] / a);
        out.bgra[i * 4 + 2] = Enc(rgba[i * 4 + 0] / a);
        out.bgra[i * 4 + 3] = 255;
    }
}

// area coverage of the disc (cx, cy, r) over pixel (x, y), 4x4 supersampled
float DiscCover(float cx, float cy, float r, int x, int y, float k)
{
    int n = 0;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
        {
            const float vx = (x + (i + 0.5f) / 4.f) / k, vy = (y + (j + 0.5f) / 4.f) / k;
            n += (vx - cx) * (vx - cx) + (vy - cy) * (vy - cy) <= r * r;
        }
    return n / 16.f;
}

const float kLensWhite = 95.f;

bool RenderLens(int i, int scale, Image& out)
{
    const int W = kThumbW * scale, H = kThumbH * scale;
    std::vector<float> src((size_t)W * H * 4), dst((size_t)W * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            float* p = &src[((size_t)y * W + x) * 4];
            const float g = 0.010f + 0.012f * (float)y / H;
            p[0] = g * 0.8f;
            p[1] = g * 0.9f;
            p[2] = g * 1.3f;
            p[3] = 1.f;
        }
    // a bright white light and a dimmer warm one, off-axis (cat eye / swirl show), each 2x2 logical px
    auto light = [&](float fx, float fy, float r, float g, float b) {
        const int x0 = (int)(fx * W), y0 = (int)(fy * H), s = std::max(1, scale * 2);
        for (int y = y0; y < y0 + s && y < H; ++y)
            for (int x = x0; x < x0 + s && x < W; ++x)
            {
                float* p = &src[((size_t)y * W + x) * 4];
                p[0] = r;
                p[1] = g;
                p[2] = b;
            }
    };
    // v2.3 polish: big discs (radius 0.31 H: ~20 logical px) so the rim, onion rings, soft edges and
    // the off-axis squeeze read at thumbnail size. Brightness: a 2x2 px light of value v spread over a
    // disc of ~1300 px (1x) gives roughly v / 130 at the centre: the white disc sits below clipping so
    // its profile (rim, rings, soft edge, bright core) shows.
    light(0.23f, 0.31f, kLensWhite, kLensWhite, kLensWhite);
    light(0.75f, 0.66f, kLensWhite * 0.55f, kLensWhite * 0.28f, kLensWhite * 0.10f);
    lenscore::Params p;
    if (!lenscore::applyLensPreset(i, p))
        return false;
    lenscore::applyQuality(lenscore::kQualityStandard, p);
    p.maxRadius = 0.31f * H;
    p.highlightBoost = 0.f;    // the kernel only: shape, rim, rings, cat eye, fringe (no highlight analysis)
    p.highlightGain = 0.f;
    p.highlightRolloff = 0.5f;
    p.threads = 2;
    // lens frame centred on the thumbnail but smaller, so both lights are clearly off-axis
    p.frameWidth = 0.8f * W;
    p.frameHeight = 0.8f * H;
    p.frameLeft = 0.1f * W;
    p.frameTop = 0.1f * H;
    if (lenscore::render(src.data(), W, H, W * 4, nullptr, 0, dst.data(), W * 4, p) != lenscore::Status::Ok)
        return false;
    ToBGRA(dst, W, H, out);
    return true;
}

bool RenderStyle(int i, int scale, Image& out)
{
    const int W = kThumbW * scale, H = kThumbH * scale;
    // a crop of a virtual 1920 x 1080 frame, 2.5 thumbnail px per frame px at 1x: the aberration
    // widths (fractions of the frame diagonal) look as they do at 1080p, magnified
    const float k = (float)W / 160.f;  // buffer px per virtual px
    std::vector<float> src((size_t)W * H * 4), dst((size_t)W * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            float* p = &src[((size_t)y * W + x) * 4];
            const float skin = DiscCover(62.f, 64.f, 44.f, x, y, k);
            const float catch_ = DiscCover(80.f, 46.f, 4.f, x, y, k);
            float r = 0.010f, g = 0.012f, b = 0.024f;  // dark navy background
            r += skin * (0.86f - r);
            g += skin * (0.52f - g);
            b += skin * (0.36f - b);
            // a 2 px dark line (line art) across the bright face
            const float vy = (y + 0.5f) / k, vx = (x + 0.5f) / k;
            if (std::fabs(vy - 70.f) < 1.2f && vx > 34.f && vx < 92.f)
                r = 0.03f, g = 0.02f, b = 0.02f;
            r += catch_ * (1.f - r);
            g += catch_ * (1.f - g);
            b += catch_ * (1.f - b);
            p[0] = r;
            p[1] = g;
            p[2] = b;
            p[3] = 1.f;
        }
    lenscore::Params p;
    if (!lenscore::applyAberrationStyle(i, p))
        return false;
    p.maxRadius = 0.f;
    p.lumAberration = 1.2f;
    p.highlightBoost = 0.f;
    p.highlightRolloff = 0.f;
    p.threads = 2;
    p.frameWidth = 1920.f * k;
    p.frameHeight = 1080.f * k;
    p.frameLeft = -880.f * k;
    p.frameTop = -480.f * k;
    if (lenscore::render(src.data(), W, H, W * 4, nullptr, 0, dst.data(), W * 4, p) != lenscore::Status::Ok)
        return false;
    ToBGRA(dst, W, H, out);
    return true;
}

const float kGrainPreviewGain = 1.45f;

bool RenderStock(int i, int scale, Image& out)
{
    const int W = kThumbW * scale, H = kThumbH * scale;
    std::vector<float> src((size_t)W * H * 4), dst((size_t)W * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            float* p = &src[((size_t)y * W + x) * 4];
            // v2.3 polish: mid-grey with a dark band (top) and a bright band (bottom), so the grain's
            // size, strength and its shadow / highlight response all show (display 0.18 / 0.50 / 0.82)
            const float fy = (float)y / H;
            const float e = fy < 0.28f ? 0.18f : fy < 0.72f ? 0.50f : 0.82f;
            const float v = std::pow(e, 2.2f);
            p[0] = p[1] = p[2] = v;
            p[3] = 1.f;
        }    lenscore::Params p;
    if (!lenscore::applyGrainStock(i, p))
        return false;
    p.maxRadius = 0.f;
    p.highlightBoost = 0.f;
    p.highlightRolloff = 0.f;
    // exaggerated for the preview, the same for every stock (the picker says so): x1.45 keeps the
    // strongest stock (1.35) inside the slider range, so the order and ratios are the real ones
    p.grain = p.grain * kGrainPreviewGain;
    p.grainScaleX = p.grainScaleY = 2.f * scale;  // magnified 2x: one full-resolution grain px = 2 logical px
    p.grainSeed = 7;
    p.threads = 2;
    if (lenscore::render(src.data(), W, H, W * 4, nullptr, 0, dst.data(), W * 4, p) != lenscore::Status::Ok)
        return false;
    ToBGRA(dst, W, H, out);
    return true;
}

std::mutex g_m;
std::map<int, std::unique_ptr<Image>> g_cache;
inline int Key(Kind k, int i, int scale)
{
    return ((int)k * 1000 + i) * 8 + std::min(7, std::max(1, scale));
}

}  // namespace

bool Render(Kind k, int i, int scale, Image& out)
{
    scale = std::min(4, std::max(1, scale));
    try
    {
        return k == LENS ? RenderLens(i, scale, out) : k == STYLE ? RenderStyle(i, scale, out) : RenderStock(i, scale, out);
    }
    catch (...)
    {
        return false;
    }
}

const Image* Get(Kind k, int i, int scale, bool renderIfMissing)
{
    if (i < 0 || i >= Count(k))
        return nullptr;
    const int key = Key(k, i, scale);
    {
        std::lock_guard<std::mutex> lk(g_m);
        const auto it = g_cache.find(key);
        if (it != g_cache.end())
            return it->second.get();
    }
    if (!renderIfMissing)
        return nullptr;
    std::unique_ptr<Image> img(new Image());
    if (!Render(k, i, scale, *img))
        return nullptr;
    std::lock_guard<std::mutex> lk(g_m);
    auto& slot = g_cache[key];
    if (!slot)
        slot = std::move(img);  // a concurrent render of the same key: keep the first
    return slot.get();          // entries are never removed: the pointer stays valid
}

namespace {
// last measured render time per (kind, scale), ms; < 0 = not measured yet
std::atomic<float> g_lastMs[kKinds][5] = {};
std::once_flag g_lastInit;
}  // namespace

bool Warm(Kind k, int scale, double budgetMs)
{
    std::call_once(g_lastInit, [] {
        for (auto& row : g_lastMs)
            for (auto& v : row)
                v.store(-1.f);
    });
    const int sc = std::min(4, std::max(1, scale));
    const auto t0 = std::chrono::steady_clock::now();
    bool all = true;
    for (int i = 0; i < Count(k); ++i)
    {
        if (Get(k, i, scale, false))
            continue;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        // never start a render that will not fit the budget (unknown cost: only with a large budget,
        // i.e. on the background worker): the UI thread must not stall on a big lens thumbnail
        const float est = g_lastMs[k][sc].load();
        if (ms >= budgetMs || (est < 0.f ? budgetMs < 1000.0 : ms + est > budgetMs))
        {
            all = false;
            continue;
        }
        const auto r0 = std::chrono::steady_clock::now();
        if (!Get(k, i, scale, true))
            all = false;
        // keep the slowest seen (a concurrent render of the same key returns early and would look cheap)
        const float took = (float)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
        g_lastMs[k][sc].store(std::max(est, took));
    }
    return all;
}
Grid Layout(int n, float w, float h)
{
    Grid g;
    n = std::max(1, n);
    const int fit = std::max(1, (int)std::floor((w + kGap) / (float)kCellMinW));
    const int cols = std::min(n, fit);
    const int rows = (n + cols - 1) / cols;
    if (rows * kCellH <= h + 0.5f && cols * kCellMinW - kGap <= w + 0.5f)
    {
        g.cols = cols;
        g.rows = rows;
        g.s = 1.f;
    }
    else
    {
        // shrink: the column count that gives the biggest cells and still fits both ways
        float best = 0.f;
        for (int c = 1; c <= n; ++c)
        {
            const int r = (n + c - 1) / c;
            const float s = std::min(w / (float)(c * kCellMinW - kGap), h / (float)(r * kCellH));
            if (s > best + 1e-4f)
            {
                best = s;
                g.cols = c;
                g.rows = r;
            }
        }
        g.s = std::max(0.25f, std::min(1.f, best));
    }
    // the cells share the width (labels get the room), up to 1.6x the minimum
    const float pitch = std::min((w + kGap * g.s) / g.cols, 1.6f * kCellMinW * g.s);
    g.cw = std::max((float)kCellMinW, pitch / g.s);
    const float gw = g.cols * g.cw * g.s - kGap * g.s;
    g.x0 = std::max(0.f, (w - gw) * 0.5f);
    g.y0 = kGap * 0.5f * g.s;
    return g;
}

namespace {
// cell content box (thumbnail + label, without the gap)
void CellBox(const Grid& g, int i, float& x, float& y, float& w, float& h)
{
    const int c = i % std::max(1, g.cols), r = i / std::max(1, g.cols);
    x = g.x0 + c * g.cw * g.s;
    y = g.y0 + r * kCellH * g.s;
    w = (g.cw - kGap) * g.s;
    h = (kThumbH + kLabelH) * g.s;
}
}  // namespace

void CellRect(const Grid& g, int i, float& x, float& y, float& tw, float& th)
{
    float bw, bh;
    CellBox(g, i, x, y, bw, bh);
    tw = kThumbW * g.s;
    th = kThumbH * g.s;
    x += std::floor((bw - tw) * 0.5f);
}

void LabelRect(const Grid& g, int i, float& x, float& y, float& lw, float& lh)
{
    float bw, bh;
    CellBox(g, i, x, y, bw, bh);
    y += kThumbH * g.s;
    lw = bw;
    lh = kLabelH * g.s;
}

int HitTest(const Grid& g, int n, float x, float y)
{
    for (int i = 0; i < n; ++i)
    {
        float cx, cy, bw, bh;
        CellBox(g, i, cx, cy, bw, bh);
        if (x >= cx && x < cx + bw && y >= cy && y < cy + bh)
            return i;
    }
    return -1;
}

int PreferredHeight(Kind k, int nominalWidth)
{
    const int n = Count(k);
    const int cols = std::max(1, std::min(n, (nominalWidth + kGap) / kCellMinW));
    const int rows = (n + cols - 1) / cols;
    return rows * kCellH + kGap;
}

// ---- label wrapping ----------------------------------------------------------------------------

namespace {
float CharEm(wchar_t c)
{
    if (c == L' ')
        return 0.30f;
    if (c >= 0xFF61 && c <= 0xFF9F)
        return 0.55f;  // half-width katakana
    if (c >= 0x3041 && c <= 0x30FA)
        return 0.92f;  // kana: a little narrower than 1 em in the UI fonts
    if (c >= 0x2E80)
        return 1.0f;   // CJK, full-width forms
    if (c >= L'A' && c <= L'Z')
        return 0.68f;
    if (c >= L'0' && c <= L'9')
        return 0.58f;
    if (wcschr(L"iljtfI.,:;'!|()[]/-", c))
        return 0.34f;
    if (c >= L'a' && c <= L'z')
        return c == L'm' || c == L'w' ? 0.86f : 0.56f;
    return 0.62f;
}
bool NoLineStart(wchar_t c)
{
    return wcschr(L"\x3001\x3002\xFF09\x300D\x30FC\x30FB)]%,.", c) != nullptr;  // 、。）」ー・ ) ] % , .
}
// script class for the second-choice breaks (between kanji and kana, hiragana and katakana)
int Script(wchar_t c)
{
    if (c >= 0x3041 && c <= 0x309F)
        return 1;  // hiragana
    if ((c >= 0x30A1 && c <= 0x30FA) || c == 0x30FC)
        return 2;  // katakana (with the long-vowel mark)
    if (c >= 0x4E00 && c <= 0x9FFF)
        return 3;  // kanji
    return 0;
}
std::wstring Trim(const std::wstring& s)
{
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == L' ' || s[a] == 0x30FB))
        ++a;
    while (b > a && s[b - 1] == L' ')
        --b;
    return s.substr(a, b - a);
}
}  // namespace

float EstimateWidth(const std::wstring& s, float fontPx)
{
    float em = 0.f;
    for (wchar_t c : s)
        em += CharEm(c);
    return em * fontPx;
}

std::vector<std::wstring> WrapLabel(const std::wstring& s0, float maxWidth, float fontPx)
{
    const std::wstring s = Trim(s0);
    std::vector<std::wstring> out;
    if (s.empty())
        return out;
    if (EstimateWidth(s, fontPx) <= maxWidth || s.size() < 2)
    {
        out.push_back(s);
        return out;
    }
    // candidate breaks, in tiers: (1) after a nakaguro (dropped when it would not fit), at a space, or
    // before an opening bracket; (2) where the script changes (kanji / hiragana / katakana). In the
    // first tier where both lines fit, the most balanced split wins; failing that, a split whose first
    // line fits (tier 1 before tier 2) and the second line is elided.
    std::wstring partA, partB;
    for (int tier = 1; tier <= 2; ++tier)
    {
        bool found = false, bestBothFit = false;
        float bestMax = 1e30f;
        std::wstring bestA, bestB;
        for (size_t k = 1; k < s.size(); ++k)
        {
            std::wstring a, b;
            if (tier == 1 && s[k] == L' ')
            {
                a = s.substr(0, k);
                b = s.substr(k + 1);
            }
            else if (tier == 1 && s[k - 1] == 0x30FB)
            {
                a = s.substr(0, k);
                if (EstimateWidth(a, fontPx) > maxWidth)
                    a = s.substr(0, k - 1);  // the separator is not needed at a line end
                b = s.substr(k);
            }
            else if (tier == 1 && (s[k] == 0xFF08 || s[k] == L'(') && s[k - 1] != L' ')
            {
                a = s.substr(0, k);
                b = s.substr(k);
            }
            else if (tier == 2 && Script(s[k - 1]) && Script(s[k]) && Script(s[k - 1]) != Script(s[k]) &&
                     !NoLineStart(s[k]))
            {
                a = s.substr(0, k);
                b = s.substr(k);
            }
            else
                continue;
            a = Trim(a);
            b = Trim(b);
            if (a.empty() || b.empty())
                continue;
            const float wa = EstimateWidth(a, fontPx), wb = EstimateWidth(b, fontPx);
            if (wa > maxWidth)
                continue;
            const bool both = wb <= maxWidth;
            const float m = std::max(wa, wb);
            if (!found || (both && !bestBothFit) || (both == bestBothFit && m < bestMax))
            {
                found = true;
                bestBothFit = both;
                bestMax = m;
                bestA = a;
                bestB = b;
            }
        }
        if (found && bestBothFit)
        {
            out.push_back(bestA);
            out.push_back(bestB);
            return out;
        }
        if (found && partA.empty())
        {
            partA = bestA;
            partB = bestB;
        }
    }
    if (!partA.empty())
    {
        out.push_back(partA);
        out.push_back(partB);
        return out;
    }    // 2) by width: the longest first line that fits, not starting line 2 with closing punctuation
    size_t k = 1;
    float acc = CharEm(s[0]) * fontPx;
    while (k < s.size() && acc + CharEm(s[k]) * fontPx <= maxWidth)
        acc += CharEm(s[k++]) * fontPx;
    while (k > 1 && k < s.size() && NoLineStart(s[k]))
        --k;
    out.push_back(Trim(s.substr(0, k)));
    const std::wstring rest = Trim(s.substr(k));
    if (!rest.empty())
        out.push_back(rest);  // still too long: the renderer ends it with an ellipsis
    return out;
}
}  // namespace obthumb
