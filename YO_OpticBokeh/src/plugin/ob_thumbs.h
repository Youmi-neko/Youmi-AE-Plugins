// Optic Bokeh v2.3: preset thumbnails for the Effect Controls pickers (Lens Preset, Aberration
// Style, Grain Stock). The images are rendered by the engine itself (lenscore) with exactly the
// values the popup writes, so they cannot misrepresent a preset. Shared by the plug-in (Drawbot)
// and tools\picker_sheet.cpp (offline PNGs).
//
// Threading: Get() renders lazily and caches per (kind, index, scale) behind a mutex; the cache is
// never touched by SMART_RENDER (only by UI events and the offline tool).
#pragma once
#include <string>
#include <vector>

namespace obthumb {

enum Kind { LENS = 0, STYLE, STOCK };
const int kKinds = 3;

// logical (1x) sizes, px. A cell = thumbnail + a two-line label below it; cells may be wider than the
// thumbnail (they share the panel width) so labels get room.
const int kThumbW = 88, kThumbH = 66, kLabelLineH = 14, kLabelH = 2 * kLabelLineH + 2, kGap = 6;
const int kCellMinW = kThumbW + kGap, kCellH = kThumbH + kLabelH + kGap;
const float kLabelFont = 11.f;  // logical px (scaled with the cell)

int Count(Kind k);                 // entries (popup items 2..), "Custom" excluded
const char* NameEn(Kind k, int i); // English UI name (the popup item text, translation key)

// BGRA 8-bit, opaque, w = kThumbW * scale, h = kThumbH * scale. scale: 1 or 2 (device px / logical px).
struct Image
{
    int w = 0, h = 0;
    std::vector<unsigned char> bgra;
};
bool Render(Kind k, int i, int scale, Image& out);

// Cached image or nullptr. renderIfMissing = false: only what is already there (no work).
const Image* Get(Kind k, int i, int scale, bool renderIfMissing);
// Renders missing thumbnails of `k` until `budgetMs` is spent; true when all are ready.
bool Warm(Kind k, int scale, double budgetMs);

// Grid of n cells in a w x h (logical) area. The cells keep their size (s = 1) while they fit;
// a narrow area shrinks them uniformly (s < 1) so the grid always fits; a wide one centres them.
struct Grid
{
    int cols = 1, rows = 1;
    float s = 1.f;        // cell scale (thumbnail, label font)
    float cw = kCellMinW; // cell width before scaling (>= kCellMinW: wide panels give labels room)
    float x0 = 0, y0 = 0; // top-left of the first cell (logical, relative to the area)
};
Grid Layout(int n, float w, float h);
// thumbnail rect of item i (centred in its cell)
void CellRect(const Grid& g, int i, float& x, float& y, float& tw, float& th);
// label area of item i: full cell width, below the thumbnail, two lines
void LabelRect(const Grid& g, int i, float& x, float& y, float& lw, float& lh);
int HitTest(const Grid& g, int n, float x, float y);  // -1 = none (same geometry as CellRect / LabelRect)

// Splits a label into at most two lines that fit `maxWidth` (logical px at font size `fontPx`):
// break after a nakaguro (U+30FB) or "/" or at a space first, else at the width limit. Widths are estimated (full-width
// characters = 1 em, ASCII ~0.56 em) because Drawbot has no text measurement; the second line is
// drawn with an end ellipsis by the renderer if it is still too long.
std::vector<std::wstring> WrapLabel(const std::wstring& s, float maxWidth, float fontPx);
float EstimateWidth(const std::wstring& s, float fontPx);
// Height (logical) the picker param reserves for a nominal panel width.
int PreferredHeight(Kind k, int nominalWidth = 300);

}  // namespace obthumb
