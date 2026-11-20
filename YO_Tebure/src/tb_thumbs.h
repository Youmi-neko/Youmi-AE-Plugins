// Tebure - preset picker thumbnails: each preset's 5 s motion path (x-y scribble of the frame centre) plus an
// amplitude bar, computed by the engine itself with the preset's own values (cannot misrepresent a preset).
// Shared by the plug-in (Drawbot vector drawing in the Effect Controls) and tools (offline preview PNG).
// Threading: Get() computes lazily and caches behind a mutex; only UI events and tools call it, never renders.
#pragma once
#include <vector>

namespace tbthumb {

struct Path
{
    std::vector<float> x, y;  // normalised to [-1, 1] (shared scale for x and y: the shape is not distorted)
    float rmsPx = 0;          // RMS displacement in px at 1920 wide (amplitude bar)
    float barFrac = 0;        // 0..1 bar length (log scale over all presets)
    int family = 0;           // 0 hand-held, 1 walking, 2 impact / vehicle, 3 film / anime
};
int Count();                    // number of presets (popup items 2..)
const Path* Get(int preset);    // nullptr if out of range

// layout (logical px): cells of a thumbnail + 2-line label, uniform scale s <= 1 when the panel is narrow
const float kThumbW = 84, kThumbH = 52, kLabelH = 28, kGap = 6;
struct Grid
{
    int cols = 1, rows = 1;
    float s = 1, cw = kThumbW + kGap, ch = kThumbH + kLabelH + kGap, x0 = 0, y0 = 0;
};
Grid Layout(int n, float w);
int PreferredHeight(int n, float w);
void CellRect(const Grid& g, int i, float& x, float& y, float& w, float& h);   // thumbnail
void LabelRect(const Grid& g, int i, float& x, float& y, float& w, float& h);  // label (2 lines)
int HitTest(const Grid& g, int n, float x, float y);                           // -1 = none

}  // namespace tbthumb
