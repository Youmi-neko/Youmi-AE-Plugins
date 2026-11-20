// Optic Bokeh Kantan (simple edition), v2.3: the front end. A pure function from the Kantan controls
// to the FULL effect's UI values (obfx::Settings), which then go through the same engine adapter
// (obfx::MakeEngineParams / MakeHqParams) and the same render code as the full effect. Nothing of the
// rendering is duplicated, and every Kantan setting is a setting the full effect can also hold (so a
// Kantan instance can be converted to the full effect exactly: tools\kantan_to_full.jsx).
//
// Anchors (bit-exact, tested by aetest\probe\host_probe against the full effect's own ReadSettings):
//   Character 100 %, Aperture 0 %, Bokeh Highlights 50 %  = the full effect with that Lens Preset
//   chosen and every other control at its default; Halation / Luminance Aberration / Grain 100 % = the
//   full effect with Halation 100 % / that Aberration Style at 100 % / that Grain Stock as written.
//
// Link version (hidden param "Kantan Version"): the curves below are version 1. A later retuning adds
// version 2 and keeps version 1 here, so saved projects never change their look.
#pragma once
#include "ob_settings.h"

namespace kantan {

const int kLinkVersionCurrent = 1;

// Kantan UI values (units as shown in the UI). Defaults = a new instance.
struct Values
{
    int lens = 1;               // popup 1.. = lenscore::kLensPresets[v-1] (no "Custom")
    int quality = obfx::QUALITY_STANDARD;
    int view = obfx::VIEW_RESULT;
    double radius = 20.0;       // px at full resolution
    bool invertMap = false;
    double focusPct = 0.0;
    int mapChannel = obfx::CH_LUMINANCE;
    double depthStrengthPct = 100.0;  // = Map Scale x 100
    double depthCurve = 1.0;          // = Map Gamma
    double focusRangePct = 0.0;       // v2.3 map features (ob_mapx.h), the same as the full effect's
    double frontPct = 100.0;
    double backPct = 100.0;
    double mapBlurPx = 0.0;
    bool focusPointOn = false;
    double focusPointX = 0.0, focusPointY = 0.0;  // layer px (the host gives the point param in these)
    int useGpu = 1;                   // v2.4 Use GPU popup (1 Auto, 2 Off): passed through
    double chromaPct = 100.0;         // Chromatic Aberration: 0 = none, 100 = the preset's own, 200 = strong
    double aperturePct = 0.0;   // 0 = wide open (the preset as designed) .. 100 = stopped down
    double characterPct = 100.0;// 0 = clean ideal lens, 100 = the preset as designed, 200 = exaggerated
    double highlightsPct = 50.0;// 0 = no highlight expansion, 50 = the preset as designed, 100 = strong
    double halationPct = 0.0;   // 0..200, 100 = the full effect's Halation 100 % with its defaults
    int laStyle = 1;            // popup 1.. = lenscore::kAberrationStyles[v-1]
    double laPct = 0.0;         // 0..200, 100 = the style at 100 %
    int grainStock = 1;         // popup 1.. = lenscore::kGrainStocks[v-1]
    double grainPct = 0.0;      // 0..200, 100 = the stock's own amount
    int edgeMode = obfx::EDGE_REPEAT;
    int version = kLinkVersionCurrent;  // hidden
};

// Slider ranges (valid = slider) of the Kantan controls.
const double kRadiusMax = 500.0, kRadiusSliderMax = 100.0;
const double kApertureMax = 100.0, kCharacterMax = 200.0, kHighlightsMax = 100.0;
const double kHalationMax = 200.0, kLaMax = 200.0, kGrainMax = 200.0, kChromaMax = 200.0;
const double kDepthStrengthMax = 2000.0, kDepthStrengthSliderMax = 400.0;

// Kantan values -> the full effect's UI values. Out-of-range inputs are clamped first.
void ToSettings(const Values& v, obfx::Settings& s);

}  // namespace kantan
