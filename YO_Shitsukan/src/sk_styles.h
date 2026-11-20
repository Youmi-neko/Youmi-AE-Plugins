// Shitsukan styles: the values the "Style" popup writes (shared by the plug-in, the CLI and the tests).
// A style writes every texture / line / watercolour control (never Amount, map, edges, view, seed, space).
// Popup value 1 = Custom (writes nothing); style s (popup s + 2) = kStyles[s].
#pragma once
#include "sk_core.h"

namespace sk {

struct Style
{
    const char* key;  // CLI name
    Params v;         // only the style fields are used
};

// the neutral base every style starts from (all effects off, sensible secondary values)
inline Params StyleBase()
{
    Params p;
    p.fine = p.mid = p.large = 0;
    p.detailSize = 1.5; p.halo = 70; p.edgeAware = 60; p.colorDetail = 30;
    p.protectLines = true; p.lineSens = 50; p.lineWidth = 3; p.lineColor = LC_COLOR; p.blackWalls = false; p.bgDetailLines = false;
    p.fillBlur = 0; p.keepEdges = 30; p.lineThick = 0; p.lineDark = 0; p.lineSoft = 0; p.lineStyle = LS_ASIS;
    p.wc = 0; p.flatten = 60; p.flattenSize = 6; p.strokes = 45; p.strokeLen = 16; p.strokeAngle = 35; p.glaze = 60;
    p.rim = 70; p.rimWidth = 4; p.gravity = 30; p.bleed = 35; p.bleedWidth = 10; p.backrun = 0; p.wobble = 2; p.wobbleScale = 40;
    p.uneven = 25; p.gran = 25; p.paper = 40; p.paperScale = 100; p.keepWhite = 75; p.dilution = 35; p.protectDetail = 70;
    p.wcMode = WCM_FILTER; p.paintScale = 16; p.merge = 50; p.loose = 50; p.liftDarks = 50; p.charge = 40; p.margin = 0; p.palette = 0;
    p.warp = 0; p.pigSet = 0; p.paintEngine = 0;
    return p;
}
inline Params MakeStyle(int s)
{
    Params p = StyleBase();
    switch (s)
    {
    case 0:  // Texture Only: clearly crisper brushwork and more local contrast, no halos
        p.fine = 100; p.mid = 60; p.large = 50;
        break;
    case 1:  // Blur Fills, Keep Lines
        p.fillBlur = 16; p.keepEdges = 25;
        break;
    case 2:  // Watercolor (Loose): region painter, large loose washes, pencil lines
        p.wc = 100; p.wcMode = WCM_PAINT; p.paintScale = 28; p.merge = 90; p.loose = 70; p.rim = 80; p.rimWidth = 4; p.bleed = 55;
        p.bleedWidth = 18; p.glaze = 40; p.gravity = 30; p.uneven = 30; p.charge = 35; p.keepWhite = 70; p.liftDarks = 60; p.dilution = 25;
        p.gran = 25; p.paper = 45; p.margin = 0; p.palette = 0; p.protectDetail = 70; p.lineStyle = LS_PENCIL; p.lineDark = -35;
        p.warp = 60;
        break;
    case 3:  // Watercolor (Careful): region painter, smaller tidy washes, ink lines
        p.wc = 100; p.wcMode = WCM_PAINT; p.paintScale = 14; p.merge = 65; p.loose = 25; p.rim = 60; p.rimWidth = 3; p.bleed = 30;
        p.bleedWidth = 10; p.glaze = 60; p.gravity = 25; p.uneven = 20; p.charge = 20; p.keepWhite = 60; p.liftDarks = 45; p.dilution = 15;
        p.gran = 20; p.paper = 35; p.lineStyle = LS_INK; p.lineDark = -15;
        break;
    case 4:  // Line and Wash Sketch
        p.wc = 100; p.wcMode = WCM_PAINT; p.paintScale = 30; p.merge = 100; p.loose = 100; p.margin = 60; p.dilution = 45; p.liftDarks = 70;
        p.rim = 70; p.bleed = 40; p.bleedWidth = 14; p.glaze = 30; p.charge = 30; p.keepWhite = 85; p.gran = 20; p.paper = 50;
        p.lineStyle = LS_INK; p.lineDark = 10;
        break;
    case 5:  // Picture Book Watercolor
        p.wc = 100; p.wcMode = WCM_PAINT; p.paintScale = 20; p.merge = 85; p.loose = 40; p.rim = 30; p.bleed = 45; p.bleedWidth = 10;
        p.glaze = 40; p.charge = 45; p.dilution = 30; p.liftDarks = 70; p.keepWhite = 70; p.gran = 15; p.paper = 30; p.palette = 12; p.pigSet = 1;  /* storybook (soft): the owner's choice of default */
        p.lineStyle = LS_PENCIL; p.lineDark = -45;
        break;
    case 6:  // Background Art (Transparent Watercolor)
        p.wc = 100; p.wcMode = WCM_PAINT; p.paintScale = 12; p.merge = 60; p.loose = 15; p.rim = 55; p.rimWidth = 3; p.bleed = 35;
        p.bleedWidth = 10; p.glaze = 70; p.charge = 25; p.dilution = 10; p.liftDarks = 35; p.keepWhite = 50; p.gran = 30; p.paper = 35;
        p.lineStyle = LS_ASIS;
        break;
    case 7:  // Wet-in-Wet
        p.wc = 100; p.wcMode = WCM_PAINT; p.paintScale = 40; p.merge = 100; p.loose = 50; p.rim = 15; p.bleed = 100; p.bleedWidth = 40; p.backrun = 60;
        p.glaze = 20; p.charge = 60; p.dilution = 35; p.liftDarks = 60; p.keepWhite = 60; p.margin = 30; p.gran = 15; p.paper = 35;
        p.lineStyle = LS_PENCIL; p.lineDark = -60;
        break;
    case 8:  // Watercolor (Old Filter): the round-2 filter (strokes, glazes, rims, paper)
        p.wc = 100; p.wcMode = WCM_FILTER; p.lineDark = -25; p.lineSoft = 0.5;
        break;
    case 9:  // Background Art Boost
        p.fine = 140; p.mid = 100; p.large = 70; p.detailSize = 1.2; p.halo = 75; p.edgeAware = 70; p.colorDetail = 40; p.protectLines = false;
        break;
    case 10:  // Smooth Skin (use with a map on the skin for faces)
        p.fine = -100; p.mid = -75; p.large = -10; p.detailSize = 2.0; p.edgeAware = 85;
        break;
    case 11:  // Soft Lines
        p.lineSoft = 1.6; p.lineDark = -35; p.lineThick = 0.3;
        break;
    // 0.6.0.3: the round-6 looks the owner asked for (refs_user\preset_request_2026-11-15.jpg = out\round6\r6_archetypes_*):
    // the round-6 engine (src\r6, compiled as sk_r6) with the round-6 values of the three styles
    case 12:  // Line and Wash (Round 6)
        p = MakeStyle(4);
        p.merge = 70; p.paintEngine = 1;
        break;
    case 13:  // Picture Book (Round 6)
        p = MakeStyle(5);
        p.merge = 55; p.bleed = 70; p.bleedWidth = 24; p.pigSet = 0; p.paintEngine = 1;
        break;
    case 14:  // BG Art (Round 6)
        p = MakeStyle(6);
        p.merge = 35; p.paintEngine = 1;
        break;
    default: break;
    }
    return p;
}

const Style kStyles[] = {{"texture", MakeStyle(0)},  {"blurfills", MakeStyle(1)}, {"wcLoose", MakeStyle(2)}, {"wcCareful", MakeStyle(3)},
                         {"wcSketch", MakeStyle(4)}, {"wcBook", MakeStyle(5)},      {"wcBgArt", MakeStyle(6)}, {"wcWet", MakeStyle(7)},
                         {"wcOld", MakeStyle(8)},    {"bgart", MakeStyle(9)},       {"skin", MakeStyle(10)},   {"softlines", MakeStyle(11)},
                         {"r6Sketch", MakeStyle(12)}, {"r6Book", MakeStyle(13)},   {"r6BgArt", MakeStyle(14)}};
const int kStyleCount = (int)(sizeof(kStyles) / sizeof(kStyles[0]));
const int kDefaultStyle = 0;  // "Texture Only"

// copy the style fields of s into p
inline void ApplyStyle(const Style& st, Params& p)
{
    const Params& s = st.v;
    p.fine = s.fine; p.mid = s.mid; p.large = s.large; p.detailSize = s.detailSize; p.halo = s.halo; p.edgeAware = s.edgeAware;
    p.colorDetail = s.colorDetail; p.protectLines = s.protectLines; p.lineSens = s.lineSens; p.lineWidth = s.lineWidth;
    p.lineColor = s.lineColor; p.blackWalls = s.blackWalls; p.bgDetailLines = s.bgDetailLines; p.fillBlur = s.fillBlur;
    p.keepEdges = s.keepEdges; p.lineThick = s.lineThick; p.lineDark = s.lineDark; p.lineSoft = s.lineSoft; p.lineStyle = s.lineStyle;
    p.wc = s.wc; p.flatten = s.flatten; p.flattenSize = s.flattenSize; p.strokes = s.strokes; p.strokeLen = s.strokeLen;
    p.strokeAngle = s.strokeAngle; p.glaze = s.glaze; p.rim = s.rim; p.rimWidth = s.rimWidth; p.gravity = s.gravity; p.bleed = s.bleed;
    p.bleedWidth = s.bleedWidth; p.backrun = s.backrun; p.wobble = s.wobble; p.wobbleScale = s.wobbleScale; p.uneven = s.uneven;
    p.gran = s.gran; p.paper = s.paper; p.paperScale = s.paperScale; p.keepWhite = s.keepWhite; p.dilution = s.dilution;
    p.protectDetail = s.protectDetail;
    p.wcMode = s.wcMode; p.paintScale = s.paintScale; p.merge = s.merge; p.loose = s.loose; p.liftDarks = s.liftDarks;
    p.charge = s.charge; p.margin = s.margin; p.palette = s.palette; p.warp = s.warp; p.pigSet = s.pigSet; p.paintEngine = s.paintEngine;
}

}  // namespace sk
