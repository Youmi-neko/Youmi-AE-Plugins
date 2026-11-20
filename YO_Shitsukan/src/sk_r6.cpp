// 0.6.0.3: the round-6 engine (the owner's confirmed look, out\round6\r6_archetypes_*), compiled from an untouched
// copy of its source (src\r6 = build\round6\src) as a private namespace sk_r6. Selected by Painting Engine = Round 6;
// the default (Current) never comes here, so every existing style and project renders exactly as before.
// Round 6 is less stable in video (its washes are re-segmented every frame) - the reason round 7 rebuilt it.
#include "sk_core.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>
#include "sk_platform.h"  // mac_port (was: emmintrin.h, windows.h, psapi.h)
#include "sk_pmath.h"

#define sk sk_r6
#include "r6/sk_core.cpp"
#undef sk

namespace sk {
static sk_r6::Params ToR6(const Params& p)
{
    sk_r6::Params q;
    q.amount = p.amount;
    q.fine = p.fine; q.mid = p.mid; q.large = p.large; q.detailSize = p.detailSize; q.halo = p.halo; q.edgeAware = p.edgeAware;
    q.colorDetail = p.colorDetail;
    q.protectLines = p.protectLines; q.lineSens = p.lineSens; q.lineWidth = p.lineWidth; q.lineColor = p.lineColor;
    q.blackWalls = p.blackWalls; q.bgDetailLines = p.bgDetailLines; q.fillBlur = p.fillBlur; q.keepEdges = p.keepEdges;
    q.lineThick = p.lineThick; q.lineDark = p.lineDark; q.lineSoft = p.lineSoft; q.lineStyle = p.lineStyle;
    q.wc = p.wc; q.wcMode = p.wcMode == WCM_PAINT ? sk_r6::WCM_PAINT : sk_r6::WCM_FILTER;
    q.paintScale = p.paintScale; q.merge = p.merge; q.loose = p.loose; q.liftDarks = p.liftDarks; q.charge = p.charge;
    q.margin = p.margin; q.palette = p.palette; q.flatten = p.flatten; q.flattenSize = p.flattenSize;
    q.strokes = p.strokes; q.strokeLen = p.strokeLen; q.strokeAngle = p.strokeAngle; q.glaze = p.glaze;
    q.rim = p.rim; q.rimWidth = p.rimWidth; q.gravity = p.gravity; q.bleed = p.bleed; q.bleedWidth = p.bleedWidth;
    q.backrun = p.backrun; q.wobble = p.wobble; q.wobbleScale = p.wobbleScale; q.uneven = p.uneven;
    q.gran = p.gran; q.paper = p.paper; q.paperScale = p.paperScale; q.keepWhite = p.keepWhite; q.dilution = p.dilution;
    q.protectDetail = p.protectDetail;
    q.mapTarget = p.mapTarget; q.edgeMode = p.edgeMode; q.view = p.view; q.seed = p.seed; q.paperFrame = p.paperFrame;
    q.linear = p.linear; q.par = p.par;
    return q;
}
bool UseRound6(const Params& p) { return p.paintEngine == 1 && p.view >= 0 && p.view < (int)sk_r6::V_COUNT; }
bool IsIdentityRound6(const Params& p) { return sk_r6::IsIdentity(ToR6(p)); }
Status RenderRound6(const Params& p, Frame& f)
{
    sk_r6::Frame g;
    g.w = f.w; g.h = f.h; g.ox = f.ox; g.oy = f.oy; g.ds = f.ds; g.rgba = f.rgba;
    g.map = p.depth ? nullptr : f.map;  // (a depth matte has no meaning in the round-6 engine)
    g.paperTex = f.paperTex;
    g.layer.l = f.layer.l; g.layer.t = f.layer.t; g.layer.r = f.layer.r; g.layer.b = f.layer.b;
    g.out = f.out; g.threads = f.threads; g.abort = f.abort;
    switch (sk_r6::Render(ToR6(p), g))
    {
    case sk_r6::ST_OK: return ST_OK;
    case sk_r6::ST_ABORTED: return ST_ABORTED;
    case sk_r6::ST_OOM: return ST_OOM;
    default: return ST_ERROR;
    }
}
}  // namespace sk
