// obgpu_lenscore.h - lenscore::describeBlur (the CPU blur stage's own setup) -> obgpu_blur_desc.
// Header-only; needs lenscore.h and obgpu.h. The ONE copy of the mapping: gpu\cpp\obgpu_desc.cpp (harness,
// self-test generator) uses it, and the wrapper's plugin\ob_gpu_desc.h (identical code, namespace obgpuw) can
// become a one-line include of this file. It lives with the GPU library because it changes together with
// obgpu_blur_desc (OBGPU_DESC_VERSION), not with the plugin.
#pragma once
#include "lenscore.h"
#include "obgpu.h"

#include <cstring>

namespace obgpuw {

// the desc of the GPU library from lenscore's own blur setup (lenscore::describeBlur)
struct Desc
{
    obgpu_blur_desc d;
    lenscore::BlurSetup s;
};

inline bool BuildDesc(const lenscore::Params& p, int w, int h, bool needEL, Desc& D, lenscore::AbortFn ab, void* user, lenscore::Status& st)
{
    st = lenscore::describeBlur(p, w, h, D.s, ab, user);
    if (st != lenscore::Status::Ok)
        return false;
    const lenscore::BlurSetup& s = D.s;
    if (s.nLevels > OBGPU_MAX_LEVELS || lenscore::BlurSetup::kMaxLayers > OBGPU_MAX_LAYERS || lenscore::BlurSetup::kTiers != OBGPU_TIERS)
        return false;
    obgpu_blur_desc& d = D.d;
    std::memset(&d, 0, sizeof d);
    d.struct_size = (uint32_t)sizeof d;
    d.version = OBGPU_DESC_VERSION;
    d.w = w;
    d.h = h;
    d.nk = s.nk;
    d.n_tiers = s.nTiers;
    d.n_levels = s.nLevels;
    d.cat = s.cat ? 1 : 0;
    d.cat_n = s.catN;
    d.clamp_neg = s.clampNeg ? 1 : 0;
    d.fringe_share = s.fringeShare ? 1 : 0;
    d.need_el = needEL ? 1 : 0;
    d.max_reach = s.maxReach;
    d.row_begin = s.rowBegin;
    d.row_end = s.rowEnd;
    d.ap_n = s.ap.n;
    d.circ_n = s.circ.n;
    d.ax = s.ax;
    d.ay = s.ay;
    d.xext = s.xext;
    d.yext = s.yext;
    d.cat_k = s.catK;
    d.cat_lim = s.catLim;
    d.cat_inv = s.catInv;
    d.level_base = s.levelBase;
    d.fcx = s.fcx;
    d.fcy = s.fcy;
    d.half_diag = s.halfDiag;
    d.pa = s.pa;
    d.max_radius = s.maxRadius;
    d.smax_g = s.smaxG;
    d.ap_ymin = s.ap.ymin;
    d.ap_ymax = s.ap.ymax;
    d.ap_inv_step = s.ap.invStep;
    d.ap_area = s.ap.area;
    d.circ_ymin = s.circ.ymin;
    d.circ_ymax = s.circ.ymax;
    d.circ_inv_step = s.circ.invStep;
    d.circ_area = s.circ.area;
    for (int a = 0; a < 2; ++a)
        for (int ch = 0; ch < 3; ++ch)
            d.ks[a][ch] = s.ks[a][ch];
    for (int a = 0; a < 2; ++a)
        for (int t = 0; t < OBGPU_TIERS; ++t)
        {
            const lenscore::BlurSetup::Layers& L = s.prof[a][t];
            obgpu_profile& o = d.prof[a][t];
            o.nl = L.nl;
            o.share = L.share ? 1 : 0;
            o.share_from = L.shareFrom;
            o.s2 = L.s2;
            o.s2out = L.s2out;
            o.s2in = L.s2in;
            for (int l = 0; l < lenscore::BlurSetup::kMaxLayers; ++l)
            {
                o.ls[l] = L.ls[l];
                o.lb[l] = L.lb[l];
            }
        }
    for (int l = 1; l < s.nLevels; ++l)
    {
        const lenscore::BlurSetup::Level& q = s.levels[l];
        obgpu_level& o = d.levels[l];
        o.f = q.f;
        o.w = q.w;
        o.h = q.h;
        o.gox = q.gox;
        o.goy = q.goy;
        o.max_reach = q.maxReach;
        o.row_lo = q.rowLo;
        o.row_hi = q.rowHi;
    }
    d.ap_table = s.ap.table.data();
    d.circ_table = s.circ.table.data();
    d.cat_a = s.catA.empty() ? nullptr : s.catA.data();
    return true;
}

}  // namespace obgpuw
