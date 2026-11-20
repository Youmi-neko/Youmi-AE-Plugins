// obgpu_prepdesc.cpp - the prepare-stage description (ABI 1.1) from lenscore's internal constants.
// The blur desc comes from lenscore::describeBlur (obgpu_desc.cpp). The prepare stage has no such hook yet,
// so this translation unit compiles core/lenscore.cpp a second time under a private namespace name
// (`obgpu_lc`) to reach computeHighlights' constants (kAdaptRadius, kWhiteRadius, kWhiteTau, ...), gridPhase
// and frameOf. Proposed hook: lenscore::describePrepare (gpu/PHASE2.md, "engine API") removes this file's trick.
#define lenscore obgpu_lc
#include "../../core/lenscore.cpp"
#undef lenscore

#include "obgpu_desc.h"
#include <cstring>

// ---- ABI 1.1: prepare-stage description (computeHighlights / prepareRow parameters, renderVersion 2,
// automatic highlight source). Mirrors core/lenscore.cpp computeHighlights + prepareRow (v2.3).
int obgpu_prepare_desc_build(const void* params, size_t paramsSize, int w, int h, int hasMap, obgpu_prepare_desc* out) {
    using namespace obgpu_lc;
    if (!params || paramsSize != sizeof(obgpu_lc::Params) || !out || w <= 0 || h <= 0) return 1;
    try {
        Params p;
        std::memcpy(&p, params, sizeof p);
        if (!paramsValid(p)) return 1;
        if (p.renderVersion < 2 || p.highlightSource != 0) return 4;  // unsupported on the GPU
        obgpu_prepare_desc& d = *out;
        std::memset(&d, 0, sizeof d);
        d.struct_size = (uint32_t)sizeof d;
        d.version = OBGPU_PREP_VERSION;
        d.w = w; d.h = h;
        d.has_map = hasMap ? 1 : 0;
        d.invert_map = p.invertMap ? 1 : 0;
        d.render_version = p.renderVersion;
        d.focus = p.focus; d.map_scale = p.mapScale; d.map_gamma = p.mapGamma; d.max_radius = p.maxRadius;
        d.hl_gain = p.highlightGain; d.hl_threshold = p.highlightThreshold;
        d.hl_on = (p.highlightBoost > 0 && p.maxRadius > 0) ? 1 : 0;
        d.boost = p.highlightBoost;
        d.knee = (float)clampd(p.highlightKnee, 0.0, 0.97);
        d.curve = (float)clampd(p.highlightCurve, 1.0, 8.0);
        d.sel = (float)clampd(p.highlightSelectivity, 0.0, 1.0);
        d.hsat = (float)(1.0 + 2.0 * clampd(p.highlightSaturation, 0.0, 1.0));
        d.ps = (float)clampd(p.pixelScale, 1e-4, 256.0);
        double fcx, fcy, hd;
        frameOf(p, w, h, fcx, fcy, hd);
        d.adapt_gain = (float)kAdaptGain;
        const int rm = std::max(2, (int)std::lround(kAdaptRadius * hd));
        d.knee_box_r = (rm + 1) / 2;
        const double wpAmt = clampd(p.whiteProtection, 0.0, 1.0);
        d.wp_on = wpAmt > 0 ? 1 : 0;
        d.wp_amt = (float)wpAmt;
        if (d.wp_on) {
            const double rw = kWhiteRadius * 2.0 * hd;
            const int f = std::max(1, (int)std::lround(rw / 16.0));
            d.wp_f = f;
            d.wp_gx = gridPhase(p, true, f); d.wp_gy = gridPhase(p, false, f);
            d.wp_gw = (w + d.wp_gx + f - 1) / f; d.wp_gh = (h + d.wp_gy + f - 1) / f;
            d.wp_rb = std::max(1, (int)std::lround(rw / f * 0.5));
        }
        d.wp_rel = (float)kWhiteRel;
        for (int k = 0; k < kWhiteLevels && k < 10; ++k) d.wp_tau[k] = (float)kWhiteTau[k];
        const double sqps = std::sqrt(clampd(p.pixelScale, 1e-4, 256.0));
        auto scales = [&](double D, float* rf, float* wv, int& n) {
            const double mult[7] = {0.5, 0.7, 1.0, 1.4, 1.8, 2.2, 2.6};
            const double wgt[7] = {0.2, 0.2, 0.2, 0.2, 0.12, 0.06, 0.02};
            n = 0;
            for (int s = 0; s < 7; ++s) {
                rf[n] = (float)std::max(1.0, (mult[s] * D - (0.25 + 0.75 * sqps)) * 0.5);  // see detailMeasureV2
                wv[n] = (float)wgt[s];
                ++n;
            }
        };
        d.det_on = d.sel > 0 ? 1 : 0;
        if (d.det_on) scales(clampd(p.detailSize, 1.0, 400.0), d.det_rf, d.det_w, d.det_n);
        const double dmin = clampd(p.detailMin, 0.0, 400.0);
        const double dminFull = dmin / std::sqrt(clampd(p.pixelScale, 1e-4, 256.0));
        if (d.det_on && dmin > 1e-3 && dminFull > 0.5) {
            d.dmin_on = 1;
            scales(std::min(std::max(dmin, 0.5), clampd(p.detailSize, 1.0, 400.0)), d.dmin_rf, d.dmin_w, d.dmin_n);
            d.dmin_ramp = (float)smooth01((dminFull - 0.5) / 2.5);
        }
        d.dots = (float)clampd(p.dotsOnly, 0.0, 1.0);
        d.dots_on = (d.det_on && d.dots > 0) ? 1 : 0;
        const double len = std::max(3.0, 2.0 * clampd(p.detailSize, 1.0, 400.0));
        d.margin = prepareMargin(p, w, h);
        for (int dir = 0; dir < 4; ++dir) d.dots_r[dir] = std::max(1, (int)std::lround(0.5 * len / (dir >= 2 ? 1.41421356 : 1.0)));
        return 0;
    } catch (...) {
        return 3;
    }
}

// ---- ABI 1.3: post-stage description. Mirrors core/lenscore.cpp postProcess (v2.3): PostCtx setup, buildGlow,
// applyHalation, applyLumAberration (modes 0-2), rolloff, and blurField / blurFieldBox / gaussField's resolution
// of a sigma into the field operation. (Proposed hook: lenscore::describePost, same idea as describeBlur.)
namespace {
void fieldBox(const obgpu_lc::Params& p, int W, int H, double sigma, int rbForce, obgpu_field_op& o) {
    using namespace obgpu_lc;
    const int f = sigma > 6.0 ? std::max(1, (int)(sigma / 3.0)) : 1;
    o.kind = 2;
    o.sigma = (float)sigma;
    o.f = f;
    o.gx = gridPhase(p, true, f);
    o.gy = gridPhase(p, false, f);
    o.gw = (W + o.gx + f - 1) / f;
    o.gh = (H + o.gy + f - 1) / f;
    const double sd = sigma / f;
    o.rb0 = rbForce > 0 ? rbForce : std::max(1, (int)std::lround((std::sqrt(4.0 * sd * sd + 1.0) - 1) * 0.5));
}
// gaussField(in, W, H, sigma) directly (no 0.3 floor)
void fieldGauss(double sigma, obgpu_field_op& o) {
    std::memset(&o, 0, sizeof o);
    o.kind = 1;
    o.sigma = (float)sigma;
    o.r = std::max(1, (int)std::ceil(3.0 * sigma));
}
void fieldOp(const obgpu_lc::Params& p, int W, int H, double sigma, obgpu_field_op& o);
// gaussOrBox: exact Gaussian <= 6 px, blurField above
void fieldGB(const obgpu_lc::Params& p, int W, int H, double sigma, obgpu_field_op& o) {
    if (sigma <= 6.0) fieldGauss(sigma, o);
    else fieldOp(p, W, H, sigma, o);
}
// blurField(in, W, H, sigma, p, ...)
void fieldOp(const obgpu_lc::Params& p, int W, int H, double sigma, obgpu_field_op& o) {
    std::memset(&o, 0, sizeof o);
    if (p.renderVersion < 2) { fieldBox(p, W, H, sigma, -1, o); return; }
    if (sigma <= 6.0) {
        o.kind = 1;
        const double s = std::max(0.3, sigma);
        o.sigma = (float)s;
        o.r = std::max(1, (int)std::ceil(3.0 * s));
        return;
    }
    const int f = std::max(1, (int)(sigma / 3.0));
    const double sd = sigma / f;
    const int r0 = std::max(1, (int)std::floor((std::sqrt(4.0 * sd * sd + 1.0) - 1) * 0.5));
    const double v0 = r0 * (r0 + 1.0), v1 = (r0 + 1.0) * (r0 + 2.0);
    const double t = obgpu_lc::clampd((sd * sd - v0) / (v1 - v0), 0.0, 1.0);
    fieldBox(p, W, H, sigma, r0, o);
    if (t > 1e-6) { o.rb1 = r0 + 1; o.t = (float)t; }
}
}  // namespace

int obgpu_post_desc_build(const void* params, size_t paramsSize, int w, int h, obgpu_post_desc* out) {
    return obgpu_post_desc_build_ex(params, paramsSize, w, h, 0u, out);
}

int obgpu_post_desc_build_ex(const void* params, size_t paramsSize, int w, int h, uint32_t flags, obgpu_post_desc* out) {
    using namespace obgpu_lc;
    if (!params || paramsSize != sizeof(obgpu_lc::Params) || !out || w <= 0 || h <= 0) return 1;
    try {
        Params p;
        std::memcpy(&p, params, sizeof p);
        if (!paramsValid(p)) return 1;
        obgpu_post_desc& d = *out;
        std::memset(&d, 0, sizeof d);
        d.struct_size = (uint32_t)sizeof d;
        d.version = OBGPU_POST_VERSION2;
        d.w = w; d.h = h;
        double fcx, fcy, hd;
        frameOf(p, w, h, fcx, fcy, hd);
        d.fcx = (float)fcx; d.fcy = (float)fcy; d.half_diag = (float)hd; d.pa = (float)aspectOf(p);
        double glowAmt = clampd(p.glow, 0.0, 1.0);
        if (glowAmt < 1e-4) glowAmt = 0;
        double vig = clampd(p.vignetting, 0.0, 1.0);
        if (vig < 1e-4) vig = 0;
        d.glow_amt = (float)glowAmt; d.vig = (float)vig;
        if (glowAmt > 0) {
            const double sigma = std::min(0.25, std::max(0.005, (double)p.glowSize)) * 2.0 * hd;
            const int f = std::max(1, (int)(sigma / 3.0));
            d.glow_f = f;
            d.glow_gox = gridPhase(p, true, f); d.glow_goy = gridPhase(p, false, f);
            d.glow_gw = (w + d.glow_gox + f - 1) / f; d.glow_gh = (h + d.glow_goy + f - 1) / f;
            const double sd = sigma / f;
            d.glow_rb = std::max(1, (int)std::lround((std::sqrt(4.0 * sd * sd + 1.0) - 1) * 0.5));
        }
        const bool hal = p.halation > 1e-5, la = p.lumAberration > 1e-5;
        d.roll_candidate = p.highlightRolloff > 1e-4 ? 1 : 0;
        d.roll_knee = (float)(1.0 - 0.5 * clampd(p.highlightRolloff, 0.0, 1.0));
        d.roll_v2 = p.renderVersion >= 2 ? 1 : 0;
        if (hal) {
            d.hal_amt = (float)clampd(p.halation, 0.0, 4.0);
            d.hal_floor = (float)std::max(1e-3, (double)p.halationThreshold);
            d.hal_relk = (float)std::pow(2.0, clampd(p.halationRelative, -4.0, 8.0));
            d.hal_soft = (float)clampd(p.halationSoftness, 0.0, 1.0);
            fieldOp(p, w, h, std::max(1.0, kHalScene * 2.0 * hd), d.hal_scene);
            const double sw = std::max(1.0, clampd(p.halationSize, 0.001, 0.2) * 2.0 * hd);
            const double st = std::max(0.8, 0.12 * sw);
            const double rim = clampd(p.halationRim, 0.0, 1.0);
            d.hal_rim = (float)rim;
            if (rim > 0) fieldOp(p, w, h, st, d.hal_tight);
            if (rim < 1) fieldOp(p, w, h, sw, d.hal_wide);
            for (int c = 0; c < 3; ++c) d.hal_tint[c] = (float)tintC(p.halationTint[c]);
        }
        const bool la34 = la && (p.laMode == 3 || p.laMode == 4) && (flags & OBGPU_POST_REQ_LA34);
        const bool laGpu = la && ((p.laMode != 3 && p.laMode != 4) || la34);
        if (la34 && p.laMode == 4) {   // applyChannelDrift (v2.3) / driftGeom
            const DriftGeom G = driftGeom(p, hd);
            d.la34 = 4;
            if (G.rl > 0) { d.dr_s1 = std::max(1, (int)std::lround(0.6 * G.rl)); d.dr_s2 = (int)std::lround(0.42 * G.rl); }
            d.dr_v2 = p.renderVersion >= 2 ? 1 : 0;
            if (d.dr_v2 && G.spread > 0) { d.dr_sp1 = std::max(1, (int)std::lround(0.6 * G.spread)); d.dr_sp2 = (int)std::lround(0.42 * G.spread); }
            d.dr_thr = (float)clampd(p.laThreshold, 0.0, 10.0);
            d.dr_curve = (float)clampd(p.laCurve, 0.25, 4.0);
            d.dr_pres = (float)clampd(p.laPreserve, 0.0, 1.0);
            d.dr_kstop = (float)(G.sm * 2.5066);
            d.dr_len = (float)G.len;
            d.dr_sm2 = (float)(2.0 * G.sm);
            d.dr_lencap = (float)std::min(G.lenCap, 1e30);
            const double sgn = p.laShiftDir == 1 ? 1.0 : -1.0;
            for (int c = 0; c < 3; ++c) d.dr_k[c] = (float)(sgn * clampd(p.laShift[c], -2.0, 2.0));
            fieldGB(p, w, h, G.sm, d.dr_soft);
            if (d.dr_sp1 > 0) fieldGB(p, w, h, G.dirSigma, d.dr_dir);
        }
        if (la34 && p.laMode == 3) {   // applyContrastRim / contrastFringe (v2.3) / rimGeom
            const RimGeom G = rimGeom(p, hd);
            d.la34 = 3;
            if (G.rl > 0) { d.rim_s1 = std::max(1, (int)std::lround(0.6 * G.rl)); d.rim_s2 = (int)std::lround(0.42 * G.rl); }
            d.rim_rn = G.rn; d.rim_rc = G.rc;
            d.rim_amt = (float)clampd(p.lumAberration, 0.0, 4.0);
            d.rim_thr = (float)clampd(p.laThreshold, 0.0, 10.0);
            d.rim_curve = (float)clampd(p.laCurve, 0.25, 4.0);
            d.rim_hs = (float)clampd(p.laHueShift, 0.0, 1.0);
            d.rim_ex = (float)(1.0 + 3.0 * clampd(p.laSharpness, 0.0, 1.0));
            d.rim_mixb = (float)clampd(p.laBrightColor, 0.0, 1.0);
            d.rim_comp = (float)clampd(p.laComplement, 0.0, 1.0);
            for (int c = 0; c < 3; ++c) d.rim_hi[c] = (float)tintC(p.laTint[c]);
            fieldOp(p, w, h, G.sReg, d.rim_reg);
            d.rim_fringe = p.laBlur > 1e-6 ? 1 : 0;
            if (d.rim_fringe) {
                const double sR = G.sf, sig[3] = {sR, 0.55 * sR, 0.25 * sR}, s0 = 0.12 * sR;
                for (int c = 0; c < 3; ++c) if (sig[c] >= 0.3) fieldGB(p, w, h, sig[c], d.rim_sig[c]);
                if (s0 >= 0.3) fieldGB(p, w, h, s0, d.rim_s0);
            } else {
                fieldGauss(G.sd, d.rim_fb);
                d.rim_bright_side = G.sb >= 0.3 ? 1 : 0;
                if (d.rim_bright_side) fieldGauss(G.sb, d.rim_fd);
                d.rim_mixb_on = d.rim_mixb > 1e-4 ? 1 : 0;
            }
        }
        if (laGpu) {
            d.la_mode = p.laMode == 1 ? 1 : (p.laMode == 2 ? 2 : 0);
            d.la_amt = (float)clampd(p.lumAberration, 0.0, 4.0);
            d.la_thr = (float)clampd(p.laThreshold, 0.0, 10.0);
            d.la_curve = (float)clampd(p.laCurve, 0.25, 4.0);
            d.la_hs = (float)clampd(p.laHueShift, 0.0, 1.0);
            for (int c = 0; c < 3; ++c) d.la_hi[c] = (float)tintC(p.laTint[c]);
            fieldOp(p, w, h, std::max(0.6, clampd(p.laWidth, 0.0005, 0.1) * 2.0 * hd), d.la_field);
        }
        // the GPU prefix of the chain glow/vig -> halation -> lum. aberration -> rolloff (-> grain: CPU)
        uint32_t s = 0;
        bool open = true;
        auto stage = [&](bool enabled, bool supported, uint32_t bit) {
            if (!open || !enabled) return;
            if (supported) s |= bit; else open = false;
        };
        stage(glowAmt > 0 || vig > 0, true, OBGPU_POST_GLOWVIG);
        stage(hal, true, OBGPU_POST_HALATION);
        stage(la, laGpu, OBGPU_POST_LUMAB);
        stage(d.roll_candidate != 0, true, OBGPU_POST_ROLLOFF);
        d.stages = s;
        return 0;
    } catch (...) {
        return 3;
    }
}

void obgpu_post_rest_params(void* params, size_t paramsSize, uint32_t done) {
    if (!params || paramsSize != sizeof(obgpu_lc::Params)) return;
    obgpu_lc::Params& p = *static_cast<obgpu_lc::Params*>(params);
    if (done & OBGPU_POST_GLOWVIG) { p.glow = 0; p.vignetting = 0; }
    if (done & OBGPU_POST_HALATION) p.halation = 0;
    if (done & OBGPU_POST_LUMAB) p.lumAberration = 0;
    if (done & OBGPU_POST_ROLLOFF) p.highlightRolloff = 0;
}