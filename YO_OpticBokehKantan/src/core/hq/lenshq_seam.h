// lenshq_seam.h - adapter between lenscore's stage seam (lenscore.h, "STAGE SEAM v2.0") and the
// quality blur engine lenshq. Header-only; needs lenscore.cpp + core/hq/*.cpp linked (the plugin
// links both). Replaces ONLY stage 2 (blur):
//
//     lenscore::prepareHighlights(...)                         -> prep        (lenscore)
//     lenshq::blurPrepared(prep, p, hp, dst, stride, eL, ...)                (this file)
//     lenscore::postProcess(prep, p, dst, stride, eL, ...)                   (lenscore)
//
// Contract kept: radius/side semantics taken verbatim from Prepared, energy conservation, output
// >= 0, identity where the kernel fits in its own pixel, frame rect (Params.frame*) for cat eye,
// eL blurred with the G-channel weights and normalised like the colour, vignetting NOT applied here
// (postProcess does it). No global state; result independent of the thread count; tile-exact with
// inputMarginHq().
#pragma once

#include "lenscore.h"
#include "lenshq.h"

#include <algorithm>

namespace lenshq {

// HqParams -> HqOptions. spectralSamples: 0 or 1..2 = RGB three-band (alpha-like, as documented in
// lenscore.h), >= 3 = that many wavelength bins for the clip edge (the interior is always integrated
// continuously), < 0 = automatic spectral (recommended for "Best").
inline HqOptions hqOptionsFrom(const lenscore::HqParams& hp) {
    HqOptions q;
    q.textureAmount = hp.textureAmount;
    q.textureScale = hp.textureScale;
    q.textureType = hp.textureType;
    q.textureSeed = hp.textureSeed;
    q.lateralCA = hp.lateralCA;
    if (hp.spectralSamples < 0) { q.fringeModel = FringeSpectral; q.spectralSamples = 0; }
    else if (hp.spectralSamples < 3) { q.fringeModel = FringeThreeBand; }
    else { q.fringeModel = FringeSpectral; q.spectralSamples = hp.spectralSamples; }
    q.applyVignetting = false;  // stage 3 (postProcess) owns vignetting
    q.invertFrontBokeh = false; // same disc orientation as Fast / Standard (see lenshq.h)
    return q;
}

// Stage 2 for Quality = Best. dst: premultiplied linear RGBA (stride in floats, >= 4*prep.w).
// eLOut: prep.w*prep.h floats, written iff non-null (pass non-null exactly when
// lenscore::postNeedsEL(p, prep)).
inline Status blurPrepared(const lenscore::Prepared& prep, const lenscore::Params& p,
                           const lenscore::HqParams& hp, float* dst, int dstStrideFloats, float* eLOut,
                           AbortFn abort = nullptr, void* user = nullptr, HqStats* stats = nullptr) {
    const int w = prep.w, h = prep.h;
    const size_t n = (size_t)w * h;
    if (w <= 0 || h <= 0 || prep.rgba.size() < n * 4 || prep.radius.size() < n || prep.side.size() < n)
        return Status::BadArgs;
    lenscore::Params pp = p;
    double rmax = 0;
    // tables are sized from the slider maxRadius (data independent = tile exact); a finite radius
    // above it (should not happen) still fits; Inf / NaN radii are treated as in focus by the engine
    for (size_t i = 0; i < n; ++i) {
        const double r = prep.radius[i];
        if (r > rmax && r <= 4000.0) rmax = r;
    }
    if (!(pp.maxRadius >= 0 && pp.maxRadius <= 4000)) pp.maxRadius = 0;
    pp.maxRadius = (float)std::min(4000.0, std::max((double)pp.maxRadius, rmax));
    HqOptions q = hqOptionsFrom(hp);
    q.radiusIn = prep.radius.data();
    q.sideIn = prep.side.data();
    q.stats = stats;
    std::vector<float> zeros;
    if (eLOut) {
        if (prep.eL.size() >= n) q.auxSrc = prep.eL.data();
        else { zeros.assign(n, 0.f); q.auxSrc = zeros.data(); }
        q.auxSrcStride = w;
        q.auxDst = eLOut;
        q.auxDstStride = w;
    }
    return render(prep.rgba.data(), w, h, w * 4, nullptr, 0, dst, dstStrideFloats, pp, q, abort, user);
}

// Blur-stage margin (px) for Best.
inline int blurMarginHq(const lenscore::Params& p, const lenscore::HqParams& hp) {
    return inputMargin(p, hqOptionsFrom(hp));
}

// Total PreRender margin for Best (same composition as lenscore::inputMargin).
inline int inputMarginHq(const lenscore::Params& p, const lenscore::HqParams& hp) {
    return lenscore::prepareMargin(p) + blurMarginHq(p, hp) + lenscore::postMargin(p) + 2;
}

} // namespace lenshq
