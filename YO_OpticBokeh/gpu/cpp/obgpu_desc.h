// obgpu_desc.h - builds an obgpu_blur_desc from lenscore Params with lenscore's OWN setup code
// (shape tables, layer profiles, levels, row range). C++ shim compiled into the host (.aex / harness).
#pragma once
#include "../include/obgpu.h"
#include <cstddef>

// params: a lenscore::Params (size checked). needEL = lenscore::postNeedsEL(p, prep).
// Fills *out; *holder owns the tables the desc points to (free with obgpu_desc_free).
// Returns 0 on success, 1 bad args / invalid params, 2 out of memory, 3 other error.
int obgpu_desc_build(const void* params, size_t paramsSize, int w, int h, int needEL,
                     obgpu_blur_desc* out, void** holder);
void obgpu_desc_free(void* holder);

// ABI 1.1: prepare-stage desc for obgpu_render_frame. Returns 0 ok, 1 bad args, 3 error, 4 unsupported on the GPU
// (renderVersion 1 or a highlight-layer source: use the CPU prepare).
int obgpu_prepare_desc_build(const void* params, size_t paramsSize, int w, int h, int hasMap, obgpu_prepare_desc* out);
template <class P>
inline int obgpuPrepareDescBuild(const P& p, int w, int h, bool hasMap, obgpu_prepare_desc* out) {
    return obgpu_prepare_desc_build(&p, sizeof(P), w, h, hasMap ? 1 : 0, out);
}

// ABI 1.3: post desc for obgpu_render_frame_post (w, h = the crop; params = the crop-shifted Params the CPU
// postProcess would get). obgpu_post_rest_params zeroes the parameters of the stages the GPU did (done bits),
// so lenscore::postProcess with the result runs exactly the rest of the chain.
int obgpu_post_desc_build(const void* params, size_t paramsSize, int w, int h, obgpu_post_desc* out);
// ABI 1.4: flags = OBGPU_POST_REQ_LA34 also puts luminance aberration modes 3 (Contrast Rim v2.1) and 4 (Channel
// Drift) on the GPU; obgpu_post_desc_build = flags 0 (1.3 behaviour: the prefix stops before them).
int obgpu_post_desc_build_ex(const void* params, size_t paramsSize, int w, int h, uint32_t flags, obgpu_post_desc* out);
template <class P>
inline int obgpuPostDescBuildEx(const P& p, int w, int h, uint32_t flags, obgpu_post_desc* out) {
    return obgpu_post_desc_build_ex(&p, sizeof(P), w, h, flags, out);
}
void obgpu_post_rest_params(void* params, size_t paramsSize, uint32_t done);
template <class P>
inline int obgpuPostDescBuild(const P& p, int w, int h, obgpu_post_desc* out) {
    return obgpu_post_desc_build(&p, sizeof(P), w, h, out);
}
template <class P>
inline void obgpuPostRestParams(P& p, uint32_t done) { obgpu_post_rest_params(&p, sizeof(P), done); }

// typed convenience wrapper (the caller includes lenscore.h)
template <class P>
inline int obgpuDescBuild(const P& p, int w, int h, bool needEL, obgpu_blur_desc* out, void** holder) {
    return obgpu_desc_build(&p, sizeof(P), w, h, needEL ? 1 : 0, out, holder);
}
