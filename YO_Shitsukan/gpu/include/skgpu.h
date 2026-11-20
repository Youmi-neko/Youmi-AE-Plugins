// Shitsukan GPU path (gpu\src\lib.rs). Every non-OK status: run the CPU stage for this frame.
// 0.6.0.5: one GPU frame per rendered frame (pooled memory, inputs uploaded once, results kept between the stages).
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { SKGPU_OK = 0, SKGPU_BUSY = 1, SKGPU_LOST = 2, SKGPU_OOM = 3, SKGPU_ERROR = 4 };
typedef struct SkGpuCtx SkGpuCtx;
typedef struct SkGpuFrame SkGpuFrame;
typedef struct SkLinesArgs
{
    int32_t w, h, rc, line_color, bg_detail, rb;
    float thr, thr_a;     // line threshold; 0.5f / 255.f (transparent)
    float crf1[3];        // RFPlain coefficients (sigma 7 ds, 3 iterations)
    float crf4[3];        // RF4 coefficients (sigma rc + 1, 3 iterations)
    const float* c;       // w*h*3 fill colour (the CPU's C before the in-paint)
    const float* a;       // w*h alpha
    float* out_alpha;     // w*h line coverage (final, after clutter / transparency)
    uint8_t* out_light;   // w*h light-line flags (NULL when not wanted)
    float* out_fh;        // w*h*3 the in-paint fill (SmoothNorm3 result)
} SkLinesArgs;
// the painter's tail: sk_paint.inc (5b) wet-in-wet, (5) density, (6) slivers, then sk_core.cpp's painter paper
typedef struct SkTailArgs
{
    int32_t stop;          // 1: through the wet-in-wet mix (out_dw, out_dm); 3: through the paper (out_p)
                           // 0.6.0.6 (the self-test): 2 through (5) density (out_dw); 4 through (6) slivers (out_dw);
                           // 5 the paper's blurred line coverage (out_dm, w*h)
    int32_t wet_on;        // run (5b)
    int32_t pa_blur;       // blur the line coverage for the paper's small-feature weight (sigma 4 ds > 0.05)
    float cwet[3], crim[3], cpa[3];  // the recursive filters' coefficients
    const uint8_t* uni;    // tail.wgsl TU after "ps" (640 bytes)
    uint32_t uni_bytes;
    const double* d64;     // tail.wgsl d64 from index 339 on
    uint32_t d64_count;
    const float* grids;    // the small grids (working-grid planes, noise grids, gradient table)
    uint32_t grids_count;
    const float* a;        // w*h layer alpha (used when the frame has none resident)
    const float* alpha;    // w*h line coverage or NULL (used when the frame has none resident)
    const float *dw, *dm, *p;                     // density (3), mean density, the fill (3)
    const float *ev, *wet, *bz, *dl, *pw, *dline, *ptex;  // dl / dline / ptex may be NULL
    float *out_dw, *out_dm, *out_p;
    int32_t ink_on;        // then sk_core.cpp 5. ink (no thickness change) over the painted colour, before out_p
    int32_t ink_soft;      // the soft line blur (RFPlain<3>, coefficients cink)
    float cink[3];
    const float* delta;    // w*h*3 the line densities
    const float* far_w;    // w*h depth far weight or NULL (no depth line scaling)
} SkTailArgs;
SkGpuCtx* skgpu_create(void);
int32_t skgpu_has_tail(SkGpuCtx* ctx);
SkGpuFrame* skgpu_frame_begin(SkGpuCtx* ctx, int32_t w, int32_t h, uint32_t wait_ms, int32_t* status);
int32_t skgpu_frame_lines(SkGpuFrame* fr, const SkLinesArgs* args);
int32_t skgpu_frame_tail(SkGpuFrame* fr, const SkTailArgs* args);
void skgpu_frame_end(SkGpuFrame* fr, int32_t ok);
int32_t skgpu_lost(SkGpuCtx* ctx);
void skgpu_trim(SkGpuCtx* ctx);
// 0.6.0.6: adapter / driver / exp self-test, one line for the diagnostic log; returns the length
int32_t skgpu_describe(SkGpuCtx* ctx, char* buf, uint32_t len);
#ifdef __cplusplus
}
#endif
