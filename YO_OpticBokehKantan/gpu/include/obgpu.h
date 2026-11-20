/* obgpu.h - C ABI of the Optic Bokeh GPU blur stage (Rust + wgpu, library "obgpu").
 *
 * Phase 1: replaces lenscore::blurAlpha (Fast / Standard quality, any renderVersion's blur stage)
 * on a GPU. Prepare and post stages stay on the CPU. Every call returns a status; nothing throws,
 * panics or blocks forever (all waits have a timeout and poll the abort callback).
 *
 * Thread safety: one obgpu_context may be used from any number of threads at once (AE multi-frame
 * rendering). Jobs share one device / queue; VRAM is reserved per job against the budget.
 *
 * Typical use (see gpu/INTEGRATION.md):
 *   obgpu_context* g = obgpu_init(&cfg, &st);               // once (global setup), may fail -> CPU only
 *   obgpu_blur_desc d; void* hold;                          // per frame, from the lenscore Params
 *   obgpu_desc_build(&params, sizeof params, prep.w, prep.h, needEL, &d, &hold);   (C++ shim)
 *   st = obgpu_blur(g, &in, &d, &out, abortFn, user);        // != OBGPU_OK -> run the CPU blur
 *   obgpu_desc_free(hold);
 *   obgpu_shutdown(g);                                       // global setdown
 */
#ifndef OBGPU_H
#define OBGPU_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define OBGPU_ABI_VERSION 1u

typedef enum obgpu_status {
    OBGPU_OK = 0,
    OBGPU_BAD_ARGS = 1,       /* invalid input / desc (caller bug) */
    OBGPU_ABORTED = 2,        /* the abort callback returned non-zero */
    OBGPU_NO_ADAPTER = 3,     /* no suitable GPU (or disabled) */
    OBGPU_DEVICE_LOST = 4,    /* device removed / reset / destroyed; the context re-creates it on a later call */
    OBGPU_OUT_OF_VRAM = 5,    /* allocation failed even with tiling at the minimum band */
    OBGPU_TIMEOUT = 6,        /* the GPU did not finish within timeout_ms (context is marked lost) */
    OBGPU_BUSY = 7,           /* could not get a VRAM reservation within wait_ms */
    OBGPU_HOST_OOM = 8,       /* host allocation failed */
    OBGPU_UNSUPPORTED = 9,    /* desc needs something this build does not implement */
    OBGPU_INTERNAL = 10,      /* any other error (validation error, panic caught at the ABI) */
    OBGPU_SELFTEST_FAILED = 11 /* (ABI 1.1) obgpu_selftest found this adapter out of tolerance: the context stays
                                  disabled, every later call returns this status -> CPU */
} obgpu_status;

/* backends bit mask (0 = platform default: DX12 on Windows, Metal on macOS, Vulkan elsewhere) */
#define OBGPU_BACKEND_DX12   1u
#define OBGPU_BACKEND_VULKAN 2u
#define OBGPU_BACKEND_METAL  4u

typedef struct obgpu_config {
    uint32_t struct_size;          /* sizeof(obgpu_config) */
    uint32_t backends;             /* OBGPU_BACKEND_* mask, 0 = default (Windows: Vulkan preferred, DX12 fallback) */
    int32_t  allow_software;       /* 1 = accept CPU / WARP / llvmpipe adapters (tests only) */
    int32_t  force_no_adapter;     /* 1 = pretend there is no adapter (failure-path tests) */
    uint64_t vram_budget_bytes;    /* total VRAM all jobs may hold; 0 = automatic (50 % of the OS budget for this process,
                                      0.75-3 GB; 1.5 GB if unknown; phase 3's 6 GB cap was reverted in phase 5). Each job plans for total / min(running +
                                      waiting jobs, max_jobs) (ABI 1.5.2), at most 60 % of what is free now split the same way, and tiles
                                      (exact row bands) above it */
    uint32_t timeout_ms;           /* max wait for one GPU submission (0 = 4000) */
    uint32_t wait_ms;              /* max wait for a job slot under contention (0 = 15000; abortable). ABI 1.5.2: an
                                      upper bound only - the router gives up earlier when the GPU can no longer win */
    int32_t  max_jobs;             /* max concurrent jobs on the device (0 = 2) */
    int32_t  verbose;              /* 1 = log to stderr */
} obgpu_config;

typedef struct obgpu_context obgpu_context;

/* ---- blur description (filled by obgpu_desc_build from lenscore's own setup code) ---- */
#define OBGPU_MAX_LAYERS 32
#define OBGPU_TIERS 8
#define OBGPU_MAX_LEVELS 8
typedef struct obgpu_profile {
    int32_t nl, share, share_from, _pad;
    double s2, s2out, s2in, _pad2;
    double ls[OBGPU_MAX_LAYERS];
    double lb[OBGPU_MAX_LAYERS];
} obgpu_profile;

typedef struct obgpu_level {
    int32_t f, w, h, gox, goy, max_reach;
    int32_t row_lo, row_hi;   /* coarse rows [row_lo, row_hi) the fine output rows interpolate from */
} obgpu_level;

#define OBGPU_DESC_VERSION 1u
typedef struct obgpu_blur_desc {
    uint32_t struct_size, version;
    int32_t w, h;                       /* prepared buffer size */
    int32_t nk, n_tiers, n_levels, cat, cat_n, clamp_neg, fringe_share, need_el;
    int32_t max_reach, row_begin, row_end;   /* fine output rows [row_begin, row_end) computed, others 0 */
    int32_t ap_n, circ_n, _pad0;
    double ax, ay, xext, yext, cat_k, cat_lim, cat_inv, level_base;
    double fcx, fcy, half_diag, pa, max_radius, smax_g;
    double ap_ymin, ap_ymax, ap_inv_step, ap_area;
    double circ_ymin, circ_ymax, circ_inv_step, circ_area;
    double ks[2][3];                    /* [front 0 / behind 1][R, G, B] kernel scale */
    obgpu_profile prof[2][OBGPU_TIERS]; /* [side][tier] */
    obgpu_level levels[OBGPU_MAX_LEVELS];   /* [1, n_levels) used */
    const double* ap_table;             /* 2*(ap_n+1) doubles: IL, IR interleaved */
    const double* circ_table;           /* 2*(circ_n+1) */
    const double* cat_a;                /* cat_n*cat_n (row j = oy index), or NULL */
} obgpu_blur_desc;

/* ---- job data ---- */
typedef struct obgpu_input {
    uint32_t struct_size;
    const float*  rgba;          /* w*h*4 premultiplied linear (lenscore::Prepared::rgba) */
    const double* radius;        /* w*h (Prepared::radius) */
    const int8_t* side;          /* w*h (Prepared::side) */
    const float*  el;            /* w*h (Prepared::eL), required if desc.need_el */
} obgpu_input;

typedef struct obgpu_output {
    uint32_t struct_size;
    float*  rgba;                /* w*h*4 premultiplied, row stride rgba_stride floats */
    int32_t rgba_stride;         /* floats, >= 4*w */
    float*  el;                  /* w*h, written if desc.need_el */
} obgpu_output;

typedef struct obgpu_stats {     /* optional per-job timings (ms) and sizes */
    double total_ms, convert_ms, upload_ms, gpu_ms, readback_ms, wait_ms;
    uint64_t vram_bytes;         /* peak bytes this job allocated */
    int32_t bands;               /* row bands (1 = untiled) */
    int32_t submissions;
} obgpu_stats;

typedef int (*obgpu_abort_fn)(void* user);   /* non-zero = abort; called on the calling thread */

/* NULL on failure (*status says why). Creating the device compiles no shaders; pipelines are
   compiled on first use per variant (obgpu_warmup compiles the common ones). */
obgpu_context* obgpu_init(const obgpu_config* cfg, obgpu_status* status);
obgpu_status   obgpu_warmup(obgpu_context* ctx);
obgpu_status   obgpu_blur(obgpu_context* ctx, const obgpu_input* in, const obgpu_blur_desc* desc,
                          const obgpu_output* out, obgpu_abort_fn abort_fn, void* user, obgpu_stats* stats);
void           obgpu_shutdown(obgpu_context* ctx);
/* adapter description ("NVIDIA GeForce RTX 4070 SUPER (Dx12, driver ...)"); returns length */
size_t         obgpu_adapter_info(obgpu_context* ctx, char* buf, size_t cap);
const char*    obgpu_status_string(obgpu_status s);
/* test hooks: 1 = destroy the device (simulated device loss), 2 = set the total VRAM budget to `value` bytes
   (0 = config), 3 = force band height `value` rows (0 = auto), 4 = artificial delay per submission `value` ms,
   5 = override timeout_ms (0 = config), 6 = (ABI 1.1) self-test fault injection: scale the GPU result by
   1 + value * 1e-6 before the comparison (proves the self-test rejects an inaccurate adapter),
   7 = (phase 3) own GPU load generator on a second device, submissions of ~value ms (0 = stop),
   8 = (ABI 1.5) print the per-pass GPU time profile collected with the environment variable OBGPU_PROFILE=1
   to stderr and reset it. Test switch (environment): OBGPU_POST_NOREUSE=1 = the phase-4 post buffer layout. */
obgpu_status   obgpu_test_hook(obgpu_context* ctx, int what, uint64_t value);

/* ---- ABI 1.1 (additive): adapter self-test ----
   Renders the embedded tiny reference cases (4 blur-stage cases + 2 whole-frame cases through the GPU prepare
   with White Protection / Dots / Min Detail; prepared input or source + lenscore CPU result, generated by
   `obgpu_harness mkselftest`: chroma fringe + cat eye + tiers, legacy cat eye + coarse levels, Fast fringe
   sharing, obstruction, front + back, soft alpha, a 300x light for the magnitude tiers) twice and compares
   with the CPU result. Pass = every value within |d| <= 5e-4 * L, mean |d| / L <= 2e-5 (L = max(1, max |cpu|
   over the 9x9 neighbourhood): HDR-aware), no NaN, both runs bit-identical. On failure the context is disabled (OBGPU_SELFTEST_FAILED from then on).
   Call once per process after obgpu_init (0.1-0.3 s incl. pipeline compiles on Vulkan). */
typedef struct obgpu_selftest_report {
    uint32_t struct_size;
    int32_t  pass;            /* 1 = within tolerance */
    int32_t  cases;
    int32_t  deterministic;   /* 1 = two runs bit-identical */
    int32_t  failed_case;     /* -1 or the first failing case */
    double   max_rel;         /* max |gpu - cpu| / L over all cases (L: see above) */
    double   mean_abs;        /* worst per-case mean |gpu - cpu| / L */
    double   ms;
    double   seed_ms_per_wu;  /* ABI 1.5: GPU busy ms per work unit measured on the frame cases (router cold-start seed) */
    double   seed_wall_ms, seed_busy_ms;  /* ABI 1.5: wall / GPU busy ms of those cases (start-up contention) */
} obgpu_selftest_report;
obgpu_status   obgpu_selftest(obgpu_context* ctx, obgpu_selftest_report* report);

/* ---- ABI 1.1 (additive): the Fast / Standard frame pipeline on the GPU ----
   obgpu_render_frame runs prepareHighlights (automatic highlight source, renderVersion 2) and the blur stage
   on the GPU with ONE upload (source window + map) and ONE readback (blurred crop + eL). postProcess stays
   on the CPU (measured 5-75 ms with the default look; see gpu/PHASE2.md). Everything the CPU postProcess
   needs from the prepare stage is returned: roll_relevant (Prepared::rollRelevant) and, if requested,
   the prepared colour and eL of the crop (needed only by the veiling glow).
   Inputs: the plugin's padded window (src premultiplied linear RGBA, map single channel) and the crop rect
   the wrapper would pass to cropPrepared; the blur desc is built for the crop (with outRowBegin/End shifted
   into it, exactly as for obgpu_blur) with need_el = (highlightRolloff > 1e-4).
   Unsupported cases return OBGPU_UNSUPPORTED before any GPU work (highlight layer sources, keepExtra
   views, renderVersion 1): run the CPU prepare + obgpu_blur (phase 1) or the CPU path. */
#define OBGPU_PREP_VERSION 1u
typedef struct obgpu_prepare_desc {     /* filled by obgpu_prepare_desc_build (cpp/obgpu_desc.cpp) */
    uint32_t struct_size, version;
    int32_t w, h;                        /* window */
    int32_t has_map, invert_map, hl_on, render_version;
    float focus, map_scale, map_gamma, max_radius;
    float hl_gain, hl_threshold, boost, knee;
    float curve, sel, hsat, ps;          /* ps = clamped pixelScale */
    float adapt_gain; int32_t knee_box_r; int32_t wp_on; float wp_amt;
    int32_t wp_f, wp_gx, wp_gy, wp_gw, wp_gh, wp_rb; float wp_rel; float _pad0;
    float wp_tau[10];
    int32_t det_on, det_n, dmin_on, dmin_n;
    float det_rf[8], det_w[8];           /* per scale: fractional radius, weight */
    float dmin_rf[8], dmin_w[8];
    float dmin_ramp, dots; int32_t dots_on; int32_t margin;   /* margin = lenscore::prepareMargin(p, w, h): halo of a prepare band */
    int32_t dots_r[4];
} obgpu_prepare_desc;

typedef struct obgpu_frame_input {
    uint32_t struct_size;
    const float* src; int32_t src_stride;          /* floats per row, >= 4*w */
    const float* map; int32_t map_stride;          /* NULL = no map */
    int32_t crop_x, crop_y, crop_w, crop_h;        /* the cropPrepared rect inside the window */
} obgpu_frame_input;

typedef struct obgpu_frame_output {
    uint32_t struct_size;
    float* rgba; int32_t rgba_stride;              /* crop_w x crop_h blurred, premultiplied */
    float* el;                                     /* crop_w * crop_h, if desc.need_el */
    float* prepared_rgba;                          /* optional crop_w*crop_h*4: prepared colour (for the CPU glow) */
    int32_t roll_relevant;                         /* out: Prepared::rollRelevant */
    uint32_t stages;                               /* out: OBGPU_STAGE_* bits done on the GPU */
    float* prepared_el;                            /* optional crop_w*crop_h: prepared eL (the CPU glow reads it too) */
} obgpu_frame_output;
#define OBGPU_STAGE_PREPARE 1u
#define OBGPU_STAGE_BLUR    2u

int obgpu_render_frame(obgpu_context* ctx, const obgpu_frame_input* in, const obgpu_prepare_desc* prep,
                       const obgpu_blur_desc* blur, obgpu_frame_output* out, obgpu_abort_fn abort_fn, void* user,
                       obgpu_stats* stats);

/* ---- ABI 1.3 (additive): post stages on the GPU ----
   lenscore::postProcess runs, in this order: glow + vignetting (postRowA) -> halation -> luminance aberration
   -> highlight rolloff -> grain. obgpu_render_frame_post runs a PREFIX of that chain on the device-resident
   blurred crop before the one readback; the wrapper then calls postProcess for the rest with the done stages'
   parameters zeroed (obgpu_post_rest_params, cpp/obgpu_desc.h), so the order is always the CPU's.
   GPU stages: glow + vignetting, halation, luminance aberration modes 0-2 (Soft Bleed / Wide Glow / Bright
   Tint and Custom 0-2), rolloff; with ABI 1.4 (obgpu_post_desc_build_ex + OBGPU_POST_REQ_LA34) also laMode 3
   (Contrast Rim v2.1: Hard Rim / Soft Glow Fringe) and laMode 4 (Channel Drift: the "Contrast Rim" / "Red / Cyan"
   styles). Without the request the prefix stops before modes 3 / 4 (1.3 behaviour). Never on the GPU: grain
   (by design: cheap per-pixel hash noise on the final image, and its exact look is defined on the CPU).
   *post_done = OBGPU_POST_* bits actually run (0 = none, e.g. not enough VRAM: the CPU does all of post);
   out->el is the eL after the GPU stages (what the CPU rolloff needs), out->stages gets OBGPU_STAGE_POST. */
#define OBGPU_STAGE_POST    4u
#define OBGPU_POST_VERSION 1u            /* (ABI 1.4 builds fill version 2: OBGPU_POST_VERSION2 below) */
#define OBGPU_POST_GLOWVIG  1u
#define OBGPU_POST_HALATION 2u
#define OBGPU_POST_LUMAB    4u
#define OBGPU_POST_ROLLOFF  8u
typedef struct obgpu_field_op {          /* lenscore blurField / gaussField of a scalar field, resolved */
    int32_t kind;                        /* 0 none, 1 exact Gaussian (gaussField), 2 frame-aligned 3-box grid (blurFieldBox) */
    int32_t r;                           /* Gaussian: radius ceil(3 sigma) */
    int32_t f, gx, gy, gw, gh;           /* box: grid factor, phase, grid size */
    int32_t rb0, rb1;                    /* box radii; rb1 > 0: the v2.2 variance interpolation between rb0 and rb1 */
    float sigma, t, _pad;
} obgpu_field_op;
typedef struct obgpu_post_desc {         /* filled by obgpu_post_desc_build (cpp/obgpu_prepdesc.cpp) */
    uint32_t struct_size, version;
    int32_t w, h;                        /* the crop */
    uint32_t stages;                     /* OBGPU_POST_* the GPU may run (a prefix of the enabled chain) */
    int32_t roll_candidate;              /* highlightRolloff > 1e-4 (the GPU adds its roll_relevant) */
    float glow_amt, vig;
    int32_t glow_f, glow_gw, glow_gh, glow_gox, glow_goy, glow_rb;
    float fcx, fcy, half_diag, pa;
    float hal_amt, hal_floor, hal_relk, hal_soft, hal_rim, hal_tint[3];
    obgpu_field_op hal_scene, hal_tight, hal_wide;
    int32_t la_mode; float la_amt, la_thr, la_curve, la_hs, la_hi[3];
    obgpu_field_op la_field;
    int32_t roll_v2; float roll_knee;
    /* ---- ABI 1.4 (version 2): luminance aberration modes 3 / 4 on the GPU. Filled only by
       obgpu_post_desc_build_ex(..., OBGPU_POST_REQ_LA34, ...); obgpu_post_desc_build leaves la34 = 0 (1.3 behaviour). */
    int32_t la34;                        /* 0 = not on the GPU, 3 = Contrast Rim v2.1, 4 = Channel Drift */
    /* Channel Drift (applyChannelDrift / driftGeom) */
    int32_t dr_s1, dr_s2;                /* line-protection closing: segment radii (rl = 0: both 0) */
    int32_t dr_v2, dr_sp1, dr_sp2;       /* renderVersion >= 2; band-spreading dilation radii (0 = none) */
    float dr_thr, dr_curve, dr_pres, dr_kstop, dr_len, dr_sm2, dr_lencap;
    float dr_k[3];                       /* signed shift per channel (retreat / advance) */
    obgpu_field_op dr_soft, dr_dir;      /* gaussOrBox(map softness), gaussOrBox(direction sigma) */
    /* Contrast Rim v2.1 (applyContrastRim / contrastFringe / rimGeom) */
    int32_t rim_s1, rim_s2, rim_rn, rim_rc;   /* closing/opening segment radii, region radii */
    int32_t rim_fringe, rim_bright_side, rim_mixb_on, _pad2;
    float rim_amt, rim_thr, rim_curve, rim_hs, rim_ex, rim_mixb, rim_comp, rim_hi[3];
    obgpu_field_op rim_reg, rim_fb, rim_fd, rim_sig[3], rim_s0;
} obgpu_post_desc;
#define OBGPU_POST_VERSION2 2u
#define OBGPU_POST_LA34     16u          /* post_done: the luminance aberration that ran was mode 3 or 4 (with LUMAB) */
#define OBGPU_POST_REQ_LA34 1u           /* obgpu_post_desc_build_ex flag */
/* size of the ABI 1.3 (version 1) desc, still accepted by obgpu_render_frame_post */
#define OBGPU_POST_DESC_V1_SIZE ((size_t)(&((obgpu_post_desc*)0)->la34))
int obgpu_render_frame_post(obgpu_context* ctx, const obgpu_frame_input* in, const obgpu_prepare_desc* prep,
                            const obgpu_blur_desc* blur, const obgpu_post_desc* post, obgpu_frame_output* out,
                            obgpu_abort_fn abort_fn, void* user, obgpu_stats* stats, uint32_t* post_done);

/* ---- ABI 1.2 (additive): adaptive routing + diagnostics ----
   The owner's machines routinely run other GPU software, and with several MFR frames in flight the GPU can be
   saturated. obgpu_route(work) says whether the next frame should go to the GPU (1) or the CPU (0), from
   exponential averages of measured GPU job wall times (every obgpu_render_frame / obgpu_blur updates them; they
   include other applications' load) and of the CPU times the wrapper reports with obgpu_report_cpu, plus the
   job-slot queue. Hysteresis: GPU -> CPU when the predicted GPU time exceeds the CPU's, CPU -> GPU below
   0.75 x (the GPU estimate follows improvements faster than slowdowns); in the GPU state a frame still goes
   to the CPU when the slot queue ahead of it would make it slower. While on the CPU every 8th call probes the
   GPU to refresh its estimate; a probe that hits the deadline doubles that interval (up to 64), a fast GPU
   job resets it.
   Deadline: once a CPU time has been reported, every GPU job (routed or a probe) is abandoned when its wall
   time passes max(1.25 x the predicted CPU time, 1 s) and returns OBGPU_BUSY (render that frame on the CPU),
   so a frame sent to a GPU that another application saturates costs at most ~2.25x a CPU frame and the
   estimate is penalised at once (jobs already in flight then get 0.5 x the CPU time). A frame waits for a busy job slot only once the GPU estimate rests on
   >= 3 samples. The deadline is re-evaluated while the job runs (a CPU report arriving meanwhile
   applies). The deadline covers every GPU entry point (obgpu_blur, obgpu_render_frame[_post]) once the wrapper has
   called obgpu_route or obgpu_report_cpu (1.5.1); a wrapper that calls neither has none (phase-1/2 behaviour).
   Cold start (ABI 1.5): GPU first. Until a CPU time is known every routed frame goes to the GPU (queueing
   for the job slots) under a cold deadline of 4 x the GPU's own time per work unit (before its first sample
   the obgpu_selftest seed), within [2 s, 15 s]; a miss returns OBGPU_BUSY, the wrapper renders that frame on
   the CPU and reports it, and from then on the comparison above applies. One CPU frame is spent on learning
   only when the GPU looks contended (its own time per work unit > 0.6 x the self-test seed; without a seed
   after 2 GPU frames). Call obgpu_selftest at start-up so the seed exists. (1.2-1.4 sent 1 frame to the GPU
   and the rest to the CPU until both sides were known: in MFR that CPU frame dominated short renders.)
   VRAM: an allocation failure / loss halves the per-job budgets (more bands), +5 % per successful job.
   Work units: obgpu_work_units(desc) - the same measure for both sides. */
double         obgpu_work_units(const obgpu_blur_desc* desc);
int            obgpu_route(obgpu_context* ctx, double work_units);
void           obgpu_report_cpu(obgpu_context* ctx, double work_units, double ms);

typedef struct obgpu_counters {
    uint32_t struct_size;
    int32_t  route_gpu;              /* routing state: 1 = GPU */
    uint64_t gpu_jobs, gpu_failed, cpu_routed, probes, cpu_reports;
    double   ema_gpu_ms_per_wu;      /* learned GPU wall ms per work unit (NaN = no sample yet) */
    double   ema_cpu_ms_per_wu;
    double   last_wall_ms;           /* last GPU job, wall (incl. transfers and waits) */
    double   last_gpu_busy_ms;       /* last GPU job, GPU busy time from timestamp queries (NaN = unsupported) */
    double   ema_busy_fraction;      /* busy / wall of GPU jobs: low = GPU shared / the job mostly waited */
    int32_t  active_jobs;            /* GPU jobs running now */
    int32_t  waiting_jobs;           /* callers waiting for a job slot */
    int32_t  timestamps;             /* 1 = the adapter supports timestamp queries */
    uint64_t deadline_aborts;        /* GPU jobs abandoned at the routing deadline (rendered on the CPU) */
    uint64_t timeouts;               /* GPU waits over timeout_ms (abandoned, the device is kept) */
    int32_t  last_prep_bands;        /* last obgpu_render_frame: prepare row bands (1 = untiled) */
    double   last_prep_rows_factor;  /* window rows prepared / rows of one untiled pass (halo recomputation) */
    double   vram_scale;             /* self-tuning factor on the job budgets (1 = none) */
    double   seed_ms_per_wu;         /* ABI 1.5: cold-start seed from obgpu_selftest (0 = none) */
    /* ABI 1.5.2 (appended): OBGPU_BUSY returned while waiting for a job slot, by reason (see obgpu_job_info) */
    uint64_t busy_wait_limit;        /* the wait passed wait_ms, or the queue moved far slower than the GPU's own pace */
    uint64_t busy_queue;             /* the GPU could no longer finish before the routing deadline (1.25 x CPU time) */
    uint64_t busy_saturated;         /* another job hit the deadline / a submission timed out meanwhile */
} obgpu_counters;
obgpu_status   obgpu_get_counters(obgpu_context* ctx, obgpu_counters* out);

/* ---- ABI 1.5.2 (additive): slot wait by prediction, and a per-job diagnostic ----
   Slot wait: a caller waiting for a job slot (MFR: more frames than max_jobs) no longer gives up after a fixed
   wait. It returns OBGPU_BUSY (render that frame on the CPU) when (a) another job hit the routing deadline or a
   submission timed out since it arrived (saturated / stuck GPU), (b) a CPU time is known and its wait so far plus
   the GPU's own time for the job would pass the routing deadline (the GPU can no longer win), or (c) the queue
   moves far slower than the GPU's own pace (wait > max(2 s, 3 x rounds ahead x own time per job)); never later
   than wait_ms (0 = 15 s). The cold-start deadline counts the job's own time from the moment it got its slot
   (1.5.1 counted the slot wait as well). A waiting caller holds no VRAM: each running job plans for
   total / min(running + waiting, max_jobs) (1.5.1: / (running + waiting)). Wrappers should pass wait_ms = 0.
   obgpu_last_job: what happened to the CALLING thread's last obgpu_blur / obgpu_render_frame[_post] call.
   Diagnostics only. */
#define OBGPU_BUSY_NONE         0
#define OBGPU_BUSY_WAIT_LIMIT   1   /* slot wait over the limit (c) */
#define OBGPU_BUSY_QUEUE        2   /* slot wait: GPU no longer expected before the deadline (b) */
#define OBGPU_BUSY_SATURATED    3   /* slot wait: another job hit the deadline / a timeout (a) */
#define OBGPU_BUSY_DEADLINE     4   /* running: routing deadline (1.25 x CPU time from the call) */
#define OBGPU_BUSY_COLD_DEADLINE 5  /* running: cold-start deadline (4 x own estimate, own time) */
typedef struct obgpu_job_info {
    uint32_t struct_size;
    int32_t  status;          /* what the call returned */
    int32_t  busy_reason;     /* OBGPU_BUSY_* when status == OBGPU_BUSY */
    int32_t  ahead;           /* jobs running + waiting when the call arrived */
    int32_t  prep_bands;      /* prepare row bands of a frame job (0 = none) */
    int32_t  _pad;
    double   slot_wait_ms;    /* waiting for a job slot */
    double   run_ms;          /* from getting the slot to the end (0 = never got one) */
    double   total_ms;        /* the whole call */
    double   deadline_ms;     /* routing deadline in force at the end (NaN = none) */
    int32_t  deadline_own;    /* 1 = it counts the job's own time (cold start), 0 = time since the call */
    int32_t  _pad2;
    double   est_gpu_ms;      /* router estimate of the GPU's own time for this job (NaN = none) */
    double   est_cpu_ms;      /* router estimate of the CPU time (NaN = unknown) */
    uint64_t budget_bytes;    /* VRAM planned for the job (0 = never got a slot) */
    double   work_units;
} obgpu_job_info;
obgpu_status   obgpu_last_job(obgpu_context* ctx, obgpu_job_info* out);

#ifdef __cplusplus
}
#endif
#endif
