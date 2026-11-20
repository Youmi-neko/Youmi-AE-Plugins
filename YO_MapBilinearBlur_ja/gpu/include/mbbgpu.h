/* Map Bilinear Blur - GPU engine (Rust + wgpu, static library). C ABI.
 * The GPU runs the same algorithm as src/mbb_core.cpp (the CPU engine stays the reference and the
 * fallback). The C++ side describes the frame (ladder, level ranges, settings: mbb::DescribeGpu) and
 * hands over the engine's level-0 buffer; the library returns the premultiplied RGBA result over O.
 * Any status other than MBBGPU_OK / MBBGPU_ABORTED means: render this frame on the CPU. */
#ifndef MBBGPU_H
#define MBBGPU_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum
{
    MBBGPU_OK = 0,
    MBBGPU_NO_ADAPTER = 1,  /* no usable GPU (or no 64-bit float support): CPU for the session */
    MBBGPU_DEVICE_LOST = 2, /* driver reset / removal: the device is re-created on a later call */
    MBBGPU_OUT_OF_VRAM = 3, /* the frame does not fit the VRAM budget */
    MBBGPU_BUSY = 4,        /* too many concurrent GPU jobs (MFR): this frame on the CPU */
    MBBGPU_TIMEOUT = 5,
    MBBGPU_ABORTED = 6,     /* the abort callback asked to stop */
    MBBGPU_BAD_ARGS = 7,
    MBBGPU_UNSUPPORTED = 8,
    MBBGPU_INTERNAL = 9,
    MBBGPU_DISABLED = 10,   /* switched off (too many losses, or a failed self test) */
    MBBGPU_NOT_READY = 11   /* still initialising in the background */
};

#define MBBGPU_MAX_ENTRIES 32

typedef struct
{
    int32_t x0, y0, w, h, extW, extH; /* stored cell range of a level (relative to D) and its extent */
} mbbgpu_range;

typedef struct
{
    uint32_t size; /* sizeof(mbbgpu_desc) */
    int32_t D[4], O[4], O1[4], R0[4]; /* l, t, r, b (layer coords at the current downsample) */
    int32_t interp, downSmooth, blend, antiRing, protect, clip;
    uint32_t ditherSeed;
    int32_t ditherFrame;
    int32_t sharpType, ring, hasAmount, top, Lp;
    int32_t pad0;
    double radius, ds, sharpAmount, sharpRadius;
    int32_t lev[MBBGPU_MAX_ENTRIES];
    double var[MBBGPU_MAX_ENTRIES];
    mbbgpu_range rg[MBBGPU_MAX_ENTRIES]; /* L = 0 .. Lp */
} mbbgpu_desc;

/* Starts device creation + pipeline compilation in the background (idempotent). */
void mbbgpu_init_async(void);
/* Blocks until initialisation finished; returns MBBGPU_OK or the init failure. */
int mbbgpu_wait_ready(void);
typedef int (*mbbgpu_abort_fn)(void* user); /* non-zero = stop; called on the calling thread only */
/* v0.3.1: waits while the device is being created, at most until window_ms after that initialisation began
 * (starts it if needed, also after a device loss / once more after a failed attempt). Returns MBBGPU_OK,
 * MBBGPU_NOT_READY (window over), MBBGPU_ABORTED, or why the GPU cannot be used (NO_ADAPTER, DISABLED, INTERNAL). */
int mbbgpu_wait_init(uint32_t window_ms, mbbgpu_abort_fn abort, void* user);
/* v0.3.1: how long a job waits for a GPU slot (MFR) before it is MBBGPU_BUSY (default 2000 ms) */
void mbbgpu_set_slot_wait_ms(uint32_t ms);
/* v0.3.1: the slot wait (ms) of this thread's last mbbgpu_render call; the last initialisation's duration (ms, -1 none) */
double mbbgpu_last_slot_wait_ms(void);
double mbbgpu_init_ms(void);
/* v0.3.1, tests: delay every following initialisation by ms */
void mbbgpu_test_init_delay_ms(uint32_t ms);
/* base5: (q*r, q*g, q*b, q*a, q) per R0 pixel; amount: signed blur amount per R0 pixel or NULL;
 * out: premultiplied RGBA over O (4 floats per pixel, dense). */
int mbbgpu_render(const mbbgpu_desc* desc, const float* base5, const float* amount, float* out, mbbgpu_abort_fn abort,
                  void* user);
/* adapter name / status text (UTF-8, static storage) */
const char* mbbgpu_info(void);
const char* mbbgpu_status_string(int status);
void mbbgpu_shutdown(void);
/* tests: 0 none, 1 = pretend no adapter, 2 = every job BUSY, 3 = every job OUT_OF_VRAM, 4 = lose the device on the next job,
 * 5 = the next initialisation fails (INTERNAL) once, 6 = every initialisation fails (v0.3.1; set before the first init) */
void mbbgpu_test_hook(int hook);

#ifdef __cplusplus
}
#endif
#endif
