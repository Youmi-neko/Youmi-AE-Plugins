// mac_port: stand-in for the GPU engine library (gpu\, Rust + wgpu) in builds without it (CMake option MBB_GPU=OFF,
// the default on macOS for the first port). Every call reports "no usable GPU", which is exactly the state the real
// library reaches on an adapter without 64-bit float shaders (every Apple GPU under Metal): the plug-in then renders
// every frame with the CPU engine (MapBilinearBlur.cpp: any status other than MBBGPU_OK -> CPU), so the picture is
// the CPU reference. Not used on Windows (the release links gpu\target\static\release\mbbgpu.lib).
#include "mbbgpu.h"

extern "C" {
void mbbgpu_init_async(void) {}
int mbbgpu_wait_ready(void) { return MBBGPU_NO_ADAPTER; }
int mbbgpu_wait_init(uint32_t, mbbgpu_abort_fn, void*) { return MBBGPU_NO_ADAPTER; }
void mbbgpu_set_slot_wait_ms(uint32_t) {}
double mbbgpu_last_slot_wait_ms(void) { return 0.0; }
double mbbgpu_init_ms(void) { return -1.0; }
void mbbgpu_test_init_delay_ms(uint32_t) {}
int mbbgpu_render(const mbbgpu_desc*, const float*, const float*, float*, mbbgpu_abort_fn, void*) { return MBBGPU_NO_ADAPTER; }
const char* mbbgpu_info(void) { return "GPU engine not built into this binary (CPU only)"; }
const char* mbbgpu_status_string(int status)
{
    switch (status)
    {
    case MBBGPU_OK: return "ok";
    case MBBGPU_NO_ADAPTER: return "no usable GPU (GPU engine not built in)";
    default: return "GPU engine not built in";
    }
}
void mbbgpu_shutdown(void) {}
void mbbgpu_test_hook(int) {}
}
