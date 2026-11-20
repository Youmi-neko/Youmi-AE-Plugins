// obgpu_desc.cpp - the GPU blur description from lenscore::describeBlur (the CPU blur stage's own setup:
// same code as blurAlpha, cannot drift), through the shared mapping include\obgpu_lenscore.h (BuildDesc, the
// same code as the wrapper's plugin\ob_gpu_desc.h; tools\gpu_desc_check.exe compared the two for 400 sets).
#include "lenscore.h"
#include "obgpu_desc.h"
#include "obgpu_lenscore.h"
#include <cstring>
#include <memory>
#include <new>

int obgpu_desc_build(const void* params, size_t paramsSize, int w, int h, int needEL, obgpu_blur_desc* out, void** holder) {
    if (!holder) return 1;
    *holder = nullptr;
    if (!params || paramsSize != sizeof(lenscore::Params) || !out || w <= 0 || h <= 0) return 1;
    try {
        lenscore::Params p;
        std::memcpy(&p, params, sizeof p);
        std::unique_ptr<obgpuw::Desc> D(new obgpuw::Desc);
        lenscore::Status st = lenscore::Status::Ok;
        if (!obgpuw::BuildDesc(p, w, h, needEL != 0, *D, nullptr, nullptr, st)) {
            if (st == lenscore::Status::BadArgs) return 1;
            if (st == lenscore::Status::OutOfMemory) return 2;
            return 3;
        }
        *out = D->d;
        *holder = D.release();
        return 0;
    } catch (const std::bad_alloc&) {
        return 2;
    } catch (...) {
        return 3;
    }
}

void obgpu_desc_free(void* holder) { delete static_cast<obgpuw::Desc*>(holder); }
