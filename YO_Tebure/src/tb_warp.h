// Tebure - projection of a camera pose to the image (exact rotation homography + translation / zoom) and the
// high-quality resampler (premultiplied float RGBA, Catmull-Rom bicubic or Lanczos-3, edge modes, motion-blur
// accumulation), auto-scale and the path-graph overlay. No After Effects dependency.
#pragma once
#include "tb_model.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace tb {

struct RectI
{
    int l = 0, t = 0, r = 0, b = 0;
    int w() const { return r - l; }
    int h() const { return b - t; }
    bool empty() const { return r <= l || b <= t; }
};
RectI Intersect(const RectI& a, const RectI& b);
RectI Union(const RectI& a, const RectI& b);

// Geometry in BUFFER pixels (downsampled layer coordinates). Pixel (x, y) covers [x, x+1) x [y, y+1).
struct Geo
{
    double widthRef = 1920; // full-res reference width (focal length in px and the px@1920 unit)
    double ds = 1;          // downsample factor (0.5 = Half), horizontal
    double dsy = 0;         // vertical downsample factor (0 = same as ds)
    double par = 1;         // pixel aspect ratio (width / height of a pixel)
    double cx = 0, cy = 0;  // pivot / principal point (buffer px, continuous)
    double focal = 35;      // mm (35 mm equivalent: 36 mm wide)
    double autoScale = 1;   // extra zoom about the pivot (edge hiding)
};

struct Mat3
{
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};
// output pixel (continuous buffer coords) -> source pixel (continuous buffer coords)
Mat3 InverseMap(const Pose& p, const Geo& g);
Mat3 Invert(const Mat3& a);
bool Apply(const Mat3& M, double x, double y, double& ox, double& oy);  // false if behind the camera
bool IsIdentity(const Mat3& M, double tol = 1e-12);

// Smallest extra zoom (>= 1) so that for EVERY pose inside the box |pose_i| <= bound_i the output frame `frame`
// (full-res px) samples only inside `content` shrunk by `margin` px. Resolution independent (computed at full
// res with ds = 1). Capped at 8.
double AutoScale(const Pose& bound, const RectI& content, double pivotX, double pivotY, double widthRef, double focal,
                 double margin = 2.0, double par = 1.0);
// The same check for one pose (tests): true if every output corner lands inside content - margin.
bool CoversFrame(const Pose& p, const Geo& g, const RectI& out, const RectI& content, double margin);

enum Edge { EDGE_AUTO = 0, EDGE_REPEAT, EDGE_MIRROR, EDGE_TRANSPARENT, EDGE_CLIP, EDGE_COUNT };
enum Interp { INTERP_BICUBIC = 0, INTERP_LANCZOS3, INTERP_COUNT };

// float buffer WITHOUT zero-initialisation (resize) - the pixel passes overwrite it anyway (0.3 speed)
struct FloatBuf
{
    std::unique_ptr<float[]> p;
    size_t n = 0;
    void resize(size_t k)
    {
        if (k != n)
            p.reset(k ? new float[k] : nullptr);
        n = k;
    }
    void assign(size_t k, float v)
    {
        resize(k);
        std::fill(p.get(), p.get() + k, v);
    }
    float* data() { return p.get(); }
    const float* data() const { return p.get(); }
    size_t size() const { return n; }
    float& operator[](size_t i) { return p[i]; }
    const float& operator[](size_t i) const { return p[i]; }
};

// premultiplied float RGBA image in buffer coordinates
struct Image
{
    RectI rect;
    FloatBuf px;  // rect.w * rect.h * 4
    float* at(int x, int y) { return &px[((size_t)(y - rect.t) * rect.w() + (x - rect.l)) * 4]; }
    const float* at(int x, int y) const { return &px[((size_t)(y - rect.t) * rect.w() + (x - rect.l)) * 4]; }
};

struct WarpJob
{
    const Image* src = nullptr;   // covers (part of) the content
    RectI content;                // the picture's true extent (edge modes act here)
    int edge = EDGE_AUTO;
    int interp = INTERP_BICUBIC;
    std::vector<Mat3> maps;       // one per motion-blur sample (equal weights)
    RectI out;                    // pixels to produce
    int threads = 1;
    std::function<bool()> abort;  // polled per row block; true = stop
    // 0.3: if set, every finished output row (premultiplied RGBA, out.w() pixels, buffer row y = out.t + i) goes here
    // instead of into dst (no full-size float output buffer, one pass less). Called from worker threads.
    std::function<void(int, const float*)> rowOut;
};
// returns false if aborted. dst->rect = job.out.
bool Warp(const WarpJob& job, Image* dst);

// source region needed for the maps over `out` (+ kernel margin), clipped to content
RectI SourceNeeded(const std::vector<Mat3>& maps, const RectI& out, const RectI& content, int edge, int interp);
// output extent of the moved content (transparent, overflowing mode)
RectI ForwardBounds(const std::vector<Mat3>& maps, const RectI& content);

// path graph overlay drawn over a premultiplied image (buffer coords): x / y (px @1920), roll (deg), zoom (%)
void DrawGraph(Image* img, const RectI& frame, double ds, const std::vector<double>& ts, const std::vector<Pose>& path,
               const Pose& bound, double focal, double widthRef);

}  // namespace tb
