#pragma once

#include "Image.h"
#include "image_visit_transform.h"

namespace gs_train
{
template <ImageMemoryLayout MEMORY_LAYOUT>
void image_depthbuffer_to_rgb(const Image<1, float, MEMORY_LAYOUT>& depthbuffer,
                              Image<3, float, MEMORY_LAYOUT>& out_colorbuffer,
                              cudaStream_t stream)
{
    CHECK_ARG(depthbuffer.size() == out_colorbuffer.size(), "depthbuffer and out_colorbuffer sizes must match");

    image_visit(
        depthbuffer,
        [out_colorbuffer] __device__(Image1fCHW & depth, int x, int y) mutable {
            const float k_log_base = 10.0f;
            float d = depth.value(x, y).r;
            // Depth to log scale
            d = log2(d + 1.0f) / log2(k_log_base);
            d = min(1.0f, d);
            out_colorbuffer.set_value(x, y, glm::vec3(d));
            return 0; // TODO temporary until I find a solution
        },
        stream);
}

/// Convert a depthbuffer to an RGB image by warping depthbuffer values in a log-scale.
template <ImageMemoryLayout MEMORY_LAYOUT>
Image<3, float, MEMORY_LAYOUT> image_depthbuffer_to_rgb( //
    const Image<1, float, MEMORY_LAYOUT>& depthbuffer,
    cudaStream_t stream)
{
    using OutImageT = Image<3, float, MEMORY_LAYOUT>;
    OutImageT out_colorbuffer = OutImageT::malloc(depthbuffer.width, depthbuffer.height);
    image_depthbuffer_to_rgb(depthbuffer, out_colorbuffer, stream);
    return out_colorbuffer;
}
} // namespace gs_train
