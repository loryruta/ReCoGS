#pragma once

#include "Image.h"
#include "image_visit_transform.h"

namespace gs_train
{
/// Convert a depthbuffer to an RGB image by warping depthbuffer values in a log-scale.
template <ImageMemoryLayout MEMORY_LAYOUT>
Image<3, float, MEMORY_LAYOUT> image_depthbuffer_to_rgb(const Image<1, float, MEMORY_LAYOUT>& depthbuffer)
{
    using OutImageT = Image<3, float, MEMORY_LAYOUT>;
    OutImageT depth_rgb = OutImageT::malloc(depthbuffer.width, depthbuffer.height);
    image_visit(depthbuffer, [depth_rgb] __device__(Image1fCHW & depth, int x, int y) mutable {
        const float k_log_base = 10.0f;
        float d = depth.value(x, y).r;
        // Depth to log scale
        d = log2(d + 1.0f) / log2(k_log_base);
        d = min(1.0f, d);
        depth_rgb.set_value(x, y, glm::vec3(d));
        return 0; // TODO temporary until I find a solution
    });
    return depth_rgb;
}
} // namespace gs_train
