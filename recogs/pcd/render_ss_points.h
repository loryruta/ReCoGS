#pragma once

#include <nvfunctional>
#include <optional>

#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "GSCamera.h"
#include "utils/image/Image.h"

namespace recogs
{
template <typename CALLBACK>
void render_ss_points(const thrust::device_vector<glm::vec3>& ws_points,
                      const GSCamera& camera,
                      int r,
                      const Image4fHWC& depthmap,
                      bool test_depth,
                      CALLBACK callback,
                      cudaStream_t stream)
{
    if (ws_points.empty()) return;

    auto f = [depthmap, r, test_depth, callback] __device__(uint32_t point_idx, int2 pixel, float view_z) mutable {
        int W = (int) depthmap.width;
        int H = (int) depthmap.height;
        uint32_t z = __float_as_uint(view_z);
        for (int ry = pixel.y - r; ry <= pixel.y + r; ++ry) {
            for (int rx = pixel.x - r; rx <= pixel.x + r; ++rx) {
                if (rx < 0 || rx >= W || ry < 0 || ry >= H) continue;
                // Depth test
                if (test_depth) {
                    uint32_t* depth_ptr = reinterpret_cast<uint32_t*>(&depthmap.data_d()[(ry * W + rx) * 4 + 3]);
                    uint32_t old_z = atomicMin(depth_ptr, z);
                    if (old_z <= z) continue;
                }
                callback(point_idx, int2(rx, ry), view_z);
            }
        }
    };
    project_points(ws_points, camera, f, stream);
}
} // namespace recogs
