#pragma once

#include <glm/glm.hpp>

#include "Image.h"

namespace recogs
{
/// Perform separate Sobel convolutions on the X and Y axes and return the derivatives.
__device__ inline void sobel_xy(const Image4fHWC& color_depth, int x, int y, float& out_dx, float& out_dy)
{
    float vals[9];
    // Read neighbor values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int xj = x + j - 1;
            int yi = y + i - 1;
            if (xj >= 0 && xj < color_depth.width && yi >= 0 && yi < color_depth.height) {
                vals[i * 3 + j] = color_depth.value(xj, yi).w;
            } else {
                vals[i * 3 + j] = 0;
            }
        }
    }
    // Compute X/Y convolutions
    out_dx = vals[0] - vals[2] + 2 * vals[3] - 2 * vals[5] + vals[6] - vals[8];
    out_dy = vals[0] + 2 * vals[1] + vals[2] - vals[6] - 2 * vals[7] - vals[8];
}

template <bool DISPLAY>
__global__ void depthmap_to_normalmap_kernel(Image4fHWC color_depth, Image4fHWC out_normal_map)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= color_depth.width || y >= color_depth.height) return;
    float dx, dy;
    sobel_xy(color_depth, x, y, dx, dy);
    glm::vec3 normal = glm::vec3(-dx, -dy, 1.0f);
    normal = glm::normalize(normal);
    float* out_ptr = &out_normal_map.data_d()[(y * color_depth.width + x) * 4];
    if constexpr (DISPLAY) {
        out_ptr[0] = normal.x * 0.5f + 0.5f;
        out_ptr[1] = normal.y * 0.5f + 0.5f;
        out_ptr[2] = normal.z * 0.5f + 0.5f;
    } else {
        out_ptr[0] = normal.x;
        out_ptr[1] = normal.y;
        out_ptr[2] = normal.z;
    }
}

/// Convert the depth values stored in the color-depth buffer to a view-space normal map.
/// \param color_depth    The input color-depth buffer
/// \param out_normal_map The output normal map (can be the same of color depth)
template <bool DISPLAY>
void depthmap_to_normalmap(const Image4fHWC& color_depth, Image4fHWC& out_normal_map, cudaStream_t stream)
{
    dim3 num_blocks{};
    num_blocks.x = div_ceil(color_depth.width, 16u);
    num_blocks.y = div_ceil(color_depth.height, 16u);
    dim3 block_dim = {16, 16};
    depthmap_to_normalmap_kernel<DISPLAY><<<num_blocks, block_dim, 0, stream>>>(color_depth, out_normal_map);
}

} // namespace recogs