#pragma once

#include "Image.h"
#include "utils/misc_utils.h"

namespace recogs
{
namespace detail
{
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
__global__ void image_downsample_kernel( //
    Image<C, T, MEMORY_LAYOUT> image,
    int num_downsample,
    Image<C, T, MEMORY_LAYOUT> out_image)
{
    uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
    int scale = 1 << num_downsample;
    int ow = image.width / scale;
    int oh = image.height / scale;
    if (ox >= ow || oy >= oh) return;

    glm::vec<C, float> sum_val{};
    float n = 0.f;
    for (uint32_t ix = ox * scale; ix < (ox + 1) * scale; ++ix) {
        for (uint32_t iy = oy * scale; iy < (oy + 1) * scale; ++iy) {
            if (ix < image.width && iy < image.height) {
                auto val = image.value(ix, iy);
                sum_val += glm::vec<C, float>(val);
                n += 1.f;
            }
        }
    }
    auto avg_val = typename Image<C, T, MEMORY_LAYOUT>::Value{sum_val / n};
    out_image.set_value(ox, oy, avg_val);
}
} // namespace detail

/// Downsample the input image by subdividing it \c num_downsample times.
/// The output image will be \c 2^num_downsample times smaller than the input image.
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_downsample(const Image<C, T, MEMORY_LAYOUT>& image,
                      int num_downsample,
                      Image<C, T, MEMORY_LAYOUT>& out_image)
{
    int scale = 1 << num_downsample;
    int out_w = image.width / scale;
    int out_h = image.height / scale;
    CHECK_ARG(out_image.width == out_w && out_image.height == out_h, "out_image must be %dx%d", out_w, out_h);
    CHECK_ARG(num_downsample >= 1 && num_downsample <= 3, "num_downsample must be >= 1 and <= 3");

    dim3 num_blocks{};
    num_blocks.x = div_ceil(out_w, 32);
    num_blocks.y = div_ceil(out_h, 32);
    dim3 block_dim{};
    block_dim.x = 32;
    block_dim.y = 32;
    detail::image_downsample_kernel<<<num_blocks, block_dim>>>(image, num_downsample, out_image);
}
} // namespace recogs
