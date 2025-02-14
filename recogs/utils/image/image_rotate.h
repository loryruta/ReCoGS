#pragma once

#include "Image.h"
#include "utils/misc_utils.h"

namespace gs_train
{
namespace detail
{
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
__global__ void image_rotate_90_clockwise_kernel(Image<C, T, MEMORY_LAYOUT> src_image,
                                                 Image<C, T, MEMORY_LAYOUT> dst_image)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src_image.width || y >= src_image.height) return;
    typename Image<C, T, MEMORY_LAYOUT>::Value val = src_image.value(x, y);
    dst_image.set_value(src_image.height - y - 1, x, val);
}
} // namespace detail

/// Rotate \c src_image 90 degrees clockwise and store the result in \c dst_image.
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_rotate_90_clockwise(const Image<C, T, MEMORY_LAYOUT>& src_image, Image<C, T, MEMORY_LAYOUT> dst_image)
{
    CHECK_ARG(dst_image.width == src_image.height && dst_image.height == src_image.width,
              "dst_image must be %dx%d",
              src_image.height,
              src_image.width);

    dim3 num_blocks;
    num_blocks.x = div_ceil(src_image.width, 32u);
    num_blocks.y = div_ceil(src_image.height, 32u);
    dim3 block_dim{32, 32};
    detail::image_rotate_90_clockwise_kernel<<<num_blocks, block_dim>>>(src_image, dst_image);
}
} // namespace gs_train
