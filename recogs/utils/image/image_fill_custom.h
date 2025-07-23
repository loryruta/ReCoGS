#pragma once

// Routine used to test window resize

#include "Image.h"

namespace recogs
{
namespace detail
{
template <typename IMAGE>
__global__ void image_fill_custom_kernel(IMAGE image, int x_num, int y_num, typename IMAGE::Value fill_value)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= min(x_num, image.width) && y <= min(y_num, image.height)) {
        image.set_value(x, y, fill_value);
    }
}
} // namespace detail

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_fill_custom(Image<C, T, MEMORY_LAYOUT>& image,
                       int x_num,
                       int y_num,
                       typename Image<C, T, MEMORY_LAYOUT>::Value fill_value,
                       cudaStream_t stream)
{
    dim3 num_blocks{};
    num_blocks.x = div_ceil(image.width, 32);
    num_blocks.y = div_ceil(image.height, 32);
    dim3 block_dim{32, 32};
    detail::image_fill_custom_kernel<Image<C, T, MEMORY_LAYOUT>>
        <<<num_blocks, block_dim, 0, stream>>>(image, x_num, y_num, fill_value);
}
} // namespace recogs
