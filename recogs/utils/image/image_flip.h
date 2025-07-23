#pragma once

#include "Image.h"

namespace recogs
{
namespace detail
{
template <typename IMAGE>
__global__ static void image_flip_x_kernel(IMAGE image)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= (image.width >> 1) || y >= image.height) return;

    auto v0 = image.value(x, y);
    auto v1 = image.value(image.width - x - 1, y);
    image.set_value(x, y, v1);
    image.set_value(image.width - x - 1, y, v0);
}

template <typename IMAGE>
__global__ static void image_flip_y_kernel(IMAGE image)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= image.width || y >= (image.height >> 1)) return;

    auto v0 = image.at(x, y);
    auto v1 = image.at(x, image.height - y - 1);
    image.set_value(x, y, v1);
    image.set_value(x, image.height - y - 1, v0);
}
} // namespace detail

/// Flip the given image horizontally
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_flip_x(const Image<C, T, MEMORY_LAYOUT>& image)
{
    dim3 num_blocks{};
    num_blocks.x = div_ceil(image.width >> 1, 32);
    num_blocks.y = div_ceil(image.height >> 1, 32);
    dim3 block_dim{32, 32};
    detail::image_flip_x_kernel<<<num_blocks, block_dim>>>(image);
}

/// Flip the given image vertically
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_flip_y(const Image<C, T, MEMORY_LAYOUT>& image)
{
    dim3 num_blocks{};
    num_blocks.x = div_ceil(image.width, 32);
    num_blocks.y = div_ceil(image.height >> 1, 32);
    dim3 block_dim{32, 32};
    detail::image_flip_y_kernel<<<num_blocks, block_dim>>>(image);
}

} // namespace recogs
