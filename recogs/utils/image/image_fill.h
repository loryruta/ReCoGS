#pragma once

namespace gs_train
{
namespace detail
{
template <typename IMAGE>
__global__ static void image_fill_kernel(IMAGE image, typename IMAGE::Value fill_value)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image.width || y >= image.height) return;
    image.set_value(x, y, fill_value);
}
} // namespace detail

template <size_t C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_fill(Image<C, T, MEMORY_LAYOUT>& image, const typename Image<C, T, MEMORY_LAYOUT>::Value& fill_value)
{
    dim3 num_blocks{div_ceil(image.width, 32u), div_ceil(image.height, 32u)};
    dim3 block_dim{32, 32};
    detail::image_fill_kernel<Image<C, T, MEMORY_LAYOUT>><<<num_blocks, block_dim>>>(image, fill_value);
}
} // namespace gs_train
