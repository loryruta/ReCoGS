#pragma once

#include "Image.h"
#include "utils/misc_utils.h"

namespace recogs
{
namespace detail
{
template <typename IMAGE, typename CALLBACK>
__global__ static void image_visit_transform_kernel(IMAGE image, CALLBACK callback)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image.width || y >= image.height) return;
    auto ret_val = callback(image, x, y); // TODO doesn't work with void
    if constexpr (std::is_same_v<decltype(ret_val), typename IMAGE::Value>) {
        image.set_value(x, y, ret_val);
    } else { // void or anything else
        callback(image, x, y);
    }
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename CALLBACK>
void image_visit_transform(const Image<C, T, MEMORY_LAYOUT>& image, CALLBACK callback, cudaStream_t stream)
{
    dim3 num_blocks{};
    num_blocks.x = div_ceil(image.width, 32u);
    num_blocks.y = div_ceil(image.height, 32u);
    dim3 block_dim{};
    block_dim.x = 32;
    block_dim.y = 32;
    image_visit_transform_kernel<Image<C, T, MEMORY_LAYOUT>, CALLBACK>
        <<<num_blocks, block_dim, 0, stream>>>(image, callback);
}
} // namespace detail

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename CALLBACK>
void image_visit(const Image<C, T, MEMORY_LAYOUT>& image, CALLBACK callback, cudaStream_t stream)
{
    detail::image_visit_transform(image, callback, stream);
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename CALLBACK>
void image_transform(Image<C, T, MEMORY_LAYOUT>& image, CALLBACK callback, cudaStream_t stream)
{
    detail::image_visit_transform(image, callback, stream);
}
} // namespace recogs
