#pragma once
#include "Image.h"

namespace gs_train
{
namespace detail
{
template <typename IMAGE, typename CALLBACK>
__global__ static void image_visit_transform_kernel(IMAGE image)
{
    using CallbackResult = typename std::result_of<CALLBACK>::type;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image.width || y >= image.height) return;
    typename IMAGE::Value val = image.value(x, y);
    if constexpr (std::is_same_v<CallbackResult, typename IMAGE::Value>) {
        image.set_value(x, y, CALLBACK(val));
    } else { // void or anything else
        CALLBACK(val);
    }
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename CALLBACK>
void image_visit_transform(const Image<C, T, MEMORY_LAYOUT>& image)
{
    dim3 num_blocks{};
    num_blocks.x = div_ceil(image.width, 32);
    num_blocks.y = div_ceil(image.height, 32);
    dim3 block_dim{};
    block_dim.x = 32;
    block_dim.y = 32;
    image_visit_transform_kernel<Image<C, T, MEMORY_LAYOUT>, CALLBACK><<<num_blocks, block_dim>>>(image);
}
} // namespace detail

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename CALLBACK>
void image_visit(const Image<C, T, MEMORY_LAYOUT>& image)
{
    using CallbackResultT = typename std::result_of<CALLBACK>::type;
    static_assert(std::is_same_v<CallbackResultT, void>, "CALLBACK must return void");
    image_visit_transform(image);
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename CALLBACK>
void image_transform(const Image<C, T, MEMORY_LAYOUT>& image)
{
    using CallbackResultT = typename std::result_of<CALLBACK>::type;
    static_assert(std::is_same_v<CallbackResultT, void>, "CALLBACK must return Image::Value");
    image_visit_transform(image);
}
} // namespace gs_train
