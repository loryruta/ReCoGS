#pragma once

#include "Image.h"
#include "utils/misc_utils.h"

namespace recogs
{
namespace detail
{
template <typename SRC_IMAGE, typename DST_IMAGE>
__global__ static void image_cast_kernel(SRC_IMAGE src_image, DST_IMAGE dst_image)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src_image.width || y >= src_image.height) return;
    typename SRC_IMAGE::Value src_val = src_image.value(x, y);
    typename DST_IMAGE::Value dst_val;
#pragma unroll
    for (int c = 0; c < glm::min(SRC_IMAGE::k_components, DST_IMAGE::k_components); ++c) {
        dst_val[c] = (typename DST_IMAGE::ValueType)(src_val[c]);
    }
    dst_image.set_value(x, y, dst_val);
}
} // namespace detail

template <int SRC_C,
          typename SRC_T,
          ImageMemoryLayout SRC_LAYOUT,
          int DST_C,
          typename DST_T,
          ImageMemoryLayout DST_LAYOUT>
void image_cast( //
    const Image<SRC_C, SRC_T, SRC_LAYOUT>& src_image,
    Image<DST_C, DST_T, DST_LAYOUT>& dst_image,
    cudaStream_t stream)
{
    CHECK_ARG(src_image.size() == dst_image.size(), "src_image must have the same dim of dst_image");

    dim3 num_blocks{};
    num_blocks.x = div_ceil<uint32_t>(src_image.width, 32);
    num_blocks.y = div_ceil<uint32_t>(src_image.height, 32);
    dim3 block_dim{};
    block_dim.x = 32;
    block_dim.y = 32;
    detail::image_cast_kernel<<<num_blocks, block_dim, 0, stream>>>(src_image, dst_image);
}
} // namespace recogs
