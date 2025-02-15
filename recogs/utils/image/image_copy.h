#pragma once

#include "Image.h"
#include "utils/AABB.h"

namespace gs_train
{
namespace detail
{
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
__global__ void image_copy_kernel(Image<C, T, MEMORY_LAYOUT> src_image,
                                  AABB2i src_region,
                                  Image<C, T, MEMORY_LAYOUT> dst_image,
                                  glm::ivec2 dst_pos)
{
    uint32_t rx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ry = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t src_rx = rx + src_region.min.x;
    uint32_t src_ry = ry + src_region.min.y;
    if (src_rx >= src_region.max.x || src_ry >= src_region.max.y) return;

    auto val = src_image.value(src_rx, src_ry);
    dst_image.set_value(dst_pos.x + rx, dst_pos.y + ry, val);
}
} // namespace detail

/// Copy the \c src_region from \c src_image to \c dst_image starting from \c dst_pos (w.r.t. top-left corner).
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_copy( //
    const Image<C, T, MEMORY_LAYOUT>& src_image,
    const AABB2i& src_region,
    Image<C, T, MEMORY_LAYOUT>& dst_image,
    const glm::ivec2& dst_pos)
{
    AABB2i src_image_aabb(glm::ivec2(0), src_image.size());
    AABB2i dst_image_aabb(glm::ivec2(0), dst_image.size());
    AABB2i dst_region(dst_pos, dst_pos + src_region.size());
    CHECK_ARG(src_image_aabb.contains(src_region), "src_region must be within src_image");
    CHECK_ARG(dst_image_aabb.contains(dst_region), "dst_pos must be within dst_image");
    dim3 num_blocks{};
    num_blocks.x = div_ceil(src_region.max.x - src_region.min.x, 32);
    num_blocks.y = div_ceil(src_region.max.y - src_region.min.y, 32);
    dim3 block_dim{};
    block_dim.x = 32;
    block_dim.y = 32;
    detail::image_copy_kernel<<<num_blocks, block_dim>>>(src_image, src_region, dst_image, dst_pos);
}

/// Copy the \c src_region from \c src_image to \c dst_image starting from the top-left corner.
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_copy( //
    const Image<C, T, MEMORY_LAYOUT>& src_image,
    const AABB2i& src_region,
    Image<C, T, MEMORY_LAYOUT>& dst_image)
{
    image_copy(src_image, src_region, dst_image, glm::ivec2{} /* dst_pos */);
}

/// Copy the entire \c src_image to \c dst_image.
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
void image_copy( //
    const Image<C, T, MEMORY_LAYOUT>& src_image,
    Image<C, T, MEMORY_LAYOUT>& dst_image)
{
    CHECK_ARG(src_image.size() == dst_image.size(), "src_image size must match dst_image");
    AABB2i src_region(glm::ivec2(0), src_image.size());
    image_copy(src_image, src_region, dst_image, glm::ivec2{} /* dst_pos */);
}

} // namespace gs_train