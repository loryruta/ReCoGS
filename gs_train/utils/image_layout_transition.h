#pragma once

#include <cassert>

namespace gs_train
{
/// Enum describing how the image is stored in memory
enum class ImageLayout { BCHW, BHWC };

namespace detail
{
template <ImageLayout SRC_LAYOUT, ImageLayout DST_LAYOUT, typename T>
__global__ void image_layout_transition_kernel(int B, int C, int H, int W, const T* img, T* out_img)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t b = blockIdx.z;
    if (x >= W || y >= H) return;
    assert(b < B);

    for (int c = 0; c < C; ++c) {
        uint32_t bchw_i = b * C * H * W + c * H * W + y * W + x;
        uint32_t bhwc_i = b * H * W * C + y * W * C + x * C + c;
        uint32_t src_i;
        uint32_t dst_i;
        if constexpr (SRC_LAYOUT == ImageLayout::BCHW) {
            src_i = bchw_i;
        } else if (SRC_LAYOUT == ImageLayout::BHWC) {
            src_i = bhwc_i;
        }
        if constexpr (DST_LAYOUT == ImageLayout::BCHW) {
            dst_i = bchw_i;
        } else if (DST_LAYOUT == ImageLayout::BHWC) {
            dst_i = bhwc_i;
        }
        out_img[dst_i] = img[src_i];
    }
}
} // namespace detail

/// Transition the image from a source layout to another layout (e.g. from BHWC to BCHW)
template <ImageLayout SRC_LAYOUT, ImageLayout DST_LAYOUT, typename T>
void image_layout_transition(int B, int C, int H, int W, const T* img, T* out_img)
{
    static_assert(SRC_LAYOUT != DST_LAYOUT, "SRC_LAYOUT is equal to DST_LAYOUT: transition is pointless");

    dim3 num_blocks{};
    num_blocks.x = div_ceil(W, 32);
    num_blocks.y = div_ceil(H, 32);
    num_blocks.z = B;
    dim3 block_dim{};
    block_dim.x = 32;
    block_dim.y = 32;
    block_dim.z = 1;
    detail::image_layout_transition_kernel<SRC_LAYOUT, DST_LAYOUT, T>
        <<<num_blocks, block_dim>>>(B, C, H, W, img, out_img);
}
} // namespace gs_train
