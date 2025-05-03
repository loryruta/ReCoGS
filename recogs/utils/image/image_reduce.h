#pragma once

#include <cuda/std/numeric>

#include "Image.h"
#include "utils/aggregate_ops.h"
#include "utils/cuda_utils.h"
#include "utils/warp_ops.h"

namespace recogs
{
namespace detail
{
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename OP>
__global__ void _image_reduce_kernel(const Image<C, T, MEMORY_LAYOUT> src, Image<C, T, MEMORY_LAYOUT> dst)
{
    using Value = typename Image<C, T, MEMORY_LAYOUT>::Value;

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t lane_id = get_lane_id();
    uint32_t warp_id = get_warp_id();

    Value val;
    if (x < src.width && y < src.height) {
        val = src.value(x, y);
    } else {
        val = Value{OP::k_identity};
    }

    /* Warp-level reduction */
    // Within a block, consider all its 32 warps and store their reduced value in shared memory
    __shared__ Value warp_vals[32];
#pragma unroll
    for (int c = 0; c < C; ++c) {
        warp_vals[warp_id][c] = warp_reduce<T, OP>(val[c]);
    }
    __syncthreads();

    /* Block-level reduction */
    // Within a block reduce the gathered per-warp reductions.
    // Only warp = 0 is charged of performing the reduction
    if (warp_id == 0) {
        Value block_val{};
#pragma unroll
        for (int c = 0; c < C; ++c) {
            T warp_val = warp_vals[lane_id][c];
            block_val[c] = warp_reduce<T, OP>(warp_val);
        }
        dst.set_value(blockIdx.x, blockIdx.y, block_val);
    }
}
} // namespace detail

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT, typename OP>
typename Image<C, T, MEMORY_LAYOUT>::Value image_reduce( //
    const Image<C, T, MEMORY_LAYOUT>& image,
    cudaStream_t stream)
{
    const uint32_t w = image.width;
    const uint32_t h = image.height;

    Image<C, T, MEMORY_LAYOUT> src_ref = image.create_ref();
    uint32_t dst_w = div_ceil(image.width, 32u);
    uint32_t dst_h = div_ceil(image.height, 32u);
    Image<C, T, MEMORY_LAYOUT> dst = Image<C, T, MEMORY_LAYOUT>::malloc(dst_w, dst_h, stream);

    while (true) {
//        printf("[DEBUG] [Image/reduce] Reducing (%d %d) -> (%d %d)\n", src.width, src.height, dst.width, dst.height);

        dim3 num_blocks;
        num_blocks.x = dst.width;
        num_blocks.y = dst.height;
        dim3 block_dim{32, 32};
        // TODO src_ref writing within itself (src_ref == dst!!!); Why it's working???
        detail::_image_reduce_kernel<C, T, MEMORY_LAYOUT, OP><<<num_blocks, block_dim, 0, stream>>>(src_ref, dst);

        if (dst.width == 1 && dst.height == 1) break;

        src_ref = dst.create_ref();
        dst.width = glm::max(src_ref.width >> 5, 1u);
        dst.height = glm::max(src_ref.height >> 5, 1u);
    }

    using Value = typename Image<C, T, MEMORY_LAYOUT>::Value;
    Value result{};
    if constexpr (MEMORY_LAYOUT == ImageMemoryLayout::CHW) {
        for (int c = 0; c < C; ++c) {
            size_t addr = c * h * w;
            CHECK_CUDA(cudaMemcpyAsync(&result[c], dst.data_d() + addr, sizeof(T), cudaMemcpyDeviceToHost, stream));
        }
    } else if constexpr (MEMORY_LAYOUT == ImageMemoryLayout::HWC) {
        size_t addr = 0;
        CHECK_CUDA(cudaMemcpyAsync(
            reinterpret_cast<void*>(&result), dst.data_d() + addr, sizeof(Value), cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return result;
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
__host__ typename Image<C, T, MEMORY_LAYOUT>::Value image_sum( //
    const Image<C, T, MEMORY_LAYOUT>& image,
    cudaStream_t stream)
{
    return image_reduce<C, T, MEMORY_LAYOUT, Sum<T>>(image, stream);
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
__host__ typename Image<C, T, MEMORY_LAYOUT>::Value image_min( //
    const Image<C, T, MEMORY_LAYOUT>& image,
    cudaStream_t stream)
{
    return image_reduce<C, T, MEMORY_LAYOUT, Min<T>>(image, stream);
}

template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT>
__host__ typename Image<C, T, MEMORY_LAYOUT>::Value image_max( //
    const Image<C, T, MEMORY_LAYOUT>& image,
    cudaStream_t stream)
{
    return image_reduce<C, T, MEMORY_LAYOUT, Max<T>>(image, stream);
}
} // namespace recogs
