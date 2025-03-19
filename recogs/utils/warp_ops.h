#pragma once

#define FULL_MASK 0xFFFFFFFF

namespace gs_train
{
/// Perform a reduction within a warp. We expect all threads to run this function.
/// \return the reduced value for the current lane. The aggregated value is held by the lane 0
template <typename T, typename OP>
__device__ T warp_reduce(T val)
{
    static const OP op; // TODO better constraint between T and OP

    assert(__activemask() == FULL_MASK);

    // Reference:
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        T other_val = __shfl_down_sync(FULL_MASK, val, offset);
        val = op(other_val, val);
    }

    return val;
}

template <typename T>
__device__ T warp_min(T value)
{
    return warp_reduce<T, Min<T>>(value);
}

template <typename T>
__device__ T warp_max(T value)
{
    return warp_reduce<T, Max<T>>(value);
}

template <typename T>
__device__ T warp_sum(T value)
{
    return warp_reduce<T, Sum<T>>(value);
}
} // namespace gslab
