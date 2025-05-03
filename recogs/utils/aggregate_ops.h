#pragma once

#include <cuda/std/numeric>

namespace recogs
{
template <typename T>
struct Sum {
    static constexpr T k_identity = T{0};

    __forceinline__ __host__ __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct Min {
    static constexpr T k_identity = cuda::std::numeric_limits<T>::max();

    __forceinline__ __host__ __device__ T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};

template <typename T>
struct Max {
    static constexpr T k_identity = cuda::std::numeric_limits<T>::lowest();

    __forceinline__ __host__ __device__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

} // namespace recogs
