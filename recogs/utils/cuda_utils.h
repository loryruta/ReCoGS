#pragma once

#include <cstdio>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "utils/misc_utils.h"

#define CHECK_CUDA(_error) recogs::check_cuda_error(_error, __FILE_NAME__, __LINE__)

#define RCGS_TPTR(_x) thrust::raw_pointer_cast((_x).data())

namespace recogs
{
namespace detail
{
template <typename FUNCTION>
__global__ void dispatch_single_thread_kernel(FUNCTION function)
{
    function();
}
} // namespace detail

inline void check_cuda_error(cudaError_t error, char const* file, int line)
{
    if (RCGS_UNLIKELY(error != cudaSuccess)) {
        fprintf(stderr, "[ERROR] CUDA error: %s (%s:%d)\n", cudaGetErrorString(error), file, line);
        exit(1);
    }
}

template <typename T>
inline T to_host(const T* ptr_d)
{
    T result;
    CHECK_CUDA(cudaMemcpy(&result, ptr_d, sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

template <typename T>
inline std::vector<T> to_host_vector(const T* ptr_d, size_t num_elements)
{
    std::vector<T> result(num_elements);
    result.resize(num_elements);
    CHECK_CUDA(cudaMemcpy(result.data(), ptr_d, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

template <typename T>
inline T* to_device(const T& value)
{
    T* ptr_d;
    CHECK_CUDA(cudaMalloc(&ptr_d, sizeof(T)));
    CHECK_CUDA(cudaMemcpy(ptr_d, &value, sizeof(T), cudaMemcpyHostToDevice));
    return ptr_d;
}

template <typename T>
inline T* to_device_array(std::span<T> values)
{
    T* ptr_d;
    CHECK_CUDA(cudaMalloc(&ptr_d, values.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(ptr_d, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr_d;
}

template <typename T>
inline void dump_device_buffer(const T* buffer, size_t num_elements, const std::filesystem::path& out_filepath)
{
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<T> buffer_data = to_host_vector<T>(buffer, num_elements);
    FILE* f = fopen(out_filepath.c_str(), "wt");
    CHECK_STATE(f);

    char entry[256];
    for (int i = 0; i < num_elements; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            sprintf(entry, "%.15f,\n", i, buffer_data[i]);
        } else if constexpr (std::is_integral_v<T>) {
            sprintf(entry, "%d,\n", i, buffer_data[i]);
        }
        fputs(entry, f);
    }
    fclose(f);

    printf("[DEBUG] [cuda_utils] Buffer written to: %s\n", out_filepath.c_str());
}

template <typename FUNCTION>
void dispatch_single_thread(FUNCTION function)
{
    detail::dispatch_single_thread_kernel<FUNCTION><<<1, 1>>>(function);
}

template <typename T>
__host__ __device__ void swap(T& inout_a, T& inout_b)
{
    T c = inout_a;
    inout_a = inout_b;
    inout_b = c;
}

__forceinline__ __device__ uint32_t get_lane_id()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ uint32_t get_warp_id()
{
    // This is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ float sign(float value) { return value < 0 ? -1 : 1; }

} // namespace recogs
