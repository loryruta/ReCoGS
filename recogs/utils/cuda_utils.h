#pragma once

#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "utils/misc_utils.h"

#define CHECK_CUDA(_error) gs_train::check_cuda_error(_error, __FILE_NAME__, __LINE__)

namespace gs_train
{
inline void check_cuda_error(cudaError_t error, char const* file, int line)
{
    if (error != cudaSuccess) {
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
void dump_device_buffer(const T* buffer, size_t num_elements, const char* out_filepath)
{
    std::vector<T> buffer_data = to_host_vector<T>(buffer, num_elements);
    FILE* f = fopen(out_filepath, "wt");
    CHECK_STATE(f);

    char entry[256];
    for (int i = 0; i < num_elements; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            sprintf(entry, "%d: %f, ", i, buffer_data[i]);
            fputs(entry, f);
        } else {
            // NOT IMPLEMENTED!
            fputs("N", f);
        }
    }
    fclose(f);
}

} // namespace gs_train
