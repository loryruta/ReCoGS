#pragma once

#include "utils/cuda_utils.h"

namespace gs_train
{
struct Buffer {
    const char* const name; // For debugging
    void* data_d = nullptr;
    size_t size = 0;

    explicit Buffer(const char* name) : name(name) {}
    ~Buffer() = default;

    template <typename T>
    [[nodiscard]] const T* data_ptr() const
    {
        return (T*) data_d;
    }

    template <typename T>
    [[nodiscard]] T* data_ptr()
    {
        return (T*) data_d;
    }

    inline bool resize(size_t new_size)
    {
        if (!data_d || size < new_size) {
            printf("Resizing CUDA buffer \"%s\" from %zu to %zu bytes\n", name, size, new_size);
            void* new_data_d;
            cudaError_t error = cudaMalloc(&new_data_d, new_size);
            if (error == cudaErrorMemoryAllocation) {
                printf("[WARNING] Can't allocate %zu bytes of memory (out of memory)\n", new_size);
                return false;
            }
            CHECK_CUDA(cudaMemcpy(new_data_d, data_d, size, cudaMemcpyDeviceToDevice));
            size = new_size;
            if (data_d) CHECK_CUDA(cudaFree(data_d));
            data_d = new_data_d;
        }
        return true;
    }

    inline void fill(int value) { CHECK_CUDA(cudaMemset(data_d, value, size)); }

    inline void destroy()
    {
        if (data_d) {
            CHECK_CUDA(cudaFree(data_d));
            data_d = nullptr;
        }
    }
};
} // namespace gs_train
