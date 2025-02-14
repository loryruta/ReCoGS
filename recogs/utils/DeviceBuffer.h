#pragma once

#include <vector>

#include "utils/cuda_utils.h"

namespace gs_train
{
/// A RAII wrapper for a buffer held in device memory
struct DeviceBuffer {
    const char* const name; // For debugging
    void* data_d = nullptr;
    size_t size = 0;

    explicit DeviceBuffer(const char* name) : name(name ? name : "unnamed") {}
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = default;
    ~DeviceBuffer() = default; // TODO DELETE THE BUFFER!

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

    /// Upload the given host data to the buffer.
    /// Fail if the buffer is small.
    template <typename T>
    void upload(const T* elements, size_t num_elements, size_t offset = 0)
    {
        size_t num_bytes = num_elements * sizeof(T);
        size_t required_bytes = offset + num_bytes;
        CHECK_STATE(size >= required_bytes, "Buffer is too small");
        CHECK_CUDA(cudaMemcpy(((uint8_t*) data_d) + offset, elements, num_bytes, cudaMemcpyHostToDevice));
    }

    /// Fit the given data within the buffer.
    /// Re-allocate the buffer if the size isn't enough.
    template <typename T>
    void fit_data(const T* elements, size_t num_elements, size_t offset = 0)
    {
        size_t required_bytes = offset + num_elements * sizeof(T);
        if (size < required_bytes) CHECK_STATE(resize(required_bytes), "Buffer resize failed");
        upload(elements, num_elements, offset);
    }

    inline void destroy()
    {
        if (data_d) {
            CHECK_CUDA(cudaFree(data_d));
            data_d = nullptr;
        }
    }

    template <typename T>
    static DeviceBuffer alloc(size_t num_elements, const char* name = nullptr)
    {
        DeviceBuffer buffer(name);
        buffer.resize(num_elements * sizeof(T));
        return buffer;
    }

    template <typename T>
    static DeviceBuffer from_data(const std::vector<T>& vec, const char* name = nullptr)
    {
        DeviceBuffer buffer = DeviceBuffer::alloc<T>(vec.size() * sizeof(T), name);
        buffer.upload<T>(vec.data(), vec.size());
        return buffer;
    }
};
} // namespace gs_train
