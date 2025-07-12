#pragma once

#include <cstdint>

#include <glm/glm.hpp>

#include "utils/cuda_utils.h"
#include "utils/misc_utils.h"

namespace recogs
{
enum class ImageMemoryLayout : uint8_t { HWC, CHW };

/// A class for interacting with a image stored in device memory.
///
/// \tparam C
///     The number of components per pixel
/// \tparam T
///     Type of pixel components
/// \tparam MEMORY_LAYOUT
///     The order in which bytes are stored in the image.
///     For example:
///     - HWC (H, W, C) is the format for stbi_write_*
///     - CHW (C, H, W) is the format for libtorch
template <int C, typename T, ImageMemoryLayout MEMORY_LAYOUT = ImageMemoryLayout::HWC>
class Image
{
    static_assert(C >= 1, "Number of components must be at least 1");

public:
    static constexpr int k_components = C;
    using ValueType = T;
    using Value = glm::vec<C, T>;
    static constexpr ImageMemoryLayout k_memory_layout = MEMORY_LAYOUT;

private:
    T* m_data_d = nullptr;

public:
    uint32_t width; // TODO use int so no explicit casting
    uint32_t height;
    bool owned = true;

    explicit Image(uint32_t width, uint32_t height, T* data_d = nullptr)
        : width(width), height(height), m_data_d(data_d)
    {
    }

    /// Copy the image to a non-owning reference.
    __host__ __device__ Image(const Image& other) // TODO delete, prefer explicitly calling create_ref()
        : width(other.width), height(other.height), m_data_d(other.m_data_d), owned(false)
    {
    }

    /// Move the image to a new owning/non-owning reference, invalidating the moved reference.
    Image(Image&& other) noexcept //
        : width(other.width), height(other.height), m_data_d(other.m_data_d), owned(other.owned)
    {
        other.m_data_d = nullptr;
    }

    /// On destruction, if a owning and valid reference, destroy the image.
    __host__ __device__ ~Image()
    {
#ifndef __CUDA_ARCH__
        if (owned && m_data_d) {
            CHECK_CUDA(cudaFree(m_data_d));
            m_data_d = nullptr;
        }
#else
        // An image on device must never own its data
        assert(!owned);
#endif
    }

    [[nodiscard]] __host__ __device__ glm::uvec2 size() const { return {width, height}; }
    [[nodiscard]] __host__ __device__ T* data_d() const { return m_data_d; }

    /// Read the pixel at (x, y)
    __device__ Value value(uint32_t x, uint32_t y) const
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        if constexpr (MEMORY_LAYOUT == ImageMemoryLayout::HWC) {
            return *(reinterpret_cast<Value*>(m_data_d) + y * width + x);
        } else {
            assert(MEMORY_LAYOUT == ImageMemoryLayout::CHW);
            Value val;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                val[c] = m_data_d[c * height * width + y * width + x];
            }
            return val;
        }
    }

    /// Set the pixel at (x, y)
    __device__ void set_value(uint32_t x, uint32_t y, const Value& val)
    {
        assert(x >= 0 && x < width);
        assert(y >= 0 && y < height);
        if constexpr (MEMORY_LAYOUT == ImageMemoryLayout::HWC) {
            *(reinterpret_cast<Value*>(m_data_d) + y * width + x) = val;
        } else {
            assert(MEMORY_LAYOUT == ImageMemoryLayout::CHW);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                m_data_d[c * height * width + y * width + x] = val[c];
            }
        }
    }

    /// Create a non-owning reference of the image.
    __host__ Image create_ref() const
    {
        Image cloned_ref(width, height, m_data_d);
        cloned_ref.owned = false;
        return cloned_ref;
    }

    __host__ void to_host(T* out_data, cudaStream_t stream) const
    {
        CHECK_CUDA(cudaMemcpyAsync( //
            out_data,
            m_data_d,
            width * height * C * sizeof(T),
            cudaMemcpyDeviceToHost,
            stream));
    }

    __host__ void to_host(std::vector<T>& out_data, cudaStream_t stream) const
    {
        out_data.resize(width * height * C);
        to_host(out_data.data(), stream);
    }

    __host__ Image& operator=(Image&& other) noexcept
    {
        CHECK_STATE(!owned, "Moving is only valid towards non-owning references");
        width = other.width;
        height = other.height;
        m_data_d = other.m_data_d;
        owned = other.owned;
    }

    /// Allocate a image owning new data (uninitialized).
    __host__ static Image malloc(uint32_t width, uint32_t height, cudaStream_t stream = 0)
    {
        Image image(width, height, nullptr);
        CHECK_CUDA(cudaMallocAsync(&image.m_data_d, width * height * C * sizeof(T), stream));
        if (stream == CU_STREAM_LEGACY) {
            // TODO not sure if needed; if we allocate asynchronously we should also deallocate asynchronously!
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        return image;
    }

    /// Create a non-owning reference to the provided data.
    __host__ static Image ref(uint32_t width, uint32_t height, T* data_d)
    {
        Image image(width, height, data_d);
        image.owned = false;
        return image;
    }
};

using Image1u = Image<1, uint8_t, ImageMemoryLayout::CHW>;
using Image1u8 = Image<1, uint8_t, ImageMemoryLayout::HWC>;
using Image1fHWC = Image<1, float, ImageMemoryLayout::HWC>;
using Image3fHWC = Image<3, float, ImageMemoryLayout::HWC>;
using Image4fHWC = Image<4, float, ImageMemoryLayout::HWC>;
using Image1fCHW = Image<1, float, ImageMemoryLayout::CHW>;
using Image3fCHW = Image<3, float, ImageMemoryLayout::CHW>;
using Image4fCHW = Image<4, float, ImageMemoryLayout::CHW>;

} // namespace recogs
