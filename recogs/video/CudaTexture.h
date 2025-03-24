#pragma once

#include "utils/image/Image.h"

namespace gs_train
{
class CudaTexture
{
private:
    const int m_width;
    const int m_height;
    cudaArray_t m_cuda_array{};
    cudaTextureObject_t m_texture_object{};

public:
    explicit CudaTexture(int width, int height);
    CudaTexture(CudaTexture&) = delete;
    CudaTexture(CudaTexture&& other) noexcept;
    ~CudaTexture();

    [[nodiscard]] glm::ivec2 resolution() const { return {m_width, m_height}; }

    [[nodiscard]] cudaTextureObject_t texture_object() const { return m_texture_object; }

    void write(const Image4fHWC& image, cudaStream_t stream);
};
} // namespace gs_train