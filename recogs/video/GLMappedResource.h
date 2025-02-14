#pragma once

#include <glad/glad.h>

namespace gs_train
{
struct GLMappedResource {
private:
    const int m_W;
    const int m_H;
    GLuint m_texture = 0;
    cudaGraphicsResource_t m_gl_resource = nullptr;
    cudaArray_t m_cuda_array = nullptr;

public:
    explicit GLMappedResource(int W, int H);
    ~GLMappedResource();

    [[nodiscard]] GLuint texture() const { return m_texture; }

    /// Write CUDA image data to the GL texture.
    /// \param image_d
    ///     Device image data with memory layout (H, W, 4).
    void write(float* image_d);
};
} // namespace gs_train
