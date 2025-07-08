#pragma once

#include <glad/glad.h>

#include "utils/image/Image.h"

namespace recogs
{
// Forward decl
class GLTextureMapped;

struct GLTextureMapped_CreateParams {
    int width;
    int height;
    // OpenGL glTexImage2D params
    GLint internalformat;
    GLenum format;
    GLenum type;
    GLint min_filter, mag_filter;
    GLint wrap_s, wrap_t;
};

struct GLTextureMapped_MappedGuard {
    friend class GLTextureMapped;

    cudaGraphicsResource_t graphics_resource{};
    cudaArray_t cuda_array{};
    cudaSurfaceObject_t surface_object{};
    cudaStream_t stream{};

    GLTextureMapped_MappedGuard(const GLTextureMapped_MappedGuard&) = delete;
    ~GLTextureMapped_MappedGuard();

private:
    GLTextureMapped_MappedGuard() = default;
    GLTextureMapped_MappedGuard(GLTextureMapped_MappedGuard&&) noexcept;
};

/// \brief RAII class for an OpenGL texture mapped to a CUDA array.
class GLTextureMapped
{
private:
    const int m_width;
    const int m_height;
    GLuint m_texture = 0;
    cudaGraphicsResource_t m_graphics_resource = nullptr;

    std::string m_name = "untitled";

public:
    GLTextureMapped(const GLTextureMapped&) = delete;
    GLTextureMapped(GLTextureMapped&&) noexcept;
    ~GLTextureMapped();

    [[nodiscard]] int width() const { return m_width; }
    [[nodiscard]] int height() const { return m_height; }
    [[nodiscard]] GLuint texture() const { return m_texture; }
    [[nodiscard]] cudaGraphicsResource_t gl_resource() const { return m_graphics_resource; }

    void set_name(const std::string& name) { m_name = name; }

    GLTextureMapped_MappedGuard map(cudaStream_t stream);

    void map(const std::function<void(cudaArray_t, cudaSurfaceObject_t)>& callback, cudaStream_t stream)
    {
        auto map_guard = map(stream);
        callback(map_guard.cuda_array, map_guard.surface_object);
    }

    /// Write CUDA image data to the GL texture.
    template <int C, typename T>
    void write(Image<C, T, ImageMemoryLayout::HWC>& image, cudaStream_t stream)
    {
        auto guard = map(stream);
        CHECK_CUDA(cudaMemcpy2DToArrayAsync( //
            guard.cuda_array,
            0, // wOffset
            0, // hOffset
            image.data_d(),
            m_width * C * sizeof(T), // spitch (tightly packed)
            m_width * C * sizeof(T), // width
            m_height,
            cudaMemcpyDeviceToDevice,
            stream));
    }

    static GLTextureMapped create(const GLTextureMapped_CreateParams& params);
    static GLTextureMapped create_rgba32f(int width, int height);

private:
    explicit GLTextureMapped(int width, int height, GLuint texture, cudaGraphicsResource_t gl_resource);
};
} // namespace recogs
