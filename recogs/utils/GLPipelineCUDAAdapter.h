#pragma once

#include <functional>

#include <glad/glad.h>

#include "GSCamera.h"
#include "disk/DiskBuffer.h"
#include "utils/image/Image.h"
#include "video/GLTextureMapped.h"
#include "video/gl_utils.h"

namespace recogs
{
class GLPipelineCUDAAdapter
{
public:
    using GLRenderFunction = std::function<void(Framebuffer&)>;

private:
    std::unique_ptr<GLTextureMapped> m_colordepth_texture;
    std::unique_ptr<Framebuffer> m_framebuffer;

    /* GL pipeline adapter objects */
    GLuint m_vao; // Empty vertex array used for screen-quad drawing
    GLuint m_color_attachment_texture{};
    GLuint m_depth_attachment_texture{};
    Program m_colordepth_to_attachments;
    Program m_attachments_to_colordepth;

public:
    ~GLPipelineCUDAAdapter() = default;

    /// Adapt the provided OpenGL pipeline to the outer CUDA pipeline:
    /// <ul>
    /// <li>The color-depth buffer is converted to an OpenGL framebuffer</li>
    /// <li>The GL pipeline is executed</li>
    /// <li>The output color and depth buffer are re-converted to the color-depth buffer</li>
    /// </ul>
    static void
    use_gl(const GSCamera& camera, Image4fHWC& color_depth, const GLRenderFunction& gl_render, cudaStream_t stream);

private:
    explicit GLPipelineCUDAAdapter();

    void setup_gl();
    void setup_screenbuffers(int width, int height);

    void
    use_gl0(const GSCamera& camera, Image4fHWC& color_depth, const GLRenderFunction& gl_pipeline, cudaStream_t stream);
};
} // namespace recogs
