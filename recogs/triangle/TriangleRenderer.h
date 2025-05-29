#pragma once

#include <functional>

#include <glad/glad.h>

#include "GSCamera.h"
#include "triangle/DiskBuffer.h"
#include "utils/image/Image.h"
#include "video/GLTextureMapped.h"
#include "video/gl_utils.h"

namespace recogs
{
class TriangleRenderer
{
private:
    /* GL pipeline adapter objects */
    GLuint m_fbo{};
    std::unique_ptr<GLTextureMapped> m_colordepth_texture;
    GLuint m_vao; // Empty vertex array used for screen-quad drawing
    GLuint m_color_attachment_texture{};
    GLuint m_depth_attachment_texture{};
    Program m_colordepth_to_attachments;
    Program m_attachments_to_colordepth;

    /* Disk rendering */
    Program m_disk_program;

public:
    explicit TriangleRenderer();
    ~TriangleRenderer() = default;

    // TODO camera to fix depths from view-space to window-space
    void render(Image4fHWC& color_depth, const std::function<void()>& gl_pipeline, cudaStream_t stream);

    void
    render_disks(const GSCamera& camera, const DiskBuffer& disk_buffer, Image4fHWC& color_depth, cudaStream_t stream);

private:
    void setup_gl();
    void setup_screenbuffers(int width, int height);
};
} // namespace recogs
