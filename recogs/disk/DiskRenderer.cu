#include "DiskRenderer.h"

#include <glm/gtc/type_ptr.hpp>

#include "utils/GLPipelineCUDAAdapter.h"

using namespace recogs;

namespace
{

const char* k_disk_vs = R"(#version 460 core

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec2 a_scale;
layout(location = 2) in vec4 a_rotation;

layout(location = 0) uniform mat4 u_camera;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out flat uint v_disk_id;

vec3 quat_rotate(vec4 q, vec3 v) {
   return v + 2.0 * cross(cross(v, q.xyz ) + q.w * v, q.xyz);
}

void main()
{
    const vec2 vertices[] = vec2[](
        vec2(-1.0, 1.0),  // 0
        vec2(1.0, 1.0),   // 1
        vec2(-1.0, -1.0), // 2
        vec2(1.0, 1.0),   // 1
        vec2(1.0, -1.0),  // 3
        vec2(-1.0, -1.0)  // 2
    );

    vec3 p = vec3(a_scale * vertices[gl_VertexID], 0);
    p = quat_rotate(a_rotation, p);
    p += a_position.xyz;

    gl_Position = u_camera * vec4(p, 1);
    v_uv = vertices[gl_VertexID];
    v_disk_id = gl_InstanceID;
}
)";

const char* k_disk_fs = R"(#version 460 core

layout(location = 0) in vec2 v_uv;
layout(location = 1) in flat uint v_disk_id;

layout(location = 0) out uint f_disk_id;
layout(location = 1) out vec2 f_uv;

void main()
{
    float r = length(v_uv);
    // Discarding prevents early depth test (possible optimization)
    if (r > 1) {
        discard;
    }

    f_disk_id = v_disk_id;
    f_uv = v_uv;
}
)";
} // namespace

DiskRenderer::DiskRenderer()
{
    /* Program */
    printf("[INFO ] [DiskRenderer] Creating shader program...\n");

    Shader vshader(GL_VERTEX_SHADER);
    vshader.source_from_str(k_disk_vs);
    vshader.compile();
    Shader fshader(GL_FRAGMENT_SHADER);
    fshader.source_from_str(k_disk_fs);
    fshader.compile();
    m_program.attach_shader(vshader);
    m_program.attach_shader(fshader);
    m_program.link();
}

void DiskRenderer::setup_screenbuffers(glm::ivec2 resolution)
{
    { // Disk ID map
        GLTextureMapped_CreateParams params{};
        params.width = resolution.x;
        params.height = resolution.y;
        params.internalformat = GL_R32UI;
        params.format = GL_RED_INTEGER;
        params.type = GL_UNSIGNED_INT;
        params.min_filter = GL_NEAREST;
        params.mag_filter = GL_NEAREST;
        params.wrap_s = GL_CLAMP_TO_EDGE;
        params.wrap_t = GL_CLAMP_TO_EDGE;
        m_disk_id_map_texture = std::make_unique<GLTextureMapped>(GLTextureMapped::create(params));
        glObjectLabel(GL_TEXTURE, m_disk_id_map_texture->texture(), -1, "DiskRenderer_DiskIdMapTexture");
    }

    { // UV
        GLTextureMapped_CreateParams params{};
        params.width = resolution.x;
        params.height = resolution.y;
        params.internalformat = GL_RG32F;
        params.format = GL_RG;
        params.type = GL_FLOAT;
        params.min_filter = GL_NEAREST;
        params.mag_filter = GL_NEAREST;
        params.wrap_s = GL_CLAMP_TO_EDGE;
        params.wrap_t = GL_CLAMP_TO_EDGE;
        m_uv_texture = std::make_unique<GLTextureMapped>(GLTextureMapped::create(params));
        glObjectLabel(GL_TEXTURE, m_uv_texture->texture(), -1, "DiskRenderer_UVTexture");
    }

    m_framebuffer = std::make_unique<Framebuffer>(resolution);
    m_framebuffer->set_name("DiskRenderer_Framebuffer");
    m_framebuffer->attach_color(0, m_disk_id_map_texture->texture());
    m_framebuffer->attach_color(1, m_uv_texture->texture());
    m_framebuffer->create_and_attach_depth_renderbuffer();
    m_framebuffer->draw_buffers({GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1});
    CHECK_STATE(m_framebuffer->status() == GL_FRAMEBUFFER_COMPLETE);
}

void DiskRenderer::fill_disk_id_map(const recogs::DiskRenderer_Params& params)
{
    CHECK_ARG(params.camera);
    CHECK_ARG(params.color_depth);
    CHECK_ARG(params.disk_buffer);
    CHECK_ARG(params.stream);

    const Camera& camera = *params.camera;
    Image4fHWC& color_depth = *params.color_depth;
    const DiskBuffer& disk_buffer = *params.disk_buffer;

    if (disk_buffer.empty()) return;

    glm::ivec2 resolution = color_depth.size();
    if (!m_framebuffer || m_framebuffer->resolution != resolution) {
        setup_screenbuffers(resolution);
    }

    GLPipelineCUDAAdapter::use_gl(
        camera,
        color_depth,
        [&](Framebuffer& framebuffer) {
            // Clear disk IDs
            GLuint clear_value = 0xFFFFFFFF;
            glClearTexImage(m_disk_id_map_texture->texture(), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &clear_value);
            // Bind local framebuffer
            m_framebuffer->bind();
            glViewport(0, 0, resolution.x, resolution.y);
            m_framebuffer->blit_from(framebuffer, GL_DEPTH_BUFFER_BIT); // Copy outer depth buffer
            // Draw disk IDs
            m_program.use();
            glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(camera.viewproj()));
            glBindVertexArray(disk_buffer.vao());
            glBindBuffer(GL_ARRAY_BUFFER, disk_buffer.vbo());
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei) disk_buffer.size());
            // Copy local depth buffer to outer
            framebuffer.blit_from(*m_framebuffer, GL_DEPTH_BUFFER_BIT);
        },
        params.stream);
}
