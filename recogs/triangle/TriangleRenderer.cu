#include "TriangleRenderer.h"

#include <fstream>

#include <cuda_runtime.h>
#include <glm/gtc/type_ptr.hpp>
#include <surface_functions.h>

#include "utils/image/image_fill.h"
#include "utils/image/image_visit_transform.h"
#include "video/gl_utils.h"

using namespace recogs;

namespace
{
const char* k_colordepth_to_attachments_vshader = R"(#version 460 core

    out vec2 v_uv;

    void main()
    {
        const vec2 k_uvs[] = vec2[](
            vec2(0.0, 0.0), // 0
            vec2(1.0, 0.0), // 1
            vec2(0.0, 1.0), // 2
            vec2(1.0, 0.0), // 1
            vec2(1.0, 1.0), // 3
            vec2(0.0, 1.0)  // 2
        );
        const vec2 k_vertices[] = vec2[](
            vec2(-1.0, 1.0),  // 0
            vec2(1.0, 1.0),   // 1
            vec2(-1.0, -1.0), // 2
            vec2(1.0, 1.0),   // 1
            vec2(1.0, -1.0),  // 3
            vec2(-1.0, -1.0)  // 2
        );
        gl_Position = vec4(k_vertices[gl_VertexID], 0, 1);
        v_uv = k_uvs[gl_VertexID];
    }
)";

const char* k_colordepth_to_attachments_fshader = R"(#version 460 core

    in vec2 v_uv;

    uniform sampler2D u_colordepth;

    layout(location = 0) out vec4 f_color;

    void main()
    {
        vec4 value = texture(u_colordepth, v_uv);
        f_color = vec4(value.rgb, 1);
        gl_FragDepth = value.a;
    }
)";

const char* k_attachments_to_colordepth_cshader = R"(#version 460 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) uniform readonly image2D u_color_attachment;
layout (r32f, binding = 1) uniform readonly image2D u_depth_attachment;

layout (rgba32f, binding = 2) uniform writeonly image2D u_colordepth;

void main() {
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

    vec3 color = imageLoad(u_color_attachment, xy).rgb;
    float depth = imageLoad(u_depth_attachment, xy).r;
    imageStore(u_colordepth, xy, vec4(color, depth));
}
)";

const char* k_disk_vs = R"(#version 460 core

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec2 a_scale;
layout(location = 2) in vec4 a_rotation;

layout(location = 0) uniform mat4 u_camera;

out vec2 v_uv;

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
    // TODO apply rotation
    p += a_position.xyz;

    gl_Position = u_camera * vec4(p, 1);
    v_uv = vertices[gl_VertexID];
}
)";

const char* k_disk_fs = R"(#version 460 core

in vec2 v_uv;

layout(location = 0) out vec4 f_color;

void main()
{
    float r = length(v_uv);
    if (r > 1) discard;
    f_color = vec4(r * 0.8 + 0.2, 0, 0, 1);
}
)";

/// Conversion from App rendering pipeline (CUDA) to GL rendering pipeline.
__global__ void convert_to_gl_kernel(const Image4fHWC colordepth, cudaSurfaceObject_t out_gl_colordepth)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= colordepth.width || y >= colordepth.height) return;
    const float* data_ptr = colordepth.data_d() + ((y * colordepth.width + x) * 4);
    float4 value;
    value.x = data_ptr[0];
    value.y = data_ptr[1];
    value.z = data_ptr[2];
    // TODO conversion from clip-space to view-space
    value.w = data_ptr[3];
    surf2Dwrite<float4>(value, out_gl_colordepth, x * sizeof(float4), y); // Fake IDE error
}

/// Conversion from GL rendering pipeline to App rendering pipeline (CUDA).
__global__ void convert_from_gl_kernel(cudaSurfaceObject_t gl_colordepth, Image4fHWC out_colordepth)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= out_colordepth.width || y >= out_colordepth.height) return;
    float* out_data_ptr = out_colordepth.data_d() + ((y * out_colordepth.width + x) * 4);
    float4 value = surf2Dread<float4>(gl_colordepth, x * sizeof(float4), y); // Fake IDE error
    out_data_ptr[0] = value.x;
    out_data_ptr[1] = value.y;
    out_data_ptr[2] = value.z;
    // TODO conversion from clip-space to view-space
    out_data_ptr[3] = value.w;
}
} // namespace

TriangleRenderer::TriangleRenderer() { setup_gl(); }

void TriangleRenderer::setup_gl()
{
    glGenVertexArrays(1, &m_vao);

    // Color-depth to GL attachments
    {
        Shader vshader(GL_VERTEX_SHADER, "colordepth_to_attachments_vshader");
        vshader.source_from_str(k_colordepth_to_attachments_vshader);
        vshader.compile();
        Shader fshader(GL_FRAGMENT_SHADER, "colordepth_to_attachments_fshader");
        fshader.source_from_str(k_colordepth_to_attachments_fshader);
        fshader.compile();
        m_colordepth_to_attachments.attach_shader(vshader);
        m_colordepth_to_attachments.attach_shader(fshader);
        m_colordepth_to_attachments.link();
    }
    // GL attachments to color-depth
    {
        Shader cshader(GL_COMPUTE_SHADER, "attachments_to_colordepth_cshader");
        cshader.source_from_str(k_attachments_to_colordepth_cshader);
        cshader.compile();
        m_attachments_to_colordepth.attach_shader(cshader);
        m_attachments_to_colordepth.link();
    }

    /* Disk rendering */
    // Program
    {
        Shader vshader(GL_VERTEX_SHADER);
        vshader.source_from_str(k_disk_vs);
        vshader.compile();
        Shader fshader(GL_FRAGMENT_SHADER);
        fshader.source_from_str(k_disk_fs);
        fshader.compile();
        m_disk_program.attach_shader(vshader);
        m_disk_program.attach_shader(fshader);
        m_disk_program.link();
    }
}

void TriangleRenderer::setup_screenbuffers(int width, int height) // TODO recreate_screenbuffers(width, height)
{
    m_colordepth_texture = std::make_unique<GLTextureMapped>(GLTextureMapped::create_rgba32f(width, height));
    m_colordepth_texture->set_name("TriangleRenderer_colordepth");

    // GL color attachment
    if (m_color_attachment_texture) glDeleteTextures(1, &m_color_attachment_texture);
    glGenTextures(1, &m_color_attachment_texture);
    glBindTexture(GL_TEXTURE_2D, m_color_attachment_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glBindTexture(GL_TEXTURE_2D, 0);

    // GL depth attachment
    if (m_depth_attachment_texture) glDeleteTextures(1, &m_depth_attachment_texture);
    glGenTextures(1, &m_depth_attachment_texture);
    glBindTexture(GL_TEXTURE_2D, m_depth_attachment_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Framebuffer
    if (m_fbo) glDeleteFramebuffers(1, &m_fbo);
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color_attachment_texture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depth_attachment_texture, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void TriangleRenderer::render(Image4fHWC& color_depth, const std::function<void()>& gl_pipeline, cudaStream_t stream)
{
    int width = (int) color_depth.width;
    int height = (int) color_depth.height;

    if (!m_colordepth_texture || (m_colordepth_texture->width() != width || m_colordepth_texture->height() != height)) {
        setup_screenbuffers(width, height);
    }

    dim3 num_blocks{};
    num_blocks.x = div_ceil(width, 16);
    num_blocks.y = div_ceil(height, 16);
    dim3 blocks_dim = {16, 16, 1};

    // Convert from CUDA color-depth to GL-mapped color-depth
    // TODO This step can be avoided if we use cudaSurfaceObject_t (GL-mapped)
    m_colordepth_texture->map(
        [&](cudaArray_t, cudaSurfaceObject_t surface_object) {
            convert_to_gl_kernel<<<num_blocks, blocks_dim, 0, stream>>>(color_depth, surface_object);
        },
        stream);
    // CUDA guarantees that any CUDA work issued before subsequent graphics operations will complete.
    // There's no need to explicitly synchronize the stream

    /* GL-mapped color-depth to GL color/depth attachments */
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        m_colordepth_to_attachments.use();
        glBindVertexArray(m_vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_colordepth_texture->texture());
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    /* Run user-defined GL pipeline */
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST); // By default, enable depth-test
        glDisable(GL_BLEND);     // By default, disable alpha blending
        gl_pipeline();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    /* GL color/depth attachments to GL-mapped color-depth */
    {
        m_attachments_to_colordepth.use();
        glBindImageTexture(0, m_color_attachment_texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        glBindImageTexture(1, m_depth_attachment_texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(2, m_colordepth_texture->texture(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glDispatchCompute(num_blocks.x, num_blocks.y, num_blocks.z);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // CUDA guarantees that any graphics call issued before cudaGraphicsMapResources() will complete before any
    // subsequent CUDA work. No need to synchronize the stream

    /* GL-mapped color-depth to CUDA color-depth */
    // TODO This step can be avoided if we use cudaSurfaceObject_t (GL-mapped)
    m_colordepth_texture->map(
        [&](cudaArray_t, cudaSurfaceObject_t surface_object) {
            convert_from_gl_kernel<<<num_blocks, blocks_dim, 0, stream>>>(surface_object, color_depth);
        },
        stream);
}

void TriangleRenderer::render_disks(const GSCamera& camera,
                                    const DiskBuffer& disk_buffer,
                                    Image4fHWC& color_depth,
                                    cudaStream_t stream)
{
    if (disk_buffer.empty()) return;
    render(
        color_depth,
        [&]() {
            m_disk_program.use();

            glClear(GL_DEPTH_BUFFER_BIT); // TODO remove

            glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(camera.viewproj()));
            glBindVertexArray(disk_buffer.vao());
            glBindBuffer(GL_ARRAY_BUFFER, disk_buffer.vbo());
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei) disk_buffer.size());
        },
        stream);
}
