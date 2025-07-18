#include "GLPipelineCUDAAdapter.h"

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
            vec2(0.0, 1.0), // 0
            vec2(1.0, 1.0), // 1
            vec2(0.0, 0.0), // 2
            vec2(1.0, 1.0), // 1
            vec2(1.0, 0.0), // 3
            vec2(0.0, 0.0)  // 2
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

    uniform sampler2D u_color_depth;
    layout(location = 1) uniform vec2 u_camera_data;

    layout(location = 0) out vec4 f_color;

    void main()
    {
        vec4 value = texture(u_color_depth, v_uv);
        float z = value.a;
        // View-space depth to Window-space
        float p22 = u_camera_data.r;
        float p32 = u_camera_data.g;
        float z_d = p22 + p32 / z; // [0, 1]
        f_color = vec4(value.rgb, 1);
        gl_FragDepth = z_d;
    }
)";

const char* k_attachments_to_colordepth_cshader = R"(#version 460 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, binding = 0) uniform readonly image2D u_color_attachment;
layout (r32f, binding = 1) uniform readonly image2D u_depth_attachment;

layout(location = 0) uniform vec2 u_camera_data;

layout (rgba32f, binding = 2) uniform writeonly image2D u_color_depth;

void main() {
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

    vec3 color = imageLoad(u_color_attachment, xy).rgb;
    // Window-space depth to View-space
    float p22 = u_camera_data.r;
    float p32 = u_camera_data.g;
    float z_d = imageLoad(u_depth_attachment, xy).r;
    float z = p32 / (z_d - p32);
    imageStore(u_color_depth, xy, vec4(color, z));
}
)";

/// Conversion from App rendering pipeline (CUDA) to GL rendering pipeline.
__global__ void convert_to_gl_kernel(const Image4fHWC colordepth, cudaSurfaceObject_t out_gl_colordepth)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int H = colordepth.height;
    if (x >= colordepth.width || y >= H) return;
    const float* data_ptr = colordepth.data_d() + ((y * colordepth.width + x) * 4);
    float4 value;
    value.x = data_ptr[0];
    value.y = data_ptr[1];
    value.z = data_ptr[2];
    value.w = data_ptr[3];
    surf2Dwrite<float4>(value, out_gl_colordepth, x * sizeof(float4), y); // Fake IDE error
}

/// Conversion from GL rendering pipeline to App rendering pipeline (CUDA).
__global__ void convert_from_gl_kernel(cudaSurfaceObject_t gl_colordepth, Image4fHWC out_colordepth)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int H = out_colordepth.height;
    if (x >= out_colordepth.width || y >= H) return;
    float* out_data_ptr = out_colordepth.data_d() + ((y * out_colordepth.width + x) * 4);
    float4 value = surf2Dread<float4>(gl_colordepth, x * sizeof(float4), y); // Fake IDE error
    out_data_ptr[0] = value.x;
    out_data_ptr[1] = value.y;
    out_data_ptr[2] = value.z;
    out_data_ptr[3] = value.w;
}

std::unique_ptr<GLPipelineCUDAAdapter> g_instance;
} // namespace

GLPipelineCUDAAdapter::GLPipelineCUDAAdapter() { setup_gl(); }

void GLPipelineCUDAAdapter::setup_gl()
{
    glGenVertexArrays(1, &m_vao);

    // Color-depth to GL attachments
    {
        printf("[INFO ] [GLPipelineCUDAAdapter] Creating \"colordepth_to_attachments\" program...\n");

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
        printf("[INFO ] [GLPipelineCUDAAdapter] Creating \"attachments_to_colordepth_cshader\" program...\n");

        Shader cshader(GL_COMPUTE_SHADER, "attachments_to_colordepth_cshader");
        cshader.source_from_str(k_attachments_to_colordepth_cshader);
        cshader.compile();
        m_attachments_to_colordepth.attach_shader(cshader);
        m_attachments_to_colordepth.link();
    }
}

void GLPipelineCUDAAdapter::setup_screenbuffers(int width, int height) // TODO recreate_screenbuffers(width, height)
{
    m_colordepth_texture = std::make_unique<GLTextureMapped>(GLTextureMapped::create_rgba32f(width, height));

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
    m_framebuffer = std::make_unique<Framebuffer>(glm::ivec2(width, height));
    m_framebuffer->attach_color(0, m_color_attachment_texture);
    m_framebuffer->attach_depth_texture(m_depth_attachment_texture);
    CHECK_STATE(m_framebuffer->status() == GL_FRAMEBUFFER_COMPLETE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLPipelineCUDAAdapter::use_gl0(const Camera& camera,
                                    Image4fHWC& color_depth,
                                    const GLRenderFunction& gl_render,
                                    cudaStream_t stream)
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

    // Compute variables for view-space depth to window-space depth
    // https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf (Section 13.8.1)
    float p22 = camera.projmatrix()[2][2];
    float p32 = camera.projmatrix()[3][2];

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
        m_framebuffer->bind();
        glViewport(0, 0, width, height);
        glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
        // Enable depth test because we want to write to the depth buffer (gl_FragDepth),
        // but disable depth test by making all fragments pass
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_ALWAYS);
        glDisable(GL_BLEND);
        m_colordepth_to_attachments.use();
        glBindVertexArray(m_vao);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_colordepth_texture->texture());
        glUniform2f(1 /* u_camera_data */, p22, p32);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    /* Run user-defined GL pipeline */
    {
        m_framebuffer->bind();
        glViewport(0, 0, width, height);
        glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
        glEnable(GL_DEPTH_TEST); // By default, enable depth-test
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND); // By default, disable alpha blending

        gl_render(*m_framebuffer);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    /* GL color/depth attachments to GL-mapped color-depth */
    {
        m_attachments_to_colordepth.use();
        glBindImageTexture(0, m_color_attachment_texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        glBindImageTexture(1, m_depth_attachment_texture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(2, m_colordepth_texture->texture(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glUniform2f(0 /* u_camera_data */, p22, p32);
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

void GLPipelineCUDAAdapter::use_gl(const Camera& camera,
                                   Image4fHWC& color_depth,
                                   const GLRenderFunction& gl_render,
                                   cudaStream_t stream)
{
    if (!g_instance) {
        g_instance = std::unique_ptr<GLPipelineCUDAAdapter>(new GLPipelineCUDAAdapter());
    }
    g_instance->use_gl0(camera, color_depth, gl_render, stream);
}
