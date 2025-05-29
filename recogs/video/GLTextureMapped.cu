#include "GLTextureMapped.h"

#include <cuda_gl_interop.h>
#include <glad/glad.h>

#include "utils/cuda_utils.h"

using namespace recogs;

GLTextureMapped::GLTextureMapped(int width, int height, GLuint texture, cudaGraphicsResource_t gl_resource)
    : m_width(width), m_height(height), m_texture(texture), m_graphics_resource(gl_resource)
{
}

GLTextureMapped::GLTextureMapped(GLTextureMapped&& other) noexcept
    : m_width(other.m_width), m_height(other.m_height), m_texture(other.m_texture),
      m_graphics_resource(other.m_graphics_resource)
{
    other.m_texture = 0;
    other.m_graphics_resource = nullptr;
}

GLTextureMapped::~GLTextureMapped()
{
    if (m_graphics_resource) CHECK_CUDA(cudaGraphicsUnmapResources(1, &m_graphics_resource));
    if (m_texture) glDeleteTextures(1, &m_texture);
}

void GLTextureMapped::map(const std::function<void(cudaArray_t, cudaSurfaceObject_t)>& scoped_fn, cudaStream_t stream)
{
    CHECK_CUDA(cudaGraphicsMapResources(1, &m_graphics_resource, stream));
    cudaArray_t cuda_array;
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, m_graphics_resource, 0, 0));

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_array;
    cudaSurfaceObject_t surface_object;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface_object, &resource_desc));

    scoped_fn(cuda_array, surface_object);

    CHECK_CUDA(cudaGraphicsUnmapResources(1, &m_graphics_resource, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // TODO delete the surface object ?
}

GLTextureMapped GLTextureMapped::create(const GLTextureMapped_CreateParams& params)
{
    GLuint texture;
    cudaGraphicsResource_t resource;

    int W = params.width;
    int H = params.height;

    // Create GL texture
    glEnable(GL_TEXTURE_2D); // TODO useless
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, params.min_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, params.mag_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, params.wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, params.wrap_t);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create GL resource
    CHECK_CUDA(
        cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    return GLTextureMapped(W, H, texture, resource);
}

GLTextureMapped GLTextureMapped::create_rgba32f(int width, int height)
{
    GLTextureMapped_CreateParams create_params{};
    create_params.width = width;
    create_params.height = height;
    create_params.internalformat = GL_RGBA32F;
    create_params.format = GL_RGBA;
    create_params.type = GL_FLOAT;
    create_params.min_filter = GL_NEAREST;
    create_params.mag_filter = GL_NEAREST;
    create_params.wrap_s = GL_CLAMP_TO_BORDER;
    create_params.wrap_t = GL_CLAMP_TO_BORDER;
    return create(create_params);
}
