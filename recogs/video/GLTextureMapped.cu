#include "GLTextureMapped.h"

#include <cuda_gl_interop.h>
#include <glad/glad.h>

#include "utils/cuda_utils.h"

using namespace recogs;

GLTextureMapped_MappedGuard::GLTextureMapped_MappedGuard(GLTextureMapped_MappedGuard&& other) noexcept
    : graphics_resource(other.graphics_resource), cuda_array(other.cuda_array), surface_object(other.surface_object),
      stream(other.stream)
{
    surface_object = 0;
}

GLTextureMapped_MappedGuard::~GLTextureMapped_MappedGuard()
{
    if (surface_object == 0) return; // Moved

    CHECK_CUDA(cudaGraphicsUnmapResources(1, &graphics_resource, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaDestroySurfaceObject(surface_object)); // TODO Destroy surface object on CUDA stream?
}

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
    if (m_graphics_resource) CHECK_CUDA(cudaGraphicsUnregisterResource(m_graphics_resource));
    if (m_texture) glDeleteTextures(1, &m_texture);
}

GLTextureMapped_MappedGuard GLTextureMapped::map(cudaStream_t stream)
{
    cudaArray_t cuda_array;
    cudaSurfaceObject_t surface_object;

    CHECK_CUDA(cudaGraphicsMapResources(1, &m_graphics_resource, stream));
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, m_graphics_resource, 0, 0));

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_array;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface_object, &resource_desc));

    GLTextureMapped_MappedGuard guard{};
    guard.graphics_resource = m_graphics_resource;
    guard.cuda_array = cuda_array;
    guard.surface_object = surface_object;
    guard.stream = stream;
    return guard;
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
    glTexImage2D(GL_TEXTURE_2D, 0, params.internalformat, W, H, 0, params.format, params.type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, params.min_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, params.mag_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, params.wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, params.wrap_t);

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
