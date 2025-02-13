#include "GLMappedResource.h"

#include <cuda_gl_interop.h>
#include <glad/glad.h>

#include "utils/cuda_utils.h"

using namespace gs_train;

GLMappedResource::GLMappedResource(int W, int H) : m_W(W), m_H(H)
{
    // Create OpenGL texture
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_W, m_H, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create a CUDA mapped resource to access the texture
    CHECK_CUDA(
        cudaGraphicsGLRegisterImage(&m_gl_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    CHECK_CUDA(cudaGraphicsMapResources(1, &m_gl_resource));
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&m_cuda_array, m_gl_resource, 0, 0));
}

GLMappedResource::~GLMappedResource()
{
    CHECK_CUDA(cudaFreeArray(m_cuda_array));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &m_gl_resource));
    glDeleteTextures(1, &m_texture);
}

void GLMappedResource::write(float* data_d)
{
    CHECK_CUDA(cudaMemcpy2DToArray(m_cuda_array,
                                   0, // wOffset
                                   0, // hOffset
                                   data_d,
                                   m_W * 4 * sizeof(float), // spitch (tightly packed)
                                   m_W * 4 * sizeof(float), // width
                                   m_H,
                                   cudaMemcpyDeviceToDevice));
}
