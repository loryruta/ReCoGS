#include "CudaTexture.h"

#include "utils/cuda_utils.h"

using namespace gs_train;

CudaTexture::CudaTexture(int width, int height) : m_width(width), m_height(height)
{
    // Alloc CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    CHECK_CUDA(cudaMallocArray(&m_cuda_array, &channelDesc, width, height));
    // Alloc resource desc
    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = m_cuda_array;
    // Alloc texture desc
    cudaTextureDesc texture_desc{};
    // TODO allow customization from outside
    texture_desc.addressMode[0] = cudaAddressModeBorder;
    texture_desc.addressMode[1] = cudaAddressModeBorder;
    texture_desc.borderColor[0] = 0;
    texture_desc.borderColor[1] = 0;
    texture_desc.borderColor[2] = 0;
    texture_desc.borderColor[3] = 0;
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = true;
    // Create texture object
    CHECK_CUDA(cudaCreateTextureObject(&m_texture_object, &resource_desc, &texture_desc, nullptr));
}

CudaTexture::CudaTexture(CudaTexture&& other) noexcept
    : m_width(other.m_width), m_height(other.m_height), m_cuda_array(other.m_cuda_array),
      m_texture_object(other.m_texture_object)
{
    m_cuda_array = {};
    m_texture_object = {};
}

CudaTexture::~CudaTexture()
{
    if (m_texture_object) {
        CHECK_CUDA(cudaDestroyTextureObject(m_texture_object));
    }
    if (m_cuda_array) {
        CHECK_CUDA(cudaFreeArray(m_cuda_array));
    }
}

void CudaTexture::write(const Image4fHWC& image, cudaStream_t stream)
{
    CHECK_ARG(image.width == m_width && image.height == m_height, "image size mismatch texture size");
    CHECK_CUDA(cudaMemcpy2DToArrayAsync( //
        m_cuda_array,
        0,
        0,
        image.data_d(),
        m_width * 4 * sizeof(float),
        m_width * 4 * sizeof(float),
        m_height,
        cudaMemcpyDeviceToDevice,
        stream));
}
