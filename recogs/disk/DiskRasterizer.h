#pragma once

#include <thrust/device_vector.h>

#include "Disks.h"
#include "GSCamera.h"
#include "utils/image/Image.h"

namespace recogs
{
class DiskRasterizer
{
private:
    thrust::device_vector<uint8_t> m_geometry_buffer;
    thrust::device_vector<uint8_t> m_binning_buffer;
    thrust::device_vector<uint8_t> m_image_buffer;

public:
    explicit DiskRasterizer() = default;
    ~DiskRasterizer() = default;

    void forward(const Disks& disks, const GSCamera& camera, Image4fHWC& color_depth, cudaStream_t stream);
};
} // namespace recogs
