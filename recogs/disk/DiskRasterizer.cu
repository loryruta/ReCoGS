#include "DiskRasterizer.h"

#include "rasterizer_impl.h"

using namespace recogs;

namespace
{
std::function<char*(size_t N)>
resize_functional(thrust::device_vector<uint8_t>& buffer, const char* name, size_t alignment)
{
    return [&buffer, name, alignment](size_t num_bytes) -> char* {
        if (num_bytes >= buffer.size()) {
            size_t new_size = div_ceil(num_bytes, alignment) * alignment;
            printf("[DEBUG] [DiskRasterizer] Resizing \"%s\" to %zu bytes\n", name, new_size);
            buffer.resize(new_size);
        }
        return reinterpret_cast<char*>(RCGS_TPTR(buffer));
    };
}
} // namespace

void DiskRasterizer::forward(const Disks& disks, const GSCamera& camera, Image4fHWC& color_depth, cudaStream_t stream)
{
    CHECK_ARG(disks.is_valid(), "Invalid disks");

    CudaRasterizer::Rasterizer::forward( //
        resize_functional(m_geometry_buffer, "geometry_buffer", 1 << 24 /* 16MB */),
        resize_functional(m_binning_buffer, "binning_buffer", 1 << 24 /* 16MB */),
        resize_functional(m_image_buffer, "image_buffer", 1 << 24 /* 16MB */),
        disks.count,
        camera.width,
        camera.height,
        (const float*) RCGS_TPTR(disks.positions),
        (const float*) RCGS_TPTR(disks.scales),
        (const float*) RCGS_TPTR(disks.rotations),
        camera.viewmatrix_d(),
        camera.projmatrix_d(),
        color_depth.data_d(),
        false, // debug
        stream);
}
