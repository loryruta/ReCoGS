#pragma once

#include "Camera.h"
#include "DiskBuffer.h"
#include "utils/image/Image.h"
#include "video/GLTextureMapped.h"
#include "video/gl_utils.h"

namespace recogs
{
namespace detail
{
/// \param write_color_func
///     A device function for writing the pixel according to the disk information.
template <typename WRITE_COLOR>
__global__ void render_disk_id_map(cudaSurfaceObject_t disk_id_map,
                                   cudaSurfaceObject_t uv_map,
                                   const Disk* disks,
                                   WRITE_COLOR write_color_func,
                                   Image4fHWC color_depth)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int W = (int) color_depth.width;
    int H = (int) color_depth.height;
    if (x >= W || y >= H) return;
    uint32_t disk_id = surf2Dread<uint32_t>(disk_id_map, x * sizeof(uint32_t), y);
    if (disk_id == UINT32_MAX) return;
    float2 uv = surf2Dread<float2>(uv_map, x * sizeof(float2), y);
    float opacity = disks[disk_id].opacity;
    float* color_ptr = &color_depth.data_d()[(y * W + x) * 4];
    write_color_func(disk_id, uv, opacity, color_ptr);
}
} // namespace detail

struct DiskRenderer_Params {
    const Camera* camera;
    Image4fHWC* color_depth;
    const DiskBuffer* disk_buffer;
    /// (Optional) A map indicating where very disk was rasterized (used e.g. for clearing).
    Image<1, uint32_t, ImageMemoryLayout::HWC>* disk_ids;
    cudaStream_t stream;
};

class DiskRenderer
{
private:
    Program m_program;
    std::unique_ptr<GLTextureMapped> m_disk_id_map_texture;
    std::unique_ptr<GLTextureMapped> m_uv_texture;
    std::unique_ptr<Framebuffer> m_framebuffer;

public:
    explicit DiskRenderer();
    ~DiskRenderer() = default;

    template <typename WRITE_COLOR>
    void render(const DiskRenderer_Params& params, WRITE_COLOR write_color_func);

private:
    void setup_screenbuffers(glm::ivec2 resolution);

    void fill_disk_id_map(const DiskRenderer_Params& params);
};

template <typename WRITE_COLOR>
void DiskRenderer::render(const DiskRenderer_Params& params, WRITE_COLOR write_color_func)
{
    fill_disk_id_map(params);

    Image4fHWC& color_depth = *params.color_depth;
    cudaStream_t stream = params.stream;

    auto disk_id_map_guard = m_disk_id_map_texture->map(stream);
    auto uv_map_guard = m_uv_texture->map(stream);

    const Disk* disks = RCGS_TPTR(params.disk_buffer->disks_d);

    dim3 num_blocks{};
    num_blocks.x = div_ceil(color_depth.width, 32);
    num_blocks.y = div_ceil(color_depth.height, 32);
    dim3 block_dim(32, 32);
    detail::render_disk_id_map<<<num_blocks, block_dim, 0, stream>>>( //
        disk_id_map_guard.surface_object,
        uv_map_guard.surface_object,
        disks,
        write_color_func,
        color_depth);
}
} // namespace recogs
