#include "Sel2d.h"

#include <glm/glm.hpp>
#include <thrust/copy.h>

#include "App.h"
#include "Sel3d.h"
#include "utils/DeviceBuffer.h"
#include "utils/bresenham.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_visit_transform.h"

using namespace recogs;

namespace
{
__device__ glm::ivec2 apply_scale_offset(glm::ivec2 p, int W, int H, float scale, glm::vec2 offset)
{
    glm::ivec2 result;
    result.x = (int) (float(p.x) * scale + float(W) * (-0.5f * scale + offset.x + 0.5f));
    result.y = (int) (float(p.y) * scale + float(H) * (-0.5f * scale + offset.y + 0.5f));
    return result;
}

__global__ void
fill_line_kernel(Image1u8 fill_image, glm::ivec2 p0, glm::ivec2 p1, int r, float scale, glm::vec2 offset)
{
    int W = (int) fill_image.width;
    int H = (int) fill_image.height;
    glm::ivec2 p0t = apply_scale_offset(p0, W, H, scale, offset);
    glm::ivec2 p1t = apply_scale_offset(p1, W, H, scale, offset);
    bresenham_draw_line_radius(p0t, p1t, r, [&] __device__(int x, int y) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        fill_image.set_value(x, y, glm::vec<1, uint8_t>(1));
    });
}

__global__ void clear_line_kernel(
    Image1u8 fill_image, Image1u8 clear_image, glm::ivec2 p0, glm::ivec2 p1, int r, float scale, glm::vec2 offset)
{
    assert(fill_image.size() == clear_image.size());
    int W = (int) fill_image.width;
    int H = (int) fill_image.height;
    glm::ivec2 p0t = apply_scale_offset(p0, W, H, scale, offset);
    glm::ivec2 p1t = apply_scale_offset(p1, W, H, scale, offset);
    bresenham_draw_line_radius(p0t, p1t, r, [&] __device__(int x, int y) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        fill_image.set_value(x, y, glm::vec<1, uint8_t>(0));
        clear_image.set_value(x, y, glm::vec<1, uint8_t>(1));
    });
}
} // namespace

Sel2d::Sel2d(const Camera& camera, const Image4fHWC& color_depth) : m_camera(camera), m_color_depth(color_depth)
{
    int W = (int) camera.width;
    int H = (int) camera.height;

    m_fill_mask = std::make_unique<Image1u8>(Image1u8::malloc(W, H, g_stream));
    m_clear_mask = std::make_unique<Image1u8>(Image1u8::malloc(W, H, g_stream));
    m_sel3d_mask = std::make_unique<Image1u8>(Image1u8::malloc(W, H, g_stream));

    // Initialize masks
    image_fill(*m_fill_mask, glm::vec<1, uint8_t>(0), g_stream);
    image_fill(*m_clear_mask, glm::vec<1, uint8_t>(0), g_stream);
    _reproject_sel3d(g_stream);

    CHECK_CUDA(cudaStreamSynchronize(g_stream));
}

Sel3d& Sel2d::sel3d() const { return g_app->sel3d(); }

void Sel2d::fill_line(glm::ivec2 p0, glm::ivec2 p1, int r, float scale, glm::vec2 offset, cudaStream_t stream)
{
    dim3 num_blocks = 1;
    dim3 block_dim(16, 16);
    fill_line_kernel<<<num_blocks, block_dim, 0, stream>>>(*m_fill_mask, p0, p1, r, scale, offset);
}

void Sel2d::clear_line(glm::ivec2 p0, glm::ivec2 p1, int r, float scale, glm::vec2 offset, cudaStream_t stream)
{
    image_fill(*m_clear_mask, glm::vec<1, uint8_t>(0), stream);
    {
        dim3 num_blocks = 1;
        dim3 block_dim(16, 16);
        clear_line_kernel<<<num_blocks, block_dim, 0, stream>>>(*m_fill_mask, *m_clear_mask, p0, p1, r, scale, offset);
    }
    // Delete elements of the 3D selection (e.g. disks)
    sel3d().clear(*m_clear_mask, m_camera, stream);
    // Re-project the 3D selection to screen
    _reproject_sel3d(stream);
}

void Sel2d::populate_sel3d(cudaStream_t stream)
{
    printf("[DEBUG] [Sel2d] Populate 3D selection; Unprojecting/filtering/appending 2D points...\n");
    // Unproject points of the fill mask and augment the 3D selection
    sel3d().append(*m_fill_mask, m_camera, m_color_depth, stream);
    // Clear points of the fill mask
    printf("[DEBUG] [Sel2d] Populate 3D selection; Clearing fill mask...\n");
    image_fill(*m_fill_mask, glm::vec<1, uint8_t>(0), stream);
    // Re-project the 3D selection mask to the 2D view
    _reproject_sel3d(stream);
}

void Sel2d::_reproject_sel3d(cudaStream_t stream)
{
    image_fill(*m_sel3d_mask, glm::vec<1, uint8_t>(0), stream);
    sel3d().project(m_camera, m_color_depth, *m_sel3d_mask, stream);
}

void Sel2d::linearize_mask(const Image1u8& mask, thrust::device_vector<glm::ivec2>& out_ss_points, cudaStream_t stream)
{
    out_ss_points.resize(mask.width * mask.height);
    thrust::device_vector<uint32_t> counter(1, 0);
    glm::ivec2* out_sspoints_d = RCGS_TPTR(out_ss_points);
    uint32_t* out_counter_d = RCGS_TPTR(counter);
    image_visit(
        mask,
        [out_sspoints_d, out_counter_d] __device__(const Image1u8& image, int x, int y) mutable {
            if (image.value(x, y).r) {
                uint32_t i = atomicAdd(out_counter_d, 1);
                out_sspoints_d[i].x = x;
                out_sspoints_d[i].y = y;
            }
            return 0; // TODO temporary
        },
        stream);
    size_t num_points = counter[0];
    out_ss_points.resize(num_points);
    printf("[DEBUG] [Sel2d] Mask linearized to %zu SS points\n", num_points);
}
