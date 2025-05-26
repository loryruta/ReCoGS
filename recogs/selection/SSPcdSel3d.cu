#include "SSPcdSel3d.h"

#include "pcd/remove_statistical_outliers.h"
#include "pcd/render_ss_points.h"
#include "utils/camera_projections.h"

using namespace recogs;

void SSPcdSel3d::append(const Image1u8& fill_mask,
                        const GSCamera& camera,
                        const Image4fHWC& color_depth,
                        cudaStream_t stream)
{
    CHECK_ARG(color_depth.size() == camera.resolution(), "depth and view must have the same resolution");
    // Linearize 2D selection to a list of 2D screen-space coordinates
    thrust::device_vector<glm::ivec2> ss_points;
    Sel2d::linearize_mask(fill_mask, ss_points, stream);
    // Unproject points to world-space
    thrust::device_vector<glm::vec3> new_points(ss_points.size());
    unproject_points(
        ss_points,
        color_depth,
        camera,
        [new_points_d = RCGS_TPTR(new_points)] __device__(uint32_t point_idx, const glm::vec3& ws_point) {
            new_points_d[point_idx] = ws_point;
        },
        stream);
    // Compact/filter newly added points (remove statistical outliers)
    size_t old_points_count = new_points.size();
    constexpr int k_nb_neighbors = 16;
    constexpr float k_std_ratio = 0.007f;
    constexpr float k_cutoff_radius = 10.0f;
    // Error? Don't worry, it's a fake IDE error (too many templates down there...)
    new_points = remove_statistical_outliers<k_nb_neighbors>(new_points, k_std_ratio, k_cutoff_radius, stream);
    printf("[DEBUG] [SSPcdSel3d] Compaction from %zu to %zu points\n", old_points_count, new_points.size());
    // Append the new points
    m_points.insert(m_points.end(), new_points.begin(), new_points.end());
    printf("[INFO ] [SSPcdSel3d] 3D selection pointcloud has %zu points\n", m_points.size());
}

void SSPcdSel3d::clear(const Image1u8& clear_mask, const GSCamera& camera, cudaStream_t stream)
{
    thrust::device_vector<bool> clear_bitmask(m_points.size(), false /* value */);
    // Project the points to screen, if they overlap the clear mask, then mark them (in a bitmask) to be deleted
    bool* clear_bitmask_d = RCGS_TPTR(clear_bitmask);
    auto f = [clear_mask, clear_bitmask_d] __device__(uint32_t idx, int2 pixel, float z) {
        if (pixel.x < 0 || pixel.x >= clear_mask.width || pixel.y < 0 || pixel.y >= clear_mask.height) return;
        if (clear_mask.value(pixel.x, pixel.y).r) {
            clear_bitmask_d[idx] = true;
        }
    };
    project_points(m_points, camera, f, stream);
    // Delete the points
    thrust::device_vector<glm::vec3> new_points(m_points.size());
    auto result_end = thrust::copy_if( //
        thrust::cuda::par.on(stream),
        m_points.begin(),
        m_points.end(),
        clear_bitmask.begin(),
        new_points.begin(),
        thrust::logical_not<bool>());
    m_points = std::move(new_points);
}

void SSPcdSel3d::project(const GSCamera& camera,
                         const Image4fHWC& depthmap,
                         Image1u8& sel3d_mask,
                         cudaStream_t stream) const
{
    auto f = [sel3d_mask] __device__(uint32_t point_idx, int2 pixel, float view_z) mutable {
        sel3d_mask.set_value(pixel.x, pixel.y, glm::vec<1, uint8_t>(1));
    };
    render_ss_points(m_points, camera, point_radius, depthmap, true /* test_depth */, f, stream);
}
