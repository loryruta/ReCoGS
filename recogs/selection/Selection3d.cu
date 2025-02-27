#include "Selection3d.h"

#include "utils/camera_projections.h"
#include "utils/remove_statistical_outliers.h"

using namespace gs_train;

void Selection3d::append( //
    const thrust::device_vector<uint32_t>& ss_points,
    const Image1fCHW& depth_map,
    const GSCamera& camera)
{
    CHECK_ARG(depth_map.size() == camera.resolution(), "depth and view must have the same resolution");

    thrust::device_vector<glm::vec3> new_points(ss_points.size());

    // Unproject points to world-space
    thrust::device_vector<uint32_t> counter(1, 0);
    glm::vec3* new_points_d = thrust::raw_pointer_cast(new_points.data());
    uint32_t* counter_d = thrust::raw_pointer_cast(counter.data());
    unproject_points(ss_points, depth_map, camera, [new_points_d, counter_d] __device__(const glm::vec3& ws_point) {
        uint32_t i = atomicAdd(counter_d, 1);
        new_points_d[i] = ws_point;
    });

    // Compact newly added points
    size_t old_points_count = new_points.size();
    constexpr int k_nb_neighbors = 16;
    constexpr float k_std_ratio = 0.007f;
    constexpr float k_cutoff_radius = 10.0f;
    // Error? Don't worry, it's a fake IDE error (too many templates down there...)
    new_points = remove_statistical_outliers<k_nb_neighbors>(new_points, k_std_ratio, k_cutoff_radius);
    printf("[INFO ] [Selection3d] Compaction from %zu to %zu points\n", old_points_count, m_points.size());

    // Append the new points
    m_points.insert(m_points.end(), new_points.begin(), new_points.end());
}

void Selection3d::project(const GSCamera& camera, Selection2d::RefMap& out_ref_map) const
{
    if (m_points.empty()) return;

    project_points(m_points, camera, [r = point_radius, out_ref_map] __device__(glm::ivec2 ss_point, float) mutable {
        uint32_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t W = out_ref_map.width;
        uint32_t H = out_ref_map.height;
        for (int ry = ss_point.y - r; ry <= ss_point.y + r; ++ry) {
            for (int rx = ss_point.x - r; rx <= ss_point.x + r; ++rx) {
                if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
                    out_ref_map.set_value(rx, ry, Selection2d::RefMap::Value{point_idx});
                }
            }
        }
    });
}

void Selection3d::clear(const thrust::device_vector<bool>& clear_bitmask)
{
    CHECK_ARG(m_points.size() == clear_bitmask.size());
    thrust::device_vector<glm::vec3> new_points;
    thrust::copy_if(
        m_points.begin(), m_points.end(), clear_bitmask.begin(), new_points.begin(), thrust::logical_not<bool>());
    m_points = std::move(new_points);
}
