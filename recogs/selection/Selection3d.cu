#include "Selection3d.h"

#include "App.h"
#include "utils/remove_statistical_outliers.h"

using namespace recogs;

Selection3d::Selection3d(App& app) : m_app(app) {}

void Selection3d::append(const thrust::device_vector<uint32_t>& ss_points,
                         const Image4fHWC& color_depth,
                         const GSCamera& camera)
{
    CHECK_ARG(color_depth.size() == camera.resolution(), "depth and view must have the same resolution");

    thrust::device_vector<glm::vec3> new_points(ss_points.size());

    // Unproject points to world-space
    thrust::device_vector<uint32_t> counter(1, 0);
    glm::vec3* new_points_d = thrust::raw_pointer_cast(new_points.data());
    uint32_t* counter_d = thrust::raw_pointer_cast(counter.data());
    unproject_points(
        ss_points,
        color_depth,
        camera,
        [new_points_d, counter_d] __device__(const glm::vec3& ws_point) {
            uint32_t i = atomicAdd(counter_d, 1);
            new_points_d[i] = ws_point;
        },
        m_app.stream());

    // Compact newly added points
    size_t old_points_count = new_points.size();
    constexpr int k_nb_neighbors = 16;
    constexpr float k_std_ratio = 0.007f;
    constexpr float k_cutoff_radius = 10.0f;
    // Error? Don't worry, it's a fake IDE error (too many templates down there...)
    new_points = remove_statistical_outliers<k_nb_neighbors>(new_points, k_std_ratio, k_cutoff_radius, m_app.stream());
    printf("[INFO ] [Selection3d] Compaction from %zu to %zu points\n", old_points_count, m_points.size());

    // Append the new points
    m_points.insert(m_points.end(), new_points.begin(), new_points.end());
}

void Selection3d::project(const GSCamera& camera, Selection2d::RefMap& out_ref_map) const
{
    project(
        camera,
        [out_ref_map] __device__(uint32_t x, uint32_t y, float) mutable {
            uint32_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
            out_ref_map.set_value(x, y, Selection2d::RefMap::Value{point_idx});
        },
        m_app.stream());
}

void Selection3d::clear(const thrust::device_vector<bool>& clear_bitmask, cudaStream_t stream)
{
    CHECK_ARG(m_points.size() == clear_bitmask.size());
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
