#pragma once

#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "GSCamera.h"
#include "Selection2d.h"
#include "utils/image/Image.h"

namespace gs_train
{
class Selection3d
{
private:
    thrust::device_vector<glm::vec3> m_points;

public:
    int point_radius = 2;

    explicit Selection3d() = default;
    ~Selection3d() = default;

    [[nodiscard]] const thrust::device_vector<glm::vec3>& points() const { return m_points; };
    [[nodiscard]] bool empty() const { return m_points.empty(); }
    [[nodiscard]] bool size() const { return m_points.size(); }

    /// Unproject the ref. screen-space points to world-space using the depth map and camera parameters.
    /// Then, append the result to the selection pointcloud.
    void append(const thrust::device_vector<uint32_t>& ss_points, const Image1fCHW& depth_map, const GSCamera& camera);

    /// Project the 3D selection to the provided reference map.
    void project(const GSCamera& ss_point, Selection2d::RefMap& out_ref_map) const;

    /// Clear the points signaled to \c true in the provided bitmask.
    void clear(const thrust::device_vector<bool>& clear_bitmask);
};
} // namespace gs_train
