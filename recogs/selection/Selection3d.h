#pragma once

#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "GSCamera.h"
#include "Selection2d.h"
#include "utils/image/Image.h"

namespace recogs
{
class Selection3d
{
    friend class Selection2d;

private:
    App& m_app;
    thrust::device_vector<glm::vec3> m_points;

public:
    /// Screen-space radius of a point
    int point_radius = 2;

    explicit Selection3d(App& app);
    ~Selection3d() = default;

    [[nodiscard]] const thrust::device_vector<glm::vec3>& points() const { return m_points; };
    [[nodiscard]] bool empty() const { return m_points.empty(); }
    [[nodiscard]] bool size() const { return m_points.size(); }

    /// Unproject the ref. screen-space points to world-space using the depth map and camera parameters.
    /// Then, append the result to the selection pointcloud.
    void append(const thrust::device_vector<uint32_t>& ss_points, const Image4fHWC& color_depth, const GSCamera& camera);

    /// Project the 3D selection to screen and invoke a device callback for any projected point.
    /// Any point outside is clipped.
    /// The thread "global index" (i.e. `block * block_dim + thread_idx`) is the pointcloud point index.
    template <typename CALLBACK>
    void project(const GSCamera& camera, CALLBACK callback, cudaStream_t stream) const;

    /// Project the 3D selection to the provided reference map.
    void project(const GSCamera& camera, Selection2d::RefMap& out_ref_map) const;

    /// Clear the points signaled to \c true in the provided bitmask.
    void clear(const thrust::device_vector<bool>& clear_bitmask, cudaStream_t stream);
};
} // namespace recogs

#include "Selection3d.inl"
