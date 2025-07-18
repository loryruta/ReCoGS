#pragma once

#include <glm/glm.hpp>
#include <thrust/device_vector.h>

#include "Camera.h"
#include "Sel3d.h"
#include "selection/Sel2d.h"
#include "utils/image/Image.h"

namespace recogs
{
/// A screen-space pointcloud selection: the points are projected to screen and a screen-space radius is applied.
/// Similar to what OpenGL does when rendering in GL_POINT mode.
class SSPcdSel3d : public Sel3d
{
private:
    thrust::device_vector<glm::vec3> m_points;

public:
    int point_radius = 2; ///< Screen-space radius of a point

    explicit SSPcdSel3d() = default;
    ~SSPcdSel3d() = default;

    [[nodiscard]] const thrust::device_vector<glm::vec3>& points() const { return m_points; };
    [[nodiscard]] bool empty() const { return m_points.empty(); }
    [[nodiscard]] size_t size() const { return m_points.size(); }

    void append(const Image1u8& fill_mask,
                const Camera& camera,
                const Image4fHWC& color_depth,
                cudaStream_t stream) override;

    void clear(const Image1u8& clear_mask, const Camera& camera, cudaStream_t stream) override;

    void project(const Camera& camera,
                 const Image4fHWC& depthmap,
                 Image1u8& sel3d_mask,
                 cudaStream_t stream) const override;
};
} // namespace recogs
