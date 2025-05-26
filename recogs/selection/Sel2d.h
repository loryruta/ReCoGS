#pragma once

#include <thrust/device_vector.h>

#include "GSCamera.h"
#include "utils/DeviceBuffer.h"
#include "utils/image/Image.h"

namespace recogs
{
// Forward decl
class Sel3d;

/// A class to edit the 3D selection from one 2D view.
/// It permits to:
///   1. Add novel points to the 3D selection.
///   2. Delete existing 3D points from the input 3D selection.
class Sel2d
{
private:
    const GSCamera& m_camera;
    const Image4fHWC& m_color_depth; ///< Rendering of the 3DGS scene and accurate depthmap estimation

    std::unique_ptr<Image1u8> m_fill_mask;
    std::unique_ptr<Image1u8> m_clear_mask;
    std::unique_ptr<Image1u8> m_sel3d_mask;

public:
    explicit Sel2d(const GSCamera& camera, const Image4fHWC& color_depth);
    ~Sel2d() = default;

    [[nodiscard]] const GSCamera& view() const { return m_camera; }
    [[nodiscard]] glm::uvec2 resolution() const { return m_camera.resolution(); }

    [[nodiscard]] const Image1u8& fill_mask() const { return *m_fill_mask; }
    [[nodiscard]] const Image1u8& clear_mask() const { return *m_clear_mask; }
    [[nodiscard]] const Image1u8& sel3d_mask() const { return *m_sel3d_mask; }

    [[nodiscard]] Sel3d& sel3d() const;

    /// Set the pixels in a line between \c p0 and \c p1, given a radius.
    void fill_line(glm::ivec2 p0, glm::ivec2 p1, int radius, float scale, glm::vec2 offset, cudaStream_t stream);
    /// Clear the selection in a line between \c p0 and \c p1 given a radius.
    void clear_line(glm::ivec2 p0, glm::ivec2 p1, int radius, float scale, glm::vec2 offset, cudaStream_t stream);
    /// Unproject every 2D point of the fill mask to the 3D selection.
    void populate_sel3d(cudaStream_t stream);

    void _reproject_sel3d(cudaStream_t stream);

    static void
    linearize_mask(const Image1u8& mask, thrust::device_vector<glm::ivec2>& out_ss_points, cudaStream_t stream);
};
} // namespace recogs
