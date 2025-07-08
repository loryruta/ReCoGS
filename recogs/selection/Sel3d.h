#pragma once

#include <nvfunctional>

#include "GSCamera.h"
#include "utils/image/Image.h"

namespace recogs
{
/// \brief A class describing a 3D selection in world-space.
class Sel3d
{
public:
    explicit Sel3d() = default;
    ~Sel3d() = default;

    /// Check whether the 3D selection is empty or not.
    virtual bool empty() const = 0;

    /// Append the points set in the fill mask to the 3D selection.
    /// \param fill_mask Mask holding the 2D points to populate the 3D selection
    /// \param camera
    /// \param depthmap  Depthmap used for unprojection
    /// \param stream
    virtual void
    append(const Image1u8& fill_mask, const GSCamera& camera, const Image4fHWC& depthmap, cudaStream_t stream) = 0;

    /// Clear the 3D selection where indicated by the clear mask.
    /// \param clear_mask Mask where set bits mean to clear the underlying representation
    /// \param camera
    /// \param stream
    virtual void clear(const Image1u8& clear_mask, const GSCamera& camera, cudaStream_t stream) = 0;

    /// Project the 3D selection to the 3D selection mask.
    /// \param camera
    /// \param depthmap   Depthmap used for depth-testing
    /// \param sel3d_mask A binary mask where the 3D selection is projected to
    /// \param stream
    virtual void
    project(const GSCamera& camera, const Image4fHWC& depthmap, Image1u8& sel3d_mask, cudaStream_t stream) const = 0;
};
} // namespace recogs
