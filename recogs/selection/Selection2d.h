#pragma once

#include <thrust/device_vector.h>

#include "GSCamera.h"
#include "utils/DeviceBuffer.h"
#include "utils/image/Image.h"

namespace gs_train
{
// Forward decl
class Selection3d;

/// A class allowing to edit the 3D selection from one view. It allows to:
/// 1. Add/delete screen-space points that are then unprojected to 3D
/// 2. Delete existing 3D points from the input 3D selection
class Selection2d
{
public:
    using NewMap = Image<1, uint8_t, ImageMemoryLayout::CHW>;
    using RefMap = Image<1, uint32_t, ImageMemoryLayout::CHW>;

private:
    Selection3d& m_selection3d;
    const GSCamera m_view;
    NewMap m_new_map; ///< Map used to store newly added points on the current view
    RefMap m_ref_map; ///< Map used to store references to the selection 3D points

    /// A bitmask telling which 3D selection point has to be cleared.
    thrust::device_vector<bool> m_clear_bitmask;

public:
    explicit Selection2d(Selection3d& selection3d, GSCamera view);
    ~Selection2d() = default;

    [[nodiscard]] const Selection3d& selection3d() const { return m_selection3d; }
    [[nodiscard]] const GSCamera& view() const { return m_view; }
    [[nodiscard]] glm::uvec2 resolution() const { return m_view.resolution(); }

    [[nodiscard]] const NewMap& new_map() const { return m_new_map; }
    [[nodiscard]] const RefMap& ref_map() const { return m_ref_map; }

    /// Set the pixels in a line between \c p0 and \c p1, given a radius.
    void fill_line(glm::ivec2 p0, glm::ivec2 p1, int radius);

    /// Clear the points in a line between \c p0 and \c p1 given a radius.
    /// \param hard
    ///     If set, delete points even from the 3D selection
    void clear_line(glm::ivec2 p0, glm::ivec2 p1, int radius, bool hard = false);

    /// Populate the 3D selection with new points:
    /// drain "new map" by unprojecting its points and adding them to the 3D selection.
    /// After this call this Selection2d object becomes invalid.
    void populate_selection3d(const Image1fCHW& depth);

private:
    void reproject_ref_map();
};
} // namespace gs_train
