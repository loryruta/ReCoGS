#pragma once

#include "GSCamera.h"
#include "SVONode.h"
#include "utils/AABB.h"
#include "utils/image/Image.h"

namespace recogs
{
class SVORenderer
{
public:
    explicit SVORenderer() = default;
    ~SVORenderer() = default;

    /// \param svo_d SVO allocated on GPU memory
    /// \param color_depth Output color/depth buffer
    void render(const AABB3f& svo_minmax, const SVONode* svo_d, const GSCamera& camera, Image4fHWC color_depth);
};
} // namespace recogs
