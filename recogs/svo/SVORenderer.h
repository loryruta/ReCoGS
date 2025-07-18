#pragma once

#include "Camera.h"
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
    void render(const SVO& svo, const Camera& camera, Image4fHWC& color_depth, cudaStream_t stream);
};
} // namespace recogs
