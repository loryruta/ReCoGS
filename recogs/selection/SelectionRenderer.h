#pragma once

#include "Selection2d.h"
#include "Selection3d.h"
#include "utils/image/Image.h"

namespace recogs
{
class SelectionRenderer
{
public:
    glm::vec3 selection3d_color = glm::vec3(0.8f, 0.8f, 0.8f);
    glm::vec3 selection2d_color = glm::vec3(1.0f, 1.0f, 1.0f);

    explicit SelectionRenderer() = default;
    ~SelectionRenderer() = default;

    void render( //
        const Selection3d& selection3d,
        const GSCamera& camera,
        Image4fHWC& out_color_depth,
        cudaStream_t stream);

    void render( //
        const Selection2d& selection2d,
        Image4fHWC& out_color_depth,
        cudaStream_t stream);
};
} // namespace recogs
