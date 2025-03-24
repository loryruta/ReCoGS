#pragma once

#include "utils/image/Image.h"

namespace gs_train
{
class Screen
{
public:
    virtual ~Screen() = default;

    /// A unique name for the screen (usually the Screen's class name).
    [[nodiscard]] virtual const char* name() const = 0;

    virtual void resize(int width, int height) = 0;
    virtual void update(float dt) = 0;
    virtual void render(Image4fHWC& out_color_depth) = 0;
    virtual void ui() = 0;
};
} // namespace gs_train
