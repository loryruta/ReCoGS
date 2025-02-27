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
    /// \param[out] out_colorbuffer
    ///     The output colorbuffer to write on. Shape must be (3, H, W)
    virtual void render(Image3fCHW& out_colorbuffer) = 0;
};
} // namespace gs_train
