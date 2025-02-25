#pragma once

#include "utils/image/Image.h"

namespace gs_train
{
class Screen
{
public:
    virtual void on_enable() {};
    virtual void on_disable() {};

    virtual void resize(int width, int height) = 0;
    virtual void update(float dt) = 0;
    /// \param[out] out_colorbuffer
    ///     The output colorbuffer to write on. Shape must be (3, H, W)
    virtual void render(Image3fCHW& out_colorbuffer) = 0;
};
} // namespace gs_train
