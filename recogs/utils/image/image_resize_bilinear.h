#pragma once

#include <glm/glm.hpp>

#include "utils/image/Image.h"

BEGIN_NAMESPACE

template <int C, typename T>
__device__ glm::vec<C, T> image_hwc_sample_bilinear(const Image<C, T, ImageMemoryLayout::HWC>& src, float x, float y)
{
    int x0 = (int) floorf(x);
    int y0 = (int) floorf(y);
    int W = src.width;
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, src.height - 1);
    float dx = x - float(x0);
    float dy = y - float(y0);
    glm::vec<C, T> v0_{}; // x = 0
    glm::vec<C, T> v1_{}; // x = 1
    // Horizontal interpolation
    { // x = 0
        glm::vec<C, T> v00{};
        glm::vec<C, T> v01{};
        for (int c = 0; c < C; ++c) {
            v00[c] = src.data_d()[(y0 * W + x0) * C + c];
            v01[c] = src.data_d()[(y0 * W + x1) * C + c];
        }
        v0_ = v00 * (1 - dx) + v01 * dx; // x = 0
    }
    { // x = 1
        glm::vec<C, T> v10{};
        glm::vec<C, T> v11{};
        for (int c = 0; c < C; c++) {
            v10[c] = src.data_d()[(y1 * W + x0) * C + c];
            v11[c] = src.data_d()[(y1 * W + x1) * C + c];
        }
        v1_ = v10 * (1 - dx) + v11 * dx; // x = 1
    }
    // Vertical interpolation
    return v0_ * (1 - dy) + v1_ * dy;
}

END_NAMESPACE