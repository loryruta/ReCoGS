#pragma once

#include "utils/misc_utils.h"

// Bresenham's Line Algorithm:
// https://youtu.be/CceepU1vIKo?t=923

namespace recogs
{
namespace detail
{
template <typename PUT_PIXEL>
__device__ void bresenham_draw_line_h(int x0, int y0, int x1, int y1, PUT_PIXEL put_pixel)
{
    if (x0 > x1) {
        swap(x0, x1);
        swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    int dir = dy < 0 ? -1 : 1;
    dy *= dir;
    if (dx != 0) {
        int y = y0;
        int p = 2 * dy - dx;
        for (int i = 0; i < dx + 1; ++i) {
            put_pixel(x0 + i, y);
            if (p >= 0) {
                y += dir;
                p -= 2 * dx;
            }
            p += 2 * dy;
        }
    }
}

template <typename PUT_PIXEL>
__device__ void bresenham_draw_line_v(int x0, int y0, int x1, int y1, PUT_PIXEL put_pixel)
{
    if (y0 > y1) {
        swap(x0, x1);
        swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = y1 - y0;
    int dir = dx < 0 ? -1 : 1;
    dx *= dir;
    if (dy != 0) {
        int x = x0;
        int p = 2 * dx - dy;
        for (int i = 0; i < dy + 1; ++i) {
            put_pixel(x, y0 + i);
            if (p >= 0) {
                x += dir;
                p -= 2 * dy;
            }
            p += 2 * dx;
        }
    }
}
} // namespace detail

/// Iterates over a line from point \c p0 to \c p1 according to the Bresenham algorithm,
/// invokes \c put_pixel for every pixel encountered.
/// Execution policy: single-threaded, does not leverage parallelism.
template <typename PUT_PIXEL>
__device__ void bresenham_draw_line(glm::ivec2 p0, glm::ivec2 p1, PUT_PIXEL put_pixel)
{
    if (abs(p1.x - p0.x) > abs(p1.y - p0.y)) {
        detail::bresenham_draw_line_h<PUT_PIXEL>(p0.x, p0.y, p1.x, p1.y, put_pixel);
    } else {
        detail::bresenham_draw_line_v<PUT_PIXEL>(p0.x, p0.y, p1.x, p1.y, put_pixel);
    }
}

/// Iterates over a line from point \c p0 to \c p1, with a radius \c r, according to the Bresenham algorithm;
/// invokes \c callback for every pixel encountered.
/// Execution policy: the line progress is sequential, "thickness" pixels are processed in parallel.
template <typename PUT_PIXEL>
__device__ void bresenham_draw_line_radius(glm::ivec2 p0, glm::ivec2 p1, int r, PUT_PIXEL callback)
{
    const int numitems_x = div_ceil(r * 2, int(blockDim.x));
    const int numitems_y = div_ceil(r * 2, int(blockDim.y));
    bresenham_draw_line(p0, p1, [&] __device__(int cx, int cy) mutable {
        int sx = cx - r + int(threadIdx.x) * numitems_x;
        int sy = cy - r + int(threadIdx.y) * numitems_y;
        int ex = sx + numitems_x;
        int ey = sy + numitems_y;
        int r2 = r * r;
        for (int x = sx; x <= ex; ++x) {
            for (int y = sy; y <= ey; ++y) {
                int d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if (d2 <= r2) callback(x, y);
            }
        }
    });
}
} // namespace recogs
