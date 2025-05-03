#pragma once

#include "utils/misc_utils.h"

namespace recogs
{
// Bresenham's Line Algorithm:
// https://youtu.be/CceepU1vIKo?t=923

template <typename PUT_PIXEL>
__device__ void _bresenham_draw_line_h(int x0, int y0, int x1, int y1, PUT_PIXEL put_pixel)
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
__device__ void _bresenham_draw_line_v(int x0, int y0, int x1, int y1, PUT_PIXEL put_pixel)
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

template <typename PUT_PIXEL>
__device__ void bresenham_draw_line(int x0, int y0, int x1, int y1, PUT_PIXEL put_pixel)
{
    if (abs(x1 - x0) > abs(y1 - y0)) {
        _bresenham_draw_line_h<PUT_PIXEL>(x0, y0, x1, y1, put_pixel);
    } else {
        _bresenham_draw_line_v<PUT_PIXEL>(x0, y0, x1, y1, put_pixel);
    }
}
} // namespace gslab
