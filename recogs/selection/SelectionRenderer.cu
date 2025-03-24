#include "SelectionRenderer.h"

#include "utils/camera_projections.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_visit_transform.h"

using namespace gs_train;

void SelectionRenderer::render( //
    const Selection3d& selection3d,
    const GSCamera& camera,
    Image4fHWC& out_color_depth,
    cudaStream_t stream)
{
    if (selection3d.empty()) return;

    // TODO the same function can also be used to apply_edit

    // clang-format off
    auto f = [
        r = selection3d.point_radius,
        color = selection3d_color,
        color_depth = out_color_depth
    ] __device__(int x, int y, float view_z) mutable { // TODO use selection3d.project(...)
        uint32_t W = color_depth.width;
        uint32_t H = color_depth.height;
        uint32_t z = __float_as_uint(view_z);
        for (int ry = y - r; ry <= y + r; ++ry) {
            for (int rx = x - r; rx <= x + r; ++rx) {
                if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
                    uint32_t* depth_ptr = reinterpret_cast<uint32_t*>(&color_depth.data_d()[ry * W * 4 + rx * 4 + 3]);
                    // Depth test
                    uint32_t old_z = atomicMin(depth_ptr, z);
                    if (old_z > z) {
                        color_depth.set_value(rx, ry, glm::vec4(color, z));
                    }
                }
            }
        }
    };
    // clang-format on
    project_points(selection3d.points(), camera, f, stream);
}

void SelectionRenderer::render( //
    const Selection2d& selection2d,
    Image4fHWC& out_color_depth,
    cudaStream_t stream)
{
    CHECK_ARG(selection2d.resolution() == out_color_depth.size());
    // Fill the colorbuffer with the 3D selection points (reference map)
    render(selection2d.selection3d(), selection2d.view(), out_color_depth, stream);
    // Fill the colorbuffer with the current view stroke (new map)
    image_visit(
        selection2d.new_map(),
        [out_color_depth,
         color = selection2d_color] __device__(const Selection2d::NewMap& new_map, int x, int y) mutable {
            if (new_map.value(x, y).r != 0) {
                out_color_depth.set_value(x, y, glm::vec4(color, 0));
            }
            return 0; // TODO temporary
        },
        stream);
}
