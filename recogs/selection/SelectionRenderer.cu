#include "SelectionRenderer.h"

#include "utils/camera_projections.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_visit_transform.h"

using namespace gs_train;

void SelectionRenderer::render( //
    const Selection3d& selection3d,
    const GSCamera& camera,
    Image3fCHW& out_colorbuffer,
    Image1fCHW& inout_depthbuffer,
    cudaStream_t stream)
{
    if (selection3d.empty()) return;

    // TODO the same function can also be used to apply_edit

    // clang-format off
    auto f = [
        r = selection3d.point_radius,
        color = selection3d_color,
        colorbuffer = out_colorbuffer,
        depthbuffer = inout_depthbuffer.data_d()
    ] __device__(int x, int y, float view_z) mutable { // TODO use selection3d.project(...)
        uint32_t W = colorbuffer.width;
        uint32_t H = colorbuffer.height;
        uint32_t z = __float_as_uint(view_z);
        for (int ry = y - r; ry <= y + r; ++ry) {
            for (int rx = x - r; rx <= x + r; ++rx) {
                if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
                    uint32_t* depth_ptr = (uint32_t*) &depthbuffer[ry * W + rx];
                    // Depth test
                    uint32_t old_z = atomicMin(depth_ptr, z);
                    if (old_z > z) {
                        colorbuffer.set_value(rx, ry, color);
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
    Image3fCHW& out_colorbuffer,
    Image1fCHW& inout_depthbuffer,
    cudaStream_t stream)
{
    CHECK_ARG(selection2d.resolution() == out_colorbuffer.size());
    // Fill the colorbuffer with the 3D selection points (reference map)
    render(selection2d.selection3d(), selection2d.view(), out_colorbuffer, inout_depthbuffer, stream);
    // Fill the colorbuffer with the current view stroke (new map)
    image_visit(
        selection2d.new_map(),
        [out_colorbuffer,
         color = selection2d_color] __device__(const Selection2d::NewMap& new_map, int x, int y) mutable {
            if (new_map.value(x, y).r != 0) {
                out_colorbuffer.set_value(x, y, color);
            }
            return 0; // TODO temporary
        },
        stream);
}
