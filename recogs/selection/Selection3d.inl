#include "utils/camera_projections.h"

namespace recogs
{
template <typename CALLBACK>
void Selection3d::project(const GSCamera& camera, CALLBACK callback, cudaStream_t stream) const
{
    if (m_points.empty()) return;

    uint32_t W = camera.width;
    uint32_t H = camera.height;
    project_points(
        m_points,
        camera,
        [W, H, r = point_radius, callback] __device__(int x, int y, float view_z) mutable {
            for (int ry = y - r; ry <= y + r; ++ry) {
                for (int rx = x - r; rx <= x + r; ++rx) {
                    if (rx >= 0 && rx < W && ry >= 0 && ry < H) {
                        callback(uint32_t(rx), uint32_t(ry), view_z);
                    }
                }
            }
        },
        stream);
}
} // namespace recogs
