#include "DiskSel3d.h"

#include "App.h"
#include "PcdSel3d.h"

using namespace recogs;

DiskPcdSel3d::DiskPcdSel3d() {}

void DiskPcdSel3d::append(const Image1u& edit_mask,
                          const GSCamera& camera,
                          const Image4fHWC& color_depth,
                          cudaStream_t stream)
{


}

void DiskPcdSel3d::clear(const Image1u& edit_mask,
                         const GSCamera& camera,
                         const Image4fHWC& color_depth,
                         cudaStream_t stream)
{
    auto clear_fn = [] __device__ (uint2 pixel, float* out_color) {
        // TODO save disk idx
    };
    // TODO test depth
    // TODO don't save depth
    App::g().disk_rasterizer().forward(*m_disks, camera, color_depth, clear_fn, stream);
}

void DiskPcdSel3d::project(const GSCamera& camera,
                           Image4fHWC& color_depth,
                           const FillPixelT& fill_pixel_fn,
                           cudaStream_t stream)
{
    App::g().disk_rasterizer().forward(*m_disks, camera, color_depth, fill_pixel_fn, stream);
}
