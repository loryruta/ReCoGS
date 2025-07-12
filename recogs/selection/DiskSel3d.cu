#include "DiskSel3d.h"

#include "App.h"
#include "CompactDiskPopulator.h"
#include "Sel2d.h"
#include "utils/image/depthmap_to_normalmap.h"

using namespace recogs;

DiskPcdSel3d::DiskPcdSel3d() {}

void DiskPcdSel3d::append(const Image1u8& fill_mask,
                          const GSCamera& camera,
                          const Image4fHWC& color_depth,
                          cudaStream_t stream)
{
    Image4fHWC normal_map = Image4fHWC::malloc(color_depth.width, color_depth.height, stream);
    depthmap_to_normalmap<false /* DISPLAY */>(color_depth, normal_map, stream);

    CompactDiskPopulator compact_disk_populator;
    m_disk_buffer = compact_disk_populator.populate(fill_mask, camera, color_depth, normal_map, stream);
}

void DiskPcdSel3d::clear(const Image1u8& clear_mask, const GSCamera& camera, cudaStream_t)
{
    // TODO
}

void DiskPcdSel3d::project(const GSCamera& camera,
                           const Image4fHWC& depthmap,
                           Image1u8& sel3d_mask,
                           cudaStream_t stream) const
{
    // TODO
    //    auto f = [depthmap, sel3d_mask] __device__(uint2 pixel, float view_z, float2 uv) mutable {
    //        int W = (int) depthmap.width;
    //        // Depth test
    //        uint32_t view_z_u32 = __float_as_uint(view_z);
    //        uint32_t* depth_ptr = reinterpret_cast<uint32_t*>(&depthmap.data_d()[(pixel.y * W + pixel.x) * 4 + 3]);
    //        uint32_t old_z = atomicMin(depth_ptr, view_z_u32);
    //        if (old_z <= view_z_u32) return;
    //        // If depth test passed, set the Selection 3D mask
    //        sel3d_mask.set_value(pixel.x, pixel.y, glm::vec<1, uint8_t>(1));
    //    };
    //    g_app->disk_rasterizer().rasterize(m_disks, camera, f, stream);
}
