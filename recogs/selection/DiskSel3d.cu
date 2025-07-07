#include "DiskSel3d.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "App.h"
#include "Sel2d.h"
#include "pcd/remove_statistical_outliers.h"
#include "utils/camera_projections.h"
#include "utils/image/depthmap_to_normalmap.h"

using namespace recogs;

namespace
{
__global__ void init_disks_kernel(const glm::ivec2* ss_points,
                                  const glm::vec3* ws_points,
                                  const uint32_t* filtered_indices,
                                  const uint32_t num_filtered_indices,
                                  const Image4fHWC normal_map,
                                  glm::mat3 inv_V,
                                  Disk* out_disks)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_filtered_indices) return;
    uint32_t idx = filtered_indices[i];
    glm::ivec2 pixel = ss_points[idx];
    glm::vec3 ws_normal = inv_V * normal_map.value(pixel.x, pixel.y);

    // In our disk renderer, disk default orientation is with normal +Z.
    // Here we find the quaternion to rotate such normal to the target world-space normal
    // Reference:
    // https://stackoverflow.com/a/1171995/7358682
    glm::vec3 rot_axis = glm::cross(glm::vec3(0, 0, -1), ws_normal);
    glm::vec4 q{};
    q.x = rot_axis.x;
    q.y = rot_axis.y;
    q.z = rot_axis.z;
    q.w = sqrt(glm::length(ws_normal) + ws_normal.z);

    Disk& disk = out_disks[i];
    disk.position = glm::vec4(ws_points[idx], 1);
    disk.scale = glm::vec2(0.008, 0.008);
    disk.rotation = glm::normalize(q);
}
} // namespace

DiskPcdSel3d::DiskPcdSel3d() {}

void DiskPcdSel3d::append(const Image1u8& fill_mask,
                          const GSCamera& camera,
                          const Image4fHWC& depthmap,
                          cudaStream_t stream)
{
    /* Linearize 2D selection */
    thrust::device_vector<glm::ivec2> ss_points;
    Sel2d::linearize_mask(fill_mask, ss_points, stream);
    /* Unproject to world-space */
    thrust::device_vector<glm::vec3> ws_points(ss_points.size());
    unproject_points(
        ss_points,
        depthmap,
        camera,
        [ws_points_d = RCGS_TPTR(ws_points)] __device__(uint32_t point_idx, const glm::vec3& ws_point) {
            ws_points_d[point_idx] = ws_point;
        },
        stream);
    /* Remove statistical outliers */
    constexpr int k_nb_neighbors = 16;
    constexpr float k_std_ratio = 0.007f;
    constexpr float k_cutoff_radius = 10.0f;
    // Error? Don't worry, it's a fake IDE error (too many templates down there...)
    thrust::device_vector<uint32_t> filtered_indices =
        remove_statistical_outliers<k_nb_neighbors>(ws_points, k_std_ratio, k_cutoff_radius, stream);
    printf("[DEBUG] [DiskSel3d] Append; Statistical outliers removal lead to %zu points\n", filtered_indices.size());
    if (filtered_indices.empty()) return;
    /* Compute normal map */
    Image4fHWC normal_map = Image4fHWC::malloc(depthmap.width, depthmap.height, stream);
    depthmap_to_normalmap<false /* DISPLAY */>(depthmap, normal_map, stream);
    printf("[DEBUG] [DiskSel3d] Append; (%d, %d) normal map computed from depth map\n",
           normal_map.width,
           normal_map.height);
    /* Initialize disks */
    int num_disks = (int) filtered_indices.size();
    printf("[INFO ] [DiskSel3d] Append; Initializing %d disks...\n", num_disks);
    thrust::device_vector<Disk> disks(num_disks);
    dim3 num_blocks = div_ceil<uint32_t>(filtered_indices.size(), 1024u);
    init_disks_kernel<<<num_blocks, 1024, 0, stream>>>( //
        RCGS_TPTR(ss_points),
        RCGS_TPTR(ws_points),
        RCGS_TPTR(filtered_indices),
        filtered_indices.size(),
        normal_map,
        camera.inv_view(),
        RCGS_TPTR(disks));
    /* Extend the disk buffer */
    std::vector<Disk> disks_host(num_disks);
    thrust::copy(thrust::cuda::par.on(stream), disks.begin(), disks.end(), disks_host.begin());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (const Disk& disk : disks_host) m_disk_buffer.emplace_back() = disk;
    /* Upload to GPU (OpenGL) */
    m_disk_buffer.upload();
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
