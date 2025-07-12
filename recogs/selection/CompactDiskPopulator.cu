#include "CompactDiskPopulator.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <thrust/gather.h>

#include "ClusterIntGrid.h"
#include "GSCamera.h"
#include "Sel2d.h"
#include "pcd/remove_statistical_outliers.h"
#include "utils/camera_projections.h"
#include "utils/image/Image.h"
#include "utils/murmur_hash.h"
#include "utils/stb_image_write.h"

using namespace recogs;

namespace
{
struct Ray3f {
    glm::vec3 o;
    glm::vec3 d;
};

/// Generate a ray intersecting the given pixel, provided with camera transformations.
__host__ __device__ void
generate_camera_ray(const glm::vec2& pix, const glm::mat4& inv_V, const glm::mat3& inv_K, Ray3f& out_ray)
{
    out_ray.o = inv_V[3]; // Camera world-space position
    out_ray.d = glm::normalize(glm::mat3(inv_V) * inv_K * glm::vec3(pix, 1.f));
}

__host__ __device__ float intersect_ray_plane(const Ray3f& ray, const glm::vec3& po, const glm::vec3& pn)
{
    return glm::dot((po - ray.o), pn) / glm::dot(ray.d, pn);
}

struct Plane {
    glm::vec3 o;
    glm::vec3 n;
};

glm::vec3 unproject_point(const glm::vec2& pix, float d, const glm::mat4& inv_V, const glm::mat3& inv_K)
{
    glm::vec3 pv = inv_K * glm::vec3(pix, 1.f);
    pv *= d / pv.z;                    // View-space
    return inv_V * glm::vec4(pv, 1.f); // World-space
}

bool is_same_plane(const Plane& a, const Plane& b)
{
    // Distance of `b.o` to the plane `a`
    float o_dist = abs(glm::dot(b.o - a.o, a.n));
    // Alignment distance between normals of `a` and `b`
    float n_dist = abs(glm::dot(a.n, b.n) - 1.0f);

    return o_dist < 0.01f && n_dist < 0.01f;
}
} // namespace

std::vector<glm::ivec2> CompactDiskPopulator::_identify_statistical_outliers(const Image1u8& fill_mask,
                                                                             const GSCamera& camera,
                                                                             const Image4fHWC& color_depth,
                                                                             cudaStream_t stream)
{
    // ----------------------------------------------------------------
    /* Linearize 2D selection */
    // ----------------------------------------------------------------

    thrust::device_vector<glm::ivec2> ss_points;
    Sel2d::linearize_mask(fill_mask, ss_points, stream);

    // ----------------------------------------------------------------
    /* Unprojection */
    // ----------------------------------------------------------------

    thrust::device_vector<glm::vec3> ws_points(ss_points.size());
    unproject_points(
        ss_points,
        color_depth,
        camera,
        [ws_points_d = RCGS_TPTR(ws_points)] __device__(uint32_t point_idx, const glm::vec3& ws_point) {
            ws_points_d[point_idx] = ws_point;
        },
        stream);

    // ----------------------------------------------------------------
    /* Remove statistical outliers */
    // ----------------------------------------------------------------

    constexpr int k_nb_neighbors = 16;
    constexpr float k_std_ratio = 0.1f;
    constexpr float k_cutoff_radius = 10.0f;
    // Error? Don't worry, it's a fake IDE error (too many templates down there...)
    thrust::device_vector<uint32_t> filtered_indices =
        remove_statistical_outliers<k_nb_neighbors>(ws_points, k_std_ratio, k_cutoff_radius, stream);

    // ----------------------------------------------------------------
    /* Gather and download to host */
    // ----------------------------------------------------------------

    std::vector<glm::ivec2> filtered_pixels_h(filtered_indices.size());
    {
        thrust::device_vector<glm::ivec2> filtered_pixels(filtered_indices.size());
        thrust::gather(thrust::cuda::par.on(stream),
                       filtered_indices.begin(),
                       filtered_indices.end(),
                       ss_points.begin(),
                       filtered_pixels.begin());
        thrust::copy(
            thrust::cuda::par.on(stream), filtered_pixels.begin(), filtered_pixels.end(), filtered_pixels_h.begin());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    return filtered_pixels_h;
}

void CompactDiskPopulator::_create_disk_at_aabb(const ClusterIntGrid_AABB& aabb,
                                                const GSCamera& camera,
                                                int W,
                                                const std::vector<float>& depths,
                                                const std::vector<float>& normals,
                                                recogs::Disk& out_disk)
{
    glm::mat4 inv_V = camera.inv_view();
    glm::mat3 inv_K = camera.inv_K();

    glm::vec2 pix = (glm::vec2(aabb.max + aabb.min) + 1.0f) * 0.5f; // Pixel center
    glm::vec2 ext = glm::vec2(aabb.max - aabb.min);                 // Extent
    int pix_id = int(pix.y) * W + int(pix.x);

    // ----------------------------------------------------------------
    /* Read depth and normal */
    // ----------------------------------------------------------------

    float depth = depths.at(pix_id * 4 + 3);
    glm::vec3 normal{};
    normal.x = normals.at(pix_id * 4 + 0);
    normal.y = normals.at(pix_id * 4 + 1);
    normal.z = normals.at(pix_id * 4 + 2);

    // ----------------------------------------------------------------
    /* Disk plane */
    // ----------------------------------------------------------------

    glm::vec3 po = unproject_point(pix, depth, inv_V, inv_K);
    glm::vec3 pn = inv_V * glm::vec4(normal, 0);
    pn = glm::normalize(pn); // Needed?
    Ray3f ray{};
    float t;

    // ----------------------------------------------------------------
    /* Disk orientation */
    // ----------------------------------------------------------------

    /* Plane tangent X */
    // 0.70711 = 1/sqrt(2) = radius of inscribed square of side 1
    float rx = ext.x / sqrt(2.0f);
    generate_camera_ray(pix + glm::vec2(rx, 0), inv_V, inv_K, ray);
    t = intersect_ray_plane(ray, po, pn); // Negative t is a rare event
    glm::vec3 tx = (ray.o + t * ray.d) - po;
    float sx = glm::length(tx);
    tx /= sx; // Normalize

    /* Plane tangent Y */
    float ry = ext.y / sqrt(2.0f);
    generate_camera_ray(pix + glm::vec2(0, ry), inv_V, inv_K, ray);
    t = intersect_ray_plane(ray, po, pn); // Negative t is a rare event
    glm::vec3 ty = (ray.o + t * ray.d) - po;
    float sy = glm::length(ty);
    ty /= sy; // Normalize

    out_disk.position = glm::vec4(po, 1);
    out_disk.scale = glm::vec2(sx, sy);
    {
        // The default disk orientation is +Z normal, +Y up, +X right.
        // We construct a rotation matrix (and a quaternion) to transform the disk local frame to the world-space disk
        // frame: plane normal, plane vertical tangent, plane horizontal tangent
        glm::mat3 rot_mat{};
        rot_mat[0] = tx;
        rot_mat[1] = ty;
        rot_mat[2] = pn;
        rot_mat = glm::transpose(rot_mat);
        glm::quat q = glm::toQuat(rot_mat);
        out_disk.rotation.x = q.x;
        out_disk.rotation.y = q.y;
        out_disk.rotation.z = q.z;
        out_disk.rotation.w = q.w;
    }
}

std::unique_ptr<DiskBuffer> CompactDiskPopulator::populate(const Image1u8& fill_mask,
                                                           const GSCamera& camera,
                                                           const Image4fHWC& color_depth,
                                                           const Image4fHWC& normal_map,
                                                           cudaStream_t stream)
{
    int W = (int) fill_mask.width;
    int H = (int) fill_mask.height;

    // ----------------------------------------------------------------
    /* Download maps */
    // ----------------------------------------------------------------

    std::vector<uint8_t> fill_mask_h(W * H, 0);
    std::vector<float> depths_h;
    std::vector<float> normals_h;
    fill_mask.to_host(fill_mask_h, stream);
    color_depth.to_host(depths_h, stream);
    normal_map.to_host(normals_h, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // ----------------------------------------------------------------
    /* Initialize fill mask (after statistical outlier removal) */
    // ----------------------------------------------------------------

//    {
//        std::vector<glm::ivec2> filtered_pixels =
//            _identify_statistical_outliers(fill_mask, camera, color_depth, stream);
//        for (glm::ivec2 pixel : filtered_pixels) {
//            fill_mask_h[pixel.y * W + pixel.x] = 1; // Mark as filled
//        }
//    }

    // ----------------------------------------------------------------
    /* Assign planes */
    // ----------------------------------------------------------------

    glm::mat4 inv_V = camera.inv_view();
    glm::mat3 inv_K = camera.inv_K();

    std::vector<Plane> planes;
    std::vector<int> plane_assignments(W * H, -1 /* Unassigned */);

    // Assign every pixel to a unique plane
    for (int y = 0; y < H; ++y) {
        printf("[DEBUG] [CompactDiskPopulator] Processing scanline %d/%d; Planes: %zu\n", y, H, planes.size());

        for (int x = 0; x < W; ++x) {
            int pix_loc = y * W + x;
            if (!fill_mask_h.at(pix_loc)) continue;

            glm::vec2 pix = glm::vec2(x, y) + 0.5f;
            float d = depths_h[pix_loc * 4 + 3];
            glm::vec3 n;
            n.x = normals_h[pix_loc * 4 + 0];
            n.y = normals_h[pix_loc * 4 + 1];
            n.z = normals_h[pix_loc * 4 + 2];
            Plane plane{};
            plane.o = unproject_point(pix, d, inv_V, inv_K);
            plane.n = inv_V * glm::vec4(n, 0);

            bool plane_assigned = false;
            for (int plane_id = 0; plane_id < planes.size(); ++plane_id) {
                if (is_same_plane(plane, planes.at(plane_id))) {
                    plane_assignments[pix_loc] = plane_id;
                    plane_assigned = true;
                    break;
                }
            }
            if (!plane_assigned) {
                plane_assignments[pix_loc] = (int) planes.size();
                planes.emplace_back(plane);
            }
        }
    }

    // ----------------------------------------------------------------
    /* Find compact rectangular regions */
    // ----------------------------------------------------------------

    ClusterIntGrid cluster_int_grid(W, H, plane_assignments.data());

    std::vector<ClusterIntGrid_AABB> aabbs = cluster_int_grid.cluster();
    printf("[DEBUG] [CompactDiskPopulator] Planes clustered into %zu AABB\n", aabbs.size());

    // ----------------------------------------------------------------
    /* Generate disks */
    // ----------------------------------------------------------------

    std::unique_ptr<DiskBuffer> disk_buffer = std::make_unique<DiskBuffer>();
    for (const ClusterIntGrid_AABB& aabb : aabbs) {
        Disk& disk = disk_buffer->disks.emplace_back();
        _create_disk_at_aabb(aabb, camera, W, depths_h, normals_h, disk);
    }
    disk_buffer->upload(stream);
    printf("[DEBUG] [CompactDiskPopulator] Disk buffer created and uploaded with %zu disks\n", disk_buffer->size());

    return disk_buffer;
}
