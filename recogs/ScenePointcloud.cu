#include "ScenePointcloud.h"

#include <fmt/format.h>

#include "App.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "svo/SVOBuilder.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_fill.h"
#include "utils/str_utils.h"

using namespace recogs;

namespace
{
__global__ void unproject_points_to_grid_kernel(Image4fHWC color_depth,
                                                glm::mat4 inv_view,
                                                glm::mat3 inv_K,
                                                float max_depth_threshold,
                                                glm::vec3 scene_min,
                                                glm::vec3 scene_ext,
                                                int octree_resolution,
                                                uint64_t* out_points,
                                                uint32_t* out_num_points)
{

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= color_depth.width || y >= color_depth.height) return;
    // Point is unprojected to world-space using depthmap
    float view_z = color_depth.value(x, y).w; // View space depth
    if (view_z <= 0 || view_z > max_depth_threshold) return;
    glm::vec3 p(x, y, 1.0f);
    p = inv_K * p;
    p *= view_z / p.z;              // View-space
    p = inv_view * glm::vec4(p, 1); // World-space
    // Normalize the point within the scene grid and obtain the voxel location
    glm::ivec3 voxel_loc = ((p - scene_min) / scene_ext) * glm::vec3(float(1 << octree_resolution));
    //printf("voxel loc: %d %d %d\n", voxel_loc.x, voxel_loc.y, voxel_loc.z);
    // Compute the morton code
    uint64_t morton_code = 0;
    for (int level = 0; level < octree_resolution; ++level) {
        morton_code |= (voxel_loc.x & 1) << (level * 3);
        morton_code |= ((voxel_loc.y & 1) << 1) << (level * 3);
        morton_code |= ((voxel_loc.z & 1) << 2) << (level * 3);
        voxel_loc.x >>= 1;
        voxel_loc.y >>= 1;
        voxel_loc.z >>= 1;
    }
    // Append the morton code to the points list
    uint32_t i = atomicAdd(out_num_points, 1);
    out_points[i] = morton_code;
}
} // namespace

void ScenePointcloud::generate(const Scene& scene, const std::vector<GSCamera>& cameras, glm::ivec2 resolution)
{
    CHECK_ARG(!cameras.empty(), "No camera provided");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Adapt cameras to the provided resolution, and upload them
    std::vector<GSCamera> adapted_cameras;
    for (const GSCamera& camera : cameras) {
        GSCamera& adapted_camera = adapted_cameras.emplace_back();
        adapted_camera.copy(camera);
        adapted_camera.set_resolution(resolution.x, resolution.y);
        adapted_camera.update(stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    Image4fHWC colordepth = Image4fHWC::malloc(resolution.x, resolution.y, stream);
    uint32_t max_num_points = resolution.x * resolution.y;
    thrust::device_vector<uint64_t> points_d(max_num_points); // Scene voxels (morton codes)
    thrust::device_vector<uint32_t> num_points_d(1);          // Number of generated voxels
    std::vector<uint64_t> points(max_num_points);             // Number of points

    StereoDepthEstimatorParams stereo_params;
    stereo_params.background_d = m_app.background_d();
    stereo_params.scene = &scene;
    // stereo_params.camera =
    stereo_params.rasterizer = &m_app.gs_rasterizer();
    stereo_params.axis = 0;
    stereo_params.b = 0.07f;
    stereo_params.inout_color_depth = &colordepth;
    stereo_params.stream = stream;
    stereo_params.debug = false;

    StereoDepthEstimator& stereo = m_app.stereo_depth_estimator();

    // Write the unprojected points to a binary file.
    // The content is divided in two sections:
    //     Size: sizeof(uint32_t)
    //     Number of cameras
    //   Offset table:
    //     Size: num_cameras * sizeof(uint32_t):
    //   Points lists:
    //     Size: varying * 3 * sizeof(float)
    //     List of world-space points as sequence of glm::vec3

    size_t num_total_unprojections = 0;

    // PointcloudHashmap pointcloud_hashmap(scene, 14 /* octree_resolution */);

    int octree_resolution = 13;
    // Hashmap for counting the number of occurrences for the same voxel
    std::unordered_map<uint64_t, uint16_t> pointcloud_hashmap;

    for (int camera_idx = 0; camera_idx < adapted_cameras.size(); ++camera_idx) {
        Stopwatch stopwatch;
        const GSCamera& camera = adapted_cameras.at(camera_idx);
        CHECK_STATE(glm::ivec2(camera.resolution()) == resolution, "All cameras must share the same resolution");
        CHECK_STATE(camera.is_uploaded());
        // Clear depth (due to depth testing occurring in StereoDepthEstimator)
        image_fill(colordepth, Image4fCHW::Value{INFINITY}, stream);
        // Estimate depth
        stereo_params.camera = &camera;
        // stereo_params.debug = camera_idx <= 4;
        // stereo_params.debug_image_prefix = fmt::format("scenepcd-{}-", camera_idx);
        stereo.estimate_hv(stereo_params);
        // Unproject points
        CHECK_CUDA(cudaMemsetAsync(RCGS_TPTR(num_points_d), 0, sizeof(uint32_t), stream));
        const float k_max_depth_threshold = 4.0f;
        dim3 num_blocks{};
        num_blocks.x = div_ceil(resolution.x, 16);
        num_blocks.y = div_ceil(resolution.y, 16);
        dim3 block_dim{};
        block_dim.x = 16;
        block_dim.y = 16;
        unproject_points_to_grid_kernel<<<num_blocks, block_dim, 0, stream>>>( //
            colordepth,
            camera.inv_view(),
            camera.inv_K(),
            k_max_depth_threshold,
            scene.min(),
            scene.extent(),
            octree_resolution,
            RCGS_TPTR(points_d),
            RCGS_TPTR(num_points_d));
        // Download to host the number of generated points and the morton codes
        uint32_t num_points;
        CHECK_CUDA(
            cudaMemcpyAsync(&num_points, RCGS_TPTR(num_points_d), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpyAsync(
            points.data(), RCGS_TPTR(points_d), num_points * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        // Compact the generated data in a hashmap
        for (int i = 0; i < num_points; ++i) {
            auto [iterator, inserted] = pointcloud_hashmap.emplace(points[i], 1);
            if (!inserted) {
                ++iterator->second;
            }
        }
        num_total_unprojections += num_points;

        printf("[INFO ] [ScenePointcloud] Camera %4d/%4zu unprojected to %7d/%7d points in %s; Hashmap size: %zu, "
               "Total unprojections: %zu)\n",
               camera_idx + 1,
               adapted_cameras.size(),
               num_points,
               max_num_points,
               stopwatch.elapsed_time_str().c_str(),
               pointcloud_hashmap.size(),
               num_total_unprojections);
    }
    // Convert hashmap voxels to a linear array
    std::vector<uint64_t> linear_voxels;
    linear_voxels.reserve(pointcloud_hashmap.size());
    for (const auto& [point, count] : pointcloud_hashmap) {
        linear_voxels.emplace_back(point);
    }
    // Write points/weights to a binary file
    FILE* f = fopen(m_output_filepath.c_str(), "w");
    uint64_t num_points = pointcloud_hashmap.size();
    fwrite(&num_points, sizeof(uint64_t), 1, f);
    printf("[INFO ] [ScenePointcloud] Writing %zu points to binary file: %s\n", num_points, m_output_filepath.c_str());
    printf("[DEBUG] [ScenePointcloud]   Writing points...\n");
    for (const auto& [morton_code, count] : pointcloud_hashmap) {
        fwrite(&morton_code, sizeof(uint64_t), 1, f);
    }
    printf("[DEBUG] [ScenePointcloud]   Writing weights...\n");
    fclose(f);

    // Build a SVO
    SVOBuilder svo_builder;
    svo_builder.build(linear_voxels.data(), linear_voxels.size(), octree_resolution, stream);

    size_t filesize = get_filesize(m_output_filepath);
    printf("[INFO ] [ScenePointcloud] Generation ended; Cameras: %zu, Points: %zu, File size: %s\n",
           adapted_cameras.size(),
           num_total_unprojections,
           num_bytes_to_string(filesize).c_str());
}
