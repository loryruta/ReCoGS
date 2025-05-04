#include "ScenePointcloud.h"

#include <fmt/format.h>

#include "App.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_fill.h"

using namespace recogs;

__global__ void unproject_points_kernel(Image4fHWC colordepth,
                                        glm::mat4 inv_view,
                                        glm::mat3 inv_K,
                                        float max_depth_threshold,
                                        float* out_unprojected_points,
                                        uint32_t* out_counter)
{

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= colordepth.width || y >= colordepth.height) return;
    float view_z = colordepth.value(x, y).w; // View space depth
    if (view_z <= 0 || view_z > max_depth_threshold) return;
    glm::vec3 p(x, y, 1.0f);
    p = inv_K * p;
    p *= view_z / p.z;              // View-space
    p = inv_view * glm::vec4(p, 1); // World-space
    uint32_t i = atomicAdd(out_counter, 1);
    out_unprojected_points[i * 3 + 0] = p.x;
    out_unprojected_points[i * 3 + 1] = p.y;
    out_unprojected_points[i * 3 + 2] = p.z;
}

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
    thrust::device_vector<float> unproj_points_d(max_num_points * 3);
    thrust::device_vector<uint32_t> unproj_points_counter_d(1);
    std::vector<float> unproj_points(max_num_points * 3);

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

    FILE* f = fopen(m_output_filepath.c_str(), "w");

    uint32_t num_cameras = adapted_cameras.size();
    fwrite(&num_cameras, sizeof(uint32_t), 1, f);
    uint32_t cameras_offset = ftell(f);
    fseek(f, (long) (num_cameras * sizeof(uint32_t)), SEEK_CUR);

    size_t num_total_points = 0;

    for (int camera_idx = 0; camera_idx < adapted_cameras.size(); ++camera_idx) {
        Stopwatch stopwatch;
        const GSCamera& camera = adapted_cameras.at(camera_idx);
        CHECK_STATE(glm::ivec2(camera.resolution()) == resolution, "All cameras must share the same resolution");
        CHECK_STATE(camera.is_uploaded());
        // Clear depth (due to depth testing occurring in StereoDepthEstimator)
        image_fill(colordepth, Image4fCHW::Value{INFINITY}, stream);
        // Estimate depth
        stereo_params.camera = &camera;
        //        stereo_params.debug = camera_idx <= 4;
        //        stereo_params.debug_image_prefix = fmt::format("scenepcd-{}-", camera_idx);
        stereo.estimate_hv(stereo_params);
        // Unproject points
        CHECK_CUDA(cudaMemsetAsync(RCGS_TPTR(unproj_points_counter_d), 0, sizeof(uint32_t), stream));
        const float k_max_depth_threshold = 4.0f;
        dim3 num_blocks{};
        num_blocks.x = div_ceil(resolution.x, 16);
        num_blocks.y = div_ceil(resolution.y, 16);
        dim3 block_dim{};
        block_dim.x = 16;
        block_dim.y = 16;
        unproject_points_kernel<<<num_blocks, block_dim, 0, stream>>>( //
            colordepth,
            camera.inv_view(),
            camera.inv_K(),
            k_max_depth_threshold,
            RCGS_TPTR(unproj_points_d),
            RCGS_TPTR(unproj_points_counter_d));
        uint32_t num_points;
        CHECK_CUDA(cudaMemcpyAsync(
            &num_points, RCGS_TPTR(unproj_points_counter_d), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        // Copy points to host memory for disk saving
        CHECK_CUDA(cudaMemcpyAsync(unproj_points.data(),
                                   RCGS_TPTR(unproj_points_d),
                                   num_points * 3 * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        num_total_points += num_points;

        printf("[INFO ] [ScenePointcloud] Camera %4d/%4zu unprojected to %7d/%7d points in %s (total points: %zu)\n",
               camera_idx + 1,
               adapted_cameras.size(),
               num_points,
               max_num_points,
               stopwatch.elapsed_time_str().c_str(),
               num_total_points);

        // Write the list of unprojected points for the current camera
        uint32_t cur_points_offset = ftell(f);
        fwrite(unproj_points.data(), 3 * sizeof(float), num_points, f);
        uint32_t next_offset = ftell(f);
        // Write where the camera's points are
        uint32_t cur_camera_offset = cameras_offset + camera_idx * sizeof(uint32_t);
        fseek(f, cur_camera_offset, SEEK_SET);
        fwrite(&cur_points_offset, sizeof(uint32_t), 1, f);
        // Back to file's end
        fseek(f, next_offset, SEEK_SET);
    }
    fclose(f);

    size_t filesize = get_filesize(m_output_filepath);
    printf("[INFO ] [ScenePointcloud] Generation ended; Cameras: %zu, Points: %zu, File size: %zu\n",
           adapted_cameras.size(),
           num_total_points,
           filesize);
}
