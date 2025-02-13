
#include <cstdio>
#include <filesystem>

#include "Camera.h"
#include "Scene.h"
#include "rasterizer/rasterizer.h"
#include "scene_io.h"
#include "utils/buffer.h"

using namespace gs_train;

namespace
{
template <typename T>
std::function<char*(size_t N)> resize_functional(Buffer& buffer)
{
    return [&buffer](size_t N) -> char* {
        bool resized = buffer.resize(N * sizeof(T));
        if (resized) {
            return reinterpret_cast<char*>(buffer.data_d);
        } else {
            return nullptr;
        }
    };
};
} // namespace

int main(int argc, char* argv[])
{
    argc--;
    if (argc != 2) {
        fprintf(stderr, "Invalid syntax: %s <scene-ply> <edit-dataset>\n", argv[0]);
        return 1;
    }
    ++argv;

    std::filesystem::path scene_ply = argv[0];
    if (!std::filesystem::exists(scene_ply)) {
        fprintf(stderr, "Invalid scene ply: %s\n", scene_ply.c_str());
        return 1;
    }
    std::filesystem::path edit_dataset_dir = argv[1];
    if (!std::filesystem::exists(edit_dataset_dir)) {
        fprintf(stderr, "Invalid dataset directory: %s\n", edit_dataset_dir.c_str());
        return 1;
    }

    std::filesystem::path output_ply = "output.ply"; // Output written at user's cwd

    Scene scene = read_scene_from_ply(scene_ply);
    Scene scene_d = scene.to_device();
    printf("Scene loaded: %s\n", std::filesystem::absolute(scene_ply).c_str());

    float* background_d;  // TODO
    int screen_w = 512;   // TODO
    int screen_h = 512;   // TODOimg_gt
    float* colorbuffer_d; // TODO
    Camera camera;        // TODO

    Buffer geometry_buffer{"geometryBuffer"};
    Buffer binning_buffer{"binningBuffer"};
    Buffer image_buffer{"imageBuffer"};
    int result = CudaRasterizer::Rasterizer::forward( //
        resize_functional<char>(geometry_buffer),
        resize_functional<char>(binning_buffer),
        resize_functional<char>(image_buffer),
        scene_d.num_vertices,
        3,  // sh_degree
        16, // M
        background_d,
        screen_w,
        screen_h,
        scene.means,
        scene.shs,
        nullptr, // colors_precomp
        scene.opacities,
        scene.scales,
        1.0f, // scale_modifier
        scene.rotations,
        nullptr, // cov3D_precomp
        camera.viewmatrix_d,
        camera.projmatrix_d,
        camera.campos_d,
        camera.tan_fovx,
        camera.tan_fovy,
        false, // prefiltered
        colorbuffer_d,
        nullptr, // radii
        false    // debug
    );
    if (result < 0) { /* TODO handle failure */
    }

    // TODO colorbuffer_d is the prediction

    const float lambda = 0.2f;

    float* img_gt;
    float* pred_diff_gt = colorbuffer_d - img_dt;

    float* dL_dpix; // TODO

    float* dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    float* dL_dmeans2D = torch::zeros({P, 3}, means2D.options());
    float* dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    float* dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    float* dL_dopacity = torch::zeros({P, 1}, means3D.options());
    float* dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    float* dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    float* dL_dscales = torch::zeros({P, 3}, means3D.options());
    float* dL_drotations = torch::zeros({P, 4}, means3D.options());

    CudaRasterizer::Rasterizer::backward( //
        scene.num_vertices,
        3,  // D
        16, // M
        3,  // TODO R
        background_d,
        screen_w,
        screen_h,
        scene.means,
        scene.shs,
        nullptr, // colors_precomp
        scene.scales,
        1.0f, // scale_modifier
        scene.rotations,
        nullptr, // cov3D_precomp
        camera.viewmatrix_d,
        camera.projmatrix_d,
        camera.campos_d,
        camera.tan_fovx,
        camera.tan_fovy,
        nullptr, // radii
        (char*) geometry_buffer.data_d,
        (char*) binning_buffer.data_d,
        (char*) image_buffer.data_d,
        dL_dpix,
        dL_dmeans2D,
        dL_dconic,
        dL_dopacity,
        dL_dcolors,
        dL_dmeans3D,
        dL_dcov3D,
        dL_dsh,
        dL_dscales,
        dL_drotations,
        false // debug
    );

    return 0;
}
