#include "GSFunc.h"

#include "GSCamera.h"
#include "scene_io.h"
#include "utils/cuda_utils.h"
#include "video/CudaImageVisualizer.h"

#define IMG_W 500
#define IMG_H 500

using namespace gs_train;

int main(int argc, char* argv[])
{
    --argc;
    if (argc != 1) {
        fprintf(stderr, "Invalid syntax: %s <scene-ply>\n", argv[0]);
        return 1;
    }
    ++argv;

    std::filesystem::path scene_filepath = argv[0];
    if (!std::filesystem::exists(scene_filepath)) {
        fprintf(stderr, "Invalid scene PLY: %s\n", scene_filepath.c_str());
        return 1;
    }

    Scene scene = read_scene_from_ply(scene_filepath);

    Camera camera{};

    std::vector<float> background{0.f, 0.f, 0.f, 0.f};
    float* background_d = to_device_array<float>(background);

    // Allocate the colorbuffer
    float* colorbuffer_d; // (1, 3, H, W)
    CHECK_CUDA(cudaMalloc(&colorbuffer_d, IMG_W * IMG_H * 3 * sizeof(float)));

    // Create the visualizer
    auto visualizer = CudaImageVisualizer::create(IMG_W, IMG_H, "GS Optimization Test");
    visualizer->set_image(IMG_W, IMG_H, colorbuffer_d, [](int W, int H, const float* img_d, float* out_img_d) {
    });
    visualizer->start();

    GSFunc gs_func{};
    gs_func.forward(IMG_W,
                    IMG_H,
                    background_d,
                    scene.num_vertices,
                    scene.means,
                    scene.shs,
                    scene.opacities,
                    scene.scales,
                    scene.rotations,
                    camera.viewmatrix_d,
                    camera.projmatrix_d,
                    camera.campos_d,
                    camera.tan_fovx,
                    camera.tan_fovy,
                    colorbuffer_d);



    return 0;
}