#include "adam.h"

#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "gs_loss.h"
#include "utils/cuda_utils.h"
#include "utils/image_layout_transition.h"
#include "utils/misc_utils.h"
#include "utils/stb_image.h"
#include "video/DrawTexture.h"
#include "video/GLMappedResource.h"
#include "video/Window.h"

using namespace gs_train;

/// Prediction visualization code
/// \param img
///     The image to display; with shape (1, 4, H, W)
void vis_main(Window& window, int H, int W, float* img)
{
    window.make_context();

    GLMappedResource gl_mapped_resource(W, H);
    DrawTexture draw_texture{};

    float* img_bhwc; // (1, H, W, 4)
    CHECK_CUDA(cudaMalloc(&img_bhwc, H * W * 4 * sizeof(float)));

    while (!window.should_close()) {
        window.poll_events();

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        /* Transit the image to BHWC */
        transit_image_layout<ImageLayout::BCHW, ImageLayout::BHWC>(1, 4, H, W, img, img_bhwc);
        CHECK_CUDA(cudaDeviceSynchronize());

        gl_mapped_resource.write(img_bhwc);
        auto [fb_w, fb_h] = window.framebuffer_size();
        draw_texture.draw(gl_mapped_resource.texture(), 0, 0, fb_w, fb_h);

        window.swap_buffers();
    }
}

float* load_gt_image(const std::filesystem::path& gt_filepath, int& B, int& C, int& H, int& W)
{
    /* Load the image */
    printf("Loading GT image: %s\n", gt_filepath.c_str());
    B = 1;
    C = 4;
    // H
    // W
    int C_in_file;
    float* img_gt = stbi_loadf(gt_filepath.c_str(), &W, &H, &C_in_file, C);
    CHECK_STATE(img_gt, "GT image loading failed");
    size_t BCHW = B * C * H * W;

    /* Upload the image to GPU (BHWC memory layout) */
    float* img_gt_bhwc_d = to_device_array(std::span(img_gt, BCHW)); // (1, H, W, 4)
    stbi_image_free(img_gt);

    /* Transition from BHWC to BCHW */
    float* img_gt_bchw_d;
    CHECK_CUDA(cudaMalloc(&img_gt_bchw_d, BCHW * sizeof(float)));
    transit_image_layout<ImageLayout::BHWC, ImageLayout::BCHW>(B, C, H, W, img_gt_bhwc_d, img_gt_bchw_d);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(img_gt_bhwc_d));

    return img_gt_bchw_d;
}

int main(int argc, char* argv[])
{
    argc--;
    if (argc != 1) {
        fprintf(stderr, "Invalid syntax: %s <gt-image>\n", argv[0]);
        exit(1);
    }
    argv++;

    std::filesystem::path img_gt_filepath = argv[0];
    CHECK_ARG(std::filesystem::exists(img_gt_filepath), "Invalid GT image path: %s", img_gt_filepath.c_str());

    // Load the GT image (target image)
    int B, C, H, W;
    float* img_gt = load_gt_image(img_gt_filepath, B, C, H, W); // (B, C, H, W)
    size_t BCHW = B * C * H * W;
    printf("Image loaded; Shape (%d, %d, %d, %d) -> Total size: %zu\n", B, C, H, W, BCHW);

    // Init prediction (i.e. parameters)
    float* img_pred; // (B, C, H, W)
    CHECK_CUDA(cudaMalloc(&img_pred, BCHW * sizeof(float)));
    CHECK_CUDA(cudaMemset(img_pred, 0, BCHW * sizeof(float)));

    // Init prediction gradients
    float* img_pred_grads; // (B, C, H, W)
    CHECK_CUDA(cudaMalloc(&img_pred_grads, BCHW * sizeof(float)));
    CHECK_CUDA(cudaMemset(img_pred_grads, 0, BCHW * sizeof(float)));

    // Create optimizer
    printf("Creating Adam optimizer...\n");
    Adam::ParamSet param_set{};
    param_set.params = img_pred;
    param_set.grads = img_pred_grads;
    param_set.lr = 0.001f;
    param_set.num_params = BCHW;
    std::vector<Adam::ParamSet> param_sets{};
    param_sets.emplace_back(param_set);
    Adam::Options options{};
    options.eps = 1e-15f;
    std::unique_ptr<Adam> adam = std::make_unique<Adam>(param_sets, options);
    printf("Adam optimizer initialized\n");

    Window window(W, H, "Image Optimization Test", false);
    std::thread vis_thread([&]() { vis_main(window, H, W, img_pred); });

    // Optimization loop
    GSLoss loss_fn{};
    for (int iter = 0; iter < 10000; ++iter) {
        float* loss_d = loss_fn.forward(B, C, H, W, img_pred, img_gt);
        float loss = to_host(loss_d);
        printf("  Iter. %05d -> Loss: %.8f\n", iter, loss);
        loss_fn.backward(B, C, H, W, img_pred, img_gt, img_pred_grads /* output */);
        adam->step();
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    printf("Bye bye!\n");

    window.set_should_close(true);
    vis_thread.join();

    adam.reset(); // Delete the optimizer first
    CHECK_CUDA(cudaFree(img_pred));
    CHECK_CUDA(cudaFree(img_pred_grads));

    return 0;
}
