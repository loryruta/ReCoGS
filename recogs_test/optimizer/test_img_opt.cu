#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include "optimizer/Adam.h"
#include "optimizer/GSLoss.h"
#include "utils/cuda_utils.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_copy.h"
#include "utils/image/image_load.h"
#include "utils/image/image_save.h"
#include "utils/misc_utils.h"
#include "utils/stb_image.h"
#include "video/CudaImageVisualizer.h"
#include "video/Window.h"

using namespace recogs;

namespace
{
struct prg {
    float a, b;

    __host__ __device__ prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

    __host__ __device__ float operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};
} // namespace

TEST_CASE("Test image optimization")
{
    std::filesystem::path img_gt_filepath = std::filesystem::path(DATA_DIR) / "frog.jpg";
    CHECK_ARG(std::filesystem::exists(img_gt_filepath), "Invalid GT image path: %s", img_gt_filepath.c_str());

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Load the GT image (target image)
    std::unique_ptr<Image3fCHW> image_gt;
    image_load_chw(img_gt_filepath, image_gt, stream);
    int W = (int) image_gt->width;
    int H = (int) image_gt->height;
    printf("Ground truth image loaded; Size: (%d, %d)\n", W, H);

    // Init prediction (i.e. parameters)
    thrust::device_vector<float> image_pred_data(W * H * 3);
    Image3fCHW image_pred = Image3fCHW::ref(W, H, RCGS_TPTR(image_pred_data));
    thrust::device_vector<float> image_pred_grads(W * H * 3, 0);
    // Fill the prediction with random [0, 1] numbers
    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(W * H * 3),
                      image_pred_data.begin(),
                      prg(0, 1));

    // Create optimizer
    std::vector<Adam::ParamSet> param_sets{};
    {
        Adam::ParamSet& params_set = param_sets.emplace_back();
        params_set.params = image_pred.data_d();
        params_set.grads = RCGS_TPTR(image_pred_grads);
        params_set.num_params = W * H * 3;
        params_set.lr = 0.001f;
    }
    Adam::Options options{};
    std::unique_ptr<Adam> adam = std::make_unique<Adam>(param_sets, options, CU_STREAM_LEGACY);

    std::shared_ptr<Window> window = std::make_shared<Window>(W, H, "Image Optimization Test", false);
    std::unique_ptr<CudaImageVisualizer> visualizer = std::make_unique<CudaImageVisualizer>(window);
    visualizer->start();
    visualizer->set_image(image_pred);

    // Optimization loop
    GSLoss loss_fn{};
    for (int iter = 0; iter < 10000; ++iter) {
        adam->zero_grad(stream);

        float loss, L1, Lssim;
        loss_fn.forward(1, 3, H, W, image_pred.data_d(), image_gt->data_d(), loss, L1, Lssim, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("  Iter. %05d; Loss: %.8f, L1: %.8f, Lssim: %.8f\n", iter, loss, L1, Lssim);

        loss_fn.backward(1, 3, H, W, image_pred.data_d(), image_gt->data_d(), RCGS_TPTR(image_pred_grads), stream);

        adam->step(stream);

        fflush(stdout);
    }
    printf("Bye bye!\n");

    CHECK_CUDA(cudaStreamDestroy(stream));

    visualizer.reset();
    adam.reset(); // Delete the optimizer first
}
