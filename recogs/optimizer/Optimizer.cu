#include "Optimizer.h"

#include <fstream>
#include <random>

#include <fmt/format.h>
#include <thrust/extrema.h>

#include "Adam.h"
#include "App.h"
#include "GSLoss.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_save.h"

using namespace gs_train;

namespace
{
__global__ void clamp_forward_kernel(float* x, size_t N, float m, float M, bool* out_clamped)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (x[i] > M) {
        x[i] = M;
        if (out_clamped) out_clamped[i] = true;
    } else if (x[i] < m) {
        x[i] = m;
        if (out_clamped) out_clamped[i] = true;
    } else {
        if (out_clamped) out_clamped[i] = false;
    }
}

void clamp_forward(float* x, size_t N, float m, float M, bool* out_clamped, cudaStream_t stream)
{
    dim3 num_blocks = (N + 1023) / 1024;
    dim3 block_dim = 1024;
    clamp_forward_kernel<<<num_blocks, block_dim, 0, stream>>>(x, N, m, M, out_clamped);
}

__global__ void clamp_backward_kernel(const bool* clamped, size_t N, float m, float M, float* dL_dx)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dL_dx[i] *= clamped[i] ? 0 : 1;
}

void clamp_backward(const bool* clamped, size_t N, float m, float M, float* dL_dx, cudaStream_t stream)
{
    dim3 num_blocks = (N + 1023) / 1024;
    dim3 block_dim = 1024;
    clamp_backward_kernel<<<num_blocks, block_dim, 0, stream>>>(clamped, N, m, M, dL_dx);
}

void print_minmax(const char* name, const thrust::device_vector<float>& vec, cudaStream_t stream)
{
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto pair = thrust::minmax_element(vec.begin(), vec.end());
    float min_ = *pair.first;
    float max_ = *pair.second;
    printf("%s: %lf %lf\n", name, min_, max_);
}
} // namespace

Optimizer::Optimizer(App& app) : m_app(app) {}

Optimizer::~Optimizer() {}

void Optimizer::start()
{
    CHECK_STATE(!m_running);

    // CUDA stream used for training
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Load training cameras, make sure they live on GPU
    std::vector<GSCamera> training_cameras = m_app.cameras();
    for (GSCamera& camera : training_cameras) {
        camera.set_resolution(m_resolution.x, m_resolution.y);
        camera.update(stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    Image3fCHW gt = Image3fCHW::malloc(m_resolution.x, m_resolution.y);
    Image3fCHW pred = Image3fCHW::malloc(m_resolution.x, m_resolution.y);
    Image1fCHW depthbuffer = Image1fCHW::malloc(m_resolution.x, m_resolution.y);

    Scene& scene = m_app.scene();

    thrust::device_vector<float> dL_dy(m_resolution.x * m_resolution.y * 3);
    // A boolean vector remembering which values of the prediction (y) has been clamped to [0, 1]
    thrust::device_vector<bool> y_clamped(m_resolution.x * m_resolution.y * 3);

    // Init optimizer
    std::vector<Adam::ParamSet> params_sets{};
    {
        Adam::ParamSet& params_set = params_sets.emplace_back();
        params_set.params = RCGS_TPTR(scene.shs_2);
        params_set.grads = RCGS_TPTR(scene.dL_dsh);
        params_set.num_params = scene.num_vertices * 16 * 3; // A lot of parameters...
        params_set.lr = 0.00025f;                            // Like in the original code
    }
    Adam::Options options{};
    options.eps = 1e-15f;
    Adam adam(params_sets, options, stream);

    GSRasterizer gs_rasterizer{};
    gs_rasterizer.debug = true;

    GSLoss loss_func{};

    // Random
    static std::random_device random_dev;
    static std::mt19937 random(random_dev());
    std::uniform_int_distribution<> training_cameras_dist(0, int(training_cameras.size() - 1));

    // Optimization loop
    printf("[INFO ] [Optimizer] Starting the optimization loop...\n");
    m_running = true;
    int iter = 1;

    const int k_num_profile_iter = 10;
    Stopwatch profile_opt;

    while (m_running) {
        int camera_idx = training_cameras_dist(random);
        const GSCamera& sampled_camera = training_cameras.at(camera_idx);

        // Compute ground truth
        gs_rasterizer.forward(
            m_app.background_d(), scene, false /* scene_2 */, sampled_camera, gt, depthbuffer, stream);
        m_app.selection3d().project( //
            sampled_camera,
            [gt, depthbuffer] __device__(uint32_t x, uint32_t y, float view_z) mutable {
                float z = depthbuffer.value(x, y).r;
                if (view_z > z) return; // Depth testing
                glm::vec3 color = gt.value(x, y);
                color = glm::vec3(0); // TODO apply any edit
                gt.set_value(x, y, color);
            },
            stream);
        clamp_forward(gt.data_d(), gt.width * gt.height * 3, 0.0f, 1.0f, nullptr, stream);

        // Compute prediction
        // NOTE: gs_rasterizer will save internal state during the forward; thus this code doesn't have to be moved!
        int num_rendered =
            gs_rasterizer.forward(m_app.background_d(), scene, true /* scene_2 */, sampled_camera, pred, stream);
        if (num_rendered == 0) continue;
        clamp_forward(pred.data_d(), m_resolution.x * m_resolution.y * 3, 0, 1, RCGS_TPTR(y_clamped), stream);

        // Compute loss
        float loss, L1, Lssim;
        loss_func.forward(1, 3, m_resolution.y, m_resolution.x, pred.data_d(), gt.data_d(), loss, L1, Lssim, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (iter % k_num_profile_iter == 0) {
            float iter_secs = float(k_num_profile_iter) / float(profile_opt.elapsed_seconds());
            printf("[DEBUG] [Optimizer] Iter. %05d (%2.1f iter/s); Loss: %.8f; L1: %.8f, SSIM: %.8f\n",
                   iter,
                   iter_secs,
                   loss,
                   L1,
                   Lssim);
            profile_opt.reset();
        }

        // Backward
        loss_func.backward( //
            1,
            3,
            m_resolution.y,
            m_resolution.x,
            pred.data_d(),
            gt.data_d(),
            RCGS_TPTR(dL_dy), // Output; Re-initialized every time
            stream);
        clamp_backward(RCGS_TPTR(y_clamped), m_resolution.x * m_resolution.y * 3, 0.0f, 1.0f, RCGS_TPTR(dL_dy), stream);
        gs_rasterizer.backward( //
            scene,
            true /* scene_2 */,
            num_rendered,
            m_app.background_d(),
            sampled_camera,
            RCGS_TPTR(dL_dy), // Input
            stream);

        adam.step(stream);
        adam.zero_grad(stream);

        ++iter;
    }
    printf("[INFO ] [Optimizer] Ended\n");

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void Optimizer::signal_stop()
{
    m_running = false;
    printf("[DEBUG] [Optimizer] Stop signaled\n");
}
