#include "Optimizer.h"

#include <fstream>
#include <random>

#include <fmt/format.h>
#include <thrust/extrema.h>

#include "Adam.h"
#include "App.h"
#include "GSLoss.h"
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
        out_clamped[i] = true;
    } else if (x[i] < m) {
        x[i] = m;
        out_clamped[i] = true;
    } else {
        out_clamped[i] = false;
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
    dL_dx[i] *= clamped ? 0 : 1;
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

Optimizer::Optimizer(App& app) : m_app(app) { load_training_cameras(); }

Optimizer::~Optimizer() {}

void Optimizer::start()
{
    CHECK_STATE(!m_running);

    // CUDA stream used for training
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    Image3fCHW gt = Image3fCHW::malloc(m_resolution.x, m_resolution.y);
    Image3fCHW pred = Image3fCHW::malloc(m_resolution.x, m_resolution.y);
    Image1fCHW depthbuffer = Image1fCHW::malloc(m_resolution.x, m_resolution.y);

    Scene& training_scene = m_app.training_scene();

    thrust::device_vector<float> dL_dy(m_resolution.x * m_resolution.y * 3);
    // A boolean vector remembering which values of the prediction (y) has been clamped to [0, 1]
    thrust::device_vector<bool> y_clamped(m_resolution.x * m_resolution.y * 3);

    // Init optimizer
    std::vector<Adam::ParamSet> params_sets{};
    {
        Adam::ParamSet& params_set = params_sets.emplace_back();
        params_set.params = RCGS_TPTR(training_scene.shs);
        params_set.grads = RCGS_TPTR(training_scene.dL_dsh);
        params_set.num_params = training_scene.num_vertices * 16 * 3; // A lot of parameters...
        params_set.lr = 0.0025f;                                      // Like in the original code
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
    std::uniform_int_distribution<> training_cameras_dist(0, int(m_training_cameras.size() - 1));

    // Optimization loop
    printf("[INFO ] [Optimizer] Starting the optimization loop...\n");
    m_running = true;
    int iter = 0;
    while (m_running) {
        int camera_idx = training_cameras_dist(random);
        const GSCamera& sampled_camera = m_training_cameras.at(0);

        // Compute prediction
        int num_rendered = gs_rasterizer.forward(m_app.background_d(), training_scene, sampled_camera, pred, stream);
        // clamp_forward(pred.data_d(), pred.width * pred.height * 3, 0.0f, 1.0f, stream);
        if (num_rendered == 0) continue;
        clamp_forward(pred.data_d(), m_resolution.x * m_resolution.y * 3, 0, 1, RCGS_TPTR(y_clamped), stream);

        // Compute ground truth
        gs_rasterizer.forward(m_app.background_d(), m_app.scene(), sampled_camera, gt, depthbuffer, stream);
        m_app.selection3d().project( //
            sampled_camera,
            [gt, depthbuffer] __device__(uint32_t x, uint32_t y, float view_z) mutable {
                float z = depthbuffer.value(x, y).r;
                if (view_z > z) return; // Depth testing
                glm::vec3 color = gt.value(x, y);
                color *= glm::vec3(1, 0, 1); // TODO apply any edit
                gt.set_value(x, y, color);
            },
            stream);
        // clamp_forward(gt.data_d(), gt.width * gt.height * 3, 0.0f, 1.0f, stream);

        float* loss_d = loss_func.forward(1, 3, m_resolution.y, m_resolution.x, pred.data_d(), gt.data_d(), stream);
        float loss;
        CHECK_CUDA(cudaMemcpyAsync(&loss, loss_d, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("[DEBUG] [Optimizer] Iter. %05d; Loss: %.8f\n", iter, loss);

        if (iter % 50 == 0) {
            printf("[DEBUG] [Optimizer] Saving gt/pred...\n");
            CHECK_CUDA(cudaStreamSynchronize(stream));
            image_save_png(gt, fmt::format("gt-{}.png", iter));
            image_save_png(pred, fmt::format("pred-{}.png", iter));
        }

        // Backward
        adam.zero_grad(stream);
        loss_func.backward( //
            1,
            3,
            m_resolution.y,
            m_resolution.x,
            pred.data_d(),
            gt.data_d(),
            RCGS_TPTR(dL_dy), // Output
            stream);
        clamp_backward(RCGS_TPTR(y_clamped), m_resolution.x * m_resolution.y * 3, 0, 1, RCGS_TPTR(dL_dy), stream);
        gs_rasterizer.backward( //
            training_scene,
            num_rendered,
            m_app.background_d(),
            sampled_camera,
            RCGS_TPTR(dL_dy), // Input
            stream);

        // Step the optimizer
        adam.step(stream);

        CHECK_CUDA(cudaStreamSynchronize(stream));

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

void Optimizer::load_training_cameras()
{
    std::filesystem::path scene_ply = m_app.scene_ply();
    std::filesystem::path scene_folder = //
        m_app.scene_ply()
            .parent_path()  // iteration_*
            .parent_path()  // point_cloud
            .parent_path(); // <scene>
    std::filesystem::path cameras_json = scene_folder / "cameras.json";
    CHECK_ARG(std::filesystem::exists(cameras_json), "cameras.json file not found at: %s", cameras_json);
    std::ifstream file(cameras_json);
    nlohmann::json json = nlohmann::json::parse(file);
    m_training_cameras.reserve(json.size());
    for (const nlohmann::json& camera_json : json) {
        GSCamera& camera = m_training_cameras.emplace_back();
        camera.deserialize(camera_json);
        camera.set_resolution(m_resolution.x, m_resolution.y);
        camera.update(m_app.stream());
    }
    CHECK_CUDA(cudaStreamSynchronize(m_app.stream()));
}
