#include "PCVNet.h"

#include "App.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_save.h"
#include "utils/image/image_visit_transform.h"

using namespace gs_train;

// Reference:
// https://github.com/NVIDIA/ProViz-AI-Samples/blob/master/onnxruntime_cpp_samples/dml_provider/src/OrtBuffer.cpp

namespace
{
void check_ort(OrtStatusPtr status)
{
    if (status != nullptr) {
        const char* msg = Ort::GetApi().GetErrorMessage(status);
        printf("[ERROR] [PCVNet] %s\n", msg);
        Ort::GetApi().ReleaseStatus(status);
        exit(1);
    }
}
} // namespace

PCVNet::PCVNet(App& app) : m_app(app)
{
    std::filesystem::path model_filename = "pcvnet.onnx";
    CHECK_ARG(std::filesystem::exists(model_filename), "pcvnet.onnx model not found");

    m_env = std::make_unique<Ort::Env>();

    const auto& api = Ort::GetApi();

    Ort::SessionOptions session_options = Ort::SessionOptions{};
    session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    /* CUDA options */
    OrtCUDAProviderOptionsV2* cuda_options;
    check_ort(api.CreateCUDAProviderOptions(&cuda_options));
    std::vector<const char*> keys{
        "cudnn_conv_use_max_workspace",
        "use_tf32",
        "do_copy_in_default_stream"
    };
    std::vector<const char*> values{
        "1",
        "0",
        "1"
    };
    check_ort(api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size()));
    cudaStream_t cuda_stream;
    CHECK_CUDA(cudaStreamCreate(&cuda_stream));
    // This implicitly sets "has_user_compute_stream"
    check_ort(api.UpdateCUDAProviderOptionsWithValue(cuda_options, "user_compute_stream", cuda_stream));
    check_ort(api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options));
    api.ReleaseCUDAProviderOptions(cuda_options);

    Stopwatch stopwatch;
    m_session = std::make_unique<Ort::Session>(*m_env, model_filename.c_str(), session_options);
    printf("[DEBUG] [PCVNet] ONNX session initialized in %s\n", stopwatch.elapsed_time_str().c_str());
}

void PCVNet::forward(const ImageT& im0, const ImageT& im1, Image1fCHW& disparity_map)
{
    CHECK_ARG(im0.size() == im1.size(), "im0 and im1 must have the same shape");
    CHECK_ARG(im0.width % 32 == 0 && im0.height % 32 == 0, "im0 and im1 must be multiple of 32");
    CHECK_ARG(im0.size() == disparity_map.size(), "im0, im1 and output must have the same size");

    uint32_t w = im0.width;
    uint32_t h = im0.height;

    Ort::MemoryInfo meminfo =
        Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0 /* id */, OrtMemType::OrtMemTypeDefault);

    int64_t input_shape[]{1, 3, h, w};
    Ort::Value im0_ort = Ort::Value::CreateTensor( //
        meminfo,
        im0.data_d(),
        3 * h * w * sizeof(float),
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Ort::Value im1_ort = Ort::Value::CreateTensor( //
        meminfo,
        im1.data_d(),
        3 * h * w * sizeof(float),
        input_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    int64_t output_shape[]{1, 1, h, w};
    Ort::Value output_ort = Ort::Value::CreateTensor( //
        meminfo,
        disparity_map.data_d(),
        h * w * sizeof(float),
        output_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    Ort::IoBinding iobinding(*m_session);
    iobinding.BindInput("im0", im0_ort);
    iobinding.BindInput("im1", im1_ort);
    iobinding.BindOutput("disparity_map", output_ort);

    {
        Stopwatch stopwatch;

        Ort::RunOptions run_options{};
        run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        m_session->Run(run_options, iobinding);

        printf("[DEBUG] [PCVNet] Forward took %s\n", stopwatch.elapsed_time_str().c_str());
    }
}
