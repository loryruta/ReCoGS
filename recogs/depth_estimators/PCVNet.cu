#include "PCVNet.h"

#include "App.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_save.h"
#include "utils/image/image_visit_transform.h"

using namespace gs_train;

// Reference:
// https://github.com/NVIDIA/ProViz-AI-Samples/blob/master/onnxruntime_cpp_samples/dml_provider/src/OrtBuffer.cpp

PCVNet::PCVNet(App& app) : m_app(app)
{
    std::filesystem::path model_filename = "pcvnet.onnx";
    CHECK_ARG(std::filesystem::exists(model_filename), "pcvnet.onnx model not found");

    m_env = std::make_unique<Ort::Env>();

    Ort::SessionOptions session_options = Ort::SessionOptions{};
    session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    OrtCUDAProviderOptions cuda_provider_options{};
    cuda_provider_options.device_id = 0;
    cuda_provider_options.user_compute_stream = CU_STREAM_LEGACY;
    // session_options.AppendExecutionProvider_CUDA(cuda_provider_options);
    OrtTensorRTProviderOptions tensorrt_provider_options{};
    tensorrt_provider_options.device_id = 0;
    // tensorrt_provider_options.trt_fp16_enable = true; // Bad qualitative results!
    tensorrt_provider_options.trt_max_partition_iterations = 1000;
    tensorrt_provider_options.trt_min_subgraph_size = 1;
    tensorrt_provider_options.trt_max_workspace_size = (size_t) 5 * (1024 * 1024 * 1024); // Up to 5GB for optimization
    tensorrt_provider_options.trt_engine_cache_enable = true;
    tensorrt_provider_options.trt_engine_cache_path = ".trt_cache";
    session_options.AppendExecutionProvider_TensorRT(tensorrt_provider_options);

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
    Ort::IoBinding iobinding{*m_session};
    iobinding.BindInput("im0", im0_ort);
    iobinding.BindInput("im1", im1_ort);
    iobinding.BindOutput("disparity_map", output_ort);
    {
        Stopwatch stopwatch;

        Ort::RunOptions run_options{};
        m_session->Run(run_options, iobinding);

        printf("[DEBUG] [PCVNet] Forward took %s\n", stopwatch.elapsed_time_str().c_str());
    }
}
