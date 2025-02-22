#include "PCVNetEngine.h"

#include <cstring>
#include <fstream>
#include <thread>

#include "NvOnnxParser.h"
#include "utils/cuda_utils.h"
#include "utils/exceptions.h"
#include "utils/misc_utils.h"

// Reference:
// https://github.com/cyrusbehr/tensorrt-cpp-api

using namespace gs_train;

namespace
{
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* message) noexcept override
    {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            printf("[ERROR] [PCVNetEngine] TensorRT: %s\n", message);
            break;
        case Severity::kWARNING:
            printf("[WARN ] [PCVNetEngine] TensorRT: %s\n", message);
            break;
        case Severity::kINFO:
            // printf("[INFO ] [PCVNetEngine] TensorRT: %s\n", message);
            break;
        case Severity::kVERBOSE:
            // printf("[DEBUG] [PCVNetEngine] TensorRT: %s\n", message);
            break;
        }
    }
};

std::string to_string(nvinfer1::Dims dims)
{
    std::string str = "(";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) str += ", ";
        str += std::to_string(dims.d[i]);
    }
    str += ")";
    return str;
}

std::string to_string(nvinfer1::TensorIOMode mode)
{
    switch (mode) {
    case nvinfer1::TensorIOMode::kNONE:
        return "NONE";
    case nvinfer1::TensorIOMode::kINPUT:
        return "INPUT";
    case nvinfer1::TensorIOMode::kOUTPUT:
        return "OUTPUT";
    }
}

std::string to_string(nvinfer1::TensorLocation location)
{
    switch (location) {
    case nvinfer1::TensorLocation::kHOST:
        return "HOST";
    case nvinfer1::TensorLocation::kDEVICE:
        return "DEVICE";
    }
}

} // namespace

void PCVNetEngine::Options::validate() const
{
    CHECK_ARG(std::filesystem::exists(onnx_filepath), "Invalid .onnx filepath: %s", onnx_filepath.c_str());
    bool check_optprofile_image_sizes = optprofile_min_image_size.x <= optprofile_opt_image_size.x &&
                                        optprofile_opt_image_size.x <= optprofile_max_image_size.x &&
                                        optprofile_min_image_size.y <= optprofile_opt_image_size.y &&
                                        optprofile_opt_image_size.y <= optprofile_max_image_size.y;
    if (!check_optprofile_image_sizes) {
        throw IllegalArgumentException("Invalid optimization profile: min <= opt <= max");
    }
}

PCVNetEngine::PCVNetEngine(Options options) : m_options(std::move(options))
{
    m_options.validate();

    m_logger = std::make_unique<Logger>();

    printf("[INFO ] [PCVNetEngine] TensorRT version: %d.%d.%d.%d\n",
           getInferLibMajorVersion(),
           getInferLibMinorVersion(),
           getInferLibPatchVersion(),
           getInferLibBuildVersion());
}

void PCVNetEngine::build()
{
    printf("[DEBUG] [PCVNetEngine] Creating the TensorRT builder...\n");
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger));
    CHECK_STATE(builder, "Failed to create the TensorRT builder");
    int max_threads = (int) std::thread::hardware_concurrency();
    builder->setMaxThreads(max_threads);
    printf("[INFO ] [PCVNetEngine] TensorRT builder will use up to %d threads\n", max_threads);

    nvinfer1::NetworkDefinitionCreationFlags flags = 0;
    flags |= (uint32_t) nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED;
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    CHECK_STATE(network, "Failed to create the TensorRT network");

    {
        printf("[DEBUG] [PCVNetEngine] Loading the ONNX model file \"%s\"...\n",
               m_options.onnx_filepath.filename().c_str());
        std::ifstream onnx_file(m_options.onnx_filepath, std::ios::binary | std::ios::ate);
        std::streamsize onnx_filesize = onnx_file.tellg();
        onnx_file.seekg(0, std::ios::beg);
        std::vector<char> onnx_filedata(onnx_filesize);
        CHECK_STATE(onnx_file.read(onnx_filedata.data(), onnx_filesize), "Can't read the ONNX file");

        printf("[DEBUG] [PCVNetEngine] Parsing the ONNX model to TensorRT...\n");
        auto parser = nvonnxparser::createParser(*network, *m_logger);
        CHECK_STATE(parser, "Failed to create the ONNX parser");
        CHECK_STATE(parser->parse(onnx_filedata.data(), onnx_filesize), "TensorRT can't parse the ONNX file");
    } // Free the memory taken by the ONNX model (and the parser)

    auto builder_cfg = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    CHECK_STATE(builder_cfg, "Failed to create TensorRT builder config");

    // Build the optimization profile for PCVNet
    printf("[DEBUG] [PCVNetEngine] Creating the TensorRT optimization profile\n");
    nvinfer1::IOptimizationProfile* opt_profile = builder->createOptimizationProfile();
    size_t num_inputs = network->getNbInputs();
    CHECK_STATE(num_inputs == 2, "Unexpected number of inputs: %zu", num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        nvinfer1::ITensor* input = network->getInput(i);
        std::string input_name = input->getName();
        nvinfer1::Dims input_dims = input->getDimensions();
        // Validation
        CHECK_STATE(input_name == "im0" || input_name == "im1", "Unexpected input name: %s", input_name.c_str());
        // Input shape is expected to be (-1, 3, -1, -1) (dynamic batch, image height and width)
        CHECK_STATE(input_dims.nbDims == 4, "Invalid number of dimensions for input");
        CHECK_STATE(input_dims.d[0] == UINT64_MAX && input_dims.d[1] == 3 && input_dims.d[2] == UINT64_MAX &&
                        input_dims.d[3] == UINT64_MAX,
                    "im0 or im1 shape is invalid");
        printf("[DEBUG] [PCVNetEngine] Setting dimensions for input \"%s\"\n", input_name.c_str());
        // Minimum optimization profile
        nvinfer1::Dims min_dims = input_dims;
        min_dims.d[0] = 1;
        min_dims.d[2] = m_options.optprofile_min_image_size.y;
        min_dims.d[3] = m_options.optprofile_min_image_size.x;
        opt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
        // Optimal optimization profile
        nvinfer1::Dims opt_dims = input_dims;
        opt_dims.d[0] = 1;
        opt_dims.d[2] = m_options.optprofile_opt_image_size.y;
        opt_dims.d[3] = m_options.optprofile_opt_image_size.x;
        opt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
        // Maximum optimization profile
        nvinfer1::Dims max_dims = input_dims;
        max_dims.d[0] = 1;
        max_dims.d[2] = m_options.optprofile_max_image_size.y;
        max_dims.d[3] = m_options.optprofile_max_image_size.x;
        opt_profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    }
    builder_cfg->addOptimizationProfile(opt_profile);
    // Enable DataType::kBF16 layer selection, with FP32 fallback.
    // This flag is only supported by NVIDIA Ampere and later GPUs
    // TODO builder_cfg->setFlag(nvinfer1::BuilderFlag::kBF16); My GPU doesn't support bfloat16 :')
    // This level determines how much effort TensorRT would take to find a better solution for performance
    // TODO builder_cfg->setTilingOptimizationLevel(nvinfer1::TilingOptimizationLevel::kFULL);
    // Set the CUDA stream that is used to profile this network
    cudaStream_t profile_stream{};
    CHECK_CUDA(cudaStreamCreate(&profile_stream));
    builder_cfg->setProfileStream(profile_stream);

    // Build the engine
    // If this call fails, it is suggested to increase the logger verbosity to kVERBOSE and try rebuilding the
    // engine. Doing so will provide you with more information on why exactly it is failing
    printf("[INFO ] [PCVNetEngine] Building the TensorRT engine (might take several minutes)...\n");
    fflush(stdout); // So we know the program arrived here!
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *builder_cfg)};
    CHECK_STATE(plan, "Failed to create the TensorRT engine");

    std::ofstream engine_file(m_options.engine_filepath, std::ofstream::binary);
    engine_file.write(reinterpret_cast<const char*>(plan->data()), (std::streamsize) plan->size());
    printf("[INFO ] [PCVNetEngine] TensorRT engine written to: %s\n", m_options.engine_filepath.c_str());

    CHECK_CUDA(cudaStreamDestroy(profile_stream));
}

void PCVNetEngine::load()
{
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(*m_logger)};
    CHECK_STATE(m_runtime, "Failed to create TensorRT runtime");

    // Load the engine from file
    std::ifstream engine_file(m_options.engine_filepath, std::ios::binary | std::ios::ate);
    std::streamsize engine_filesize = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_filedata(engine_filesize);
    CHECK_STATE(engine_file.read(engine_filedata.data(), engine_filesize),
                "Failed to load TensorRT engine: %s",
                m_options.engine_filepath);
    engine_file.close();

    // Load the engine to TensorRT
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        m_runtime->deserializeCudaEngine(engine_filedata.data(), engine_filesize));
    CHECK_STATE(m_engine, "Failed to load TensorRT engine");
    engine_filedata.clear();

    // Create the execution context
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    CHECK_STATE(m_context, "Failed to create TensorRT context");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    size_t num_io_tensors = m_engine->getNbIOTensors();
    m_io_buffers.resize(num_io_tensors);
    for (int i = 0; i < num_io_tensors; ++i) {
        const char* tensor_name = m_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode tensor_mode = m_engine->getTensorIOMode(tensor_name);
        nvinfer1::Dims tensor_shape = m_engine->getTensorShape(tensor_name);
        nvinfer1::DataType tensor_datatype = m_engine->getTensorDataType(tensor_name);
        nvinfer1::TensorLocation tensor_location = m_engine->getTensorLocation(tensor_name);

        printf("[DEBUG] [PCVNetEngine] %s tensor \"%s\" #%d (%s) of shape %s",
               to_string(tensor_mode).c_str(),
               tensor_name,
               i,
               to_string(tensor_location).c_str(),
               to_string(tensor_shape).c_str());

        if (tensor_mode == nvinfer1::TensorIOMode::kINPUT && std::strcmp(tensor_name, "im0") == 0) {
            m_im0_tensor_idx = i;
            printf(" - deferred\n");
            continue;
        } else if (tensor_mode == nvinfer1::TensorIOMode::kINPUT && std::strcmp(tensor_name, "im1") == 0) {
            m_im1_tensor_idx = i;
            printf(" - deferred\n");
            continue;
        } else if (tensor_mode == nvinfer1::TensorIOMode::kOUTPUT && std::strcmp(tensor_name, "disparity_map") == 0) {
            m_disparity_map_tensor_idx = i;
            printf(" - deferred\n");
            continue;
        }

        // Calculate the buffer size
        size_t buffer_size = 1;
        for (int d = 0; d < tensor_shape.nbDims; ++d) {
            int64_t dim = tensor_shape.d[d];
            if (dim <= 0) {
                if (d == 0) {
                    dim = 1;
                } else if (d == 2) {
                    dim = m_options.optprofile_max_image_size.y;
                } else if (d == 3) {
                    dim = m_options.optprofile_max_image_size.x;
                } else {
                    throw IllegalStateException("Unknown how to allocate dynamic dimension: %d"); // TODO %d
                }
            }
            buffer_size *= dim;
        }
        buffer_size *= sizeof(float);

        // TODO FP16? At the moment we're only supporting float32
        CHECK_STATE(tensor_datatype == nvinfer1::DataType::kFLOAT, "Unexpected tensor data type");

        void* buffer;
        CHECK_CUDA(cudaMalloc(&buffer, buffer_size));
        printf(" - allocated %zu bytes\n", buffer_size);
        m_io_buffers[i] = buffer;
        bool status;
        status = m_context->setTensorAddress(tensor_name, m_io_buffers.at(i)), assert(status);
    }

    CHECK_ARG(m_im0_tensor_idx >= 0, "im0 input tensor not found");
    CHECK_ARG(m_im1_tensor_idx >= 0, "im1 input tensor not found");
    CHECK_ARG(m_disparity_map_tensor_idx >= 0, "disparity_map output tensor not found");

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void PCVNetEngine::build_or_load()
{
    std::filesystem::path engine_filepath = m_options.engine_filepath;
    if (!std::filesystem::exists(engine_filepath)) {
        printf("[INFO ] [PCVNetEngine] Engine file not found at: %s\n", engine_filepath.c_str());
        build();
    }
    printf("[INFO ] [PCVNetEngine] Loading engine file: %s\n", engine_filepath.c_str());
    load();
}

void PCVNetEngine::infer(const Image3fCHW& im0,
                         const Image3fCHW& im1,
                         Image1fCHW& out_disparity_map,
                         cudaStream_t stream)
{
    assert(im0.size() == im1.size());
    assert(im0.size() == out_disparity_map.size());
    // IMPORTANT: input/output size must be divisible by 32 or:
    // "Cask Pooling Runner Execute Failure"
    assert(im0.width % 32 == 0);
    assert(im0.height % 32 == 0);

    // Set input dimensions (width and height are dynamically sized)
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = 1; // Batch size is always 1 (too much memory consumed otherwise)
    dims.d[1] = 3; // RGB
    dims.d[2] = im0.height;
    dims.d[3] = im0.width;
    m_context->setInputShape("im0", dims);
    m_context->setInputShape("im1", dims);
    assert(m_context->allInputDimensionsSpecified());

    // Set input/output buffer references
    bool status;
    status = m_context->setTensorAddress("im0", im0.data_d()), assert(status);
    status = m_context->setTensorAddress("im1", im1.data_d()), assert(status);
    status = m_context->setTensorAddress("disparity_map", out_disparity_map.data_d()), assert(status);

    // Run inference
    status = m_context->enqueueV3(stream), assert(status);
}
