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
    CHECK_STATE(network->getNbInputs() == 2, "Expected 2 inputs: im0 and im1");
    CHECK_STATE(network->getNbOutputs() == 1, "Expected 1 output: disparity_map");
    CHECK_STATE(std::strcmp(network->getInput(0)->getName(), "im0") == 0, "Expected input 0 to be im0");
    CHECK_STATE(std::strcmp(network->getInput(1)->getName(), "im1") == 0, "Expected input 1 to be im1");
    CHECK_STATE(std::strcmp(network->getOutput(0)->getName(), "disparity_map") == 0,
                "Expected output 0 to be disparity_map");

    auto builder_cfg = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    CHECK_STATE(builder_cfg, "Failed to create TensorRT builder config");

    nvinfer1::IOptimizationProfile* opt_profile{};
    nvinfer1::Dims opt_dims{};

    // Create the optimization profile for horizontal input
    opt_profile = builder->createOptimizationProfile();
    opt_dims.nbDims = 4;
    opt_dims.d[0] = 1;
    opt_dims.d[1] = 3;
    opt_dims.d[2] = k_io_height;
    opt_dims.d[3] = k_io_width;
    opt_profile->setDimensions("im0", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("im0", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("im0", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    opt_profile->setDimensions("im1", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("im1", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("im1", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    opt_dims.d[1] = 1;
    opt_profile->setDimensions("disparity_map", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("disparity_map", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("disparity_map", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    builder_cfg->addOptimizationProfile(opt_profile);

    // Create the optimization profile for vertical input
    opt_profile = builder->createOptimizationProfile();
    opt_dims.nbDims = 4;
    opt_dims.d[0] = 1;
    opt_dims.d[1] = 3;
    opt_dims.d[2] = k_io_width;
    opt_dims.d[3] = k_io_height;
    opt_profile->setDimensions("im0", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("im0", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("im0", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    opt_profile->setDimensions("im1", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("im1", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("im1", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    opt_dims.d[1] = 1;
    opt_profile->setDimensions("disparity_map", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("disparity_map", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("disparity_map", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    builder_cfg->addOptimizationProfile(opt_profile);

    // Enable DataType::kBF16 layer selection, with FP32 fallback.
    // This flag is only supported by NVIDIA Ampere and later GPUs
    // TODO My GPU doesn't support bfloat16 :')
    builder_cfg->setFlag(nvinfer1::BuilderFlag::kFP16);

    // This level determines how much effort TensorRT would take to find a better solution for performance.
    // NOTE: only seen working with kNONE; otherwise "Skipping tactic" and the plan doesn't build
    // builder_cfg->setTilingOptimizationLevel(nvinfer1::TilingOptimizationLevel::kFULL);

    printf("[DEBUG] [PCVNetEngine] Tiling optimization level: %d\n", builder_cfg->getTilingOptimizationLevel());
    builder_cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (size_t) 6 * (1 << 30) /* 6GB */);

    // Build the engine
    printf("[INFO ] [PCVNetEngine] Building the TensorRT engine (might take several minutes)...\n");
    fflush(stdout); // So we know the program arrived here!
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *builder_cfg)};
    CHECK_STATE(plan, "Failed to create the TensorRT engine");

    std::ofstream engine_file(m_options.engine_filepath, std::ofstream::binary);
    engine_file.write(reinterpret_cast<const char*>(plan->data()), (std::streamsize) plan->size());
    printf("[INFO ] [PCVNetEngine] TensorRT engine written to: %s\n", m_options.engine_filepath.c_str());
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

    // Validate I/O tensors
    size_t num_io_tensors = m_engine->getNbIOTensors();
    CHECK_STATE(num_io_tensors, num_io_tensors == 3, "Expected 3 I/O tensors: im0, im1 and disparity_map");
    for (int i = 0; i < num_io_tensors; ++i) {
        std::string name = m_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name.c_str());
        nvinfer1::Dims shape = m_engine->getTensorShape(name.c_str());
        nvinfer1::DataType datatype = m_engine->getTensorDataType(name.c_str());
        nvinfer1::TensorLocation location = m_engine->getTensorLocation(name.c_str());

        if (name == "im0") {
            CHECK_STATE(mode == nvinfer1::TensorIOMode::kINPUT, "im0 must be a input tensor");
            CHECK_STATE(shape.nbDims == 4 && shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == -1 && shape.d[3] == -1,
                        "im0 shape must be (1, 3, -1, -1)");
        } else if (name == "im1") {
            CHECK_STATE(mode == nvinfer1::TensorIOMode::kINPUT, "im1 must be a input tensor");
            CHECK_STATE(shape.nbDims == 4 && shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == -1 && shape.d[3] == -1,
                        "im1 shape must be (1, 3, -1, -1)");
        } else if (name == "disparity_map") {
            CHECK_STATE(mode == nvinfer1::TensorIOMode::kOUTPUT, "disparity_map must be an output tensor");
            //            CHECK_STATE(shape.nbDims == 4 && shape.d[0] == 1 && shape.d[1] == 1 && shape.d[2] == -1 &&
            //            shape.d[3] == -1,
            //                        "disparity_map shape must be (1, 1, -1, -1)");
        } else {
            throw IllegalStateException("Unrecognized tensor \"%s\""); // TODO name
        }

        CHECK_STATE(datatype == nvinfer1::DataType::kFLOAT, "%s must be a FLOAT tensor", name);
        CHECK_STATE(location == nvinfer1::TensorLocation::kDEVICE, "%s must be a DEVICE tensor", name);
    }
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
    bool is_rotated = im0.width == k_io_height;
    if (is_rotated) {
        assert(im0.height == k_io_width);
    } else {
        assert(im0.width == k_io_width && im0.height == k_io_height);
    }

    // IMPORTANT: input/output size must be divisible by 32 or:
    // "Cask Pooling Runner Execute Failure"

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
    m_context->setOptimizationProfileAsync(is_rotated /* horizontal vs vertical optimization profile */, stream);
    status = m_context->enqueueV3(stream), assert(status);
}
