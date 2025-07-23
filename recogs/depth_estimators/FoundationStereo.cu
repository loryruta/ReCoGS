#include "FoundationStereo.h"

#include <cinttypes>
#include <cstring>
#include <fstream>
#include <thread>

#include <NvOnnxParser.h>

#include "utils/exceptions.h"
#include "utils/misc_utils.h"
#include "utils/nvinfer_utils.h"

// Reference:
// https://github.com/cyrusbehr/tensorrt-cpp-api

USING_NAMESPACE

FoundationStereo::FoundationStereo(FoundationStereo_Options options) : m_options(std::move(options))
{
    m_logger = std::make_unique<SimpleNvInferLogger>("FoundationStereo");

    printf("[INFO ] [FoundationStereo] TensorRT version: %d.%d.%d.%d\n",
           getInferLibMajorVersion(),
           getInferLibMinorVersion(),
           getInferLibPatchVersion(),
           getInferLibBuildVersion());
}

FoundationStereo::~FoundationStereo() { printf("[DEBUG] [FoundationStereo] Destroying\n"); }

void FoundationStereo::build()
{
    printf("[DEBUG] [FoundationStereo] Creating the TensorRT builder...\n");
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*m_logger));
    CHECK_STATE(builder, "Failed to create the TensorRT builder");
    int max_threads = (int) std::thread::hardware_concurrency();
    builder->setMaxThreads(max_threads);
    printf("[INFO ] [FoundationStereo] TensorRT builder will use up to %d threads\n", max_threads);

    nvinfer1::NetworkDefinitionCreationFlags flags = 0;
    flags |= (uint32_t) nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED;
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    CHECK_STATE(network, "Failed to create the TensorRT network");

    {
        printf("[DEBUG] [FoundationStereo] Loading the ONNX model file \"%s\"...\n",
               m_options.onnx_filepath.filename().c_str());
        std::ifstream onnx_file(m_options.onnx_filepath, std::ios::binary | std::ios::ate);
        std::streamsize onnx_filesize = onnx_file.tellg();
        onnx_file.seekg(0, std::ios::beg);
        std::vector<char> onnx_filedata(onnx_filesize);
        CHECK_STATE(onnx_file.read(onnx_filedata.data(), onnx_filesize), "Can't read the ONNX file");

        printf("[DEBUG] [FoundationStereo] Parsing the ONNX model to TensorRT...\n");
        auto parser = nvonnxparser::createParser(*network, *m_logger);
        CHECK_STATE(parser, "Failed to create the ONNX parser");
        CHECK_STATE(parser->parse(onnx_filedata.data(), onnx_filesize), "TensorRT can't parse the ONNX file");
    } // Free the memory taken by the ONNX model (and the parser)

    // ----------------------------------------------------------------
    /* Validation */
    // ----------------------------------------------------------------

    CHECK_STATE(network->getNbInputs() == 2);
    std::string tensor_name;
    nvinfer1::Dims dims;
    // left
    tensor_name = network->getInput(0)->getName();
    CHECK_STATE(tensor_name == "left", "Input 0 name is {} (expected: left)", tensor_name);
    dims = network->getInput(0)->getDimensions();
    CHECK_STATE(dims.nbDims == 4 && dims.d[0] == -1 && dims.d[1] == 3 && dims.d[2] != -1 && dims.d[3] != -1);
    m_height = (int) dims.d[2];
    m_width = (int) dims.d[3];

    printf("[INFO ] [FoundationStereo] I/O dimension is (%d, %d)\n", m_width, m_height);

    // right
    tensor_name = network->getInput(1)->getName();
    CHECK_STATE(tensor_name == "right", "Input 1 name is {} (expected: right)", tensor_name);
    CHECK_STATE(dims.nbDims == 4 && dims.d[0] == -1 && dims.d[1] == 3 && dims.d[2] == m_height && dims.d[3] == m_width);
    // disp
    CHECK_STATE(network->getNbOutputs() == 1);
    tensor_name = network->getOutput(0)->getName();
    CHECK_STATE(tensor_name == "disp", "Output 0 name is {} (expected: disp)", tensor_name);
    CHECK_STATE(dims.nbDims == 4 && dims.d[0] == -1 && dims.d[1] == 3 && dims.d[2] == m_height && dims.d[3] == m_width);

    auto builder_cfg = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    CHECK_STATE(builder_cfg, "Failed to create TensorRT builder config");

    // Enable DataType::kBF16 layer selection, with FP32 fallback.
    // This flag is only supported by NVIDIA Ampere and later GPUs
    builder_cfg->setFlag(nvinfer1::BuilderFlag::kFP16);

    // This level determines how much effort TensorRT would take to find a better solution for performance.
    // NOTE: only seen working with kNONE; otherwise "Skipping tactic" and the plan doesn't build
    // builder_cfg->setTilingOptimizationLevel(nvinfer1::TilingOptimizationLevel::kFULL);

    printf("[DEBUG] [FoundationStereo] Tiling optimization level: %d\n", builder_cfg->getTilingOptimizationLevel());
    builder_cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (size_t) 6 * (1 << 30) /* 6GB */);

    // ----------------------------------------------------------------
    /* Optimization profile */
    // ----------------------------------------------------------------

    nvinfer1::IOptimizationProfile* opt_profile{};
    nvinfer1::Dims opt_dims{};

    opt_profile = builder->createOptimizationProfile();
    opt_dims.nbDims = 4;
    opt_dims.d[0] = 1;
    opt_dims.d[1] = 3;
    opt_dims.d[2] = m_height;
    opt_dims.d[3] = m_width;
    opt_profile->setDimensions("left", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("left", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("left", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    opt_profile->setDimensions("right", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("right", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("right", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    opt_dims.d[1] = 1;
    opt_profile->setDimensions("disp", nvinfer1::OptProfileSelector::kMIN, opt_dims);
    opt_profile->setDimensions("disp", nvinfer1::OptProfileSelector::kOPT, opt_dims);
    opt_profile->setDimensions("disp", nvinfer1::OptProfileSelector::kMAX, opt_dims);
    builder_cfg->addOptimizationProfile(opt_profile);

    // ----------------------------------------------------------------
    /* Build the engine */
    // ----------------------------------------------------------------

    printf("[INFO ] [FoundationStereo] Building the TensorRT engine (might take several minutes)...\n");
    fflush(stdout); // So we know the program arrived here!
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *builder_cfg)};
    CHECK_STATE(plan, "Failed to create the TensorRT engine");

    // ----------------------------------------------------------------
    /* Write the engine to file */
    // ----------------------------------------------------------------

    std::ofstream engine_file(m_options.engine_filepath, std::ofstream::binary);
    engine_file.write(reinterpret_cast<const char*>(plan->data()), (std::streamsize) plan->size());
    printf("[INFO ] [FoundationStereo] TensorRT engine written to: %s\n", m_options.engine_filepath.c_str());
}

void FoundationStereo::load()
{
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(*m_logger)};
    CHECK_STATE(m_runtime, "Failed to create TensorRT runtime");

    // Load the engine from file
    std::ifstream engine_file(m_options.engine_filepath, std::ios::binary | std::ios::ate);
    std::streamsize engine_filesize = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_filedata(engine_filesize);
    CHECK_STATE(engine_file.read(engine_filedata.data(), engine_filesize),
                "Failed to load TensorRT engine: {}",
                m_options.engine_filepath.string());
    engine_file.close();

    // Load the engine to TensorRT
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        m_runtime->deserializeCudaEngine(engine_filedata.data(), engine_filesize));
    CHECK_STATE(m_engine, "Failed to load TensorRT engine");
    engine_filedata.clear();

    // Create the execution context
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    CHECK_STATE(m_context, "Failed to create TensorRT context");

    // ----------------------------------------------------------------
    /* Validate I/O tensors */
    // ----------------------------------------------------------------

    CHECK_STATE(m_engine->getNbIOTensors() == 3);

    nvinfer1::Dims shape;

    /* left */
    CHECK_STATE(m_engine->getTensorIOMode("left") == nvinfer1::TensorIOMode::kINPUT);
    shape = m_engine->getTensorShape("left");
    m_height = (int) shape.d[2];
    m_width = (int) shape.d[3]; // Initialize dimensions
    printf("[INFO ] [FoundationStereo] Input dimension is (%d, %d)\n", m_width, m_height);
    CHECK_STATE(shape.d[0] == 1 && shape.d[1] == 3);
    CHECK_STATE(m_engine->getTensorDataType("left") == nvinfer1::DataType::kFLOAT);
    CHECK_STATE(m_engine->getTensorLocation("left") == nvinfer1::TensorLocation::kDEVICE);

    /* right */
    CHECK_STATE(m_engine->getTensorIOMode("right") == nvinfer1::TensorIOMode::kINPUT);
    shape = m_engine->getTensorShape("right");
    CHECK_STATE(shape.d[0] == 1 && shape.d[1] == 3 && shape.d[2] == m_height && shape.d[3] == m_width);
    CHECK_STATE(m_engine->getTensorDataType("right") == nvinfer1::DataType::kFLOAT);
    CHECK_STATE(m_engine->getTensorLocation("right") == nvinfer1::TensorLocation::kDEVICE);

    /* disp */
    CHECK_STATE(m_engine->getTensorIOMode("disp") == nvinfer1::TensorIOMode::kOUTPUT);
    shape = m_engine->getTensorShape("disp");
    CHECK_STATE(shape.d[0] == 1 && shape.d[1] == 1 && shape.d[2] == m_height && shape.d[3] == m_width);
    CHECK_STATE(m_engine->getTensorDataType("disp") == nvinfer1::DataType::kFLOAT);
    CHECK_STATE(m_engine->getTensorLocation("disp") == nvinfer1::TensorLocation::kDEVICE);
}

void FoundationStereo::build_or_load()
{
    if (m_runtime) return; // Already loaded
    std::filesystem::path engine_filepath = m_options.engine_filepath;
    if (!std::filesystem::exists(engine_filepath)) {
        printf("[INFO ] [FoundationStereo] Engine file not found at: %s\n", engine_filepath.c_str());
        build();
    }
    printf("[INFO ] [FoundationStereo] Loading engine file: %s\n", engine_filepath.c_str());
    load();
}

void FoundationStereo::infer(const Image3fCHW& im0,
                             const Image3fCHW& im1,
                             Image1fCHW& out_disparity_map,
                             cudaStream_t stream)
{
    build_or_load();

    CHECK_ARG(im0.size() == glm::ivec2(m_width, m_height));
    CHECK_ARG(im1.size() == glm::ivec2(m_width, m_height));
    CHECK_ARG(out_disparity_map.size() == glm::ivec2(m_height, m_width));

    CHECK_STATE(m_context->allInputDimensionsSpecified());

    CHECK_STATE(m_context->setTensorAddress("left", im0.data_d()));
    CHECK_STATE(m_context->setTensorAddress("right", im1.data_d()));
    CHECK_STATE(m_context->setTensorAddress("disp", out_disparity_map.data_d()));

    CHECK_STATE(m_context->enqueueV3(stream));
}
