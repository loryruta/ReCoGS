#pragma once

#include <filesystem>

#include <NvInfer.h>
#include <glm/glm.hpp>

#include "utils/image/Image.h"

BEGIN_NAMESPACE

struct FoundationStereo_Options {
    std::filesystem::path onnx_filepath;   ///< Input .onnx filepath
    std::filesystem::path engine_filepath; ///< TensorRT optimized engine filepath
};

/// Class wrapping the TensorRT engine for Foundation Stereo:
/// https://nvlabs.github.io/FoundationStereo/
class FoundationStereo
{
private:
    const FoundationStereo_Options m_options;

    int m_width = -1;
    int m_height = -1;

    std::unique_ptr<nvinfer1::ILogger> m_logger;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

public:
    explicit FoundationStereo(FoundationStereo_Options options);
    ~FoundationStereo();

    [[nodiscard]] int width() const { return m_width; }
    [[nodiscard]] int height() const { return m_height; }

    void build_or_load();
    void infer(const Image3fCHW& im0, const Image3fCHW& im1, Image1fCHW& out_disparity_map, cudaStream_t stream);

private:
    void build();
    void load();
};

END_NAMESPACE
