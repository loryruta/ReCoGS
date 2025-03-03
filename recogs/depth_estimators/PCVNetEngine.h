#pragma once

#include <filesystem>

#include <glm/glm.hpp>

#include "NvInfer.h"
#include "utils/image/Image.h"

namespace gs_train
{

struct PCVNetEngine {
public:
    struct Options {
        std::filesystem::path onnx_filepath;
        glm::ivec2 optprofile_min_image_size;
        glm::ivec2 optprofile_opt_image_size;
        glm::ivec2 optprofile_max_image_size;
        std::filesystem::path engine_filepath;

        void validate() const;
    };

private:
    const Options m_options;

    std::unique_ptr<nvinfer1::ILogger> m_logger;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

public:
    // PCVNet TensorRT engine is built and optimized to work with fixed size input of the following dimensions.
    // Or alternatively, their rotated version (i.e. 90deg rotated; HxW) to support vertical stereo matching
    static constexpr uint32_t k_io_width = 1088;
    static constexpr uint32_t k_io_height = 768;

    explicit PCVNetEngine(Options options);
    ~PCVNetEngine() = default;

    void build_or_load();
    void infer( //
        const Image3fCHW& im0,
        const Image3fCHW& im1,
        Image1fCHW& out_disparity_map,
        cudaStream_t stream);

private:
    void build();
    void load();
};

} // namespace gs_train