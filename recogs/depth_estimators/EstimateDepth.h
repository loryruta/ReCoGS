#pragma once

#include <glm/glm.hpp>

#include "GSCamera.h"
#include "PCVNetEngine.h"
#include "Scene.h"
#include "utils/image/Image.h"
#include "utils/image/image_copy.h"

namespace gs_train
{
// Forward decl
class App;

/// Estimate the depth of a view in a 3DGS scene using PCVNet
class EstimateDepth
{
public:
    struct Options {
        /// If set, all images generated during the depth estimation process will be saved.
        bool debug = false;
        std::string image_prefix{};
    };

    enum class Axis { H, V };

private:
    App& m_app;
    const Options m_options;

    std::unique_ptr<PCVNetEngine> m_pcvnet_engine;

    GSCamera m_rview;
    DeviceBuffer m_im0{"EstimateDepth/im0"};
    DeviceBuffer m_im1{"EstimateDepth/im1"};
    DeviceBuffer m_pcvnet_im0{"EstimateDepth/pcvnet_im0"};
    DeviceBuffer m_pcvnet_im1{"EstimateDepth/pcvnet_im1"};
    DeviceBuffer m_depth{"EstimateDepth/depth"};

public:
    explicit EstimateDepth(App& app, Options options);
    ~EstimateDepth() = default;

    Image1fCHW operator()( //
        const GSCamera& camera,
        Axis axis,
        float b,
        AABB2i region = AABB2i{},
        int num_downsample = 0);
};
} // namespace gs_train
