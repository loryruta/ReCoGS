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

/// Estimate the depth of a view in a GS scene using Stereo Matching.
class StereoDepthEstimator
{
public:
    struct Options {
        /// If set, all images generated during the depth estimation process will be saved.
        bool debug = false;
        std::string debug_image_prefix{};
    };

    enum class Axis { H, V };

private:
    App& m_app;
    const Options m_options;

    std::unique_ptr<PCVNetEngine> m_pcvnet_engine;

    DeviceBuffer m_im0{"EstimateDepth/im0"};
    DeviceBuffer m_im1{"EstimateDepth/im1"};
    DeviceBuffer m_pcvnet_im0{"EstimateDepth/pcvnet_im0"};                     // Padded/rotated
    DeviceBuffer m_pcvnet_im1{"EstimateDepth/pcvnet_im1"};                     // Padded/rotated
    DeviceBuffer m_pcvnet_disparity_map{"EstimateDepth/pcvnet_disparity_map"}; // Padded/rotated

public:
    bool debug;
    std::string debug_image_prefix;

    explicit StereoDepthEstimator(App& app, Options options);
    ~StereoDepthEstimator() = default;

    /// Estimate depth by performing stereo matching on the given axis (i.e. horizontal/vertical).
    /// \param[inout] out_depth
    ///     The output depth-buffer. It must be pre-allocated and its resolution must match the camera's.
    ///     Its values are "min-ed" with new estimates therefore it's uninitialized if filled with INFINITY values.
    void estimate_single_axis(const GSCamera& camera, Axis axis, float b, Image1fCHW& inout_depth, cudaStream_t stream);

    /// Estimate depth by performing horizontal and vertical stereo matching.
    /// \param[inout] inout_depth
    ///     The output depth-buffer. It must be pre-allocated and its resolution must match the camera's.
    ///     Its values are "min-ed" with new estimates therefore it's uninitialized if filled with INFINITY values.
    void estimate_hv(const GSCamera& camera, float b, Image1fCHW& inout_depth, cudaStream_t stream);
};
} // namespace gs_train
