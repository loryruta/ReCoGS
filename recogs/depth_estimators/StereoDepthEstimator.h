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

    thrust::device_vector<float> m_im0;
    thrust::device_vector<float> m_im1;
    thrust::device_vector<float> m_pcvnet_im0;           // Padded/rotated
    thrust::device_vector<float> m_pcvnet_im1;           // Padded/rotated
    thrust::device_vector<float> m_pcvnet_disparity_map; // Padded/rotated

public:
    bool debug;
    std::string debug_image_prefix;

    explicit StereoDepthEstimator(App& app, Options options);
    ~StereoDepthEstimator() = default;

    /// Estimate depth by performing stereo matching on the given axis (horizontal or vertical).
    /// \param[inout] inout_color_depth
    ///     The output color/depth (only depth is written).
    void estimate_single_axis(
        const GSCamera& camera, Axis axis, float b, Image4fHWC& inout_color_depth, cudaStream_t stream);

    /// Estimate depth by performing horizontal and vertical stereo matching.
    /// \param[inout] inout_color_depth
    ///     The output color/depth (only depth is written).
    void estimate_hv(const GSCamera& camera, float b, Image4fHWC& inout_color_depth, cudaStream_t stream);
};
} // namespace gs_train
