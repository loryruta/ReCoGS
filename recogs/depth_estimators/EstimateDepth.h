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
        std::string debug_image_prefix{};
    };

    enum class Axis { H, V };

private:
    App& m_app;
    const Options m_options;

    std::unique_ptr<PCVNetEngine> m_pcvnet_engine;

    GSCamera m_rview; // TODO define GSCamera copy constructor!
    DeviceBuffer m_im0{"EstimateDepth/im0"};
    DeviceBuffer m_im1{"EstimateDepth/im1"};
    DeviceBuffer m_pcvnet_im0{"EstimateDepth/pcvnet_im0"};                     // Padded/rotated
    DeviceBuffer m_pcvnet_im1{"EstimateDepth/pcvnet_im1"};                     // Padded/rotated
    DeviceBuffer m_pcvnet_disparity_map{"EstimateDepth/pcvnet_disparity_map"}; // Padded/rotated

public:
    bool debug;
    std::string debug_image_prefix;

    explicit EstimateDepth(App& app, Options options);
    ~EstimateDepth() = default;

    /// Estimate depth by performing stereo matching on one axis (horizontal or vertical).
    /// \param[inout] out_depth
    ///     The output depth-buffer. It must be pre-allocated and its resolution must match the camera's.
    ///     Its values are "min-ed" with new estimates therefore it's uninitialized if filled with INFINITY values.
    void estimate_single_axis(const GSCamera& camera, Axis axis, float b, Image1fCHW& inout_depth);

    /// Estimate depth by performing a horizontal and vertical stereo matching.
    /// \param[inout] inout_depth
    ///     The output depth-buffer. It must be pre-allocated and its resolution must match the camera's.
    ///     Its values are "min-ed" with new estimates therefore it's uninitialized if filled with INFINITY values.
    void estimate_hv(const GSCamera& camera, float b, Image1fCHW& inout_depth);
};
} // namespace gs_train
