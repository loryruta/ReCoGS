#pragma once

#include <mutex>

#include <glm/glm.hpp>

#include "GSCamera.h"
#include "GSRasterizer.h"
#include "PCVNetEngine.h"
#include "Scene.h"
#include "utils/image/Image.h"
#include "utils/image/image_copy.h"

namespace recogs
{
// Forward decl
class App;

struct StereoDepthEstimatorParams {
    const float* background_d;
    const Scene* scene;
    const GSCamera* camera;
    GSRasterizer* rasterizer;
    int axis = 0;                   ///< 0 = Horizontal, 1 = Vertical
    float b = 0.07f;                ///< The stereo baseline
    Image4fHWC* inout_color_depth;  ///< The output color/depth (only depth is written)
    cudaStream_t stream;            ///< The CUDA stream to run inference on
    bool debug = false;             ///< If set all images generated during depth estimation will be saved to files
    std::string debug_image_prefix; ///< Prefix placed before the image filename for debug

    [[nodiscard]] bool is_valid() const { return true; } // TODO
};

/// Estimate the depth of a view in a GS scene using Stereo Matching.
class StereoDepthEstimator
{
private:
    App& m_app;

    std::unique_ptr<PCVNetEngine> m_pcvnet_engine;

    thrust::device_vector<float> m_im0;
    thrust::device_vector<float> m_im1;
    thrust::device_vector<float> m_pcvnet_im0;           // Padded/rotated
    thrust::device_vector<float> m_pcvnet_im1;           // Padded/rotated
    thrust::device_vector<float> m_pcvnet_disparity_map; // Padded/rotated

    std::mutex m_mutex;

public:
    explicit StereoDepthEstimator(App& app);
    ~StereoDepthEstimator() = default;

    /// Estimate depth by performing stereo matching on the given axis (horizontal or vertical).
    /// \param[inout] inout_color_depth
    ///     The output color/depth (only depth is written).
    void estimate_single_axis(const StereoDepthEstimatorParams& params);

    /// Estimate depth by performing horizontal and vertical stereo matching.
    void estimate_hv(StereoDepthEstimatorParams& params);
};
} // namespace recogs
