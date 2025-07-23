#pragma once

#include <mutex>

#include <glm/glm.hpp>

#include "Camera.h"
#include "DepthEstimator.h"
#include "PCVNetHV.h"
#include "utils/image/Image.h"
#include "utils/image/image_copy.h"

namespace recogs
{
// Forward decl
class App;

struct PCVNetHV_DepthEstimatorParams : public DepthEstimatorParams {
    enum Axis : uint8_t { Axis_Horizontal = 0, Axis_Vertical, Axis_Both } axis = Axis_Both;

    void validate() const override;
};

/// Estimate the depth of a view in a GS scene using Stereo Matching.
class PCVNetHV_DepthEstimator : public DepthEstimator
{
private:
    std::unique_ptr<PCVNetHV> m_engine;

    Camera m_rview;
    thrust::device_vector<float> m_im0;
    thrust::device_vector<float> m_im1;
    thrust::device_vector<float> m_pcvnet_im0;           // Padded/rotated
    thrust::device_vector<float> m_pcvnet_im1;           // Padded/rotated
    thrust::device_vector<float> m_pcvnet_disparity_map; // Padded/rotated

    std::mutex m_mutex;

public:
    explicit PCVNetHV_DepthEstimator(std::unique_ptr<PCVNetHV>&& engine);

    DepthEstimatorType type() const override { return DepthEstimatorType::PCVNetHV; };

    /// Estimate depth by performing stereo matching on the given axis (horizontal or vertical).
    /// \param[inout] inout_color_depth
    ///     The output color/depth (only depth is written).
    void estimate_single_axis(const PCVNetHV_DepthEstimatorParams& params);
    /// Estimate depth by performing horizontal and vertical stereo matching.
    void estimate_hv(PCVNetHV_DepthEstimatorParams& params);

    void estimate(const DepthEstimatorParams& params) override;
};
} // namespace recogs
