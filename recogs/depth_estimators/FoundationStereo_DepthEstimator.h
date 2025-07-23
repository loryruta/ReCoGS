#pragma once

#include "Camera.h"
#include "DepthEstimator.h"
#include "FoundationStereo.h"
#include "utils/image/Image.h"

BEGIN_NAMESPACE

class FoundationStereo_DepthEstimator : public DepthEstimator
{
private:
    std::unique_ptr<FoundationStereo> m_engine;

    /* Scratch memory */
    std::unique_ptr<Image3fCHW> m_left;
    std::unique_ptr<Image3fCHW> m_right;
    std::unique_ptr<Image1fCHW> m_disp;
    Camera m_im1_camera;
    thrust::device_vector<float> m_im1_data;

public:
    explicit FoundationStereo_DepthEstimator(std::unique_ptr<FoundationStereo>&& engine);

    [[nodiscard]] DepthEstimatorType type() const override { return DepthEstimatorType::FoundationStereo; };

    /// \param camera Camera of im0
    void estimate(const DepthEstimatorParams& params) override;
};

END_NAMESPACE
