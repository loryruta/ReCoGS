#pragma once

#include "fused_ssim/ssim_if.h"

namespace gs_train
{
class GSLoss
{
private:
    static constexpr float k_lambda = 0.2f;

    FusedSSIM m_fused_ssim;

    DeviceBuffer m_tmp_buffer{"gsloss/tmp_buffer"};
    /* Forward */
    float* m_L1_avg{};
    float* m_Ldssim_avg{};
    /// Forward output
    float* m_loss{};
    /* Backward */
    DeviceBuffer m_dL_dmap{"gsloss/dL_dmap"};
    /// Backward output: gradient of loss w.r.t. image prediction
    DeviceBuffer m_dL_dy{"gsloss/dL_dy"};

public:
    explicit GSLoss();
    ~GSLoss() = default;

    /// Run the forward pass for the Gaussian Splatting loss.
    /// \param B Number of batches
    /// \param C Number of channels
    /// \param H Height of the image
    /// \param W Width of the image
    /// \param img_pred Image predicted by Gaussian Splatting of shape (B, C, H, W)
    /// \param img_gt Ground truth image of shape (B, C, H, W)
    /// \return A device pointer to the loss, a single scalar
    float* forward(int B, int C, int H, int W, const float* img_pred, const float* img_gt);

    /// Run the backward pass for the Gaussian Splatting loss.
    /// \param[out] out_dL_dy
    ///     A device pointer to the gradient of the loss w.r.t. the predicted image
    void backward(int B, int C, int H, int W, const float* img_pred, const float* img_gt, float* out_dL_dy);
};
} // namespace gs_train
