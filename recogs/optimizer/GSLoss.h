#pragma once

#include <thrust/device_vector.h>

#include "fused_ssim/ssim_if.h"
#include "utils/DeviceBuffer.h"
#include "utils/image/Image.h"

namespace gs_train
{
class GSLoss
{
private:
    static constexpr float k_lambda = 0.2f; // Weight to give to D-SSIM

    FusedSSIM m_fused_ssim;

public:
    // Forward
    thrust::device_vector<float> L1_sum;
    thrust::device_vector<float> Lssim_sum;
    /// In the forward, we remember signs used for the `abs` backward.
    thrust::device_vector<int8_t> signs;
    thrust::device_vector<float> loss;

    // Backward
    thrust::device_vector<float> dL_dmap;

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
    void forward(const Image3fCHW& img_pred,
                 const Image3fCHW& img_gt,
                 float& out_loss,
                 float& out_L1,
                 float& out_Lssim,
                 cudaStream_t stream);

    /// Run the backward pass for the Gaussian Splatting loss.
    /// \param[out] out_dL_dy
    ///     A device pointer to the gradient of the loss w.r.t. the predicted image
    void backward(const Image3fCHW& img_pred, const Image3fCHW& img_gt, float* out_dL_dy, cudaStream_t stream);
};
} // namespace gs_train
