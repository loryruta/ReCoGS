#pragma once

#include <thrust/device_vector.h>

#include "ssim.h"
#include "utils/cuda_utils.h"

namespace recogs
{
class FusedSSIM
{
private:
    thrust::device_vector<float> m_ssim_map;
    thrust::device_vector<float> m_dm_dmu1;
    thrust::device_vector<float> m_dm_dsigma1_sq;
    thrust::device_vector<float> m_dm_dsigma12;

    thrust::device_vector<float> m_dL_dimg1;

public:
    explicit FusedSSIM() = default;
    ~FusedSSIM() = default;

    [[nodiscard]] const float* ssim_map() const { return RCGS_TPTR(m_ssim_map); }

    /// Compare img1 and img2 and store the SSIM map. Read it with \c ssim_map().
    float* forward( //
        float C1,
        float C2,
        int B,
        int CH,
        int H,
        int W,
        const float* img1,
        const float* img2,
        cudaStream_t stream);

    /// \param[in] dL_dmap Gradient of the loss w.r.t. the output of the forward (SSIM map).
    float* backward( //
        float C1,
        float C2,
        int B,
        int CH,
        int H,
        int W,
        const float* img1,
        const float* img2,
        const float* dL_dmap,
        cudaStream_t stream);
};
} // namespace recogs
