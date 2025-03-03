#pragma once

#include "ssim.h"

#include "utils/DeviceBuffer.h"

namespace gs_train
{
class FusedSSIM
{
private:
    DeviceBuffer m_ssim_map{"fusedssim/ssim_map"};
    DeviceBuffer m_dm_dmu1{"fusedssim/dm_dmu1"};
    DeviceBuffer m_dm_dsigma1_sq{"fusedssim/dm_dsigma1_sq"};
    DeviceBuffer m_dm_dsigma12{"fusedssim/dm_dsigma12"};

    DeviceBuffer m_dL_dimg1{"fusedssim/dL_dimg1"};

public:
    explicit FusedSSIM() = default;
    ~FusedSSIM() = default;

    [[nodiscard]] const float* ssim_map() const { return m_ssim_map.data_ptr<float>(); }

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
        bool train,
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
} // namespace gs_train
