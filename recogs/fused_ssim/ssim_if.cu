#include "ssim_if.h"

// Block size
#define BX 32
#define BY 32

using namespace gs_train;

float* FusedSSIM::forward( //
    float C1,
    float C2,
    int B,
    int CH,
    int H,
    int W,
    const float* img1,
    const float* img2,
    bool train)
{
    m_ssim_map.resize(B * CH * H * W * sizeof(float));
    m_dm_dmu1.resize(B * CH * H * W * sizeof(float));
    m_dm_dsigma1_sq.resize(B * CH * H * W * sizeof(float));
    m_dm_dsigma12.resize(B * CH * H * W * sizeof(float));
    if (train) {
        m_dm_dmu1.fill(0);
        m_dm_dsigma1_sq.fill(0);
        m_dm_dsigma12.fill(0);
    }
    dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
    dim3 block(BX, BY, 1);
    fusedssimCUDA<<<grid, block>>>( //
        H,
        W,
        CH,
        C1,
        C2,
        const_cast<float*>(img1),
        const_cast<float*>(img2),
        m_ssim_map.data_ptr<float>(),
        m_dm_dmu1.data_ptr<float>(),
        m_dm_dsigma1_sq.data_ptr<float>(),
        m_dm_dsigma12.data_ptr<float>());
    return m_ssim_map.data_ptr<float>();
}

float* FusedSSIM::backward( //
    float C1,
    float C2,
    int B,
    int CH,
    int H,
    int W,
    const float* img1,
    const float* img2,
    const float* dL_dmap)
{
    m_dL_dimg1.resize(B * CH * H * W * sizeof(float));
    m_dL_dimg1.fill(0);
    dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
    dim3 block(BX, BY, 1);
    fusedssim_backwardCUDA<<<grid, block>>>( //
        H,
        W,
        CH,
        C1,
        C2,
        const_cast<float*>(img1),
        const_cast<float*>(img2),
        const_cast<float*>(dL_dmap),
        m_dL_dimg1.data_ptr<float>(),
        m_dm_dmu1.data_ptr<float>(),
        m_dm_dsigma1_sq.data_ptr<float>(),
        m_dm_dsigma12.data_ptr<float>());
    return m_dL_dimg1.data_ptr<float>();
}
