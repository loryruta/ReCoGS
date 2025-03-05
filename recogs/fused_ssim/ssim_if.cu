#include "ssim_if.h"

#include "utils/cuda_utils.h"

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
    cudaStream_t stream)
{
    m_ssim_map.resize(B * CH * H * W);
    m_dm_dmu1.resize(B * CH * H * W);
    m_dm_dsigma1_sq.resize(B * CH * H * W);
    m_dm_dsigma12.resize(B * CH * H * W);

    thrust::fill(thrust::cuda::par.on(stream), m_ssim_map.begin(), m_ssim_map.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), m_dm_dmu1.begin(), m_dm_dmu1.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), m_dm_dsigma1_sq.begin(), m_dm_dsigma1_sq.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream), m_dm_dsigma12.begin(), m_dm_dsigma12.end(), 0);

    dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
    dim3 block(BX, BY, 1);
    fusedssimCUDA<<<grid, block, 0, stream>>>( //
        H,
        W,
        CH,
        C1,
        C2,
        const_cast<float*>(img1),
        const_cast<float*>(img2),
        RCGS_TPTR(m_ssim_map),
        RCGS_TPTR(m_dm_dmu1),
        RCGS_TPTR(m_dm_dsigma1_sq),
        RCGS_TPTR(m_dm_dsigma12));
    return RCGS_TPTR(m_ssim_map);
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
    const float* dL_dmap,
    cudaStream_t stream)
{
    m_dL_dimg1.resize(B * CH * H * W);
    thrust::fill(thrust::cuda::par.on(stream), m_dL_dimg1.begin(), m_dL_dimg1.end(), 0);

    dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
    dim3 block(BX, BY, 1);
    fusedssim_backwardCUDA<<<grid, block, 0, stream>>>( //
        H,
        W,
        CH,
        C1,
        C2,
        const_cast<float*>(img1),
        const_cast<float*>(img2),
        const_cast<float*>(dL_dmap),
        RCGS_TPTR(m_dL_dimg1),
        RCGS_TPTR(m_dm_dmu1),
        RCGS_TPTR(m_dm_dsigma1_sq),
        RCGS_TPTR(m_dm_dsigma12));
    return RCGS_TPTR(m_dL_dimg1);
}
