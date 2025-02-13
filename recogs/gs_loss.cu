#include "gs_loss.h"

#include "utils/misc_utils.h"

#define FUSEDSSIM_C1 (0.01f * 0.01f)
#define FUSEDSSIM_C2 (0.03f * 0.03f)

#define NUM_THREADS 1024

using namespace gs_train;

GSLoss::GSLoss()
{
    m_tmp_buffer.resize(3 * sizeof(float));

    m_L1_avg = m_tmp_buffer.data_ptr<float>();
    m_Ldssim_avg = m_tmp_buffer.data_ptr<float>() + 1;
    m_loss = m_tmp_buffer.data_ptr<float>() + 2;
}

__global__ void forward_kernel_1( //
    size_t BCHW,
    const float* img_pred,
    const float* img_gt,
    const float* ssim_map,
    float* out_L1_avg,
    float* out_Ldssim_avg)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BCHW) return;

    float abs = std::abs(img_pred[i] - img_gt[i]);
    atomicAdd(out_L1_avg, abs / float(BCHW));
    atomicAdd(out_Ldssim_avg, ssim_map[i] / float(BCHW));
}

__global__ void forward_kernel_2( //
    float lambda,
    const float* L1_avg,
    const float* Ldssim_avg,
    float* out_loss)
{
    *out_loss = (1.0f - lambda) * (*L1_avg) + lambda * (1.0f - *Ldssim_avg);
}

float* GSLoss::forward(int B, int C, int H, int W, const float* img_pred, const float* img_gt)
{
    float* ssim_map = m_fused_ssim.forward(FUSEDSSIM_C1, FUSEDSSIM_C2, B, C, H, W, img_pred, img_gt, true /* train */);
    size_t BCHW = B * C * H * W;
    CHECK_CUDA(cudaMemset(m_L1_avg, 0, sizeof(float)));
    CHECK_CUDA(cudaMemset(m_Ldssim_avg, 0, sizeof(float)));
    dim3 num_blocks = div_ceil(BCHW, size_t(NUM_THREADS));
    dim3 block_dim = NUM_THREADS;
    forward_kernel_1<<<num_blocks, block_dim>>>(BCHW, img_pred, img_gt, ssim_map, m_L1_avg, m_Ldssim_avg);
    forward_kernel_2<<<1, 1>>>(k_lambda, m_L1_avg, m_Ldssim_avg, m_loss);
    return m_loss;
}

template <typename T>
__global__ void fill_kernel(size_t N, T value, T* out_buffer)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out_buffer[i] = value;
}

__global__ void backward_kernel( //
    size_t BCHW,
    const float* img_pred,
    const float* img_gt,
    float lambda,
    const float* dLdssim_dy,
    float* out_dL_dy)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BCHW) return;
    float dif = img_pred[i] - img_gt[i];
    float sgn = dif > 1e-8f ? 1.0f : (dif < -1e-8f ? -1.0f : 0.0f);
    float invN = 1.0f / float(BCHW);
    out_dL_dy[i] = invN * ((1.0f - lambda) * sgn - lambda * dLdssim_dy[i]);
}

void GSLoss::backward( //
    int B,
    int C,
    int H,
    int W,
    const float* img_pred,
    const float* img_gt,
    float* out_dL_dy)
{
    size_t BCHW = B * C * H * W;
    m_dL_dmap.resize(BCHW * sizeof(float));
    /* Compute dLdssim/dy */
    dim3 num_blocks = div_ceil(BCHW, size_t(NUM_THREADS));
    dim3 block_dim = NUM_THREADS;
    fill_kernel<float><<<num_blocks, block_dim>>>(BCHW, 1.0f, m_dL_dmap.data_ptr<float>());
    float* dLdssim_dy =
        m_fused_ssim.backward(FUSEDSSIM_C1, FUSEDSSIM_C2, B, C, H, W, img_pred, img_gt, m_dL_dmap.data_ptr<float>());
    /* Compute dL/dy */
    backward_kernel<<<num_blocks, block_dim>>>(BCHW, img_pred, img_gt, k_lambda, dLdssim_dy, out_dL_dy);
}
