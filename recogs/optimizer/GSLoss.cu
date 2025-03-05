#include "GSLoss.h"

#include "utils/misc_utils.h"

#define FUSEDSSIM_C1 (0.01f * 0.01f)
#define FUSEDSSIM_C2 (0.03f * 0.03f)

#define NUM_THREADS 1024

using namespace gs_train;

GSLoss::GSLoss()
{
    L1_sum.resize(1);
    Lssim_sum.resize(1);
    loss.resize(1);
}

__global__ void forward_kernel_1( //
    size_t BCHW,
    const float* img_pred,
    const float* img_gt,
    const float* ssim_map,
    int8_t* out_sign,
    float* out_L1_sum,
    float* out_Lssim_sum)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BCHW) return;
    float y = img_pred[i];
    float yhat = img_gt[i];
    float dif = y - yhat;
    out_sign[i] = dif < 0.0f ? -1 : (dif > 0.0f ? 1 : 0); // Save the sign for the backward
    atomicAdd(out_L1_sum, abs(dif));
    atomicAdd(out_Lssim_sum, ssim_map[i]);
}

__global__ void forward_kernel_2( //
    size_t BCHW,
    float lambda,
    float* L1_sum,
    float* Lssim_sum,
    float* out_loss)
{
    *L1_sum /= float(BCHW);
    *Lssim_sum /= float(BCHW);
    *out_loss = (1.0f - lambda) * *L1_sum + lambda * (1.0f - *Lssim_sum);
}

void GSLoss::forward(int B,
                     int C,
                     int H,
                     int W,
                     const float* img_pred,
                     const float* img_gt,
                     float& out_loss,
                     float& out_L1,
                     float& out_Lssim,
                     cudaStream_t stream)
{
    float* ssim_map = m_fused_ssim.forward(FUSEDSSIM_C1, FUSEDSSIM_C2, B, C, H, W, img_pred, img_gt, stream);
    size_t BCHW = B * C * H * W;
    signs.resize(BCHW);
    float* L1_sum_d = RCGS_TPTR(L1_sum);
    float* Lssim_sum_d = RCGS_TPTR(Lssim_sum);
    float* loss_d = RCGS_TPTR(loss);
    CHECK_CUDA(cudaMemsetAsync(L1_sum_d, 0, sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(Lssim_sum_d, 0, sizeof(float), stream));
    dim3 num_blocks = (BCHW + 1023) / 1024;
    dim3 block_dim = 1024;
    forward_kernel_1<<<num_blocks, block_dim, 0, stream>>>(
        BCHW, img_pred, img_gt, ssim_map, RCGS_TPTR(signs), L1_sum_d, Lssim_sum_d);
    forward_kernel_2<<<1, 1, 0, stream>>>(BCHW, k_lambda, L1_sum_d, Lssim_sum_d, loss_d);
    CHECK_CUDA(cudaMemcpyAsync(&out_loss, loss_d, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&out_L1, L1_sum_d, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&out_Lssim, Lssim_sum_d, sizeof(float), cudaMemcpyDeviceToHost, stream));
}

__global__ void backward_kernel( //
    size_t BCHW,
    const int8_t* sign,
    float lambda,
    const float* dLssim_dy,
    float* out_dL_dy)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BCHW) return;
    float dL1_dy = (1.0f - lambda) / float(BCHW) * float(sign[i]);
    out_dL_dy[i] = dL1_dy + dLssim_dy[i];
}

void GSLoss::backward( //
    int B,
    int C,
    int H,
    int W,
    const float* img_pred,
    const float* img_gt,
    float* out_dL_dy,
    cudaStream_t stream)
{
    size_t BCHW = B * C * H * W;
    // Compute dLdssim_dy
    dL_dmap.resize(BCHW);
    thrust::fill(thrust::cuda::par.on(stream), dL_dmap.begin(), dL_dmap.end(), -k_lambda / float(BCHW));
    float* dLssim_dy =
        m_fused_ssim.backward(FUSEDSSIM_C1, FUSEDSSIM_C2, B, C, H, W, img_pred, img_gt, RCGS_TPTR(dL_dmap), stream);
    // Compute dL_dy
    dim3 num_blocks = (BCHW + 1023) / 1024;
    dim3 block_dim = 1024;
    backward_kernel<<<num_blocks, block_dim, 0, stream>>>(BCHW, RCGS_TPTR(signs), k_lambda, dLssim_dy, out_dL_dy);
}
