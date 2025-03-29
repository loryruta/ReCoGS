#include "Adam.h"

#include <cstdint>

#include "utils/cuda_utils.h"
#include "utils/misc_utils.h"

#define NUM_THREADS 512

using namespace gs_train;

namespace
{
__global__ void step_kernel( //
    uint32_t N,
    float* __restrict__ inout_params,
    const float* __restrict__ grads,
    float lr,
    float* __restrict__ m,
    float* __restrict__ v,
    float beta1,
    float beta2,
    float beta1_decayed,
    float beta2_decayed,
    float eps,
    float weight_decay,
    bool amsgrad,
    bool maximize)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Read from global memory
    float param = inout_params[i];
    float gt = grads[i];
    float mi = m[i];
    float vi = v[i];
    // Compute
    mi = beta1 * mi + (1.0f - beta1) * gt;
    vi = beta2 * vi + (1.0f - beta2) * gt * gt;
    float mhat = mi / (1.0f - beta1_decayed);
    float vhat = vi / (1.0f - beta2_decayed);
    param = param - lr * mhat / (__fsqrt_rn(vhat) + eps);
    // Write to global memory
    inout_params[i] = param;
    m[i] = mi;
    v[i] = vi;
}

__global__ void zero_grad_kernel(size_t N, float* __restrict__ out_grads)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out_grads[i] = 0;
}

} // namespace

Adam::Adam(const std::vector<ParamSet>& params_sets, const Options& options, cudaStream_t stream)
    : m_params_sets(params_sets), m_options(options)
{
    CHECK_ARG(params_sets.size() == 1, "Only one ParamSet is supported");
    CHECK_ARG(params_sets[0].is_valid(), "Parameter set isn't valid");
    m_N = params_sets[0].num_params;
    CHECK_CUDA(cudaMallocAsync(&m_m, m_N * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&m_v, m_N * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(m_m, 0, m_N * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(m_v, 0, m_N * sizeof(float), stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    m_beta1_decayed = options.beta1;
    m_beta2_decayed = options.beta2;
}

Adam::~Adam()
{
    printf("[DEBUG] [Adam] Destroying...\n");
    CHECK_CUDA(cudaFree(m_m));
    CHECK_CUDA(cudaFree(m_v));
    printf("[DEBUG] [Adam] Destroyed\n");
}

void Adam::zero_grad(cudaStream_t stream)
{
    dim3 num_blocks = (m_N + 1023) / 1024;
    dim3 block_dim = 1024;
    zero_grad_kernel<<<num_blocks, block_dim, 0, stream>>>(m_N, m_params_sets[0].grads);
}

void Adam::step(cudaStream_t stream)
{
    dim3 num_blocks = div_ceil<uint32_t>(m_N, NUM_THREADS);
    dim3 block_dim = NUM_THREADS;
    step_kernel<<<num_blocks, block_dim, 0, stream>>>( //
        m_N,
        m_params_sets[0].params,
        m_params_sets[0].grads,
        m_params_sets[0].lr,
        m_m,
        m_v,
        m_options.beta1,
        m_options.beta2,
        m_beta1_decayed,
        m_beta2_decayed,
        m_options.eps,
        m_options.weight_decay,
        m_options.amsgrad,
        m_options.maximize);

    ++m_t;
    m_beta1_decayed *= m_options.beta1;
    m_beta2_decayed *= m_options.beta2;
}
