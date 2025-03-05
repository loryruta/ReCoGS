#include "Adam.h"

#include <cassert>
#include <cstdint>

#include <glm/glm.hpp>

#include "utils/cuda_utils.h"
#include "utils/misc_utils.h"

#define NUM_THREADS 1024

using namespace gs_train;

namespace
{
__global__ void step_kernel( //
    size_t N,
    float** __restrict__ inout_params,
    const float** __restrict__ grads,
    const float* __restrict__ lrs,
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
    float param = *inout_params[i];
    float gt = *grads[i];
    float mi = m[i];
    float vi = v[i];
    // Compute
    mi = beta1 * mi + (1.0f - beta1) * gt;
    vi = beta2 * vi + (1.0f - beta2) * gt * gt;
    float mhat = mi / (1.0f - beta1_decayed);
    float vhat = vi / (1.0f - beta2_decayed);
    param = param - lrs[i] * mhat / (__fsqrt_rn(vhat) + eps);
    // Write to global memory
    *inout_params[i] = param;
    m[i] = mi;
    v[i] = vi;
}

__global__ void init_kernel( //
    size_t N,
    const Adam::ParamSet* param_sets,
    size_t num_param_sets,
    float** out_params,
    float** out_grads,
    float* out_lrs,
    float* out_m,
    float* out_v)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    size_t param_set_idx = 0;
    size_t start_param_idx = 0;
    for (; param_set_idx < num_param_sets; ++param_set_idx) {
        const Adam::ParamSet& param_set = param_sets[param_set_idx];
        if ((start_param_idx + param_set.num_params) > i) break;
        start_param_idx += param_set.num_params;
    }
    assert(param_set_idx < num_param_sets); // Wrong algorithm otherwise
    const Adam::ParamSet& param_set = param_sets[param_set_idx];
    assert(i >= start_param_idx && (i - start_param_idx) < param_set.num_params);
    out_params[i] = param_set.params + (i - start_param_idx);
    out_grads[i] = param_set.grads + (i - start_param_idx);
    out_lrs[i] = param_set.lr;
    out_m[i] = 0.0f;
    out_v[i] = 0.0f;
}

__global__ void zero_grad_kernel(size_t N, float** out_grads)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    *out_grads[i] = 0;
}

} // namespace

Adam::Adam(std::span<ParamSet> param_sets, const Options& options, cudaStream_t stream) : m_options(options)
{
    m_N = 0;
    for (const ParamSet& param_set : param_sets) {
        CHECK_ARG(param_set.is_valid(), "Parameter set isn't valid");
        m_N += param_set.num_params;
    }
    CHECK_CUDA(cudaMallocAsync(&m_params, m_N * sizeof(float*), stream));
    CHECK_CUDA(cudaMallocAsync(&m_grads, m_N * sizeof(float*), stream));
    CHECK_CUDA(cudaMallocAsync(&m_lrs, m_N * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&m_m, m_N * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&m_v, m_N * sizeof(float), stream));

    /* Linearize parameters sets */
    const Adam::ParamSet* param_set_d = to_device_array(param_sets); // TODO use stream
    dim3 num_blocks = div_ceil<size_t>(m_N, NUM_THREADS);
    dim3 block_dim = NUM_THREADS;
    init_kernel<<<num_blocks, block_dim, 0, stream>>>(
        m_N, param_set_d, param_sets.size(), m_params, m_grads, m_lrs, m_m, m_v);
    CHECK_CUDA(cudaDeviceSynchronize()); // TODO resolve with TODO above

    m_beta1_decayed = options.beta1;
    m_beta2_decayed = options.beta2;
}

Adam::~Adam()
{
    printf("[DEBUG] [Adam] Destroying...\n");
    CHECK_CUDA(cudaFree(m_params));
    CHECK_CUDA(cudaFree(m_grads));
    CHECK_CUDA(cudaFree(m_lrs));
    CHECK_CUDA(cudaFree(m_m));
    CHECK_CUDA(cudaFree(m_v));
    printf("[DEBUG] [Adam] Destroyed\n");
}

void Adam::zero_grad(cudaStream_t stream)
{
    dim3 num_blocks = (m_N + 1023) / 1024;
    dim3 block_dim = 1024;
    zero_grad_kernel<<<num_blocks, block_dim, 0, stream>>>(m_N, m_grads);
}

void Adam::step(cudaStream_t stream)
{
    dim3 num_blocks = div_ceil(m_N, size_t(NUM_THREADS));
    dim3 block_dim = NUM_THREADS;
    step_kernel<<<num_blocks, block_dim, 0, stream>>>( //
        m_N,
        m_params,
        (const float**) m_grads,
        m_lrs,
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
