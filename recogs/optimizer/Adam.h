#pragma once

#include <cstdint>
#include <vector>

// Reference:
// https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

namespace gs_train
{
class Adam
{
public:
    struct Options {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0;
        bool amsgrad = false;  // Unused
        bool maximize = false; // Unused
        float eps = 1e-08f;
    };

    struct ParamSet {
        float* params{};
        float* grads{};
        size_t num_params{};
        float lr = 1e-4f;

        [[nodiscard]] bool is_valid() const { return params && grads && num_params && lr > 0.0f; }
    };

private:
    const Options m_options;

    std::vector<ParamSet> m_params_sets;
    uint32_t m_N{};
    float* m_m{}; ///< First moment
    float* m_v{}; ///< Second moment
    int m_t = 1;
    float m_beta1_decayed{};
    float m_beta2_decayed{};

public:
    explicit Adam(const std::vector<ParamSet>& params_sets, const Options& options, cudaStream_t stream);
    ~Adam();

    void zero_grad(cudaStream_t stream);
    void step(cudaStream_t stream);
};
} // namespace gs_train