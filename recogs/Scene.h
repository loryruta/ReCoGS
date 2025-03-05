#pragma once

#include <thrust/device_vector.h>

#include "utils/DeviceBuffer.h"

namespace gs_train
{
struct Scene {
    int num_vertices;

    // Parameters
    thrust::device_vector<float> means;
    // thrust::device_vector<float> m_normals;
    thrust::device_vector<float> shs;
    thrust::device_vector<float> opacities;
    thrust::device_vector<float> scales;
    thrust::device_vector<float> rotations;

    // Optimized SHs
    thrust::device_vector<float> shs_2;

    // Gradients
    thrust::device_vector<float> dL_dmean2D;
    thrust::device_vector<float> dL_dconic;
    thrust::device_vector<float> dL_dopacity;
    thrust::device_vector<float> dL_dcolor;
    thrust::device_vector<float> dL_dmean3D;
    thrust::device_vector<float> dL_dcov3D;
    thrust::device_vector<float> dL_dsh;
    thrust::device_vector<float> dL_dscale;
    thrust::device_vector<float> dL_drot;

    explicit Scene(int num_vertices);
    ~Scene() = default;

    [[nodiscard]] bool is_prepared_for_training() const { return !dL_dmean2D.empty(); }
    void prepare_for_training();
    void zero_grad(cudaStream_t stream);

    /// Number of bytes occupied in memory by the scene
    [[nodiscard]] size_t num_bytes() const;
};
} // namespace gs_train
