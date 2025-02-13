#pragma once

#include "gs_loss.h"

namespace gs_train
{
class GSFunc
{
    GSLoss m_gs_loss;

    Buffer m_geometry_buffer{"gs/geometry_buffer"};
    Buffer m_binning_buffer{"gs/binning_buffer"};
    Buffer m_image_buffer{"gs/image_buffer"};

public:
    explicit GSFunc() = default;
    ~GSFunc() = default;

    /// \return Number of gaussians rendered. If zero, backward pass shall be skipped
    int forward(int W,
                int H,
                float* background,
                int num_vertices,
                float* means,
                float* shs,
                float* opacities,
                float* scales,
                float* rotations,
                float* viewmatrix,
                float* projmatrix,
                float* campos,
                float tan_fovx,
                float tan_fovy,
                float* out_colorbuffer);

    void backward(float* out_dL_dmean2D,
                  float* out_dL_dconic,
                  float* out_dL_dopacity,
                  float* out_dL_dcolor,
                  float* dL_dmean3D,
                  float* dL_dcov3D,
                  float* dL_dsh,
                  float* dL_dscale,
                  float* dL_drot);
};
} // namespace gs_train
