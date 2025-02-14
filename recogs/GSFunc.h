#pragma once

#include "gs_loss.h"

namespace gs_train
{
class GSFunc
{
    GSLoss m_gs_loss;

    DeviceBuffer m_geometry_buffer{"GSFunc/geometry_buffer"};
    DeviceBuffer m_binning_buffer{"GSFunc/binning_buffer"};
    DeviceBuffer m_image_buffer{"GSFunc/image_buffer"};

public:
    explicit GSFunc() = default;
    ~GSFunc() = default;

    /// \return Number of gaussians rendered. If zero, backward pass shall be skipped
    int forward(int W,
                int H,
                const float* background,
                int num_vertices,
                const float* means,
                const float* shs,
                const float* opacities,
                const float* scales,
                const float* rotations,
                const float* viewmatrix,
                const float* projmatrix,
                const float* campos,
                float tan_fovx,
                float tan_fovy,
                float* out_colorbuffer);
};
} // namespace gs_train
