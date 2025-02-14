#pragma once

#include "GSCamera.h"
#include "Scene.h"
#include "gs_loss.h"

namespace gs_train
{
class GSRasterizer
{
    GSLoss m_gs_loss;

    DeviceBuffer m_geometry_buffer{"GSFunc/geometry_buffer"};
    DeviceBuffer m_binning_buffer{"GSFunc/binning_buffer"};
    DeviceBuffer m_image_buffer{"GSFunc/image_buffer"};

public:
    explicit GSRasterizer() = default;
    ~GSRasterizer() = default;

    /// \param[out] out_colorbuffer
    ///     Output colorbuffer of shape (3, H, W)
    /// \return
    ///     Number of gaussians rendered. If zero, backward pass shall be skipped
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

    /// \param[out] out_colorbuffer
    ///     Output colorbuffer of shape (3, H, W)
    int forward(const float* background_d, const Scene& scene, const GSCamera& camera, float* out_colorbuffer);
};
} // namespace gs_train
