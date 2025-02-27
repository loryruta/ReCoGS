#pragma once

#include "GSCamera.h"
#include "Scene.h"
#include "gs_loss.h"
#include "utils/image/Image.h"

#include <thrust/device_vector.h>

namespace gs_train
{
/// An interface class to the diff_gaussian_rasterizer code.
/// Reference:
/// https://github.com/graphdeco-inria/diff-gaussian-rasterization
class GSRasterizer
{
    GSLoss m_gs_loss;

    thrust::device_vector<char> m_geometry_buffer;
    thrust::device_vector<char> m_binning_buffer;
    thrust::device_vector<char> m_image_buffer;

public:
    explicit GSRasterizer() = default;
    ~GSRasterizer() = default;

    int forward(const float* background_d, const Scene& scene, const GSCamera& camera, Image3fCHW& out_colorbuffer);

    /// Frontend function for performing the GS forward.
    /// \param[out] out_colorbuffer Output colorbuffer
    /// \param[out] out_depthbuffer Output depthbuffer
    int forward(const float* background_d,
                const Scene& scene,
                const GSCamera& camera,
                Image3fCHW& out_colorbuffer,
                Image1fCHW& out_depthbuffer);

private:
    /// \param[out] out_colorbuffer Output colorbuffer of shape (3, H, W)
    /// \param[out] out_depthbuffer Output depthbuffer of shape (1, H, W)
    /// \return Number of gaussians rendered
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
                float* out_colorbuffer,
                float* out_depthbuffer);
};
} // namespace gs_train
