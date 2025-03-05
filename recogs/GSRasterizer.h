#pragma once

#include <thrust/device_vector.h>

#include "GSCamera.h"
#include "Scene.h"
#include "optimizer/GSLoss.h"
#include "utils/image/Image.h"

namespace gs_train
{
/// An interface class to the diff_gaussian_rasterizer code.
/// Reference:
/// https://github.com/graphdeco-inria/diff-gaussian-rasterization
class GSRasterizer
{
    thrust::device_vector<uint8_t> m_geometry_buffer;
    thrust::device_vector<uint8_t> m_binning_buffer;
    thrust::device_vector<uint8_t> m_image_buffer;

public:
    bool debug = false;

    explicit GSRasterizer() = default;
    ~GSRasterizer() = default;

    int forward(const float* background_d,
                const Scene& scene,
                bool scene_2,
                const GSCamera& camera,
                Image3fCHW& out_colorbuffer,
                cudaStream_t stream);

    /// Frontend function for performing the GS forward.
    /// \param[out] out_colorbuffer Output colorbuffer
    /// \param[out] out_depthbuffer Output depthbuffer
    int forward(const float* background_d,
                const Scene& scene,
                bool scene_2,
                const GSCamera& camera,
                Image3fCHW& out_colorbuffer,
                Image1fCHW& out_depthbuffer,
                cudaStream_t stream);

    /// Frontend function for performing the GS backward.
    /// \note Output gradients are stored within the scene.
    /// \param scene
    /// \param scene_2
    ///     Whether to enable the second set of parameters within the scene.
    /// \param num_rendered
    ///     Number of gaussians rendered during forward (i.e. forward return value), internally called "R".
    /// \param background_d
    /// \param camera
    /// \param dL_dy
    ///     Gradient of loss w.r.t. the rendered image.
    void backward( //
        Scene& scene,
        bool scene_2,
        int num_rendered,
        const float* background_d,
        const GSCamera& camera,
        const float* dL_dy,
        cudaStream_t stream);

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
                float* out_depthbuffer,
                cudaStream_t stream);

    void backward( //
        int P,
        int R,
        const float* background_d,
        int W,
        int H,
        const float* means3D,
        const float* shs,
        const float* scales,
        const float* rotations,
        const float* viewmatrix,
        const float* projmatrix,
        const float* campos,
        float tan_fovx,
        float tan_fovy,
        const float* dL_dpix,
        float* dL_dmean2D,
        float* dL_dconic,
        float* dL_dopacity,
        float* dL_dcolor,
        float* dL_dmean3D,
        float* dL_dcov3D,
        float* dL_dsh,
        float* dL_dscale,
        float* dL_drot,
        cudaStream_t stream);
};
} // namespace gs_train
