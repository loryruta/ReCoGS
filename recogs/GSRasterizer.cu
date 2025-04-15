#include "GSRasterizer.h"

#include "rasterizer/rasterizer.h"

using namespace gs_train;

namespace
{
std::function<char*(size_t N)>
resize_functional(thrust::device_vector<uint8_t>& buffer, const char* name, size_t alignment)
{
    return [&buffer, name, alignment](size_t num_bytes) -> char* {
        if (num_bytes >= buffer.size()) {
            size_t new_size = div_ceil(num_bytes, alignment) * alignment;
            printf("[DEBUG] [GSRasterizer] Resizing %s to %zu bytes\n", name, new_size);
            buffer.resize(new_size);
        }
        return reinterpret_cast<char*>(RCGS_TPTR(buffer));
    };
}
} // namespace

int GSRasterizer::forward( //
    int W,
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
    float* out_color_depth,
    cudaStream_t stream)
{
    int num_rendered = CudaRasterizer::Rasterizer::forward( //
        resize_functional(m_geometry_buffer, "geometry_buffer", 1 << 24 /* 16MB */),
        resize_functional(m_binning_buffer, "binning_buffer", 1 << 24 /* 16MB */),
        resize_functional(m_image_buffer, "image_buffer", 1 << 24 /* 16MB */),
        num_vertices,
        3,  // sh_degree
        16, // M
        background,
        W,
        H,
        means,
        shs,
        nullptr, // colors_precomp
        opacities,
        scales,
        1.0f, // scale_modifier
        rotations,
        nullptr, // cov3D_precomp
        viewmatrix,
        projmatrix,
        campos,
        tan_fovx,
        tan_fovy,
        false, // prefiltered
        out_color_depth,
        nullptr, // radii
        debug,
        show_borders,
        border_size,
        stream);
    return num_rendered;
}

int GSRasterizer::forward(const float* background_d,
                          const Scene& scene,
                          bool scene_2,
                          const GSCamera& camera,
                          Image4fHWC& out_color_depth,
                          cudaStream_t stream)
{
    return forward( //
        (int) out_color_depth.width,
        (int) out_color_depth.height,
        background_d,
        scene.num_vertices,
        RCGS_TPTR(scene.means),
        RCGS_TPTR(scene_2 ? scene.shs_2 : scene.shs),
        RCGS_TPTR(scene.opacities),
        RCGS_TPTR(scene.scales),
        RCGS_TPTR(scene.rotations),
        camera.viewmatrix_d(),
        camera.projmatrix_d(),
        camera.campos_d(),
        camera.tan_fovx(),
        camera.tan_fovy(),
        out_color_depth.data_d(),
        stream);
}

void GSRasterizer::backward( //
    Scene& scene,
    bool scene_2,
    int num_rendered,
    const float* background_d,
    const GSCamera& camera,
    const float* dL_dy,
    cudaStream_t stream)
{
    CHECK_ARG(scene.is_prepared_for_training(), "Scene gradients not allocated. Call scene.prepare_for_training()");
    backward( //
        scene.num_vertices,
        num_rendered,
        background_d,
        camera.width,
        camera.height,
        RCGS_TPTR(scene.means),
        RCGS_TPTR(scene_2 ? scene.shs_2 : scene.shs),
        RCGS_TPTR(scene.scales),
        RCGS_TPTR(scene.rotations),
        camera.viewmatrix_d(),
        camera.projmatrix_d(),
        camera.campos_d(),
        camera.tan_fovx(),
        camera.tan_fovy(),
        dL_dy,
        RCGS_TPTR(scene.dL_dmean2D),
        RCGS_TPTR(scene.dL_dconic),
        RCGS_TPTR(scene.dL_dopacity),
        RCGS_TPTR(scene.dL_dcolor),
        RCGS_TPTR(scene.dL_dmean3D),
        RCGS_TPTR(scene.dL_dcov3D),
        RCGS_TPTR(scene.dL_dsh),
        RCGS_TPTR(scene.dL_dscale),
        RCGS_TPTR(scene.dL_drot),
        stream);
}

void GSRasterizer::backward( //
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
    cudaStream_t stream)
{
    CHECK_ARG(P > 0);
    CHECK_ARG(R > 0);

    CHECK_CUDA(cudaMemsetAsync(dL_dmean2D, 0, P * 3 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dconic, 0, P * 4 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dopacity, 0, P * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dcolor, 0, P * 3 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dmean3D, 0, P * 3 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dcov3D, 0, P * 6 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dsh, 0, P * 16 * 3 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_dscale, 0, P * 3 * sizeof(float), stream));
    CHECK_CUDA(cudaMemsetAsync(dL_drot, 0, P * 4 * sizeof(float), stream));

    CudaRasterizer::Rasterizer::backward( //
        P,
        3,
        16,
        R,
        background_d,
        W,
        H,
        means3D,
        shs,
        nullptr, // colors_precomp
        scales,
        1.0f,
        rotations,
        nullptr, // cov3D_precomp
        viewmatrix,
        projmatrix,
        campos,
        tan_fovx,
        tan_fovy,
        nullptr, // radii
        reinterpret_cast<char*>(RCGS_TPTR(m_geometry_buffer)),
        reinterpret_cast<char*>(RCGS_TPTR(m_binning_buffer)),
        reinterpret_cast<char*>(RCGS_TPTR(m_image_buffer)),
        dL_dpix,
        dL_dmean2D,
        dL_dconic,
        dL_dopacity,
        dL_dcolor,
        dL_dmean3D,
        dL_dcov3D,
        dL_dsh,
        dL_dscale,
        dL_drot,
        debug,
        stream);
}
