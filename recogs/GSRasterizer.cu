#include "GSRasterizer.h"

#include "rasterizer/rasterizer.h"

using namespace gs_train;

namespace
{
template <typename T>
std::function<char*(size_t N)> resize_functional(DeviceBuffer& buffer)
{
    return [&buffer](size_t N) -> char* {
        CHECK_STATE(buffer.resize(N * sizeof(T)));
        return buffer.data_ptr<char>();
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
    float* out_colorbuffer,
    float* out_depthbuffer)
{
    int num_rendered = CudaRasterizer::Rasterizer::forward( //
        resize_functional<char>(m_geometry_buffer),
        resize_functional<char>(m_binning_buffer),
        resize_functional<char>(m_image_buffer),
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
        out_colorbuffer,
        out_depthbuffer,
        nullptr, // radii
        false    // debug
    );
    return num_rendered;
}

int GSRasterizer::forward(const float* background_d,
                          const Scene& scene,
                          const GSCamera& camera,
                          Image3fCHW& out_colorbuffer)
{
    return forward( //
        camera.width,
        camera.height,
        background_d,
        scene.num_vertices,
        scene.means.data_ptr<float>(),
        scene.shs.data_ptr<float>(),
        scene.opacities.data_ptr<float>(),
        scene.scales.data_ptr<float>(),
        scene.rotations.data_ptr<float>(),
        camera.viewmatrix_d(),
        camera.projmatrix_d(),
        camera.campos_d(),
        camera.tan_fovx(),
        camera.tan_fovy(),
        out_colorbuffer.data_d(),
        nullptr);
}

int GSRasterizer::forward(const float* background_d,
                          const Scene& scene,
                          const GSCamera& camera,
                          Image3fCHW& out_colorbuffer,
                          Image1fCHW& out_depthbuffer)
{
    return forward( //
        camera.width,
        camera.height,
        background_d,
        scene.num_vertices,
        scene.means.data_ptr<float>(),
        scene.shs.data_ptr<float>(),
        scene.opacities.data_ptr<float>(),
        scene.scales.data_ptr<float>(),
        scene.rotations.data_ptr<float>(),
        camera.viewmatrix_d(),
        camera.projmatrix_d(),
        camera.campos_d(),
        camera.tan_fovx(),
        camera.tan_fovy(),
        out_colorbuffer.data_d(),
        out_depthbuffer.data_d());
}
