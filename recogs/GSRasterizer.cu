#include "GSRasterizer.h"

#include "rasterizer/rasterizer.h"

using namespace gs_train;

namespace
{
std::function<char*(size_t N)>
resize_functional(thrust::device_vector<char>& buffer, const char* name, size_t alignment)
{
    return [&buffer, name, alignment](size_t num_bytes) -> char* {
        if (num_bytes >= buffer.size()) {
            size_t new_size = div_ceil(num_bytes, alignment) * alignment;
            printf("[DEBUG] [GSRasterizer] Resizing %s to %zu bytes\n", name, new_size);
            buffer.resize(new_size);
        }
        return thrust::raw_pointer_cast(buffer.data());
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
