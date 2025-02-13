#include "gs_func.h"

#include "rasterizer/rasterizer.h"

using namespace gs_train;

namespace
{
template <typename T>
std::function<char*(size_t N)> resize_functional(Buffer& buffer)
{
    return [&buffer](size_t N) -> char* { CHECK_STATE(buffer.resize(N * sizeof(T))); };
}
} // namespace

int GSFunc::forward(int W,
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
                    float* out_colorbuffer)
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
        nullptr, // radii
        false    // debug
    );
    return num_rendered;
}

GSFunc
