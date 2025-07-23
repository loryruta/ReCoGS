#include "FoundationStereo_DepthEstimator.h"

#include "GSRasterizer.h"
#include "utils/image/image_misc.h"
#include "utils/image/image_resize_bilinear.h"
#include "utils/image/image_save.h"

USING_NAMESPACE

namespace
{
__global__ void downsample_input(Image4fHWC src, Image3fCHW dst)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst.width || y >= dst.height) return;
    float sx = float(src.width) / float(dst.width);
    float sy = float(src.height) / float(dst.height);
    float src_x = (float(x) + 0.5f) * sx - 0.5f; // x in src space
    float src_y = (float(y) + 0.5f) * sy - 0.5f; // y in src space
    glm::vec4 v = image_hwc_sample_bilinear(src, src_x, src_y);
    dst.set_value(x, y, v);
}

__global__ void disparity_to_depth_kernel(Image1fHWC disparity_map, float Sx, float b, Image4fHWC out_colordepth)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int W = out_colordepth.width;
    if (x >= W || y >= out_colordepth.height) return;
    float sx = float(disparity_map.width) / float(W);
    float sy = float(disparity_map.height) / float(out_colordepth.height);
    float src_x = (float(x) + 0.5f) * sx - 0.5f; // x in src space
    float src_y = (float(y) + 0.5f) * sy - 0.5f; // y in src space
    float disparity = image_hwc_sample_bilinear(disparity_map, src_x, src_y).r;
    float depth = (Sx * b) / disparity;
    out_colordepth.data_d()[(y * W + x) * 4 + 3] = depth;
}
} // namespace

FoundationStereo_DepthEstimator::FoundationStereo_DepthEstimator(std::unique_ptr<FoundationStereo>&& engine)
    : m_engine(std::move(engine))
{
    m_engine->build_or_load();

    int W = m_engine->width();
    int H = m_engine->height();

    m_left = std::make_unique<Image3fCHW>(Image3fCHW::malloc(W, H));
    m_right = std::make_unique<Image3fCHW>(Image3fCHW::malloc(W, H));
    m_disp = std::make_unique<Image1fCHW>(Image1fCHW::malloc(W, H));
}

void FoundationStereo_DepthEstimator::estimate(const DepthEstimatorParams& params)
{
    Image4fHWC& inout_colordepth = *params.inout_colordepth;
    int W = inout_colordepth.width;
    int H = inout_colordepth.height;
    const Camera& camera = *params.camera;
    GSRasterizer& gs_rasterizer = *params.gs_rasterizer;
    float b = params.b;
    bool debug = params.debug;
    cudaStream_t stream = params.stream;

    int engine_W = m_engine->width();
    int engine_H = m_engine->height();

    // ----------------------------------------------------------------
    // Render im1
    // ----------------------------------------------------------------

    size_t required_im1_size = 4 * W * H;
    if (required_im1_size > m_im1_data.size()) {
        m_im1_data.resize(required_im1_size);
    }
    Image4fHWC im1(W, H, RCGS_TPTR(m_im1_data));
    im1.owned = false;
    m_im1_camera.copy(camera, 0 /* No update */);
    m_im1_camera.position += camera.right() * b;
    m_im1_camera.update(stream);
    gs_rasterizer.forward(params.background_d, *params.scene, false /* scene_2 */, m_im1_camera, im1, stream);

    // ----------------------------------------------------------------
    // Resize (im0, im1) to engine input dim
    // ----------------------------------------------------------------

    {
        dim3 num_blocks(div_ceil(engine_W, 16), div_ceil(engine_H, 30));
        dim3 block_dim(16, 16);
        downsample_input<<<num_blocks, block_dim, 0, stream>>>(inout_colordepth, *m_left);
        downsample_input<<<num_blocks, block_dim, 0, stream>>>(im1, *m_right);
    }

    if (debug) {
        image_save(*m_left, "DepthEstimator_FoundationStereo_left.png", stream);
        image_save(*m_right, "DepthEstimator_FoundationStereo_right.png", stream);
    }

    // ----------------------------------------------------------------
    // Inference
    // ----------------------------------------------------------------

    m_engine->infer(*m_left, *m_right, *m_disp, stream);

    if (debug) {
        Image3fCHW disparity_rgb = image_scalar_to_rgb(*m_disp, 0.1f, stream);
        image_save(disparity_rgb, "DepthEstimator_FoundationStereo_disp.png", stream);
    }

    // ----------------------------------------------------------------
    // Resize disparity to input size and convert disparity to depth
    // ----------------------------------------------------------------

    Image1fHWC disp_hwc = Image1fHWC::ref(m_engine->width(), m_engine->height(), m_disp->data_d());

    {
        dim3 num_blocks(div_ceil(W, 16), div_ceil(H, 16));
        dim3 block_dim(16, 16);
        disparity_to_depth_kernel<<<num_blocks, block_dim, 0, stream>>>(disp_hwc, camera.fx, b, inout_colordepth);
    }
}
