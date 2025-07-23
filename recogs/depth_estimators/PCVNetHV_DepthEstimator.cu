#include "PCVNetHV_DepthEstimator.h"

#include <fmt/format.h>

#include "App.h"
#include "GSRasterizer.h"
#include "utils/image/image_downsample.h"
#include "utils/image/image_misc.h"
#include "utils/image/image_resize_bilinear.h"
#include "utils/image/image_save.h"

USING_NAMESPACE

namespace
{
/// Given an input image for PCVNet (either im0 or im1), adapt it to PCVNet requirements.
/// This involves rotating and resizing the input image.
__global__ void adapt_input_kernel(Image4fHWC src, bool rotate90cw, Image3fCHW dst)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst.width || y >= dst.height) return;
    // ----------------------------------------------------------------
    /* Rotate the input image by 90deg clockwise */
    // ----------------------------------------------------------------
    if (rotate90cw) {
        x = y;
        y = dst.width - x - 1;
    }
    // ----------------------------------------------------------------
    /* Bilinear sample from src */
    // ----------------------------------------------------------------
    float sx = float(src.width) / float(dst.width);
    float sy = float(src.height) / float(dst.height);
    if (rotate90cw) {
        swap(sx, sy);
    }
    float src_x = (float(x) + 0.5f) * sx; // x in src space
    float src_y = (float(y) + 0.5f) * sy; // y in src space
    glm::vec<3, float> value = image_hwc_sample_bilinear(src, src_x, src_y);
    dst.set_value(x, y, value);
}

/// Write the unrotated/unpadded disparity map to an output depth map.
/// \param rotated90cw Whether the disparity map is rotated of 90deg CW
__global__ void prepare_pcvnet_output_kernel(Image1fHWC pcvnet_disparity_map,
                                             bool rotated90cw,
                                             float Sx,
                                             float b,
                                             bool overwrite_depth,
                                             Image4fHWC inout_colordepth)
{
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= inout_colordepth.width || y >= inout_colordepth.height) return;
    // ----------------------------------------------------------------
    /* Rotate the input image by 90deg clockwise */
    // ----------------------------------------------------------------
    if (rotated90cw) {
        x = pcvnet_disparity_map.width - y - 1;
        y = x;
    }
    // ----------------------------------------------------------------
    /* Bilinear sample from src */
    // ----------------------------------------------------------------
    float sx = float(pcvnet_disparity_map.width) / float(inout_colordepth.width);
    float sy = float(pcvnet_disparity_map.height) / float(inout_colordepth.height);
    float src_x = (float(x) + 0.5f) * sx; // x in src space
    float src_y = (float(y) + 0.5f) * sy; // y in src space
    float disparity = image_hwc_sample_bilinear(pcvnet_disparity_map, src_x, src_y).r;
    // ----------------------------------------------------------------
    /* Conversion of the disparity map to depthmap */
    // ----------------------------------------------------------------
    // Reference:
    // - Personal notes
    // - https://johnwlambert.github.io/stereo/
    float depth = (Sx * b) / (sx * disparity);
    float* depth_ptr = &inout_colordepth.data_d()[(y * inout_colordepth.width + x) * 4 + 3];
    if (overwrite_depth) {
        *depth_ptr = depth;
    } else {
        *depth_ptr = min(*depth_ptr, depth);
    }
}
} // namespace

void PCVNetHV_DepthEstimatorParams::validate() const { DepthEstimatorParams::validate(); }

PCVNetHV_DepthEstimator::PCVNetHV_DepthEstimator(std::unique_ptr<PCVNetHV>&& engine) : m_engine(std::move(engine))
{
    m_engine->build_or_load();

    const int padded_w = PCVNetHV::k_io_width;
    const int padded_h = PCVNetHV::k_io_height;
    m_pcvnet_im0.resize(padded_w * padded_h * 3);
    m_pcvnet_im1.resize(padded_w * padded_h * 3);
    m_pcvnet_disparity_map.resize(padded_w * padded_h);
}

void PCVNetHV_DepthEstimator::estimate_single_axis(const PCVNetHV_DepthEstimatorParams& params)
{
    const float* background_d = params.background_d;
    const Scene& scene = *params.scene;
    const Camera& camera = *params.camera;
    GSRasterizer& rasterizer = *params.gs_rasterizer;
    cudaStream_t stream = params.stream;
    float b = params.b;
    Image4fHWC& inout_colordepth = *params.inout_colordepth;
    bool overwrite_depth = params.overwrite_depth;
    bool debug = params.debug;
    int axis = params.axis;

    std::string axis_str = axis == PCVNetHV_DepthEstimatorParams::Axis::Axis_Horizontal ? "h" : "v";

    std::lock_guard<std::mutex> lock(m_mutex);

    // Configure App's GSRasterizer
    rasterizer.show_borders = false;

    dim3 num_blocks{};
    dim3 block_dim{};

    int width = camera.width;
    int height = camera.height;

    // ----------------------------------------------------------------
    // Render im0
    // ----------------------------------------------------------------

    m_im0.resize(width * height * 4);
    Image4fHWC im0 = Image4fHWC::ref(width, height, RCGS_TPTR(m_im0));
    rasterizer.forward(background_d, scene, false /* scene_2 */, camera, im0, stream);

    // ----------------------------------------------------------------
    // Render im1
    // ----------------------------------------------------------------

    m_im1.resize(width * height * 4);
    Image4fHWC im1 = Image4fHWC::ref(width, height, RCGS_TPTR(m_im1));
    m_rview = camera.clone();
    m_rview.position += (axis == 0 ? camera.right() : -camera.up()) * b;
    m_rview.update(stream);
    rasterizer.forward(background_d, scene, false /* scene_2 */, m_rview, im1, stream);
    if (debug) {
        image_save(im0, fmt::format("DepthEstimator_PCVNetHV_{}_im0.png", axis_str), stream);
        image_save(im1, fmt::format("DepthEstimator_PCVNetHV_{}_im1.png", axis_str), stream);
    }

    // ----------------------------------------------------------------
    // Downsample im0, im1
    // ----------------------------------------------------------------

    int pcvnet_w = PCVNetHV::k_io_width;
    int pcvnet_h = PCVNetHV::k_io_height;
    assert(pcvnet_w % 32 == 0); // Must be 32-padded
    assert(pcvnet_h % 32 == 0); // Must be 32-padded
    bool rotate90cw = axis == PCVNetHV_DepthEstimatorParams::Axis_Vertical;
    if (rotate90cw) {
        swap(pcvnet_w, pcvnet_h);
    }
    Image3fCHW pcvnet_im0 = Image3fCHW::ref(pcvnet_w, pcvnet_h, RCGS_TPTR(m_pcvnet_im0));
    Image3fCHW pcvnet_im1 = Image3fCHW::ref(pcvnet_w, pcvnet_h, RCGS_TPTR(m_pcvnet_im1));
    num_blocks.x = div_ceil(pcvnet_w, 16);
    num_blocks.y = div_ceil(pcvnet_h, 16);
    block_dim = {16, 16};
    adapt_input_kernel<<<num_blocks, block_dim, 0, stream>>>(im0, rotate90cw, pcvnet_im0);
    adapt_input_kernel<<<num_blocks, block_dim, 0, stream>>>(im1, rotate90cw, pcvnet_im1);
    if (debug) {
        image_save(pcvnet_im0, fmt::format("DepthEstimator_PCVNetHV_{}_adapted_im0.png", axis_str), stream);
        image_save(pcvnet_im1, fmt::format("DepthEstimator_PCVNetHV_{}_adapted_im1.png", axis_str), stream);
    }

    // ----------------------------------------------------------------
    // Infer PCVNet
    // ----------------------------------------------------------------

    Image1fCHW pcvnet_disparity_map = Image1fCHW::ref(pcvnet_w, pcvnet_h, RCGS_TPTR(m_pcvnet_disparity_map));
    Image1fHWC pcvnet_disparity_map_hwc = Image1fHWC::ref(pcvnet_w, pcvnet_h, RCGS_TPTR(m_pcvnet_disparity_map));
    m_engine->infer(pcvnet_im0, pcvnet_im1, pcvnet_disparity_map, stream);
    if (debug) {
        Image3fCHW dispmap_rgb = image_scalar_to_rgb(pcvnet_disparity_map, 0.01f, stream);
        image_save(dispmap_rgb, fmt::format("DepthEstimator_PCVNetHV_{}_adapted_disparity.png", axis_str), stream);
    }

    // Undo rotation and padding of disparity map and compute depth map in a single kernel call
    num_blocks.x = div_ceil(width, 16);
    num_blocks.y = div_ceil(height, 16);
    block_dim = {16, 16};
    if (axis == PCVNetHV_DepthEstimatorParams::Axis_Horizontal) {
        prepare_pcvnet_output_kernel<<<num_blocks, block_dim, 0, stream>>>(
            pcvnet_disparity_map_hwc, false /* rotated90cw */, camera.fx, b, overwrite_depth, inout_colordepth);
    } else if (axis == PCVNetHV_DepthEstimatorParams::Axis_Vertical) {
        prepare_pcvnet_output_kernel<<<num_blocks, block_dim, 0, stream>>>(
            pcvnet_disparity_map_hwc, true /* rotated90cw */, camera.fy, b, overwrite_depth, inout_colordepth);
    } else {
        throw IllegalArgumentException("Invalid axis");
    }

    if (debug) {
        Image4fHWC depth_rgb = image_depth_to_rgb(inout_colordepth, 0.1f, stream);
        image_save(depth_rgb, fmt::format("DepthEstimator_PCVNetHV_{}_depth.png", axis_str), stream);
    }
}

void PCVNetHV_DepthEstimator::estimate_hv(PCVNetHV_DepthEstimatorParams& params)
{
    params.validate();

    params.axis = PCVNetHV_DepthEstimatorParams::Axis_Horizontal;
    estimate_single_axis(params);
    params.axis = PCVNetHV_DepthEstimatorParams::Axis_Vertical;
    params.overwrite_depth = false;
    estimate_single_axis(params);
}

void PCVNetHV_DepthEstimator::estimate(const DepthEstimatorParams& params)
{
    params.validate();

    PCVNetHV_DepthEstimatorParams params_{};
    params_.background_d = params.background_d;
    params_.scene = params.scene;
    params_.gs_rasterizer = params.gs_rasterizer;
    params_.inout_colordepth = params.inout_colordepth;
    params_.camera = params.camera;
    params_.b = 0.07f;
    params_.overwrite_depth = params.overwrite_depth;
    params_.debug = params.debug;
    // params_.axis = PCVNetHV_DepthEstimatorParams::Axis_Both;
    estimate_hv(params_);
}
