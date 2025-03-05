#include "StereoDepthEstimator.h"

#include <fmt/format.h>

#include "App.h"
#include "utils/image/image_downsample.h"
#include "utils/image/image_misc.h"
#include "utils/image/image_save.h"

using namespace gs_train;

namespace
{
/// Given im0 and im1, prepare the input for PCVNet.
/// This involves writing the input images rotated and padded to the output.
__global__ void prepare_pcvnet_input_kernel( //
    Image3fCHW im0,
    Image3fCHW im1,
    bool rotate90cw,
    Image3fCHW out_pcvnet_im0,
    Image3fCHW out_pcvnet_im1)
{
    assert(im0.size() == im1.size());
    assert(out_pcvnet_im0.size() == out_pcvnet_im1.size());
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_pcvnet_im0.width || y >= out_pcvnet_im0.height) return;

    uint32_t sx;
    uint32_t sy;
    if (rotate90cw) {
        sx = y;
        sy = out_pcvnet_im0.width - x - 1;
    } else {
        sx = x;
        sy = y;
    }
    if (sx >= im0.width || sy >= im0.height) {
        out_pcvnet_im0.set_value(x, y, Image3fCHW::Value{0}); // Zero padding
        out_pcvnet_im1.set_value(x, y, Image3fCHW::Value{0});
    } else {
        out_pcvnet_im0.set_value(x, y, im0.value(sx, sy));
        out_pcvnet_im1.set_value(x, y, im1.value(sx, sy));
    }
}

/// Write the unrotated/unpadded disparity map to an output depth map.
/// \param rotated90cw Whether the disparity map is rotated of 90deg CW
__global__ void prepare_pcvnet_output_kernel( //
    Image1fCHW pcvnet_disparity_map,
    bool rotated90cw,
    float Sx,
    float b,
    Image1fCHW inout_depth)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inout_depth.width || y >= inout_depth.height) return;

    Image1fCHW::Value disparity;
    if (rotated90cw) {
        uint32_t sx = pcvnet_disparity_map.width - y - 1;
        uint32_t sy = x;
        disparity = pcvnet_disparity_map.value(sx, sy);
    } else {
        disparity = pcvnet_disparity_map.value(x, y);
    }
    // Conversion of the disparity map to depth map:
    // Reference:
    // - Personal notes
    // - https://johnwlambert.github.io/stereo/
    float depth = (Sx * b) / disparity.r;
    float old_depth = inout_depth.value(x, y).r;
    float new_depth = min(depth, old_depth); // Aggregation used for horizontal/vertical
    inout_depth.set_value(x, y, Image1fCHW::Value{new_depth});
}
} // namespace

StereoDepthEstimator::StereoDepthEstimator(App& app, Options options)
    : m_app(app), m_options(std::move(options)), debug(options.debug), debug_image_prefix(options.debug_image_prefix)
{
    PCVNetEngine::Options pcvnet_engine_options{};
    pcvnet_engine_options.onnx_filepath = "pcvnet.onnx";
    pcvnet_engine_options.engine_filepath = "pcvnet.engine";
    m_pcvnet_engine = std::make_unique<PCVNetEngine>(pcvnet_engine_options);
    m_pcvnet_engine->build_or_load();
}

void StereoDepthEstimator::estimate_single_axis(
    const GSCamera& camera, Axis axis, float b, Image1fCHW& inout_depth, cudaStream_t stream)
{
    dim3 num_blocks{};
    dim3 block_dim{};

    int width = camera.width;
    int height = camera.height;

    // Render im0
    m_im0.resize(width * height * 3 * sizeof(float));
    m_im1.resize(width * height * 3 * sizeof(float));
    Image3fCHW im0 = Image3fCHW::ref(width, height, m_im0.data_ptr<float>());
    m_app.gs_rasterizer().forward(m_app.background_d(), m_app.scene(), false /* scene_2 */, camera, im0, stream);

    // Render im1
    GSCamera rview = camera;
    rview.position += (axis == Axis::H ? camera.right() : -camera.up()) * b;
    rview.update(stream);
    Image3fCHW im1 = Image3fCHW::ref(width, height, m_im1.data_ptr<float>());
    m_app.gs_rasterizer().forward(m_app.background_d(), m_app.scene(), false /* scene_2 */, rview, im1, stream);
    if (debug) {
        image_save_png(im0, fmt::format("estimatedepth-{}im0.png", debug_image_prefix));
        image_save_png(im1, fmt::format("estimatedepth-{}im1.png", debug_image_prefix));
    }

    // Pad im0, im1
    int padded_width = PCVNetEngine::k_io_width;
    int padded_height = PCVNetEngine::k_io_height;
    bool rotate90cw = axis == Axis::V;
    if (rotate90cw) {
        swap(padded_width, padded_height);
    }
    m_pcvnet_im0.resize(padded_width * padded_height * 3 * sizeof(float));
    m_pcvnet_im1.resize(padded_width * padded_height * 3 * sizeof(float));
    Image3fCHW pcvnet_im0 = Image3fCHW::ref(padded_width, padded_height, m_pcvnet_im0.data_ptr<float>());
    Image3fCHW pcvnet_im1 = Image3fCHW::ref(padded_width, padded_height, m_pcvnet_im1.data_ptr<float>());
    num_blocks.x = padded_width >> 4;
    num_blocks.y = padded_height >> 4;
    block_dim = {16, 16};
    prepare_pcvnet_input_kernel<<<num_blocks, block_dim, 0, stream>>>(im0, im1, rotate90cw, pcvnet_im0, pcvnet_im1);
    if (debug) {
        image_save_png(pcvnet_im0, fmt::format("estimatedepth-{}pcvnet-im0.png", debug_image_prefix));
        image_save_png(pcvnet_im1, fmt::format("estimatedepth-{}pcvnet-im1.png", debug_image_prefix));
    }

    // Run PCVNet inference
    // IMPORTANT: must be dispatched on the same CUDA stream
    m_pcvnet_disparity_map.resize(padded_width * padded_height * sizeof(float));
    Image1fCHW pcvnet_disparity_map =
        Image1fCHW::ref(padded_width, padded_height, m_pcvnet_disparity_map.data_ptr<float>());
    m_pcvnet_engine->infer(pcvnet_im0, pcvnet_im1, pcvnet_disparity_map, stream);
    if (debug) {
        Image3fCHW disparity_map_rgb = image_depthbuffer_to_rgb(pcvnet_disparity_map, stream);
        image_save_png(disparity_map_rgb, fmt::format("estimatedepth-{}pcvnet-disparity-map.png", debug_image_prefix));
    }

    // Undo rotation and padding of disparity map and compute depth map in a single kernel call
    num_blocks.x = div_ceil(width, 16);
    num_blocks.y = div_ceil(height, 16);
    block_dim = {16, 16};
    if (axis == Axis::H) {
        prepare_pcvnet_output_kernel<<<num_blocks, block_dim, 0, stream>>>(
            pcvnet_disparity_map, false /* rotated90cw */, camera.fx, b, inout_depth);
    } else {
        prepare_pcvnet_output_kernel<<<num_blocks, block_dim, 0, stream>>>(
            pcvnet_disparity_map, true /* rotated90cw */, camera.fy, b, inout_depth);
    }
    if (debug) {
        Image3fCHW depth_rgb = image_depthbuffer_to_rgb(inout_depth, stream);
        image_save_png(depth_rgb, fmt::format("estimatedepth-{}depth.png", debug_image_prefix));
    }
}

void StereoDepthEstimator::estimate_hv(const GSCamera& camera, float b, Image1fCHW& inout_depth, cudaStream_t stream)
{
    estimate_single_axis(camera, Axis::H, b, inout_depth, stream);
    estimate_single_axis(camera, Axis::V, b, inout_depth, stream);
}
