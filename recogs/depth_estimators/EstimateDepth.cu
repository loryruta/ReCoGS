#include "EstimateDepth.h"

#include <fmt/format.h>

#include "App.h"
#include "utils/image/image_downsample.h"
#include "utils/image/image_save.h"
#include "utils/image/image_visit_transform.h"

using namespace gs_train;

namespace
{
/// Given im0 and im1, prepare the input for PCVNet.
/// This involves writing the input images to a clipped, downsampled and 32-padded output.
__global__ void prepare_pcvnet_input_kernel( //
    Image3fCHW im0,
    Image3fCHW im1,
    AABB2i region,
    int num_downsample,
    Image3fCHW out_pcvnet_im0,
    Image3fCHW out_pcvnet_im1)
{
    // Dispatched for every pixel of the downsampled region

    assert(im0.size() == im1.size());
    assert(out_pcvnet_im0.size() == out_pcvnet_im1.size());
    uint32_t ox = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_pcvnet_im0.width || oy >= out_pcvnet_im0.height) return;
    glm::ivec2 rmin = region.min;
    glm::ivec2 rmax = region.max;
    glm::vec3 im0_avg{};
    glm::vec3 im1_avg{};
    float n = 0.f;
    uint32_t ix = (ox << num_downsample) + rmin.x;
    uint32_t iy = (oy << num_downsample) + rmin.y;
    for (; ix < ((ox + 1) << num_downsample) + rmin.x; ++ix) {
        for (; iy < ((oy + 1) << num_downsample) + rmin.y; ++iy) {
            if (ix < region.max.x && iy < region.max.y) {
                im0_avg += im0.value(ix, iy);
                im1_avg += im1.value(ix, iy);
                n += 1.f;
            }
        }
    }
    if (n == 0.f) return; // Possible?
    im0_avg /= n;
    im1_avg /= n;
    out_pcvnet_im0.set_value(ox, oy, im0_avg);
    out_pcvnet_im1.set_value(ox, oy, im1_avg);
}
} // namespace

EstimateDepth::EstimateDepth(App& app, Options options) : m_app(app), m_options(std::move(options))
{
    PCVNetEngine::Options pcvnet_engine_options{};
    pcvnet_engine_options.onnx_filepath = "pcvnet_quant.onnx";
    pcvnet_engine_options.optprofile_min_image_size = glm::ivec2(200, 200);
    pcvnet_engine_options.optprofile_opt_image_size = glm::ivec2(1080, 720);
    pcvnet_engine_options.optprofile_max_image_size = glm::ivec2(1920, 1080); // 1080p
    pcvnet_engine_options.engine_filepath = "pcvnet.engine";
    pcvnet_engine_options.fp16 = true;
    m_pcvnet_engine = std::make_unique<PCVNetEngine>(pcvnet_engine_options);
    m_pcvnet_engine->build_or_load();
}

Image1fCHW EstimateDepth::operator()( //
    const GSCamera& camera,
    Axis axis,
    float b,
    AABB2i region,
    int num_downsample)
{
    int W = camera.width;
    int H = camera.height;
    if (!region.valid()) region = AABB2i(glm::ivec2(0), glm::ivec2(W, H));

    // Render im0
    m_im0.resize(W * H * 3 * sizeof(float));
    m_im1.resize(W * H * 3 * sizeof(float));
    Image3fCHW im0 = Image3fCHW::ref(W, H, m_im0.data_ptr<float>());
    m_app.gs_rasterizer().forward(m_app.background_d(), m_app.scene(), camera, m_im0.data_ptr<float>());
    // Render im1
    m_rview = camera;
    m_rview.position += (axis == Axis::H ? camera.right() : camera.up()) * b;
    m_rview.update();
    Image3fCHW im1 = Image3fCHW::ref(W, H, m_im1.data_ptr<float>());
    m_app.gs_rasterizer().forward(m_app.background_d(), m_app.scene(), m_rview, m_im1.data_ptr<float>());
    if (m_options.debug) {
        image_save_png(im0, fmt::format("estimatedepth-{}im0.png", m_options.image_prefix));
        image_save_png(im1, fmt::format("estimatedepth-{}im1.png", m_options.image_prefix));
    }

    // Clip/downsample/pad the image to PCVNet im0, im1
    int out_region_w = glm::max(div_ceil(region.size().x >> num_downsample, 32) * 32, 32);
    int out_region_h = glm::max(div_ceil(region.size().y >> num_downsample, 32) * 32, 32);
    m_pcvnet_im0.resize(out_region_w * out_region_h * 3 * sizeof(float));
    m_pcvnet_im1.resize(out_region_w * out_region_h * 3 * sizeof(float));
    dim3 num_blocks{};
    num_blocks.x = div_ceil(out_region_w, 16);
    num_blocks.y = div_ceil(out_region_h, 16);
    dim3 block_dim{16, 16};
    Image3fCHW pcvnet_im0 = Image3fCHW::ref(out_region_w, out_region_h, m_pcvnet_im0.data_ptr<float>());
    Image3fCHW pcvnet_im1 = Image3fCHW::ref(out_region_w, out_region_h, m_pcvnet_im1.data_ptr<float>());
    prepare_pcvnet_input_kernel<<<num_blocks, block_dim>>>(im0, im1, region, num_downsample, pcvnet_im0, pcvnet_im1);
    if (m_options.debug) {
        image_save_png(pcvnet_im0, fmt::format("estimatedepth-{}pcvnet-im0.png", m_options.image_prefix));
        image_save_png(pcvnet_im1, fmt::format("estimatedepth-{}pcvnet-im1.png", m_options.image_prefix));
    }

    // Run PCVNet
    // IMPORTANT: must be dispatched on the same CUDA stream
    m_depth.resize(out_region_w * out_region_h * sizeof(float));
    Image1fCHW depth = Image1fCHW::ref(out_region_w, out_region_h, m_depth.data_ptr<float>());
    m_pcvnet_engine->infer(pcvnet_im0, pcvnet_im1, depth);

    // TODO manage vertical stereo matching

    float Rx = (float) camera.width;
    float Sx = camera.tan_fovx();
    image_transform(depth, [Rx, Sx, b] __device__(Image1fCHW & disparity_map, uint32_t x, uint32_t y) {
        // Conversion of the disparity map to depth map:
        // Reference:
        // - Personal notes!
        // - https://johnwlambert.github.io/stereo/
        float disparity = disparity_map.value(x, y).r;
        float depth = (Rx * Sx * abs(b) * 0.5f) / disparity;
        return Image1fCHW::Value{depth};
    });
    return depth;
}
