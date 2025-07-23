#include "DepthEstimator.h"

#include "FoundationStereo_DepthEstimator.h"
#include "PCVNetHV_DepthEstimator.h"
#include "utils/Stopwatch.h"

USING_NAMESPACE

namespace
{
std::mutex g_mutex;
std::unique_ptr<DepthEstimator> g_depth_estimator;
} // namespace

void DepthEstimatorParams::validate() const
{
    CHECK_ARG(background_d);
    CHECK_ARG(scene);
    CHECK_ARG(gs_rasterizer);
    CHECK_ARG(inout_colordepth);
    CHECK_ARG(camera);
}

DepthEstimator& DepthEstimator::get()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_depth_estimator) {
        g_depth_estimator = std::make_unique<Gaussians_DepthEstimator>();
    }
    return *g_depth_estimator;
}

void DepthEstimator::set(DepthEstimatorType type)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_depth_estimator) {
        printf("[DEBUG] [DepthEstimator] Destroying %s\n", DepthEstimatorType_name(g_depth_estimator->type()));
        g_depth_estimator.reset();
    }

    if (type == DepthEstimatorType::Gaussians) {
        g_depth_estimator = std::make_unique<Gaussians_DepthEstimator>();
    } else if (type == DepthEstimatorType::PCVNetHV) {
        printf("[INFO ] [DepthEstimator] Loading PCVNetHV model...\n");
        Stopwatch stopwatch{};
        PCVNetHV_Options options{};
        options.onnx_filepath = "pcvnet.onnx";
        options.engine_filepath = "pcvnet.engine";
        auto pcvnet_hv = std::make_unique<PCVNetHV>(options);
        pcvnet_hv->build_or_load();
        printf("[INFO ] [DepthEstimator] PCVNetHV loaded in %s\n", stopwatch.elapsed_time_str().c_str());
        //
        g_depth_estimator = std::make_unique<PCVNetHV_DepthEstimator>(std::move(pcvnet_hv));
    } else {
        printf("[INFO ] [DepthEstimator] Loading Foundation Stereo model...\n");
        Stopwatch stopwatch{};
        FoundationStereo_Options options{};
        options.onnx_filepath = "foundation_stereo_736_1088.onnx";
        options.engine_filepath = "foundation_stereo_736_1088.engine";
        auto foundation_stereo = std::make_unique<FoundationStereo>(options);
        foundation_stereo->build_or_load();
        printf("[INFO ] [DepthEstimator] Foundation Stereo loaded in %s\n", stopwatch.elapsed_time_str().c_str());
        //
        g_depth_estimator = std::make_unique<FoundationStereo_DepthEstimator>(std::move(foundation_stereo));
    }
}
