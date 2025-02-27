#include <catch.hpp>
#include <catch2/catch_approx.hpp>
#include <fmt/format.h>

#include "App.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_save.h"

using namespace gs_train;

namespace
{
void init_camera_to_camera0_train_scene(GSCamera& camera, glm::ivec2 resolution)
{
    camera.position = {-3.0090f, -0.1109f, -3.7528f};
    camera.rotation =
        glm::quat_cast(glm::transpose(glm::mat3{{0.8761342012188561f, 0.06925962026449778f, 0.47706599800804744f},
                                                {-0.047474218398951024f, 0.9972110940209488f, -0.05758673934988211f},
                                                {-0.4797239414934442f, 0.02780537650095985f, 0.8769787916452907f}}));
    float aspect = float(resolution.y) / 1090.0f;
    camera.fx = 1159.588073303806f / (1959 * 0.5f) * aspect;
    camera.fy = 1164.6601287484507f / (1090 * 0.5f) * aspect;
    camera.width = resolution.x;
    camera.height = resolution.y;
    camera.update();
}
} // namespace

TEST_CASE("EstimateDepth benchmark")
{
    std::filesystem::path train_ply =
        std::filesystem::path(DATA_DIR) / "scenes" / "train" / "point_cloud" / "iteration_30000" / "point_cloud.ply";
    App::Params app_params{};
    app_params.scene_ply = train_ply;
    std::unique_ptr<App> app = std::make_unique<App>(app_params);

    StereoDepthEstimator::Options options{};
    StereoDepthEstimator stereo(*app, options);

    glm::ivec2 resolution(1080, 720);
    GSCamera camera;
    init_camera_to_camera0_train_scene(camera, resolution);

    const float b = 0.07f;

    Image1fCHW depth = Image1fCHW::malloc(resolution.x, resolution.y);

    // Warm up iterations
    for (int i = 0; i < 10; ++i) {
        stereo.estimate_single_axis(camera, StereoDepthEstimator::Axis::H, b, depth);
    }

    // Measure inference time
    for (int i = 0; i < 15; ++i) {
        Stopwatch stopwatch;
        stereo.estimate_single_axis(camera, StereoDepthEstimator::Axis::H, b, depth);
        printf("[INFO ] Iteration %d; EstimateDepth took %s\n", i, stopwatch.elapsed_time_str().c_str());
    }
}

TEST_CASE("EstimateDepth horizontal/vertical")
{
    std::filesystem::path train_ply =
        std::filesystem::path(DATA_DIR) / "scenes" / "train" / "point_cloud" / "iteration_30000" / "point_cloud.ply";
    App::Params app_params{};
    app_params.scene_ply = train_ply;
    std::unique_ptr<App> app = std::make_unique<App>(app_params);

    StereoDepthEstimator stereo(*app, {});
    stereo.debug = true;

    glm::ivec2 resolution(1080, 720);
    GSCamera camera;
    init_camera_to_camera0_train_scene(camera, resolution);

    const float b = 0.07f;

    printf("Running horizontal stereo matching (see qualitatively the results)...");
    stereo.debug_image_prefix = "h-";
    stereo.estimate_single_axis(camera, StereoDepthEstimator::Axis::H, b);

    printf("Running vertical stereo matching (see qualitatively the results)...");
    stereo.debug_image_prefix = "v-";
    stereo.estimate_single_axis(camera, StereoDepthEstimator::Axis::V, b);
}
