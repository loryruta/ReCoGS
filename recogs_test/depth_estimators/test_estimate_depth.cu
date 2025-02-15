#include <catch.hpp>
#include <catch2/catch_approx.hpp>
#include <fmt/format.h>

#include "App.h"
#include "depth_estimators/EstimateDepth.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_save.h"

using namespace gs_train;

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
    camera.width = 1080;
    camera.height = 720;
    camera.update();
}

TEST_CASE("EstimateDepth perf")
{
    std::filesystem::path train_ply =
        std::filesystem::path(DATA_DIR) / "scenes" / "train" / "point_cloud" / "iteration_30000" / "point_cloud.ply";
    App::Params app_params{};
    app_params.scene_ply = train_ply;
    std::unique_ptr<App> app = std::make_unique<App>(app_params);

    EstimateDepth::Options options{};
    options.debug = false;
    EstimateDepth estimate_depth(*app, options);

    glm::ivec2 resolution(1080, 720);
    GSCamera camera;
    init_camera_to_camera0_train_scene(camera, resolution);

    for (int i = 0; i < 4; ++i) {
        const float b = 0.07f;
        Stopwatch stopwatch;
        Image1fCHW depth = estimate_depth(camera, EstimateDepth::Axis::H, b, AABB2i{} /* region */, 0 /* downsample */);
        printf("[INFO ] -------------------------------- Iteration %d; EstimateDepth took %s\n",
               i,
               stopwatch.elapsed_time_str().c_str());

        if (i == 3) { // Last
            Image3fCHW depth_rgb = Image3fCHW::malloc(depth.width, depth.height);
            image_visit(depth, [depth_rgb] __device__(Image1fCHW & depth, int x, int y) mutable {
                const float k_log_base = 10.0f;
                float d = depth.value(x, y).r;
                // Depth to log scale
                d = log2(d + 1.0f) / log2(k_log_base);
                d = min(1.0f, d);
                depth_rgb.set_value(x, y, glm::vec3(d));
                return 0; // TODO temporary until I find a solution
            });
            image_save_png(depth_rgb, "estimatedepth-perf-depth.png");
        }
    }
}

TEST_CASE("EstimateDepth test region/downsample")
{
    // Initialize the app once
    static std::unique_ptr<App> s_app{};
    if (!s_app) {
        std::filesystem::path train_ply = std::filesystem::path(DATA_DIR) / "scenes" / "train" / "point_cloud" /
                                          "iteration_30000" / "point_cloud.ply";
        App::Params app_params{};
        app_params.scene_ply = train_ply;
        s_app = std::make_unique<App>(app_params);
    }

    glm::ivec2 resolution(1080, 720);
    GSCamera camera;
    init_camera_to_camera0_train_scene(camera, resolution);

    AABB2i region = GENERATE(AABB2i{}, AABB2i({234, 262}, {618, 498}));
    int num_downsample = GENERATE(0, 1, 2);

    EstimateDepth::Options options{};
    options.debug = true;
    options.image_prefix = fmt::format("{}-x{}-", region.valid() ? "Y" : "N", 1 << num_downsample);
    EstimateDepth estimate_depth(*s_app, options);

    const float b = 0.07f;
    Image1fCHW depth = estimate_depth(camera, EstimateDepth::Axis::H, b, region, num_downsample);
    Image3fCHW depth_rgb = Image3fCHW::malloc(depth.width, depth.height);
    image_visit(depth, [depth_rgb] __device__(Image1fCHW & depth, int x, int y) mutable {
        const float k_log_base = 10.0f;
        float d = depth.value(x, y).r;
        // Depth to log scale
        d = log2(d + 1.0f) / log2(k_log_base);
        d = min(1.0f, d);
        depth_rgb.set_value(x, y, glm::vec3(d));
        return 0; // TODO temporary until I find a solution
    });
    std::filesystem::path depth_filepath =
        fmt::format("estimatedepth-{}-x{}-depth.png", region.valid() ? "Y" : "N", 1 << num_downsample);
    image_save_png(depth_rgb, depth_filepath);
}
