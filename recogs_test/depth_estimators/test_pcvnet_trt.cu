#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include "depth_estimators/PCVNetEngine.h"
#include "utils/Stopwatch.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_load.h"
#include "utils/image/image_save.h"

// Only testing what is strictly required for the tool (engine building and inference).
// Extensive testing with different configurations (and results) are done in Python

using namespace gs_train;

TEST_CASE("PCVNet TensorRT")
{
    const std::string image_pair_name = "Sword1-perfect";
    const std::filesystem::path image_pair_dir =
        std::filesystem::path(DATA_DIR) / "middlebury_14_train" / image_pair_name;

    std::unique_ptr<Image3fCHW> im0;
    std::unique_ptr<Image3fCHW> im1;
    image_load_chw<3, float>(image_pair_dir / "im0_1280.png", im0, CU_STREAM_LEGACY);
    image_load_chw<3, float>(image_pair_dir / "im1_1280.png", im1, CU_STREAM_LEGACY);
    REQUIRE(im0->size() == im1->size());

    PCVNetEngine::Options options{};
    options.onnx_filepath = "pcvnet_quant.onnx";
    options.optprofile_min_image_size = glm::ivec2(200, 200);
    options.optprofile_opt_image_size = glm::ivec2(1080, 720);
    options.optprofile_max_image_size = glm::ivec2(1920, 1080); // 1080p
    options.engine_filepath = "pcvnet.engine";
    PCVNetEngine engine(options);
    engine.build_or_load();

    Image1fCHW disparity_map = Image1fCHW::malloc(im0->width, im0->height);
    // TODO im0/im1 padding!!!

    for (int i = 0; i < 15; ++i) {
        Stopwatch stopwatch;
        engine.infer(*im0, *im1, disparity_map, CU_STREAM_LEGACY);
        printf("%02d. Elapsed time: %s\n", i, stopwatch.elapsed_time_str().c_str());
    }
}
