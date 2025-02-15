#include <cstdint>
#include <filesystem>

#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include "utils/image/image_copy.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_load.h"
#include "utils/image/image_save.h"

using namespace gs_train;

TEST_CASE("image_copy - koala")
{
    std::filesystem::path in_filepath = std::filesystem::path(DATA_DIR) / "koala.jpg";
    CHECK_STATE(std::filesystem::exists(in_filepath));

    using Image3HWC = Image<3, uint8_t, ImageMemoryLayout::HWC>;

    std::unique_ptr<Image3HWC> image;
    image_load(in_filepath.c_str(), image);
    printf("Loaded image of size %dx%d\n", image->width, image->height);

    std::unique_ptr<Image3HWC> copied_image;

    // Exact copy
    copied_image = std::make_unique<Image3HWC>(Image3HWC::malloc(image->width, image->height));
    image_copy(*image, *copied_image);
    image_save_png(*copied_image, "koala_copy_exact.jpg");

    // Region copy
    copied_image = std::make_unique<Image3HWC>(Image3HWC::malloc(500, 500));
    image_fill(*copied_image, {255u, 0, 0});
    AABB2i face_region(glm::ivec2(223, 42), glm::ivec2(460, 216));
    image_copy(*image, face_region, *copied_image, glm::ivec2(0) /* dst_pos */);
    image_save_png(*copied_image, "koala_copy_region.jpg");
}
