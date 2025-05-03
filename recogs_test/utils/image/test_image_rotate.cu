#include <filesystem>

#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include "utils/image/image_load.h"
#include "utils/image/image_rotate.h"
#include "utils/image/image_save.h"

using namespace recogs;

TEST_CASE("image_rotate_90_clockwise - qualitative")
{
    std::filesystem::path in_filepath = std::filesystem::path(DATA_DIR) / "koala.jpg";
    CHECK_STATE(std::filesystem::exists(in_filepath));
    std::filesystem::path out_filepath = "./koala_rot90deg.jpg";

    using ImageT = Image<3, uint8_t, ImageMemoryLayout::HWC>;

    std::unique_ptr<ImageT> image;
    image_load(in_filepath.c_str(), image, CU_STREAM_LEGACY);
    printf("Loaded image of size %dx%d\n", image->width, image->height);

    ImageT rotated_image = ImageT::malloc(image->height, image->width);
    image_rotate_90_clockwise(*image, rotated_image);

    image_save_png(*image, "koala_original.jpg");
    image_save_png(rotated_image, out_filepath.c_str());
}
