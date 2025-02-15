#include <filesystem>

#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include "utils/image/image_downsample.h"
#include "utils/image/image_load.h"
#include "utils/image/image_save.h"

using namespace gs_train;

TEST_CASE("image_downsample - koala")
{
    std::filesystem::path in_filepath = std::filesystem::path(DATA_DIR) / "koala.jpg";
    CHECK_STATE(std::filesystem::exists(in_filepath));

    using ImageT = Image<3, uint8_t, ImageMemoryLayout::HWC>;

    std::unique_ptr<ImageT> image;
    image_load(in_filepath.c_str(), image);
    printf("Loaded image of size %dx%d\n", image->width, image->height);

    ImageT downsampled_image_x2 = ImageT::malloc(image->width >> 1, image->height >> 1);
    image_downsample(*image, 1, downsampled_image_x2);
    ImageT downsampled_image_x4 = ImageT::malloc(image->width >> 2, image->height >> 2);
    image_downsample(*image, 2, downsampled_image_x4);
    ImageT downsampled_image_x8 = ImageT::malloc(image->width >> 3, image->height >> 3);
    image_downsample(*image, 3, downsampled_image_x8);

    image_save_png(*image, "koala_downsample_x1.png");
    image_save_png(downsampled_image_x2, "koala_downsample_x2.png");
    image_save_png(downsampled_image_x4, "koala_downsample_x4.png");
    image_save_png(downsampled_image_x8, "koala_downsample_x8.png");
}
