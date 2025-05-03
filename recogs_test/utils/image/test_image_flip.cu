#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include "utils/image/image_flip.h"
#include "utils/cuda_utils.h"

using namespace recogs;

TEST_CASE("Image flip X")
{
    SECTION("Odd width")
    {
        Image1u image = Image1u::malloc(3, 2);
        dispatch_single_thread([=] __device__() mutable {
            image.set_value(0, 0, Image1u::Value{2});
            image.set_value(1, 0, Image1u::Value{7});
            image.set_value(2, 0, Image1u::Value{5});
            image.set_value(0, 1, Image1u::Value{1});
            image.set_value(1, 1, Image1u::Value{7});
            image.set_value(2, 1, Image1u::Value{8});
        });

        std::vector<uint8_t> image_data;

        image.to_host(image_data);
        CHECK(image_data[0] == 2); // 1st row
        CHECK(image_data[1] == 7);
        CHECK(image_data[2] == 5);
        CHECK(image_data[3] == 1); // 2nd row
        CHECK(image_data[4] == 7);
        CHECK(image_data[5] == 8);

        image_flip_x(image);

        image.to_host(image_data);
        CHECK(image_data[0] == 5); // 1st row
        CHECK(image_data[1] == 7);
        CHECK(image_data[2] == 2);
        CHECK(image_data[3] == 8); // 2nd row
        CHECK(image_data[4] == 7);
        CHECK(image_data[5] == 1);
    }

    SECTION("Even width")
    {
        Image1u image = Image1u::malloc(2, 2);
        dispatch_single_thread([=] __device__() mutable {
            image.set_value(0, 0, Image1u::Value{33});
            image.set_value(1, 0, Image1u::Value{44});
            image.set_value(0, 1, Image1u::Value{33});
            image.set_value(1, 1, Image1u::Value{44});
        });

        std::vector<uint8_t> image_data;

        image.to_host(image_data);
        CHECK(image_data[0] == 33); // 1st row
        CHECK(image_data[1] == 44);
        CHECK(image_data[2] == 33); // 2nd row
        CHECK(image_data[3] == 44);

        image_flip_x(image);

        image.to_host(image_data);
        CHECK(image_data[0] == 44); // 1st row
        CHECK(image_data[1] == 33);
        CHECK(image_data[2] == 44); // 2nd row
        CHECK(image_data[3] == 33);
    }
}

// TODO image_flipy not tested
