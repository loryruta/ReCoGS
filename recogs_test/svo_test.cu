#include <catch.hpp>
#include <catch2/catch_approx.hpp>

#include "svo/svo_utils.h"

using namespace recogs;

TEST_CASE("MortonCode In/Out conversion")
{
    CHECK(from_morton_code(243) == glm::ivec3(5, 7, 2));
    CHECK(from_morton_code(448) == glm::ivec3(4, 4, 4));
    CHECK(from_morton_code(0) == glm::ivec3(0, 0, 0));

    CHECK(to_morton_code(5, 7, 2) == 243);
    CHECK(to_morton_code(4, 4, 4) == 448);
    CHECK(to_morton_code(0, 0, 0) == 0);

    for (int x = 0; x < 15; ++x) {
        for (int y = 0; y < 15; ++y) {
            for (int z = 0; z < 15; ++z) {
                uint64_t morton_code = to_morton_code(x, y, z);
                glm::ivec3 voxel_loc = from_morton_code(morton_code);
                if (x != voxel_loc.x) CHECK(x == voxel_loc.x);
                if (y != voxel_loc.y) CHECK(y == voxel_loc.y);
                if (z != voxel_loc.z) CHECK(z == voxel_loc.z);
            }
        }
    }
}
