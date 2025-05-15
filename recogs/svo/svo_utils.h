#pragma once

#include <glm/glm.hpp>

namespace recogs
{
__host__ __device__ uint64_t to_morton_code(int x, int y, int z)
{
    uint64_t morton_code = 0;
    for (int i = 0;; i += 3) {
        morton_code |= (x & 1) << i;
        morton_code |= ((y & 1) << 1) << i;
        morton_code |= ((z & 1) << 2) << i;
        x >>= 1;
        y >>= 1;
        z >>= 1;
        if (!x && !y && !z) break;
    }
    return morton_code;
}

__host__ __device__ glm::ivec3 from_morton_code(uint64_t morton_code)
{
    glm::ivec3 loc{};
    for (int i = 0;; ++i) {
        loc.x |= (int) ((morton_code & 1) << i);
        loc.y |= (int) ((morton_code & 2) >> 1) << i;
        loc.z |= (int) ((morton_code & 4) >> 2) << i;
        morton_code >>= 3;
        if (!morton_code) break;
    }
    return loc;
}
}