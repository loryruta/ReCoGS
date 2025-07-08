#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include <glm/glm.hpp>

#define RCGS_LIKELY(x) __builtin_expect(!!(x), 1)
#define RCGS_UNLIKELY(x) __builtin_expect(!!(x), 0)

namespace recogs
{
template <typename INT>
#ifdef __CUDACC__
__forceinline__ __host__ __device__ INT div_ceil(INT a, INT b)
#else
INT div_ceil(INT a, INT b)
#endif
{
    if (a == 0) return 0;
    return 1 + ((a - 1) / b); // if a != 0
}

inline size_t get_filesize(const std::filesystem::path& filepath)
{
    FILE* f = fopen(filepath.c_str(), "r");
    fseek(f, 0L, SEEK_END);
    return ftell(f);
}

inline uint32_t round_to_next_power_of_2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

struct ivec3_hash {
    std::size_t operator()(const glm::ivec3& value) const
    {
        std::size_t h1 = std::hash<int>()(value.x);
        std::size_t h2 = std::hash<int>()(value.y);
        std::size_t h3 = std::hash<int>()(value.z);
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
} // namespace recogs
