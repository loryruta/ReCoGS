#pragma once

#include <bit>

#include <glm/glm.hpp>

// Reference:
// https://gist.github.com/mpottinger/54d99732d4831d8137d178b4a6007d1a

namespace recogs
{
inline glm::uvec4 murmurHash41(uint src)
{
    const uint M = 0x5bd1e995u;
    glm::uvec4 h = glm::uvec4(1190494759u, 2147483647u, 3559788179u, 179424673u);
    src *= M;
    src ^= src >> 24u;
    src *= M;
    h *= M;
    h = h ^ src;
    h = h ^ (h >> 13u);
    h *= M;
    h = h ^ (h >> 15u);
    return h;
}

inline glm::vec4 hash41(uint src)
{
    glm::uvec4 h = murmurHash41(src);
    return glm::uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0f;
}

inline glm::vec4 hash41(float src) { return hash41(glm::floatBitsToUint(src)); }
} // namespace recogs
