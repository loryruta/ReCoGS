#pragma once

#include <cstdint>

#include "SVONode.h"

namespace recogs
{
class SVOBuilder
{
public:
    explicit SVOBuilder() = default;
    ~SVOBuilder() = default;

    void build(const uint64_t* points, size_t num_points, int resolution, cudaStream_t stream);
};
} // namespace recogs
