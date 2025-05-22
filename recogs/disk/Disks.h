#pragma once

#include <glm/glm.hpp>
#include <thrust/device_vector.h>

namespace recogs
{
struct Disks {
    int count = 0;
    thrust::device_vector<glm::vec3> positions;
    thrust::device_vector<glm::vec2> scales;
    thrust::device_vector<glm::vec4> rotations;

    Disks() = default;
    Disks(const Disks&) = delete;
    Disks(Disks&&) noexcept = default;

    [[nodiscard]] bool is_valid() const
    {
        return positions.size() == count && scales.size() == count && rotations.size() == count;
    }
};
} // namespace recogs
