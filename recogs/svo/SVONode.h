#pragma once

#include <glm/glm.hpp>
#include <thrust/device_vector.h>

namespace recogs
{
/// A node of the sparse voxel octree.
struct SVONode {
    /// The mask indicating the children of this node
    uint32_t children_mask{}; // 32-bit because we have to do atomics
    /// The address of the first child of this node.
    /// UINT32_MAX is a special value to indicate a marked node.
    union {
        uint32_t first_child_offset = UINT32_MAX;
        uint32_t data;
    };

    __forceinline__ __host__ __device__ bool is_parent() const { return first_child_offset & 0x80000000; }
    __forceinline__ __host__ __device__ bool is_leaf() const { return !is_parent(); }
};

/// A struct representing a SVO in world-space.
struct SVO {
    glm::vec3 min;
    glm::vec3 max;
    thrust::device_vector<SVONode> nodes;
};

} // namespace recogs
