#pragma once

namespace recogs
{
struct SVONode {
    /// The mask indicating the children of this node
    uint32_t children_mask{}; // 32-bit because we have to do atomics
    /// The address of the first child of this node.
    /// UINT32_MAX is a special value to indicate a marked node.
    union {
        uint32_t first_child_offset = UINT32_MAX;
        uint32_t data;
    };
};
} // namespace recogs
