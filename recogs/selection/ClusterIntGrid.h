#pragma once

#include <cstdint>
#include <filesystem>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "utils/exceptions.h"

namespace recogs
{
struct ClusterIntGrid_AABB {
    glm::ivec2 min = glm::ivec2(INT32_MAX); ///< The AABB min point (inclusive)
    glm::ivec2 max = glm::ivec2(INT32_MIN); ///< The AABB max point (inclusive)
    int v = -1;                             ///< Value for which the AABB was created
    int nv = 0;                             ///< Number of values `v` contained in the AABB

    [[nodiscard]] glm::ivec2 size() const { return max - min + 1; }

    [[nodiscard]] bool valid() const { return min.x <= max.x && min.y <= max.y; }

    [[nodiscard]] float value_size_ratio() const
    {
        glm::ivec2 size_ = size();
        return float(nv) / float(size_.x * size_.y);
    }

    [[nodiscard]] bool overlaps(const ClusterIntGrid_AABB& other) const
    {
        return min.x <= other.max.x && max.x >= other.min.x && min.y <= other.max.y && max.y >= other.min.y;
    }

    void merge(const ClusterIntGrid_AABB& other)
    {
        CHECK_ARG(other.valid());
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
        // v not merged
        if (v == other.v) {
            nv += other.nv;
        }
    }
};

/// Cluster a 2D grid of integers into a set of AABB.
class ClusterIntGrid
{
private:
    const int m_width;
    const int m_height;
    const int* m_values;

    std::vector<ClusterIntGrid_AABB> m_aabbs;
    std::vector<int> m_aabb_id_map;

public:
    explicit ClusterIntGrid(int width, int height, const int* data);
    ~ClusterIntGrid() = default;

    std::vector<ClusterIntGrid_AABB> cluster();

private:
    bool expand_aabb(ClusterIntGrid_AABB& aabb, int aabb_id);

    void check_overlapping_aabbs() ;
    void compute_nv(ClusterIntGrid_AABB& aabb);

    void save_int_image(const std::filesystem::path& filepath, int W, int H, const int* data);
    void save_values_image(const std::filesystem::path& filepath);
    void save_aabb_id_map_image(const std::filesystem::path& filepath);
};
} // namespace recogs
