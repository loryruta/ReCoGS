#include "ClusterIntGrid.h"

#include <algorithm>
#include <cassert>
#include <random>

#include "utils/exceptions.h"
#include "utils/murmur_hash.h"
#include "utils/stb_image_write.h"

using namespace recogs;

ClusterIntGrid::ClusterIntGrid(int width, int height, const int* values)
    : m_width(width), m_height(height), m_values(values)
{
}

std::vector<ClusterIntGrid_AABB> ClusterIntGrid::cluster()
{
    save_values_image("ClusterIntGrid_Values.png"); // DEBUG

    m_aabb_id_map.resize(m_width * m_height, -1 /* Unassigned */);

    // ----------------------------------------------------------------
    /* Prepare random access order */
    // ----------------------------------------------------------------

    std::vector<glm::ivec2> locations{};
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            int v = m_values[y * m_width + x];
            if (v != -1) {
                locations.emplace_back(x, y);
            }
        }
    }
    {
        auto rng = std::default_random_engine{};
        std::shuffle(locations.begin(), locations.end(), rng);
    }

    // ----------------------------------------------------------------
    /* 1st expansion round (1/10 sampled locations) */
    // ----------------------------------------------------------------

    for (glm::ivec2 location : locations) {
        if (m_aabb_id_map[location.y * m_width + location.x] != -1) continue; // Already assigned
        int pix = location.y * m_width + location.x;
        assert(m_aabb_id_map[pix] == -1);
        int aabb_id = (int) m_aabbs.size();
        ClusterIntGrid_AABB& aabb = m_aabbs.emplace_back();
        aabb.min = location;
        aabb.max = location;
        aabb.v = m_values[pix];
        aabb.nv = 1;
        m_aabb_id_map[pix] = aabb_id;
        while (expand_aabb(aabb, aabb_id)) ;
    }

    save_aabb_id_map_image("ClusterIntGrid_AABBIdMap_Expand.png"); // DEBUG

    // DEBUG
    printf("[DEBUG] [ClusterIntGrid] Checking overlapping AABBs (count: %zu)\n", m_aabbs.size());
    check_overlapping_aabbs();

    return m_aabbs;
}

bool ClusterIntGrid::expand_aabb(ClusterIntGrid_AABB& aabb, int aabb_id)
{
    int nv_l; // Count of local nv (Number of matching Values)
    int nv_r;
    int nv_t;
    int nv_b;

    glm::ivec2 aabb_size = aabb.size();

    { /* Expand left */
        nv_l = 0;
        for (int y = aabb.min.y; y <= aabb.max.y; ++y) {
            int x = aabb.min.x - 1;
            if (x < 0) {
                nv_l = -1;
                break;
            }
            int a = m_aabb_id_map[y * m_width + x];
            if (a != -1) {
                nv_l = -1;
                break;
            }
            int v = m_values[y * m_width + x];
            if (v == aabb.v) ++nv_l;
        }
    }
    { /* Expand top */
        nv_t = 0.f;
        for (int x = aabb.min.x; x <= aabb.max.x; ++x) {
            int y = aabb.min.y - 1;
            if (y < 0) {
                nv_t = -1;
                break;
            }
            int a = m_aabb_id_map[y * m_width + x];
            if (a != -1) {
                nv_t = -1;
                break;
            }
            int v = m_values[y * m_width + x];
            if (v == aabb.v) ++nv_t;
        }
    }
    { /* Expand right */
        nv_r = 0.f;
        for (int y = aabb.min.y; y <= aabb.max.y; ++y) {
            int x = aabb.max.x + 1;
            if (x >= m_width) {
                nv_r = -1;
                break;
            }
            int a = m_aabb_id_map[y * m_width + x];
            if (a != -1) {
                nv_r = -1;
                break;
            }
            int v = m_values[y * m_width + x];
            if (v == aabb.v) ++nv_r;
        }
    }
    { /* Expand bottom */
        nv_b = 0.f;
        for (int x = aabb.min.x; x <= aabb.max.x; ++x) {
            int y = aabb.max.y + 1;
            if (y >= m_height) {
                nv_b = -1;
                break;
            }
            int a = m_aabb_id_map[y * m_width + x];
            if (a != -1) {
                nv_b = -1;
                break;
            }
            int v = m_values[y * m_width + x];
            if (v == aabb.v) ++nv_b;
        }
    }

    // ----------------------------------------------------------------
    /* Evaluate the best score (matching values over AABB area) */
    // ----------------------------------------------------------------

    float score_l = nv_l == -1 ? -INFINITY : float(nv_l + aabb.nv) / float((aabb_size.x + 1) * aabb_size.y);
    float score_r = nv_r == -1 ? -INFINITY : float(nv_r + aabb.nv) / float((aabb_size.x + 1) * aabb_size.y);
    float score_t = nv_t == -1 ? -INFINITY : float(nv_t + aabb.nv) / float(aabb_size.x * (aabb_size.y + 1));
    float score_b = nv_b == -1 ? -INFINITY : float(nv_b + aabb.nv) / float(aabb_size.x * (aabb_size.y + 1));
    float best_score = glm::max(score_l, glm::max(score_r, glm::max(score_t, score_b)));

    if (best_score == -INFINITY) return false; // No expansion was possible
    if (best_score < 0.9f) return false;       // Expansion not worth (would cause less than 90% matching values)

    // clang-format off
        if      (best_score == score_l) --aabb.min.x, aabb.nv += nv_l; // Left
        else if (best_score == score_t) --aabb.min.y, aabb.nv += nv_t; // Top
        else if (best_score == score_r) ++aabb.max.x, aabb.nv += nv_r; // Right
        else if (best_score == score_b) ++aabb.max.y, aabb.nv += nv_b; // Bottom
    // clang-format on

    // aabb.nv = (aabb.max.y - aabb.min.y + 1) * (aabb.max.x - aabb.min.x + 1);

    // Re-assign AABB ID map
    for (int y = aabb.min.y; y <= aabb.max.y; ++y) {
        for (int x = aabb.min.x; x <= aabb.max.x; ++x) {
            m_aabb_id_map[y * m_width + x] = aabb_id;
        }
    }
    return true;
}

void ClusterIntGrid::check_overlapping_aabbs()
{
    int num_overlapping = 0;
    for (int ai = 0; ai < m_aabbs.size(); ++ai) {
        for (int bi = 0; bi < m_aabbs.size(); ++bi) {
            if (ai == bi) continue; // Skip self
            const ClusterIntGrid_AABB& a = m_aabbs.at(ai);
            const ClusterIntGrid_AABB& b = m_aabbs.at(bi);
            if (a.overlaps(b)) {
                // clang-format off
                printf("[WARN ] [ClusterIntGrid] AABB #%d and #%d overlap! A: (%d, %d) -> (%d, %d); B: (%d, %d) -> (%d, %d)\n",
                       ai, bi,
                       a.min.x, a.min.y, a.max.x, a.max.y,
                       b.min.x, b.min.y, b.max.x, b.max.y);
                // clang-format on
                ++num_overlapping;
                break;
            }
        }
    }
    if (num_overlapping > 0) {
        throw IllegalStateException("AABBs are overlapping: {}", num_overlapping);
    }
}

void ClusterIntGrid::compute_nv(ClusterIntGrid_AABB& aabb)
{
    aabb.nv = 0;
    for (int y = aabb.min.y; y <= aabb.max.y; ++y) {
        for (int x = aabb.min.x; x <= aabb.max.x; ++x) {
            int v = m_values[y * m_width + x];
            if (v == aabb.v) ++aabb.nv;
        }
    }
}

void ClusterIntGrid::save_int_image(const std::filesystem::path& filepath, int W, int H, const int* values)
{
    std::vector<uint8_t> image_data(W * H * 3);
    for (int pix_loc = 0; pix_loc < W * H; ++pix_loc) {
        uint32_t v = values[pix_loc];
        glm::vec3 color = hash41(v); // [0, 1]
        image_data[pix_loc * 3 + 0] = (uint8_t) glm::clamp(color.r * 255.99f, 0.f, 255.f);
        image_data[pix_loc * 3 + 1] = (uint8_t) glm::clamp(color.g * 255.99f, 0.f, 255.f);
        image_data[pix_loc * 3 + 2] = (uint8_t) glm::clamp(color.b * 255.99f, 0.f, 255.f);
    }
    stbi_write_png(filepath.string().c_str(), W, H, 3, image_data.data(), W * 3);
}

void ClusterIntGrid::save_aabb_id_map_image(const std::filesystem::path& filepath)
{
    save_int_image(filepath, m_width, m_height, m_aabb_id_map.data());
}

void ClusterIntGrid::save_values_image(const std::filesystem::path& filepath)
{
    save_int_image(filepath, m_width, m_height, m_values);
}
