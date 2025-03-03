#include "Selection2d.h"

#include <glm/glm.hpp>
#include <thrust/copy.h>

#include "App.h"
#include "Selection3d.h"
#include "utils/DeviceBuffer.h"
#include "utils/bresenham.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_visit_transform.h"

using namespace gs_train;

namespace
{
template <bool IS_FILLING, bool IS_CLEARING_HARD = false>
__global__ void fill_clear_kernel( //
    glm::ivec2 p0,
    glm::ivec2 p1,
    int r,
    Selection2d::NewMap new_map,
    Selection2d::RefMap ref_map,
    bool* clear_bitmask)
{
    assert(new_map.size() == ref_map.size());
    int W = new_map.width;
    int H = new_map.height;

    // Executed with 1 block and 16x16 tile size.
    // Pixels between p0 and p1 are processed sequentially by all threads filling or clearing is performed
    // collaboratively

    // How many pixels a thread iterates along X/Y axes
    int num_items_x = div_ceil(r * 2, int(blockDim.x));
    int num_items_y = div_ceil(r * 2, int(blockDim.y));
    bresenham_draw_line(p0.x, p0.y, p1.x, p1.y, [&] __device__(int cx, int cy) mutable {
        int sx = cx - r + int(threadIdx.x) * num_items_x;
        int sy = cy - r + int(threadIdx.y) * num_items_y;
        int ex = sx + num_items_x;
        int ey = sy + num_items_y;
        int r2 = r * r;
        for (int x = sx; x <= ex; ++x) {
            for (int y = sy; y <= ey; ++y) {
                if (x < 0 || y < 0 || x >= W || y >= H) continue; // Outside the mask
                int d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if (d2 > r2) continue; // Outside the circle

                if constexpr (IS_FILLING) {
                    new_map.set_value(x, y, Selection2d::NewMap::Value{1});
                } else {
                    new_map.set_value(x, y, Selection2d::NewMap::Value{0});
                    if constexpr (IS_CLEARING_HARD) {
                        uint32_t point_idx = ref_map.value(x, y).r;
                        if (point_idx != UINT32_MAX) {
                            clear_bitmask[point_idx] = true;
                        }
                    }
                }
            }
        }
    });
}
} // namespace

Selection2d::Selection2d(Selection3d& selection3d, GSCamera view)
    : m_app(selection3d.m_app), m_selection3d(selection3d), m_view(std::move(view)),
      m_new_map(NewMap::malloc(m_view.width, m_view.height)), m_ref_map(RefMap::malloc(m_view.width, m_view.height))
{
    // Init the new map
    image_fill(m_new_map, Selection2d::NewMap::Value{0});
    // Project selection 3D to the reference map
    reproject_ref_map();
    // Init the clear bitmask
    m_clear_bitmask.resize(m_selection3d.points().size(), false);
}

void Selection2d::fill_line(glm::ivec2 p0, glm::ivec2 p1, int r)
{
    dim3 num_blocks = 1;
    dim3 block_dim(16, 16);
    fill_clear_kernel<true /* Filling */><<<num_blocks, block_dim, 0, m_app.stream()>>>( //
        p0,
        p1,
        r,
        m_new_map,
        m_ref_map,
        thrust::raw_pointer_cast(m_clear_bitmask.data()));
}

void Selection2d::clear_line(glm::ivec2 p0, glm::ivec2 p1, int r, bool hard)
{
    cudaStream_t stream = m_app.stream();
    dim3 num_blocks = 1;
    dim3 block_dim(16, 16);
    if (hard) {
        fill_clear_kernel<false /* Clearing */, true /* Hard */><<<num_blocks, block_dim, 0, stream>>>( //
            p0,
            p1,
            r,
            m_new_map,
            m_ref_map,
            thrust::raw_pointer_cast(m_clear_bitmask.data()));
    } else {
        fill_clear_kernel<false /* Clearing */, false /* Hard */><<<num_blocks, block_dim, 0, stream>>>( //
            p0,
            p1,
            r,
            m_new_map,
            m_ref_map,
            thrust::raw_pointer_cast(m_clear_bitmask.data()));
    }

    if (hard) {
        // Clear the 3D selection points using the bitmask
        m_selection3d.clear(m_clear_bitmask);
        // Re-init the bitmask
        m_clear_bitmask.resize(m_selection3d.points().size());
        thrust::fill(thrust::cuda::par.on(stream), m_clear_bitmask.begin(), m_clear_bitmask.end(), false);
        // Re-project the reference map
        reproject_ref_map();
    }
}

void Selection2d::populate_selection3d(const Image1fCHW& depth)
{
    cudaStream_t stream = m_app.stream();

    thrust::device_vector<uint32_t> ss_points(m_view.width * m_view.height);
    // Flatten "new map" to a list of screen-space points
    thrust::device_vector<uint32_t> counter(1);
    uint32_t* ss_points_d = thrust::raw_pointer_cast(ss_points.data());
    uint32_t* counter_d = thrust::raw_pointer_cast(counter.data());
    image_visit(
        m_new_map,
        [counter_d, ss_points_d] __device__(const Selection2d::NewMap& new_map, int x, int y) {
            if (new_map.value(x, y).r > 0) {
                uint32_t i = atomicAdd(counter_d, 1);
                ss_points_d[i] = (x & 0xFFFF) << 16 | (y & 0xFFFF);
            }
            return 0; // TODO temporary
        },
        stream);
    ss_points.resize(to_host(counter_d));
    // TODO resampling? to avoid too dense points
    // Unproject screen-space points to world-space and add them to the 3D selection
    m_selection3d.append(ss_points, depth, m_view);
}

void Selection2d::reproject_ref_map()
{
    image_fill(m_ref_map, Selection2d::RefMap::Value{UINT32_MAX});
    m_selection3d.project(m_view, m_ref_map);
}
