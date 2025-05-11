#include "SVOBuilder.h"

#include <cuda/std/utility>
#include <thrust/device_vector.h>

#include "utils/Stopwatch.h"
#include "utils/cuda_utils.h"
#include "utils/misc_utils.h"
#include "utils/str_utils.h"

using namespace recogs;

namespace
{
/// Return the address of the node containing the point at level within the SVO.
/// \param level The SVO level (range: [0, resolution[)
__device__ cuda::std::pair<uint32_t, uint8_t> traverse_svo(const SVONode* svo, uint64_t point, int level)
{
    uint32_t node_idx = 0;
    uint8_t child_idx = point & 0x7;
    for (int i = 1; i <= level; ++i) {
        uint8_t children_mask = svo[node_idx].children_mask;
        uint8_t num_children_before = __popc(children_mask & ((1 << child_idx) - 1));
        node_idx = svo[node_idx].first_child_offset + num_children_before;
        child_idx = (point >> (i * 3)) & 0x7;
    }
    return {node_idx, child_idx};
}

/// Kernel for marking the nodes of a given level as in need for children allocation.
/// The execution policy is one thread for every point.
__global__ void mark_kernel(const uint64_t* points, size_t num_points, int level, SVONode* svo)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;
    auto [node_idx, child_idx] = traverse_svo(svo, points[i], level);
    atomicOr(&svo[node_idx].children_mask, 1 << child_idx); // Mark
}

/// Kernel for counting and indexing the nodes for the next level, provided with the current one.
/// The execution policy is one thread for every node of the current level.
__global__ void
count_and_index_kernel(SVONode* svo, uint32_t curr_level_offset, uint32_t next_level_offset, uint32_t* num_allocations)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (curr_level_offset + i >= next_level_offset) return;
    SVONode& node = svo[curr_level_offset + i];
    uint32_t num_children = __popc(node.children_mask); // Number of marked nodes
    assert(num_children);
    uint32_t alloc_idx = atomicAdd(num_allocations, num_children);
    node.first_child_offset = next_level_offset + alloc_idx;
}

/// Kernel for writing the leaves of the SVO. Leaves are valued with indices to the points list.
/// The execution policy is one thread for every point.
__global__ void write_leaves_kernel(const uint64_t* points, size_t num_points, SVONode* svo, int level)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;
    auto [node_idx, child_idx] = traverse_svo(svo, points[i], level);
    svo[node_idx].data = i; // Used to refer to the point
}
} // namespace

void SVOBuilder::build(const uint64_t* points, size_t num_points, int resolution, cudaStream_t stream)
{
    Stopwatch stopwatch;

    size_t svo_bytesize = 1024 * 1024; // 1MB
    SVONode* svo_d;
    CHECK_CUDA(cudaMallocAsync(&svo_d, svo_bytesize, stream));
    uint32_t* num_allocations_d;
    CHECK_CUDA(cudaMallocAsync(&num_allocations_d, sizeof(uint32_t), stream));

    dim3 num_blocks;

    uint32_t curr_level_offset = 0;
    uint32_t next_level_offset = 1;
    for (int level = 0; level < resolution - 1; ++level) {
        uint32_t num_curr_level_nodes = next_level_offset - curr_level_offset;

        printf("[DEBUG] [SVOBuilder] Level %d/%d, Range: [%d, %d[ (%d nodes)\n",
               level + 1,
               resolution,
               curr_level_offset,
               next_level_offset,
               num_curr_level_nodes);
        // Clear current level
        CHECK_CUDA(cudaMemsetAsync(svo_d + curr_level_offset, 0, num_curr_level_nodes * sizeof(SVONode), stream));
        // Mark current level
        num_blocks.x = div_ceil<size_t>(num_points, 1024);
        mark_kernel<<<num_blocks, 1024, 0, stream>>>(points, num_points, level, svo_d);
        // Count and index allocations
        CHECK_CUDA(cudaMemsetAsync(num_allocations_d, 0, sizeof(uint32_t), stream));
        num_blocks.x = div_ceil<size_t>(num_curr_level_nodes, 1024);
        count_and_index_kernel<<<num_blocks, 1024, 0, stream>>>(
            svo_d, curr_level_offset, next_level_offset, num_allocations_d);
        uint32_t num_allocations;
        CHECK_CUDA(
            cudaMemcpyAsync(&num_allocations, num_allocations_d, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_STATE(num_allocations > 0); // Illegal state: algorithm issue

        printf("[DEBUG] [SVOBuilder]   Indexed level %d/%d, Allocated children: %d; Octree size: %s\n",
               level + 1,
               resolution,
               num_allocations,
               num_bytes_to_string(next_level_offset * sizeof(SVONode)).c_str());

        curr_level_offset = next_level_offset;
        next_level_offset += num_allocations;

        // Need re-allocation
        if (next_level_offset * sizeof(SVONode) > svo_bytesize) {
            size_t new_svo_bytesize = round_to_next_power_of_2(next_level_offset * sizeof(SVONode));
            printf("[INFO ] [SVOBuilder]   Resizing SVO from %s to %s\n",
                   num_bytes_to_string(svo_bytesize).c_str(),
                   num_bytes_to_string(new_svo_bytesize).c_str());
            SVONode* new_svo_d;
            CHECK_CUDA(cudaMallocAsync(&new_svo_d, new_svo_bytesize, stream));
            CHECK_CUDA(cudaMemcpyAsync(new_svo_d, svo_d, svo_bytesize, cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA(cudaFreeAsync(svo_d, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            svo_bytesize = new_svo_bytesize;
            svo_d = new_svo_d;
        }
    }
    // Write leaves
    num_blocks.x = div_ceil<size_t>(num_points, 1024);
    write_leaves_kernel<<<num_blocks, 1024, 0, stream>>>(points, num_points, svo_d, resolution - 1);
    CHECK_CUDA(cudaFreeAsync(num_allocations_d, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("[INFO ] [SVOBuilder] Leaves written; SVO built in %s using %s\n",
           stopwatch.elapsed_time_str().c_str(),
           num_bytes_to_string(svo_bytesize).c_str());
}
