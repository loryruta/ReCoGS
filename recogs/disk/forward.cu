/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "auxiliary.h"
#include "forward.h"
#include <cooperative_groups.h>

#define CUTOFF_DISTANCE 1.0f

namespace cg = cooperative_groups;

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(const float3& p_orig,
                                 const glm::vec2 scale,
                                 const glm::vec4 rot,
                                 const float* projmatrix,
                                 const float* viewmatrix,
                                 const int W,
                                 const int H,
                                 glm::mat3& T,
                                 float3& normal)
{
    glm::mat3 R = quat_to_rotmat(rot);
    glm::mat3 S = scale_to_mat(scale, 1.f /* scale_modifier */);
    glm::mat3 L = R * S;
    // center of Gaussians in the camera coordinate
    glm::mat3x4 splat2world =
        glm::mat3x4(glm::vec4(L[0], 0.0), glm::vec4(L[1], 0.0), glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1));
    // clang-format off
    glm::mat4 world2ndc = glm::mat4(projmatrix[0], projmatrix[4],  projmatrix[8],  projmatrix[12],
                                    projmatrix[1], projmatrix[5],  projmatrix[9],  projmatrix[13],
                                    projmatrix[2], projmatrix[6],  projmatrix[10], projmatrix[14],
                                    projmatrix[3], projmatrix[7],  projmatrix[11], projmatrix[15]);
    glm::mat3x4 ndc2pix = glm::mat3x4(glm::vec4(float(W) / 2.0, 0.0,            0.0, float(W - 1) / 2.0),
                                      glm::vec4(0.0,            float(H) / 2.0, 0.0, float(H - 1) / 2.0),
                                      glm::vec4(0.0,            0.0,            0.0, 1.0));
    // clang-format on
    T = glm::transpose(splat2world) * world2ndc * ndc2pix;
    normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(glm::mat3 T, float cutoff, float2& point_image, float2& extent)
{
    glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
    float d = glm::dot(t, T[2] * T[2]);
    if (d == 0.0) return false;
    glm::vec3 f = (1 / d) * t;

    glm::vec2 p = glm::vec2(glm::dot(f, T[0] * T[2]), glm::dot(f, T[1] * T[2]));

    glm::vec2 h0 = p * p - glm::vec2(glm::dot(f, T[0] * T[0]), glm::dot(f, T[1] * T[1]));

    glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
    point_image = {p.x, p.y};
    extent = {h.x, h.y};
    return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void preprocessCUDA(int P,
                               const float* orig_points,
                               const glm::vec2* scales,
                               const glm::vec4* rotations,
                               const float* viewmatrix,
                               const float* projmatrix,
                               int W,
                               int H,
                               int* radii,
                               float2* points_xy_image,
                               float* transMats,
                               float4* normal_opacity,
                               dim3 grid,
                               uint32_t* tiles_touched,
                               bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) return;

    // Compute transformation matrix
    glm::mat3 T; // Matrix warping from tangent plane to pixel space
    float3 normal;
    compute_transmat(
        ((float3*) orig_points)[idx], scales[idx], rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
    float3* T_ptr = (float3*) transMats;
    T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
    T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
    T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};

    float cos = -sumf3(p_view * normal);
    if (cos == 0) return;
    float multiplier = cos > 0 ? 1 : -1;
    normal = multiplier * normal;

    // Compute center and radius
    float2 point_image;
    float radius;
    {
        float2 extent;
        bool ok = compute_aabb(T, CUTOFF_DISTANCE, point_image, extent);
        if (!ok) return;
        radius = max(extent.x, extent.y);
    }

    uint2 rect_min, rect_max;
    getRect(point_image, radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

    radii[idx] = (int) radius;
    points_xy_image[idx] = point_image;
    normal_opacity[idx] = {normal.x, normal.y, normal.z, 1.f}; // Last component was opacity
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y) renderCUDA(const uint2* __restrict__ ranges,
                                                               const uint32_t* __restrict__ point_list,
                                                               int W,
                                                               int H,
                                                               const float2* __restrict__ points_xy_image,
                                                               const float* __restrict__ transMats,
                                                               const float4* __restrict__ normal_opacity,
                                                               float* __restrict__ out_color)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float) pix.x, (float) pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_normal_opacity[BLOCK_SIZE];
    __shared__ float3 collected_Tu[BLOCK_SIZE];
    __shared__ float3 collected_Tv[BLOCK_SIZE];
    __shared__ float3 collected_Tw[BLOCK_SIZE];

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE) break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
            collected_Tu[block.thread_rank()] = {
                transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
            collected_Tv[block.thread_rank()] = {
                transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
            collected_Tw[block.thread_rank()] = {
                transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // First compute two homogeneous planes, See Eq. (8)
            const float2 xy = collected_xy[j];
            const float3 Tu = collected_Tu[j];
            const float3 Tv = collected_Tv[j];
            const float3 Tw = collected_Tw[j];
            // Transform the two planes into local u-v system.
            // See:
            // https://github.com/hbb1/diff-surfel-rasterization/issues/15#issuecomment-2195228658
            float3 k = pix.x * Tw - Tu;
            float3 l = pix.y * Tw - Tv;
            // Cross product of two planes is a line, Eq. (9)
            float3 p = cross(k, l);
            if (p.z == 0.0) continue;
            // Perspective division to get the intersection (u,v), Eq. (10)
            float2 s = {p.x / p.z, p.y / p.z};
            // Compute depth
            float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
            // If a point is too small, its depth is not reliable?
            if (depth < near_n) continue;
            float s_dist = glm::length(glm::vec2(s.x, s.y));
            if (inside) {
                if (s_dist > CUTOFF_DISTANCE) {
                    out_color[pix_id * 4 + 0] = 1.0f;
                    out_color[pix_id * 4 + 1] = 0.f;
                    out_color[pix_id * 4 + 2] = 0.f;
                    out_color[pix_id * 4 + 3] = depth;
                } else {
                    out_color[pix_id * 4 + 0] = 1.0f;
                    out_color[pix_id * 4 + 1] = 1.0f;
                    out_color[pix_id * 4 + 2] = 1.0f;
                    out_color[pix_id * 4 + 3] = depth;
                }
            }
            // TODO pixel-wise depth test
            break;
        }
    }
}

void FORWARD::preprocess(int P,
                         const float* means3D,
                         const glm::vec2* scales,
                         const glm::vec4* rotations,
                         const float* viewmatrix,
                         const float* projmatrix,
                         int W,
                         int H,
                         int* radii,
                         float2* means2D,
                         float* transMats,
                         float4* normal_opacity,
                         const dim3 grid,
                         uint32_t* tiles_touched,
                         bool prefiltered,
                         cudaStream_t stream)
{
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256, 0, stream>>>(P,
                                                                      means3D,
                                                                      scales,
                                                                      rotations,
                                                                      viewmatrix,
                                                                      projmatrix,
                                                                      W,
                                                                      H,
                                                                      radii,
                                                                      means2D,
                                                                      transMats,
                                                                      normal_opacity,
                                                                      grid,
                                                                      tiles_touched,
                                                                      prefiltered);
}

void FORWARD::render(dim3 grid,
                     dim3 block,
                     const uint2* ranges,
                     const uint32_t* point_list,
                     int W,
                     int H,
                     const float2* means2D,
                     const float* transMats,
                     const float4* normal_opacity,
                     float* out_color,
                     cudaStream_t stream)
{
    renderCUDA<NUM_CHANNELS>
        <<<grid, block, 0, stream>>>(ranges, point_list, W, H, means2D, transMats, normal_opacity, out_color);
}
