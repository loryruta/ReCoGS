/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "SVORenderer.h"

#include "GSCamera.h"
#include "utils/cuda_utils.h"

using namespace recogs;

// Reference:
// https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees-analysis-extensions-and-implementation

// TODO Current issues:
//   1. The SVO renders when its behind the camera
//   2. How to UV-map individual voxels for shadowing?

#define CAST_STACK_DEPTH 21
#define EPSILON exp2f(-CAST_STACK_DEPTH)
#define MAX_RAYCAST_ITERATIONS 10000

namespace
{
struct Ray3f {
    glm::vec3 o;
    glm::vec3 d;
};

struct CastResult {
    float t;
    int depth;
    int iter;
    int idx;
};

__device__ bool
test_ray_aabb(const Ray3f& ray, const glm::vec3& min_, const glm::vec3& max_, float& out_tmin, float& out_tmax)
{
    glm::vec3 tmin_xyz = (min_ - ray.o) / ray.d;
    glm::vec3 tmax_xyz = (max_ - ray.o) / ray.d;
    glm::vec3 t1 = min(tmin_xyz, tmax_xyz);
    glm::vec3 t2 = max(tmin_xyz, tmax_xyz);
    out_tmin = max(max(t1.x, t1.y), t1.z);       // tmin
    out_tmax = min(min(t2.x, t2.y), t2.z);       // tmax
    return out_tmin <= out_tmax && out_tmax > 0; // Intersecting and not behind
}

__constant__ glm::vec3 k_octet_colormap[]{
    glm::vec3(0, 0, 1),
    glm::vec3(0, 1, 0),
    glm::vec3(1, 0, 0),
    glm::vec3(1, 0, 1),
    glm::vec3(0, 1, 1),
    glm::vec3(1, 1, 0),
    glm::vec3(0.5, 0.5, 0.5),
    glm::vec3(1, 1, 1),
};

__device__ bool cast_ray(
    const Ray3f& ray, const glm::vec3& svo_min, const glm::vec3& svo_max, const SVONode* svo, CastResult& cast_result)
{
    // Perform a ray-aabb intersection test to determine tmin and tmax of the SVO
    float tmin, tmax;
    if (!test_ray_aabb(ray, svo_min, svo_max, tmin, tmax)) return false;

    glm::vec3 size = svo_max - svo_min;

    // Precompute the coefficients of t_x(x), t_y(y), and t_z(z)
    float tx_coef = 1.0f / ray.d.x;
    float ty_coef = 1.0f / ray.d.y;
    float tz_coef = 1.0f / ray.d.z;
    float tx_bias = -tx_coef * ray.o.x;
    float ty_bias = -ty_coef * ray.o.y;
    float tz_bias = -tz_coef * ray.o.z;

    int dir_mask = 0;
    if (ray.d.x > 0.0f) dir_mask |= 1;
    if (ray.d.y > 0.0f) dir_mask |= 2;
    if (ray.d.z > 0.0f) dir_mask |= 4;

    // Initialize the current voxel to the first child of the root
    uint32_t cur_node_addr = 0;
    int depth = 0;
    float scale = 0.5f;

    glm::vec3 center = (svo_min + svo_max) * 0.5f;

    int idx = 0;
    // Project octree center position to understand which node to enter first
    if (center.x * tx_coef + tx_bias > tmin) idx |= 1;
    if (center.y * ty_coef + ty_bias > tmin) idx |= 2;
    if (center.z * tz_coef + tz_bias > tmin) idx |= 4;
    glm::vec3 corner = center;
    if ((idx & 1) == 0) corner.x += sign(ray.d.x) * size.x * scale;
    if ((idx & 2) == 0) corner.y += sign(ray.d.y) * size.y * scale;
    if ((idx & 4) == 0) corner.z += sign(ray.d.z) * size.z * scale;

    struct Record {
        uint32_t parent_idx;
        int idx;
        glm::vec3 corner;
    };
    Record stack[CAST_STACK_DEPTH];

    int iter = 0;

    // Traverse voxels along the ray as long as the current ray stays within the octree
    while (true) {
        ++iter;
        if (iter == 256) break;
        const SVONode& cur_parent = svo[cur_node_addr];
        assert(cur_parent.is_parent()); // Must be a parent
        // Process voxel if the corresponding bit in valid mask is set and the active t-span is non-empty.
        // This is where the magic happens (see the paper for more details):
        // https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf
        uint8_t child_idx = idx ^ dir_mask;
        uint8_t child_mask = 1 << child_idx;
        uint8_t child_bit = cur_parent.children_mask & child_mask; // Child exist?
        if (child_bit) {
            uint32_t value = cur_parent.first_child_offset & 0x7FFFFFFF;
            uint8_t child_addr = value + __popc(cur_parent.children_mask & (child_mask - 1));
            const SVONode& child_node = svo[child_addr];
            if (child_node.is_leaf()) {
                cast_result.t = tmin;
                cast_result.depth = depth;
                cast_result.iter = iter;
                cast_result.idx = idx;
                return true; // Hit a child node!
            } else {
                // ---------------------------------------------------------------- PUSH
                if (depth >= CAST_STACK_DEPTH) return false;
                stack[depth].parent_idx = cur_node_addr;
                stack[depth].idx = idx;
                stack[depth].corner = corner;
                depth++;
                scale *= 0.5f;
                // Compute the center of the sub-node (using the recovered corner position)
                center.x = corner.x - sign(ray.d.x) * size.x * scale;
                center.y = corner.y - sign(ray.d.y) * size.y * scale;
                center.z = corner.z - sign(ray.d.z) * size.z * scale;
                // Initialize the first node `idx` to enter
                idx = 0;
                if (center.x * tx_coef + tx_bias > tmin) idx |= 1, corner.x -= sign(ray.d.x) * size.x * scale;
                if (center.y * ty_coef + ty_bias > tmin) idx |= 2, corner.y -= sign(ray.d.y) * size.y * scale;
                if (center.z * tz_coef + tz_bias > tmin) idx |= 4, corner.z -= sign(ray.d.z) * size.z * scale;
                //
                cur_node_addr = child_addr;
                continue;
            }
        }
        // ---------------------------------------------------------------- ADVANCE
    advance:
        // Determine maximum t-value of the cube by evaluating tx(), ty(), and tz() at its corner
        float tx_corner = corner.x * tx_coef + tx_bias;
        float ty_corner = corner.y * ty_coef + ty_bias;
        float tz_corner = corner.z * tz_coef + tz_bias;
        float tmax_corner = fminf(fminf(tx_corner, ty_corner), tz_corner);

        // Step along the ray
        int step_mask = 0;
        if (tx_corner <= tmax_corner) {
            step_mask ^= 1;
            corner.x += sign(ray.d.x) * size.x * scale;
        }
        if (ty_corner <= tmax_corner) {
            step_mask ^= 2;
            corner.y += sign(ray.d.y) * size.y * scale;
        }
        if (tz_corner <= tmax_corner) {
            step_mask ^= 4;
            corner.z += sign(ray.d.z) * size.z * scale;
        }

        tmin = tmax_corner;
        idx ^= step_mask;

        if ((idx & step_mask) != 0) {
            // ---------------------------------------------------------------- POP
            if (depth == 0) return false;
            --depth;
            scale *= 2.0f;
            cur_node_addr = stack[depth].parent_idx;
            idx = stack[depth].idx;
            corner = stack[depth].corner;
            // When we pop, we don't want to enter again the same parent node; so we jump to the advance algorithm with
            // a beautiful goto
            goto advance;
        }
    }
    return false;
}

__global__ void cast_ray_kernel(glm::vec3 svo_min,
                                glm::vec3 svo_max,
                                const SVONode* svo,
                                glm::vec3 cam_pos,
                                glm::mat4 cam_view,
                                glm::mat4 cam_proj,
                                Image4fHWC color_depth)
{
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= color_depth.width || py >= color_depth.height) return;

    glm::vec4 frag_coord;
    frag_coord.x = (float(px) / float(color_depth.width)) * 2.0f - 1.0f;
    frag_coord.y = (float(py) / float(color_depth.height)) * 2.0f - 1.0f;
    frag_coord.z = 1.0f;
    frag_coord.w = 1.0f;

    // Generate ray
    Ray3f ray{};
    ray.o = cam_pos;
    ray.d = glm::normalize(glm::mat3(glm::inverse(cam_view)) * (glm::inverse(cam_proj) * frag_coord));
    if (fabsf(ray.d.x) < EPSILON) ray.d.x = copysignf(EPSILON, ray.d.x);
    if (fabsf(ray.d.y) < EPSILON) ray.d.y = copysignf(EPSILON, ray.d.y);
    if (fabsf(ray.d.z) < EPSILON) ray.d.z = copysignf(EPSILON, ray.d.z);

    // Ray casting against SVO
    CastResult cast_result{};
    bool hit = cast_ray(ray, svo_min, svo_max, svo, cast_result);
    if (!hit) {
        color_depth.set_value(px, py, glm::vec4(0));
        return;
    }

    int resolution = 1 << (cast_result.depth + 1);
    glm::vec3 hit_pos = ray.o + cast_result.t * ray.d;

    glm::vec3 voxel_size = (svo_max - svo_min) / float(resolution);
    //    if (RCGS_B0T0)
    //        printf("size: %f %f %f, resolution: %d, voxel_size %f %f %f\n",
    //               (svo_max - svo_min).x,
    //               (svo_max - svo_min).y,
    //               (svo_max - svo_min).z,
    //               resolution,
    //               voxel_size.x,
    //               voxel_size.y,
    //               voxel_size.z);

    // glm::vec3 color = glm::mod(hit_pos - svo_min, voxel_size) / voxel_size;
    glm::vec3 color = k_octet_colormap[cast_result.idx];
    color_depth.set_value(px, py, glm::vec4(color, 1));

    //    if (RCGS_B0T0)
    //        printf(
    //            "t: %f, depth: %d, res: %d, hit_pos %f %f %f, voxel size: %f %f %f\n",
    //            cast_result.t,
    //            cast_result.depth,
    //            resolution,
    //            hit_pos.x,
    //            hit_pos.y,
    //            hit_pos.z,
    //            voxel_size.x,
    //            voxel_size.y,
    //            voxel_size.z);
}
} // namespace

void SVORenderer::render(const SVO& svo, const GSCamera& camera, Image4fHWC& color_depth, cudaStream_t stream)
{
    glm::ivec2 resolution = camera.resolution();
    dim3 num_blocks;
    num_blocks.x = div_ceil(resolution.x, 16);
    num_blocks.y = div_ceil(resolution.y, 16);
    dim3 blocks_dim;
    blocks_dim.x = 16;
    blocks_dim.y = 16;
    cast_ray_kernel<<<num_blocks, blocks_dim, 0, stream>>>(
        svo.min, svo.max, RCGS_TPTR(svo.nodes), camera.position, camera.viewmatrix(), camera.projmatrix(), color_depth);
}
