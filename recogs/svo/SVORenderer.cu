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

#define CAST_STACK_DEPTH 23
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
    float3 pos;
    int iter;
    const SVONode* node;
    int child_idx;
    int stack_ptr;
};

__device__ bool
cast_ray(const glm::vec3& oct_min, const glm::vec3& oct_max, const Ray3f& ray, const SVONode* svo, CastResult& result)
{
    int iter = 0;

    glm::vec3 oct_mid = (oct_max + oct_min) / 2.0f;

    // Precompute the coefficients of tx(x), ty(y), and tz(z).

    float tx_coef = 1.0f / ray.d.x;
    float ty_coef = 1.0f / ray.d.y;
    float tz_coef = 1.0f / ray.d.z;
    float tx_bias = -tx_coef * ray.o.x;
    float ty_bias = -ty_coef * ray.o.y;
    float tz_bias = -tz_coef * ray.o.z;

    // Select octant mask to mirror the coordinate system so that ray direction is negative along each axis.
    int dir_mask = 7;
    if (ray.d.x > 0.0f) dir_mask ^= 1;
    if (ray.d.y > 0.0f) dir_mask ^= 2;
    if (ray.d.z > 0.0f) dir_mask ^= 4;

    // Initialize the active span of t-values.

    // TODO calculated using ray-aabb intersection test REWRITE THIS CODE DOWN HERE!
    float t_min = fmaxf(fmaxf(2.0f * tx_coef - tx_bias, 2.0f * ty_coef - ty_bias), 2.0f * tz_coef - tz_bias);
    float t_max = fminf(fminf(tx_coef - tx_bias, ty_coef - ty_bias), tz_coef - tz_bias);

    t_min = fmaxf(t_min, 0.0f); // TODO
    t_max = fminf(t_max, 1.0f); // TODO

    // Initialize the current voxel to the first child of the root.
    const SVONode* parent = svo;
    const SVONode* child_descriptor = nullptr; // Invalid until fetched
    int idx = 0;
    float3 pos = make_float3(1.0f, 1.0f, 1.0f);
    int depth = CAST_STACK_DEPTH - 1;
    float scale_exp2 = 0.5f; // exp2f(scale - s_max)

    // Project octree center position to understand which node to enter
    if (oct_mid.x * tx_coef + tx_bias > t_min) idx ^= 1, pos.x = oct_mid.x;
    if (oct_mid.y * ty_coef + ty_bias > t_min) idx ^= 2, pos.y = oct_mid.y;
    if (oct_mid.z * tz_coef + tz_bias > t_min) idx ^= 4, pos.z = oct_mid.z;

    struct Record {
        uint32_t parent_idx;
        float t_max;
    };
    Record stack[CAST_STACK_DEPTH];

    // Traverse voxels along the ray as long as the current ray stays within the octree
    while (true) {
#if (MAX_RAYCAST_ITERATIONS > 0)
        if (iter > MAX_RAYCAST_ITERATIONS) break;
#endif
        // Fetch child descriptor unless it is already valid.
        if (!child_descriptor) child_descriptor = parent;

        // Determine maximum t-value of the cube by evaluating tx(), ty(), and tz() at its corner
        float tx_corner = pos.x * tx_coef + tx_bias;
        float ty_corner = pos.y * ty_coef + ty_bias;
        float tz_corner = pos.z * tz_coef + tz_bias;
        float tc_max = fminf(fminf(tx_corner, ty_corner), tz_corner);

        // Process voxel if the corresponding bit in valid mask is set and the active t-span is non-empty.
        // This is where the magic happens (see the paper for more details):
        // https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf

        int child_idx = idx ^ dir_mask;
        int child_bit = int(child_descriptor->children_mask) & (1 << child_idx);
        if (child_bit != 0 && t_min <= t_max) {
            // INTERSECT
            // Intersect active t-span with the cube and evaluate tx(), ty(), and tz() at the center of the voxel
            float tv_max = fminf(t_max, tc_max);
            float half = scale_exp2 * 0.5f;
            float tx_center = half * tx_coef + tx_corner;
            float ty_center = half * ty_coef + ty_corner;
            float tz_center = half * tz_coef + tz_corner;

            // Descend to the first child if the resulting t-span is non-empty
            if (t_min <= tv_max) {
                // PUSH
                // Write current parent to the stack
                if (depth >= CAST_STACK_DEPTH) return false; // ERROR: no more size in the stack

                // Find child descriptor corresponding to the current voxel
                parent = svo + child_descriptor->first_child_offset + child_idx;

                // Select child voxel that the ray enters first
                idx = 0;
                depth--;
                scale_exp2 = half;

                if (tx_center > t_min) idx ^= 1, pos.x += scale_exp2;
                if (ty_center > t_min) idx ^= 2, pos.y += scale_exp2;
                if (tz_center > t_min) idx ^= 4, pos.z += scale_exp2;

                // Update active t-span and invalidate cached child descriptor.
                t_max = tv_max;
                child_descriptor = nullptr;
                continue;
            }
        }

        // ADVANCE
        // Step along the ray
        int step_mask = 0;
        if (tx_corner <= tc_max) step_mask ^= 1, pos.x += sign(ray.d.x) * scale_exp2;
        if (ty_corner <= tc_max) step_mask ^= 2, pos.y += sign(ray.d.y) * scale_exp2;
        if (tz_corner <= tc_max) step_mask ^= 4, pos.z += sign(ray.d.z) * scale_exp2;

        // Update active t-span and flip bits of the child slot index
        t_min = tc_max;
        idx ^= step_mask;

        // Proceed with pop if the bit flips disagree with the ray direction
        if ((idx & step_mask) != 0) {
            // POP
            // Find the highest differing bit between the two positions.

            unsigned int differing_bits = 0;
            if ((step_mask & 1) != 0) differing_bits |= __float_as_int(pos.x) ^ __float_as_int(pos.x + scale_exp2);
            if ((step_mask & 2) != 0) differing_bits |= __float_as_int(pos.y) ^ __float_as_int(pos.y + scale_exp2);
            if ((step_mask & 4) != 0) differing_bits |= __float_as_int(pos.z) ^ __float_as_int(pos.z + scale_exp2);
            depth = (__float_as_int((float) differing_bits) >> 23) - 127;        // position of the highest bit
            scale_exp2 = __int_as_float((depth - CAST_STACK_DEPTH + 127) << 23); // exp2f(scale - s_max)

            // Restore parent voxel from the stack
            parent = svo + stack[depth].parent_idx;
            t_max = stack[depth].t_max;
            depth--;

            // Round cube position and extract child slot index
            int shx = __float_as_int(pos.x) >> depth;
            int shy = __float_as_int(pos.y) >> depth;
            int shz = __float_as_int(pos.z) >> depth;
            pos.x = __int_as_float(shx << depth);
            pos.y = __int_as_float(shy << depth);
            pos.z = __int_as_float(shz << depth);
            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);

            // Prevent same parent from being stored again and invalidate cached child descriptor.
            child_descriptor = nullptr;
        }
    }

    // Indicate miss if we are outside the octree
#if (MAX_RAYCAST_ITERATIONS > 0)
    if (depth >= CAST_STACK_DEPTH || iter > MAX_RAYCAST_ITERATIONS)
#endif
    {
        t_min = 2.0f;
    }

    // Output results
    result.t = t_min;
    result.iter = iter;
    result.pos.x = fminf(fmaxf(ray.o.x + t_min * ray.d.x, pos.x + EPSILON), pos.x + scale_exp2 - EPSILON);
    result.pos.y = fminf(fmaxf(ray.o.y + t_min * ray.d.y, pos.y + EPSILON), pos.y + scale_exp2 - EPSILON);
    result.pos.z = fminf(fmaxf(ray.o.z + t_min * ray.d.z, pos.z + EPSILON), pos.z + scale_exp2 - EPSILON);
    result.node = parent;
    result.child_idx = idx ^ dir_mask ^ 7;
    result.stack_ptr = depth;
    return true;
}

__global__ void cast_ray_kernel(
    glm::vec3 oct_min, glm::vec3 oct_max, const SVONode* svo, glm::mat4 cam_viewproj, Image4fHWC color_depth)
{
    uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t py = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= color_depth.width || py >= color_depth.height) return;

    // Generate ray
    Ray3f ray{};
    ray.o = -cam_viewproj[3];
    ray.d = glm::normalize(glm::inverse(cam_viewproj) * glm::vec4());
    // Get rid of small ray direction components to avoid division by zero
    if (fabsf(ray.d.x) < EPSILON) ray.d.x = copysignf(EPSILON, ray.d.x);
    if (fabsf(ray.d.y) < EPSILON) ray.d.y = copysignf(EPSILON, ray.d.y);
    if (fabsf(ray.d.z) < EPSILON) ray.d.z = copysignf(EPSILON, ray.d.z);

    color_depth.set_value(px, py, glm::vec4(0, 1, 0, 1));

    // Ray casting against the SVO
    //    CastResult cast_result{};
    //    CastStack stack{};
    //    cast_ray(oct_min, oct_max, ray, svo, cast_result, stack);
}
} // namespace

void SVORenderer::render(const AABB3f& svo_minmax, const SVONode* svo_d, const GSCamera& camera, Image4fHWC color_depth)
{
    glm::ivec2 resolution = camera.resolution();
    dim3 num_blocks;
    num_blocks.x = div_ceil(resolution.x, 16);
    num_blocks.y = div_ceil(resolution.y, 16);
    dim3 blocks_dim;
    blocks_dim.x = 16;
    blocks_dim.y = 16;
    cast_ray_kernel<<<num_blocks, blocks_dim>>>(svo_minmax.min, svo_minmax.max, svo_d, camera.viewproj(), color_depth);
}
