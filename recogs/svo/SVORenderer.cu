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

__device__ bool cast_ray(const Ray3f& ray,
                         const glm::vec3& svo_min,
                         const glm::vec3& svo_max,
                         const SVONode* svo,
                         CastResult& result,
                         glm::vec3& out_color)
{
    // Perform a ray-aabb intersection test to determine tmin and tmax of the SVO
    float tmin, tmax;
    if (!test_ray_aabb(ray, svo_min, svo_max, tmin, tmax)) return false;

    int iter = 0;
    glm::vec3 svo_mid = (svo_min + svo_max) * 0.5f;
    glm::vec3 svo_ext = svo_max - svo_min;

    // Precompute the coefficients of t_x(x), t_y(y), and t_z(z)
    float tx_coef = 1.0f / ray.d.x;
    float ty_coef = 1.0f / ray.d.y;
    float tz_coef = 1.0f / ray.d.z;
    float tx_bias = -tx_coef * ray.o.x;
    float ty_bias = -ty_coef * ray.o.y;
    float tz_bias = -tz_coef * ray.o.z;

    int dir_mask = 0;
    if (ray.d.x > 0.0f) dir_mask ^= 1;
    if (ray.d.y > 0.0f) dir_mask ^= 2;
    if (ray.d.z > 0.0f) dir_mask ^= 4;

    // Initialize the current voxel to the first child of the root
    const SVONode* parent = svo;
    const SVONode* child_descriptor = nullptr; // Invalid until fetched
    int depth = CAST_STACK_DEPTH - 1;
    float scale_exp2 = 0.5f;

    int idx = 0;
    // Project octree center position to understand which node to enter first
    if (svo_mid.x * tx_coef + tx_bias > tmin) idx |= 1;
    if (svo_mid.y * ty_coef + ty_bias > tmin) idx |= 2;
    if (svo_mid.z * tz_coef + tz_bias > tmin) idx |= 4;

    glm::vec3 corner = svo_mid;
    if ((idx & 1) == 0) corner.x += sign(ray.d.x) * svo_ext.x * scale_exp2;
    if ((idx & 2) == 0) corner.y += sign(ray.d.y) * svo_ext.y * scale_exp2;
    if ((idx & 4) == 0) corner.z += sign(ray.d.z) * svo_ext.z * scale_exp2;

    struct Record {
        uint32_t parent_idx;
        float t_max;
    };
    Record stack[CAST_STACK_DEPTH];

    // Traverse voxels along the ray as long as the current ray stays within the octree
    while (true) {
        // Fetch child descriptor unless it is already valid.
        if (!child_descriptor) child_descriptor = parent;

        // Process voxel if the corresponding bit in valid mask is set and the active t-span is non-empty.
        // This is where the magic happens (see the paper for more details):
        // https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf

        int child_idx = idx ^ dir_mask;
        // assert(child_idx <= 7);
        int child_bit = int(child_descriptor->children_mask) & (1 << child_idx); // Child exist?
        if (child_bit) {
            out_color = k_octet_colormap[child_idx];
            return true;
        }

        //        if (child_bit && tmin <= tmax) {
        //            // INTERSECT
        //            // Intersect active t-span with the cube and evaluate tx(), ty(), and tz() at the center of the
        //            voxel float tv_max = fminf(tmax, tc_max); float half = scale_exp2 * 0.5f; float tx_center = half *
        //            tx_coef + tx_corner; float ty_center = half * ty_coef + ty_corner; float tz_center = half *
        //            tz_coef + tz_corner;
        //
        //            // Descend to the first child if the resulting t-span is non-empty
        //            if (tmin <= tv_max) {
        //                // PUSH
        //                // Write current parent to the stack
        //                if (depth >= CAST_STACK_DEPTH) return false; // ERROR: no more size in the stack
        //
        //                // Find child descriptor corresponding to the current voxel
        //                parent = svo + child_descriptor->first_child_offset + child_idx;
        //
        //                // Select child voxel that the ray enters first
        //                idx = 0;
        //                depth--;
        //                scale_exp2 = half;
        //
        //                if (tx_center > tmin) idx ^= 1, pos.x += scale_exp2;
        //                if (ty_center > tmin) idx ^= 2, pos.y += scale_exp2;
        //                if (tz_center > tmin) idx ^= 4, pos.z += scale_exp2;
        //
        //                // Update active t-span and invalidate cached child descriptor.
        //                tmax = tv_max;
        //                child_descriptor = nullptr;
        //                continue;
        //            }
        //        }

        // ---------------------------------------------------------------- ADVANCE

        // Determine maximum t-value of the cube by evaluating tx(), ty(), and tz() at its corner
        float tx_corner = corner.x * tx_coef + tx_bias;
        float ty_corner = corner.y * ty_coef + ty_bias;
        float tz_corner = corner.z * tz_coef + tz_bias;
        float tmax_corner = fminf(fminf(tx_corner, ty_corner), tz_corner);

        // Step along the ray
        int step_mask = 0;
        if (tx_corner <= tmax_corner) {
            step_mask ^= 1;
            corner.x += sign(ray.d.x) * svo_ext.x * scale_exp2;
        }
        if (ty_corner <= tmax_corner) {
            step_mask ^= 2;
            corner.y += sign(ray.d.y) * svo_ext.y * scale_exp2;
        }
        if (tz_corner <= tmax_corner) {
            step_mask ^= 4;
            corner.z += sign(ray.d.z) * svo_ext.z * scale_exp2;
        }

        // Update active t-span and flip bits of the child slot index
        tmin = tmax_corner;
        idx ^= step_mask;

        // Proceed with pop if the bit flips disagree with the ray direction
        if ((idx & step_mask) != 0) {
            // ---------------------------------------------------------------- POP
            // Find the highest differing bit between the two positions.
            return false;

            unsigned int differing_bits = 0;
            if ((step_mask & 1) != 0)
                differing_bits |= __float_as_int(corner.x) ^ __float_as_int(corner.x + scale_exp2);
            if ((step_mask & 2) != 0)
                differing_bits |= __float_as_int(corner.y) ^ __float_as_int(corner.y + scale_exp2);
            if ((step_mask & 4) != 0)
                differing_bits |= __float_as_int(corner.z) ^ __float_as_int(corner.z + scale_exp2);
            depth = (__float_as_int((float) differing_bits) >> 23) - 127;        // position of the highest bit
            scale_exp2 = __int_as_float((depth - CAST_STACK_DEPTH + 127) << 23); // exp2f(scale - s_max)

            // Restore parent voxel from the stack
            parent = svo + stack[depth].parent_idx;
            tmax = stack[depth].t_max;
            depth--;

            // Round cube position and extract child slot index
            int shx = __float_as_int(corner.x) >> depth;
            int shy = __float_as_int(corner.y) >> depth;
            int shz = __float_as_int(corner.z) >> depth;
            corner.x = __int_as_float(shx << depth);
            corner.y = __int_as_float(shy << depth);
            corner.z = __int_as_float(shz << depth);
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
        return false;
    }

    // Output results
    //    result.t = t_min;
    result.iter = iter;
    //    result.pos.x = fminf(fmaxf(ray.o.x + t_min * ray.d.x, pos.x + EPSILON), pos.x + scale_exp2 - EPSILON);
    //    result.pos.y = fminf(fmaxf(ray.o.y + t_min * ray.d.y, pos.y + EPSILON), pos.y + scale_exp2 - EPSILON);
    //    result.pos.z = fminf(fmaxf(ray.o.z + t_min * ray.d.z, pos.z + EPSILON), pos.z + scale_exp2 - EPSILON);
    result.node = parent;
    result.child_idx = idx ^ dir_mask ^ 7;
    result.stack_ptr = depth;
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
    glm::vec3 color;
    bool intersecting = cast_ray(ray, svo_min, svo_max, svo, cast_result, color);
    if (intersecting) {
        color_depth.set_value(px, py, glm::vec4(color, 1));
    } else {
        color_depth.set_value(px, py, glm::vec4(0));
    }
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
