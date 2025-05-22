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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P,
                const float* orig_points,
                const glm::vec2* scales,
                const glm::vec4* rotations,
                const float* viewmatrix,
                const float* projmatrix,
                int W,
                int H,
                int* radii,
                float2* points_xy_image,
                // float* isovals,
                // float3* normals,
                float* transMats,
                float4* normal_opacity,
                dim3 grid,
                uint32_t* tiles_touched,
                bool prefiltered,
                cudaStream_t stream);

// Main rasterization method.
void render(dim3 grid,
            dim3 block,
            const uint2* ranges,
            const uint32_t* point_list,
            int W,
            int H,
            const float2* points_xy_image,
            const float* transMats,
            const float4* normal_opacity,
            float* out_color,
            cudaStream_t stream);
} // namespace FORWARD

#endif
