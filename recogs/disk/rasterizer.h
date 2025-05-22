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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <vector>

namespace CudaRasterizer
{
class Rasterizer
{
public:
    static int forward(std::function<char*(size_t)> geometryBuffer,
                       std::function<char*(size_t)> binningBuffer,
                       std::function<char*(size_t)> imageBuffer,
                       int P,
                       int width,
                       int height,
                       const float* means3D,
                       const float* scales,
                       const float* rotations,
                       const float* viewmatrix,
                       const float* projmatrix,
                       float* out_color,
                       bool debug,
                       cudaStream_t stream);
};
}; // namespace CudaRasterizer

#endif
