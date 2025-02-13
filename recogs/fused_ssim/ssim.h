#pragma once

#include <cstdio>
#include <tuple>
#include <string>

__global__ void fusedssimCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    float* img1,
    float* img2,
    float* ssim_map,
    float* dm_dmu1 = nullptr,
    float* dm_dsigma1_sq = nullptr,
    float* dm_dsigma12 = nullptr
);

__global__ void fusedssim_backwardCUDA(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    float* img1,
    float* img2,
    float *dL_dmap,
    float *dL_dimg1,
    float* dm_dmu1 = nullptr,
    float* dm_dsigma1_sq = nullptr,
    float* dm_dsigma12 = nullptr
);
