#pragma once

namespace gs_train
{
struct Scene {
    int num_vertices;
    float* means;
    float* normals;
    float* shs;
    float* opacities;
    float* scales;
    float* rotations;

    Scene to_host();
    Scene to_device();
};
} // namespace gs_editor
