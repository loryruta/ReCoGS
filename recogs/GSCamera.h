#pragma once

#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>

#include "utils/DeviceBuffer.h"

namespace gs_train
{
/// A class representing a camera compatible with the 3DGS pipeline
class GSCamera
{
private:
    /* Computed */
    thrust::device_vector<float> m_viewmatrix;
    thrust::device_vector<float> m_projmatrix;
    thrust::device_vector<float> m_campos;

public:
    glm::vec3 position{};
    glm::quat rotation{};
    float fx = 0.1f;
    float fy = 0.1f;
    int width = 1080;
    int height = 720;

    explicit GSCamera();
    ~GSCamera() = default;

    [[nodiscard]] glm::uvec2 resolution() const { return {width, height}; }

    [[nodiscard]] float tan_fovx() const { return 1.0f / fx; }
    [[nodiscard]] float tan_fovy() const { return 1.0f / fy; }

    [[nodiscard]] glm::vec3 right() const { return rotation * glm::vec3(1, 0, 0); }
    [[nodiscard]] glm::vec3 up() const { return rotation * glm::vec3(0, 1, 0); };
    [[nodiscard]] glm::vec3 forward() const { return rotation * glm::vec3(0, 0, 1); };

    /// Return the view matrix (or extrinsic parameters matrix)
    [[nodiscard]] glm::mat4 viewmatrix() const;
    /// Return the inverse of the view matrix
    [[nodiscard]] glm::mat4 inv_view() const;
    /// Return the intrinsic parameters matrix
    [[nodiscard]] glm::mat3 K() const;
    /// Return the inverse of the intrinsic parameters matrix
    [[nodiscard]] glm::mat3 inv_K() const;

    [[nodiscard]] const float* viewmatrix_d() const { return thrust::raw_pointer_cast(m_viewmatrix.data()); }
    [[nodiscard]] const float* projmatrix_d() const { return thrust::raw_pointer_cast(m_projmatrix.data()); }
    [[nodiscard]] const float* campos_d() const { return thrust::raw_pointer_cast(m_campos.data()); }

    void set_resolution(int width, int height);

    /// Update computed values such as view and projection matrices
    void update();
};
} // namespace gs_train
