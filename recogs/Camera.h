#pragma once

#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>

#include "utils/DeviceBuffer.h"

namespace recogs
{
// Forward decl
struct CameraData;

/// A class representing a camera compatible with the 3DGS rasterizer from INRIA.
class Camera
{
private:
    /* Computed */
    float m_tan_fovx;
    float m_tan_fovy;
    glm::mat4 m_view_matrix;
    glm::mat4 m_proj_matrix;
    glm::mat4 m_view_proj;
    thrust::device_vector<float> m_viewmatrix;
    thrust::device_vector<float> m_projmatrix;
    thrust::device_vector<float> m_campos;

public:
    /* Extrinsic params */
    /// Camera position in world-space.
    glm::vec3 position{0, 0, 0};
    glm::quat quaternion{0, 0, 0, 1};
    float yaw = 0.f;   ///< Rotation around the world-up axis.
    float pitch = 0.f; ///< Rotation around the right axis.
    /* Intrinsic params */
    float fx = 1159.588073303806f;
    float fy = 1164.6601287484507f;
    int width = 1959;
    int height = 1090;
    float znear = 0.01f;
    float zfar = 1000.0f;

    explicit Camera();
    explicit Camera(const CameraData& data);
    Camera(const Camera&) = delete;
    Camera(Camera&&) noexcept = default;
    ~Camera() = default;

    [[nodiscard]] glm::ivec2 resolution() const { return {width, height}; }
    [[nodiscard]] float tan_fovx() const { return m_tan_fovx; }
    [[nodiscard]] float tan_fovy() const { return m_tan_fovy; }

    /// Return the view matrix (or extrinsic parameters matrix).
    [[nodiscard]] const glm::mat4& viewmatrix() const { return m_view_matrix; }
    [[nodiscard]] const glm::mat4& projmatrix() const { return m_proj_matrix; }
    /// Return the view/projection matrix.
    [[nodiscard]] const glm::mat4& viewproj() const { return m_view_proj; }
    /// Update the camera rotation quaternion from yaw/pitch/roll attributes.
    /// This must be called to apply the rotation from such attributes.
    void update_quaternion_from_yaw_pitch_roll();

    /// Return the inverse of the view matrix.
    [[nodiscard]] glm::mat4 inv_view() const;

    [[nodiscard]] glm::vec3 right() const { return inv_view()[0]; }
    [[nodiscard]] glm::vec3 up() const { return inv_view()[1]; }
    [[nodiscard]] glm::vec3 forward() const { return inv_view()[2]; }

    /// Return the intrinsic parameters matrix.
    [[nodiscard]] glm::mat3 K() const;
    /// Return the inverse of the intrinsic parameters matrix.
    [[nodiscard]] glm::mat3 inv_K() const;

    [[nodiscard]] const float* viewmatrix_d() const { return thrust::raw_pointer_cast(m_viewmatrix.data()); }
    [[nodiscard]] const float* projmatrix_d() const { return thrust::raw_pointer_cast(m_projmatrix.data()); }
    [[nodiscard]] const float* campos_d() const { return thrust::raw_pointer_cast(m_campos.data()); }

    void set_resolution(int width, int height);

    void copy(const CameraData& data, cudaStream_t stream = nullptr);
    /// Copy another camera object into the current one.
    void copy(const Camera& other, cudaStream_t stream = nullptr);
    /// Clone the camera to a new one.
    Camera clone(cudaStream_t stream = nullptr) const;

    /// Check whether the camera parameters were uploaded to GPU.
    [[nodiscard]] bool is_uploaded() const { return !m_viewmatrix.empty(); }
    /// Update computed values such as view and projection matrices.
    void update(cudaStream_t stream);

    void deserialize(nlohmann::json json);
    void log_info() const;

    Camera& operator=(Camera&& other) noexcept;
};
} // namespace recogs
