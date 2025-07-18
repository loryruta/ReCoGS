#include "Camera.h"

#include <glm/gtc/type_ptr.hpp>

#include "scene_io.h"

using namespace recogs;

Camera::Camera()
{
    m_viewmatrix.resize(16);
    m_projmatrix.resize(16);
    m_campos.resize(3);
}

Camera::Camera(const CameraData& data)
    : position(data.position), quaternion(data.rotation), fx(data.fx), fy(data.fy), width(data.width), height(data.height)
{
    m_viewmatrix.resize(16);
    m_projmatrix.resize(16);
    m_campos.resize(3);
}

glm::mat3 Camera::K() const
{
    glm::mat3 K{};
    K[0][0] = fx;
    K[1][1] = fy;
    K[2][2] = 1.0f;
    K[2][0] = float(width) * 0.5f;
    K[2][1] = float(height) * 0.5f;
    return K;
}

void Camera::update_quaternion_from_yaw_pitch_roll()
{
    glm::mat4 orientation = glm::identity<glm::mat4>();
    orientation = glm::rotate(orientation, yaw, glm::vec3(0, 1, 0));
    orientation = glm::rotate(orientation, -pitch, glm::vec3(1, 0, 0));
    quaternion = glm::quat_cast(orientation);
}

glm::mat4 Camera::inv_view() const
{
    glm::mat4 world2cam = glm::mat3_cast(quaternion);
    world2cam[3] = glm::vec4(position, 1.0f);
    return world2cam;
}

glm::mat3 Camera::inv_K() const { return glm::inverse(K()); }

void Camera::set_resolution(int new_width, int new_height)
{
    float aspect = float(new_height) / float(height);
    fx *= aspect;
    fy *= aspect;
    width = new_width;
    height = new_height;
}

void Camera::copy(const CameraData& data, cudaStream_t stream)
{
    position = data.position;
    quaternion = data.rotation;
    fx = data.fx;
    fy = data.fy;
    width = data.width;
    height = data.height;
    if (stream) update(stream);
}

void Camera::copy(const Camera& other, cudaStream_t stream)
{
    position = other.position;
    quaternion = other.quaternion;
    fx = other.fx;
    fy = other.fy;
    width = other.width;
    height = other.height;
    if (stream) update(stream);
}

Camera Camera::clone(cudaStream_t stream) const
{
    Camera cloned;
    cloned.copy(*this, stream);
    return cloned;
}

void Camera::update(cudaStream_t stream)
{
    m_tan_fovx = (float(width) * 0.5f) / fx;
    m_tan_fovy = (float(height) * 0.5f) / fy;

    // View matrix
    m_view_matrix = glm::inverse(inv_view());

    // Projection matrix
    float top = m_tan_fovy * znear;
    float bottom = -top;
    float right = m_tan_fovx * znear;
    float left = -right;
    const float zsign = 1.0f;
    m_proj_matrix = {};
    m_proj_matrix[0][0] = 2.0f * znear / (right - left);
    m_proj_matrix[1][1] = 2.0f * znear / (top - bottom);
    m_proj_matrix[2][0] = (right + left) / (right - left);
    m_proj_matrix[2][1] = (top + bottom) / (top - bottom);
    m_proj_matrix[2][3] = zsign;
    m_proj_matrix[2][2] = zsign * zfar / (zfar - znear);
    m_proj_matrix[3][2] = -(zfar * znear) / (zfar - znear);
    m_view_proj = m_proj_matrix * m_view_matrix;

    CHECK_CUDA(cudaMemcpyAsync(
        RCGS_TPTR(m_viewmatrix), glm::value_ptr(m_view_matrix), 16 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(
        RCGS_TPTR(m_projmatrix), glm::value_ptr(m_view_proj), 16 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(
        RCGS_TPTR(m_campos), glm::value_ptr(position), 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void Camera::deserialize(nlohmann::json json)
{
    width = json["width"];
    height = json["height"];
    // Position
    for (int i = 0; i < 3; ++i) {
        position[i] = json["position"][i];
    }
    // Rotation
    glm::mat3 rot_mat{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rot_mat[c][r] = json["rotation"][r][c];
        }
    }
    quaternion = glm::quat_cast(rot_mat); // Rotation matrix -> Quaternion
    fx = json["fx"];
    fy = json["fy"];
}

void Camera::log_info() const
{
    glm::vec3 forward_ = forward();
    glm::vec3 up_ = up();
    // clang-format off
    printf("[DEBUG] [GSCamera] Position: (%f, %f, %f), Forward: (%.1f, %.1f, %.1f), Up: (%.1f, %.1f, %.1f)\n",
           position.x, position.y, position.z,
           forward_.x, forward_.y, forward_.z,
           up_.x, up_.y, up_.z);
    // clang-format on
}

Camera& Camera::operator=(Camera&& other) noexcept
{
    position = other.position;
    quaternion = other.quaternion;
    fx = other.fx;
    fy = other.fy;
    width = other.width;
    height = other.height;
    // Computed attributes
    m_tan_fovx = other.m_tan_fovx;
    m_tan_fovy = other.m_tan_fovy;
    m_view_matrix = other.m_view_matrix;
    m_view_proj = other.m_view_proj;
    m_viewmatrix = std::move(other.m_viewmatrix);
    m_projmatrix = std::move(other.m_projmatrix);
    m_campos = std::move(other.m_campos);
    return *this;
}
