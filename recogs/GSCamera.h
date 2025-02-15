#pragma once

#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

#include "utils/DeviceBuffer.h"

namespace gs_train
{
/// A class representing a camera compatible with the 3DGS pipeline
class GSCamera
{
private:
    /* Computed */
    DeviceBuffer m_viewmatrix{"GSCamera/viewmatrix"};
    DeviceBuffer m_projmatrix{"GSCamera/projmatrix"};
    DeviceBuffer m_campos{"GSCamera/campos"};

public:
    glm::vec3 position{};
    glm::quat rotation{};
    float fx = 0.1f;
    float fy = 0.1f;
    int width = 1080;
    int height = 720;

    explicit GSCamera();
    GSCamera(const GSCamera&) = delete; // TODO define
    GSCamera(GSCamera&&) = default;
    ~GSCamera() = default;

    [[nodiscard]] float tan_fovx() const { return 1.0f / fx; }
    [[nodiscard]] float tan_fovy() const { return 1.0f / fy; }

    [[nodiscard]] glm::vec3 right() const { return rotation * glm::vec3(1, 0, 0); }
    [[nodiscard]] glm::vec3 up() const { return rotation * glm::vec3(0, 1, 0); };
    [[nodiscard]] glm::vec3 forward() const { return rotation * glm::vec3(0, 0, 1); };

    [[nodiscard]] const float* viewmatrix_d() const { return m_viewmatrix.data_ptr<float>(); }
    [[nodiscard]] const float* projmatrix_d() const { return m_projmatrix.data_ptr<float>(); }
    [[nodiscard]] const float* campos_d() const { return m_campos.data_ptr<float>(); }

    void set_resolution(int width, int height);

    /// Update computed values such as view and projection matrices
    void update();

    GSCamera& operator=(const GSCamera& other);
};
} // namespace gs_train
