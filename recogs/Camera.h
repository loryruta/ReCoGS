#pragma once

#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>

namespace gs_train
{
struct Camera
{
    glm::vec3 position{};
    glm::mat3 rotation = glm::identity<glm::mat3>();
    float fx = 1.0f;
    float fy = 1.0f;
    float width = 1080.0f;
    float height = 720.0f;
    /* Computed */
    glm::mat3 intrinsic_params{};
    float* viewmatrix_d;
    float* projmatrix_d;
    float* campos_d;
    float tan_fovx;
    float tan_fovy;

    [[nodiscard]] glm::vec3 right() const { return rotation[0]; }
    [[nodiscard]] glm::vec3 up() const { return rotation[1]; };
    [[nodiscard]] glm::vec3 forward() const { return rotation[2]; };

    void update();

    void from_json(nlohmann::json data);
};
} // namespace gs_train
