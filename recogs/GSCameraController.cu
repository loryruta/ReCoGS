#include "GSCameraController.h"

using namespace gs_train;

GSCameraController::GSCameraController(std::shared_ptr<Window> window, GSCamera& camera)
    : m_window(std::move(window)), m_camera(camera)
{
}

void GSCameraController::update(float dt)
{
    bool updated = false;

    bool capture_input = m_window->cursor_mode() == GLFW_CURSOR_DISABLED;
    if (!capture_input) {
        if (m_last_cursor_pos) m_last_cursor_pos.reset();
        return;
    }

    /* Update position */
    const float k_movement_speed = 1.f; // m/s
    float dp = k_movement_speed * dt;
    if (m_window->is_key_pressed(GLFW_KEY_LEFT_CONTROL)) dp *= 10.f;
    glm::vec3 position = m_camera.position;
    if (m_window->is_key_pressed(GLFW_KEY_W)) m_camera.position += m_camera.forward() * dp;
    if (m_window->is_key_pressed(GLFW_KEY_S)) m_camera.position -= m_camera.forward() * dp;
    if (m_window->is_key_pressed(GLFW_KEY_A)) m_camera.position -= m_camera.right() * dp;
    if (m_window->is_key_pressed(GLFW_KEY_D)) m_camera.position += m_camera.right() * dp;
    if (m_window->is_key_pressed(GLFW_KEY_LEFT_SHIFT)) m_camera.position += m_camera.up() * dp;
    if (m_window->is_key_pressed(GLFW_KEY_SPACE)) m_camera.position -= m_camera.up() * dp; // Y is negative
    updated = m_camera.position != position;

    /* Update rotation */
    const float k_rotation_speed = 0.06f; // rad/s
    glm::dvec2 cur_pos = m_window->cursor_pos();
    if (m_last_cursor_pos) {
        glm::dvec2 dcur_pos = cur_pos - *m_last_cursor_pos;
        if (dcur_pos.x != 0 || dcur_pos.y != 0) {
            glm::quat drot = glm::identity<glm::quat>();
            drot = glm::rotate(drot, (float) dcur_pos.x * k_rotation_speed * dt, glm::vec3(0, 1, 0));
            drot = glm::rotate(drot, (float) -dcur_pos.y * k_rotation_speed * dt, glm::vec3(1, 0, 0));
            m_camera.rotation = m_camera.rotation * drot;
            updated = true;
        }
    }
    m_last_cursor_pos = cur_pos;

    if (updated) m_camera.update();
}
