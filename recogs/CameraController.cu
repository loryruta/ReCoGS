#include "CameraController.h"

#include "utils/imgui_utils.h"

using namespace recogs;

CameraController::CameraController(Window& window, Camera& camera) : m_window(window), m_camera(camera)
{
    m_mouse_button_callback = window.add_mouse_button_callback([&](int button, int action, int mods) {
        // The user wants to interact with ImGui, so no capture...
        if (imgui_want_ui_interaction()) return;

        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            window.set_cursor_mode(GLFW_CURSOR_DISABLED);
        }
    });

    m_key_callback = window.add_key_callback([&](int key, int scancode, int action, int mods) {
        if (imgui_want_ui_interaction()) return;

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            if (is_cursor_captured()) {
                window.set_cursor_mode(GLFW_CURSOR_NORMAL);
                m_last_cursor_pos.reset();
            }
        }
    });
}

CameraController::~CameraController()
{
    m_window.remove_mouse_button_callback(m_mouse_button_callback);
    m_window.remove_key_callback(m_key_callback);
}

bool CameraController::is_cursor_captured() const { return m_window.cursor_mode() == GLFW_CURSOR_DISABLED; }

bool CameraController::update(float dt)
{
    // The cursor is not captured! So no update needed!
    if (!is_cursor_captured()) return false;

    bool updated = false;

    // Update position
    const float k_movement_speed = 1.f; // m/s
    float dp = k_movement_speed * dt;
    if (m_window.is_key_pressed(GLFW_KEY_LEFT_CONTROL)) dp *= 10.f;
    glm::vec3 position = m_camera.position;
    if (m_window.is_key_pressed(GLFW_KEY_W)) m_camera.position += m_camera.forward() * dp;
    if (m_window.is_key_pressed(GLFW_KEY_S)) m_camera.position -= m_camera.forward() * dp;
    if (m_window.is_key_pressed(GLFW_KEY_A)) m_camera.position -= m_camera.right() * dp;
    if (m_window.is_key_pressed(GLFW_KEY_D)) m_camera.position += m_camera.right() * dp;
    if (m_window.is_key_pressed(GLFW_KEY_LEFT_SHIFT)) m_camera.position += m_camera.up() * dp;
    if (m_window.is_key_pressed(GLFW_KEY_SPACE)) m_camera.position -= m_camera.up() * dp; // Y is negative
    updated = m_camera.position != position;

    // Update rotation
    const float k_rotation_speed = 0.06f; // rad/s
    glm::dvec2 cur_pos = m_window.cursor_pos();
    if (m_last_cursor_pos) {
        glm::vec2 dcur_pos = cur_pos - *m_last_cursor_pos;
        if (dcur_pos.x != 0 || dcur_pos.y != 0) {
            m_camera.yaw += dcur_pos.x * k_rotation_speed * dt;
            m_camera.pitch += dcur_pos.y * k_rotation_speed * dt;
            m_camera.update_quaternion_from_yaw_pitch_roll();
            updated = true;
        }
    }
    m_last_cursor_pos = cur_pos;

    return updated;
}

void CameraController::ui()
{
    if (is_cursor_captured()) {
        imgui_disable_ui_interaction();
    }
}
