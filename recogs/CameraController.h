#pragma once

#include <optional>

#include "Camera.h"
#include "video/Window.h"

namespace recogs
{
class CameraController
{
private:
    Window& m_window;
    Camera& m_camera;
    std::optional<glm::dvec2> m_last_cursor_pos;

    int m_mouse_button_callback;
    int m_key_callback;

public:
    explicit CameraController(Window& window, Camera& camera);
    ~CameraController();

    [[nodiscard]] bool is_cursor_captured() const;

    void reset() { m_last_cursor_pos.reset(); }
    /// Update the camera based on mouse/keyboard input of a dt.
    bool update(float dt);
    /// Function that has to be called when drawing UI.
    void ui();
};
} // namespace recogs