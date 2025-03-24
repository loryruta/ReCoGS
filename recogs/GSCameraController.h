#pragma once

#include <optional>

#include "GSCamera.h"
#include "video/Window.h"

namespace gs_train
{
class GSCameraController
{
private:
    Window& m_window;
    GSCamera& m_camera;
    std::optional<glm::dvec2> m_last_cursor_pos;

    int m_mouse_button_callback;
    int m_key_callback;

public:
    explicit GSCameraController(Window& window, GSCamera& camera);
    ~GSCameraController();

    [[nodiscard]] bool is_cursor_captured() const;

    void reset() { m_last_cursor_pos.reset(); }
    /// Update the camera based on mouse/keyboard input of a dt.
    bool update(float dt);
    /// Function that has to be called when drawing UI.
    void ui();
};
} // namespace gs_train