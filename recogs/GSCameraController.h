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

public:
    explicit GSCameraController(Window& window, GSCamera& camera);
    ~GSCameraController() = default;

    void update(float dt);
};
} // namespace gs_train