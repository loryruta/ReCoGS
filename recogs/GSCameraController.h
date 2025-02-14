#pragma once

#include <optional>

#include "GSCamera.h"
#include "video/Window.h"

namespace gs_train
{
class GSCameraController
{
private:
    const std::shared_ptr<Window> m_window;
    GSCamera& m_camera;

    std::optional<glm::dvec2> m_last_cursor_pos;

public:
    explicit GSCameraController(std::shared_ptr<Window> window, GSCamera& camera);
    ~GSCameraController() = default;

    void update(float dt);
};
} // namespace gs_train