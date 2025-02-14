#pragma once

#include "GSCamera.h"
#include "GSCameraController.h"
#include "Screen.h"

namespace gs_train
{
// Forward decl
class App;

/// Main screen when the app opens:
/// allow the user to navigate the 3DGS scene and visualize the selection pointcloud if any
class MainScreen : public Screen
{
private:
    App& m_app;

    GSCamera m_camera{};
    std::unique_ptr<GSCameraController> m_camera_controller;

    int m_key_callback_id = -1;
    int m_mouse_button_callback_id = -1;

public:
    explicit MainScreen(App& app);
    ~MainScreen();

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(float* out_colorbuffer) override;
};
} // namespace gs_train