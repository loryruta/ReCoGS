#pragma once

#include "GSCamera.h"
#include "GSCameraController.h"
#include "Screen.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "gui/StereoTest.h"
#include "gui/TrainingCamerasSlider.h"
#include "utils/image/Image.h"
#include "video/GLMappedResource.h"
#include "video/ImageSlider.h"

namespace recogs
{
// Forward decl
class App;

/// Main screen when the app opens:
/// allow the user to navigate the 3DGS scene and visualize the selection pointcloud if any
class MainScreen : public Screen
{
private:
    static constexpr int k_training_cameras_preview_resolution = 64;

    App& m_app;

    GSCamera m_camera{};
    std::unique_ptr<GSCameraController> m_camera_controller;

    int m_key_callback = -1;

    bool m_view_selection = false;

    ui::StereoTest m_ui_stereo_test;
    std::unique_ptr<ui::TrainingCamerasSlider> m_training_cameras_ui;

public:
    explicit MainScreen(App& app, std::optional<GSCamera> initial_view = std::nullopt);
    ~MainScreen() override;

    [[nodiscard]] const char* name() const override { return "MainScreen"; }

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image4fHWC& out_color_depth) override;
    void ui() override;

private:
    void render_training_cameras();
};
} // namespace recogs