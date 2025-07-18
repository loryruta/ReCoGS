#pragma once

#include "Camera.h"
#include "CameraController.h"
#include "Screen.h"
#include "TrainingCameraPool.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "gui/StereoTest.h"
#include "gui/TrainingCamerasSlider.h"
#include "utils/image/Image.h"
#include "video/GLTextureMapped.h"
#include "video/ImageSlider.h"

namespace recogs
{
// Forward decl
class App;

/// Main screen when the app opens:
/// allow the user to navigate the 3DGS scene and visualize the selection pointcloud if any
class MainScreen : public Screen
{
    friend class SelectScreen;

private:
    static constexpr int k_training_cameras_preview_resolution = 64;

    Camera m_camera{};
    std::unique_ptr<CameraController> m_camera_controller;

    int m_key_callback = -1;

    bool m_view_selection = false;

    ui::StereoTest m_ui_stereo_test;
    std::unique_ptr<ui::TrainingCamerasSlider> m_training_cameras_ui;

    std::unique_ptr<Image1u8> m_sel3d_mask;

    enum RenderDiskMode : int {
        HashedDiskId = 0, ///< Render the disks of different colors according to their ID.
        Mask,             ///< Render the translucent region covered by disks in white.
        Tint,             ///< Apply a color tint to the region covered by disks.
    } m_render_disk_mode = RenderDiskMode::Mask;

    std::unique_ptr<TrainingCameraPool> m_training_camera_cache;

    int m_selected_training_camera_idx = -1;


public:
    explicit MainScreen(std::optional<Camera> initial_view = std::nullopt);
    ~MainScreen() override;

    [[nodiscard]] const char* name() const override { return "MainScreen"; }

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image4fHWC& color_depth) override;
    void ui() override;

    void _render_sel3d(Image4fHWC& color_depth);
    void _render_disk_sel3d(Image4fHWC& color_depth);
    void _render_user_camera(Image4fHWC& color_depth);
    void _render_training_camera(int camera_idx, Image4fHWC& color_depth);

private:
    void ui_3d_selection();

    void _add_depth_epsilon(Image4fHWC& color_depth, float epsilon);
    void render_training_camera_image(const Camera& camera, Image4fHWC& out_image, cudaStream_t stream);
};
} // namespace recogs