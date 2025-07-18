#pragma once

#include "Camera.h"
#include "CameraController.h"
#include "Screen.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "selection/Sel2d.h"
#include "utils/image/Image.h"
#include "video/CudaTexture.h"

namespace recogs
{
// Forward decl
class App;

// Forward decl
class SelectScreen;

///
struct SelectScreen_Toolbar {
    SelectScreen& m_parent;

    GLuint m_select_texture;
    GLuint m_paintbrush_texture;
    GLuint m_eraser_texture;

    explicit SelectScreen_Toolbar(SelectScreen& parent);
    ~SelectScreen_Toolbar();

    void ui();
};

///
class SelectScreen : public Screen
{
    friend struct SelectScreen_Toolbar;

private:
    static constexpr float k_select_drag_speed = 0.1f;  // Drag speed applied in normalized space
    static constexpr float k_select_zoom_speed = 0.07f; // Scale speed applied in normalized  space

    Camera m_camera;
    std::unique_ptr<Image4fHWC> m_color_depth;

    std::unique_ptr<Image4fHWC> m_camera_colorbuffer;
    std::unique_ptr<CudaTexture> m_camera_texture;
    std::unique_ptr<CudaTexture> m_cuda_texture;

    glm::vec2 m_camera_offset{}; ///< Offset of the view pivoted at the center
    float m_camera_scale = 1.0f; ///< Scale of the view

    int m_key_callback = -1;
    int m_scroll_callback = -1;

    std::unique_ptr<Sel2d> m_sel2d;

    std::optional<glm::dvec2> m_last_cursor_pos_l{}; /// Cursor position of the previous frame
    std::optional<glm::dvec2> m_last_cursor_pos_r{}; /// Cursor position of the previous frame

    std::unique_ptr<SelectScreen_Toolbar> m_toolbar;

    enum class Mode { BRUSH, ERASE } m_mode = Mode::BRUSH;

public:
    explicit SelectScreen(Camera camera);
    ~SelectScreen() override;

    [[nodiscard]] const char* name() const override { return "SelectScreen"; }

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image4fHWC& out_color_depth) override;
    void ui() override;

    /// Render the 3DGS scene and an accurate depthmap for it (e.g. using Stereo Matching).
    void _render_scene_and_accurate_depth();

    /// Render the 3DGS scene together with the 2D selection, and 3D selection, then blit the colorbuffer to the camera
    /// texture, used for displaying.
    CudaTexture& _render_camera_texture();

    void _on_key(int key, int scancode, int action, int mods);
    void _on_scroll(double xoffset, double yoffset);
};
} // namespace recogs