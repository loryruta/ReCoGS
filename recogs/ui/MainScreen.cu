#include "MainScreen.h"

#include <imgui.h>

#include "App.h"
#include "GSRasterizer.h"
#include "gui/Header.h"
#include "ui/SelectScreen.h"
#include "utils/image/depthmap_to_normalmap.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_save.h"
#include "utils/imgui_utils.h"

using namespace recogs;

MainScreen::MainScreen(std::optional<GSCamera> initial_view)
{
    // Init camera
    if (initial_view) {
        m_camera = std::move(*initial_view);
    } else if (!g_app->cameras().empty()) {
        m_camera.copy(g_app->cameras().at(18), g_stream);
    } else {
        m_camera.copy(GSCamera{}, g_stream);
    }

    m_camera_controller = std::make_unique<GSCameraController>(g_app->window(), m_camera);

    // Action listener
    Window& window = g_app->window();
    m_key_callback = window.add_key_callback([&](int key, int scancode, int action, int mods) {
        if (imgui_want_ui_interaction()) {
            return;
        }
        if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
            m_view_selection = !m_view_selection;
        }
        if (key == GLFW_KEY_H && action == GLFW_PRESS) {
            window.set_cursor_mode(GLFW_CURSOR_NORMAL);
            g_app->set_screen(std::make_shared<SelectScreen>(m_camera.clone(g_stream)));
        }
    });

    // Init training cameras UI
    m_training_cameras_ui = std::make_unique<ui::TrainingCamerasSlider>(k_training_cameras_preview_resolution);
    m_training_cameras_ui->on_select = [this](int i) {
        printf("[DEBUG] [MainScreen] Setting view to training camera: %d\n", i);
        glm::ivec2 resolution = g_app->window().framebuffer_size();
        m_camera.copy(g_app->cameras().at(i), g_stream);
        m_camera.set_resolution(resolution.x, resolution.y);
        m_camera.update(g_stream);
    };
}

MainScreen::~MainScreen()
{
    Window& window = g_app->window();
    CHECK_STATE(window.remove_key_callback(m_key_callback));
}

void MainScreen::resize(int width, int height)
{
    m_camera.set_resolution(width, height);
    m_camera.update(g_stream);

    m_sel3d_mask = std::make_unique<Image1u8>(Image1u8::malloc(width, height, g_stream));

    printf("[DEBUG] [MainScreen] Screen resized to (%d, %d)\n", width, height);
}

void MainScreen::update(float dt)
{
    if (!imgui_want_ui_interaction()) {
        bool updated = m_camera_controller->update(dt);
        if (updated) {
            m_camera.update(g_stream);
            if (m_ui_stereo_test.current_capture) {
                m_ui_stereo_test.current_capture = 0;
                g_app->show_depth = false;
            }
        }
    }
}

void MainScreen::_render_sel3d(Image4fHWC& color_depth)
{
    // Project the 3D selection to a 2D mask (also performs depth test)
    image_fill(*m_sel3d_mask, glm::vec<1, uint8_t>(0), g_stream);
    g_app->sel3d().project(m_camera, color_depth, *m_sel3d_mask, g_stream);
    // Use the 2D mask to apply the edit
    image_visit(
        color_depth,
        [mask = *m_sel3d_mask] __device__(Image4fHWC & image, int x, int y) {
            if (mask.value(x, y).r) {
                image.set_value(x, y, glm::vec4(1)); // TODO apply edit ??
            }
            return 0; // TODO Fix
        },
        g_stream);
}

void MainScreen::render(Image4fHWC& color_depth)
{
    Scene& scene = g_app->scene();

    if (m_ui_stereo_test.current_capture == ui::StereoTest::Capture_NONE) {
        // Clear depth
        image_fill(color_depth, glm::vec4(1, 0, 0, INFINITY), g_stream);
        // Render 3DGS scene
        g_app->gs_rasterizer().forward(
            g_app->background_d(), scene, true /* scene_2 */, m_camera, color_depth, g_stream);
        // Render selection
        if (m_view_selection) _render_sel3d(color_depth);
    }

    // Capture stereo
    if (m_ui_stereo_test.capture != ui::StereoTest::Capture_NONE) {
        StereoDepthEstimatorParams stereo_params;
        stereo_params.background_d = g_app->background_d();
        stereo_params.scene = &g_app->scene();
        stereo_params.camera = &m_camera;
        stereo_params.rasterizer = &g_app->gs_rasterizer();
        stereo_params.b = 0.07f;
        stereo_params.inout_color_depth = &color_depth;
        stereo_params.stream = g_stream;
        stereo_params.debug = false;

        if (m_ui_stereo_test.capture == ui::StereoTest::Capture_HORIZONTAL) {
            stereo_params.axis = 0;
            g_app->stereo_depth_estimator().estimate_single_axis(stereo_params);
        } else if (m_ui_stereo_test.capture == ui::StereoTest::Capture_VERTICAL) {
            stereo_params.axis = 1;
            g_app->stereo_depth_estimator().estimate_single_axis(stereo_params);
        } else if (m_ui_stereo_test.capture == ui::StereoTest::Capture_HV) {
            g_app->stereo_depth_estimator().estimate_hv(stereo_params);
        }
    }

    if (m_ui_stereo_test.capture != ui::StereoTest::Capture_NONE) { // Only once and display it
        m_ui_stereo_test.current_capture = m_ui_stereo_test.capture;
        m_ui_stereo_test.capture = ui::StereoTest::Capture_NONE;
        g_app->show_depth = true;
    }

    if (m_ui_stereo_test.show_normalmap) {
        depthmap_to_normalmap<true /* Display */>(color_depth, color_depth, g_stream);
    }
}

void MainScreen::ui()
{
    constexpr static int k_footer_height = 70;

    m_camera_controller->ui();

    glm::vec2 resolution = g_app->window().framebuffer_size();

    // Header
    ui::Header header;
    header.height = 70;
    header.section("Depth Test", [this]() { m_ui_stereo_test.ui(); });
    header.section("View Settings", []() {
        GSRasterizer& rasterizer = g_app->gs_rasterizer();
        ImGui::Checkbox("Show borders", &rasterizer.show_borders);
        ImGui::SliderFloat("Border size", &rasterizer.border_size, 0.001f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
    });
    header.ui();

    // Footer
    if (ImGui::Begin(
            "##footer", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {
        ImGui::SetWindowSize(ImVec2(resolution.x, k_footer_height));
        ImGui::SetWindowPos(ImVec2(0, resolution.y - k_footer_height));

        m_training_cameras_ui->ui();
    }
    ImGui::End();
}
