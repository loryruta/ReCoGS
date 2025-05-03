#include "MainScreen.h"

#include <imgui.h>

#include "App.h"
#include "GSRasterizer.h"
#include "gui/Header.h"
#include "ui/SelectScreen.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_save.h"
#include "utils/imgui_utils.h"

using namespace recogs;

MainScreen::MainScreen(App& app, std::optional<GSCamera> initial_view) : m_app(app)
{
    // Init camera
    if (initial_view) {
        m_camera = std::move(*initial_view);
    } else if (!app.cameras().empty()) {
        m_camera.copy(app.cameras().at(18), m_app.stream());
    } else {
        m_camera.copy(GSCamera{}, m_app.stream());
    }
    m_camera_controller = std::make_unique<GSCameraController>(m_app.window(), m_camera);

    // Action listener
    Window& window = m_app.window();
    m_key_callback = window.add_key_callback([&](int key, int scancode, int action, int mods) {
        if (imgui_want_ui_interaction()) return;

        if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
            m_view_selection = !m_view_selection;
        }

        if (key == GLFW_KEY_H && action == GLFW_PRESS) {
            window.set_cursor_mode(GLFW_CURSOR_NORMAL);
            m_app.set_screen(std::make_shared<SelectScreen>(m_app, m_camera));
        }
    });

    // Init training cameras UI
    m_training_cameras_ui = std::make_unique<ui::TrainingCamerasSlider>(m_app, k_training_cameras_preview_resolution);
    m_training_cameras_ui->on_select = [this](int i) {
        printf("[DEBUG] [MainScreen] Setting view to training camera: %d\n", i);
        glm::ivec2 resolution = m_app.window().framebuffer_size();
        m_camera.copy(m_app.cameras().at(i), m_app.stream());
        m_camera.set_resolution(resolution.x, resolution.y);
        m_camera.update(m_app.stream());
    };
}

MainScreen::~MainScreen()
{
    Window& window = m_app.window();
    CHECK_STATE(window.remove_key_callback(m_key_callback));
}

void MainScreen::resize(int width, int height)
{
    printf("[DEBUG] [MainScreen] Resized window to (%d, %d)\n", width, height);

    m_camera.set_resolution(width, height);
    m_camera.update(m_app.stream());
}

void MainScreen::update(float dt)
{
    if (!imgui_want_ui_interaction()) {
        bool updated = m_camera_controller->update(dt);
        if (updated) {
            m_camera.update(m_app.stream());
            if (m_ui_stereo_test.current_capture) {
                m_ui_stereo_test.current_capture = 0;
                m_app.show_depth = false;
            }
        }
    }
}

void MainScreen::render(Image4fHWC& out_color_depth)
{
    Scene& scene = m_app.scene();
    cudaStream_t stream = m_app.stream();

    if (m_ui_stereo_test.current_capture == ui::StereoTest::Capture_NONE) {
        // Clear depth
        image_fill(out_color_depth, glm::vec4(1, 0, 0, INFINITY), stream);
        // Render 3DGS scene
        m_app.gs_rasterizer().forward(
            m_app.background_d(), scene, true /* scene_2 */, m_camera, out_color_depth, stream);
        // Render selection
        if (m_view_selection) {
            m_app.selection_renderer().render(m_app.selection3d(), m_camera, out_color_depth, stream);
        }
    }

    // Capture stereo
    if (m_ui_stereo_test.capture != ui::StereoTest::Capture_NONE)
    {
        StereoDepthEstimatorParams stereo_params;
        stereo_params.background_d = m_app.background_d();
        stereo_params.scene = &m_app.scene();
        stereo_params.camera = &m_camera;
        stereo_params.rasterizer = &m_app.gs_rasterizer();
        stereo_params.b = 0.07f;
        stereo_params.inout_color_depth = &out_color_depth;
        stereo_params.stream = stream;
        stereo_params.debug = false;

        if (m_ui_stereo_test.capture == ui::StereoTest::Capture_HORIZONTAL) {
            stereo_params.axis = 0;
            m_app.stereo_depth_estimator().estimate_single_axis(stereo_params);
        } else if (m_ui_stereo_test.capture == ui::StereoTest::Capture_VERTICAL) {
            stereo_params.axis = 1;
            m_app.stereo_depth_estimator().estimate_single_axis(stereo_params);
        } else if (m_ui_stereo_test.capture == ui::StereoTest::Capture_HV) {
            m_app.stereo_depth_estimator().estimate_hv(stereo_params);
        }
    }

    if (m_ui_stereo_test.capture != ui::StereoTest::Capture_NONE) { // Only once and display it
        m_ui_stereo_test.current_capture = m_ui_stereo_test.capture;
        m_ui_stereo_test.capture = ui::StereoTest::Capture_NONE;
        m_app.show_depth = true;
    }
}

void MainScreen::ui()
{
    constexpr static int k_footer_height = 70;

    m_camera_controller->ui();

    glm::vec2 resolution = m_app.window().framebuffer_size();

    // Header
    ui::Header header;
    header.height = 70;
    header.section("Depth Test", [this]() { m_ui_stereo_test.ui(); });
    header.section("View Settings", [this]() {
        GSRasterizer& rasterizer = m_app.gs_rasterizer();
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

void MainScreen::render_training_cameras() {}
