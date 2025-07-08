#include "MainScreen.h"

#include <imgui.h>

#include "App.h"
#include "GSRasterizer.h"
#include "gui/Header.h"
#include "selection/DiskSel3d.h"
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
    printf("[DEBUG] [MainScreen] Bye bye\n");
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

void MainScreen::_render_disk_sel3d(Image4fHWC& color_depth)
{
    DiskPcdSel3d* disk_sel3d = dynamic_cast<DiskPcdSel3d*>(&g_app->sel3d());
    CHECK_ARG(disk_sel3d, "Selection 3D is not a DiskPcdSel3d");
    const DiskBuffer& disk_buffer = disk_sel3d->disk_buffer();
    if (disk_buffer.empty()) return;

    DiskRenderer_Params params{};
    params.camera = &m_camera;
    params.color_depth = &color_depth;
    params.disk_buffer = &disk_buffer;
    params.stream = g_stream;
    DiskRenderer& disk_renderer = g_app->disk_renderer();
    switch (m_render_disk_mode) {
    case RenderDiskMode::HashedDiskId:
        disk_renderer.render(params, [] __device__(uint32_t disk_id, float2 uv, float opacity, float* inout_color) {
            // Reference:
            // https://gist.github.com/mpottinger/54d99732d4831d8137d178b4a6007d1a#file-murmurhash-glsl-L213-L229
            // murmurHash41
            const uint M = 0x5bd1e995u;
            glm::uvec4 h = glm::uvec4(1190494759u, 2147483647u, 3559788179u, 179424673u);
            disk_id *= M;
            disk_id ^= disk_id >> 24u;
            disk_id *= M;
            h *= M;
            h = h ^ disk_id;
            h = h ^ (h >> 13u);
            h *= M;
            h = h ^ (h >> 15u);
            // hash41
            glm::uvec4 enc = h & 0x007fffffu | 0x3f800000u;
            inout_color[0] = __uint_as_float(enc.x) - 1.0f;
            inout_color[1] = __uint_as_float(enc.y) - 1.0f;
            inout_color[2] = __uint_as_float(enc.z) - 1.0f;
            inout_color[2] = 1.0f;
        });
        break;
    case RenderDiskMode::Mask:
        disk_renderer.render(params, [] __device__(uint32_t disk_id, float2 uv, float opacity, float* inout_color) {
            inout_color[0] = sqrtf(uv.x * uv.x + uv.y * uv.y);
            inout_color[1] = 0.f;
            inout_color[2] = 0.f;
            inout_color[3] = 1.f;
        });
        break;
    case RenderDiskMode::Tint:
        disk_renderer.render(params, [] __device__(uint32_t disk_id, float2 uv, float opacity, float* inout_color) {
            inout_color[0] = sqrtf(uv.x * uv.x + uv.y * uv.y);
            inout_color[1] = 0.f;
            inout_color[2] = 0.f;
            inout_color[3] = 1.f;
        });
        break;
    default:
        break;
    }
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
        if (m_view_selection) {
            _render_disk_sel3d(color_depth);
        }
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

    ui_3d_selection();

    // Footer
    if (ImGui::Begin(
            "##footer", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {
        ImGui::SetWindowSize(ImVec2(resolution.x, k_footer_height));
        ImGui::SetWindowPos(ImVec2(0, resolution.y - k_footer_height));

        m_training_cameras_ui->ui();
    }
    ImGui::End();
}

void MainScreen::ui_3d_selection()
{
    if (ImGui::Begin("3D Selection")) {
        ImGui::Checkbox("View 3D selection", &m_view_selection);

        // Render disks
        ImGui::RadioButton("Hashed Disk ID##RenderDisksMode", (int*) &m_render_disk_mode, RenderDiskMode::HashedDiskId);
        ImGui::RadioButton("Mask##RenderDisksMode", (int*) &m_render_disk_mode, RenderDiskMode::Mask);
        ImGui::RadioButton("Tint##RenderDisksMode", (int*) &m_render_disk_mode, RenderDiskMode::Tint);
    }
    ImGui::End();
}
