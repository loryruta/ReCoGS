#include "MainScreen.h"

#include <imgui.h>

#include "App.h"
#include "GSRasterizer.h"
#include "ui/SelectScreen.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_save.h"
#include "utils/imgui_utils.h"

using namespace gs_train;

MainScreen::MainScreen(App& app, std::optional<GSCamera> initial_view) : m_app(app)
{
    // Init camera
    if (initial_view) {
        m_camera = *initial_view;
    } else if (!app.cameras().empty()) {
        m_camera = app.cameras().at(18);
    } else {
        m_camera = GSCamera{};
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

    // Init training cameras image slider
    size_t num_images = m_app.cameras().size();
    m_training_cameras_ui.image_slider =
        std::make_unique<ImageSlider>(num_images, k_training_cameras_ui_image_resolution);
    m_training_cameras_ui.image_slider->on_image_click = [&](int index) {
        printf("[DEBUG] [MainScreen] Setting view to training camera %d\n", index);
        m_camera = m_app.cameras().at(index);
        glm::ivec2 resolution = m_app.window().framebuffer_size();
        m_camera.set_resolution(resolution.x, resolution.y);
        m_camera.update(m_app.stream());
    };
    CHECK_CUDA(cudaStreamCreate(&m_training_cameras_ui.stream)); // Maybe a custom stream is not needed...
}

MainScreen::~MainScreen()
{
    CHECK_CUDA(cudaStreamDestroy(m_training_cameras_ui.stream));

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
    if (m_ui_stereo_test.capture == ui::StereoTest::Capture_HORIZONTAL) {
        printf("[DEBUG] [MainScreen] Capturing horizontal stereo...\n");
        m_app.stereo_depth_estimator().estimate_single_axis(
            m_camera, StereoDepthEstimator::Axis::H, 0.07f, out_color_depth, stream);
    } else if (m_ui_stereo_test.capture == ui::StereoTest::Capture_VERTICAL) {
        printf("[DEBUG] [MainScreen] Capturing vertical stereo...\n");
        m_app.stereo_depth_estimator().estimate_single_axis(
            m_camera, StereoDepthEstimator::Axis::V, 0.07f, out_color_depth, stream);
    } else if (m_ui_stereo_test.capture == ui::StereoTest::Capture_HV) {
        printf("[DEBUG] [MainScreen] Capturing horizontal/vertical stereo...\n");
        m_app.stereo_depth_estimator().estimate_hv(m_camera, 0.07f, out_color_depth, stream);
    }
    if (m_ui_stereo_test.capture != ui::StereoTest::Capture_NONE) { // Only once and display it
        m_ui_stereo_test.current_capture = m_ui_stereo_test.capture;
        m_ui_stereo_test.capture = ui::StereoTest::Capture_NONE;
        m_app.show_depth = true;
    }

    // Update visible training cameras
    if (m_app.ui_enabled) {
        bool update = false;
        update |= m_training_cameras_ui.image_slider->start_texture_index() != m_training_cameras_ui.start_index;
        update |= m_training_cameras_ui.image_slider->end_texture_index() != m_training_cameras_ui.end_index;
        if (update) {
            render_training_cameras();
        }
    }
}

void MainScreen::ui()
{
    m_camera_controller->ui();

    glm::vec2 resolution = m_app.window().framebuffer_size();

    m_ui_stereo_test.ui();

    if (ImGui::Begin("Training cameras",
                     nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {
        ImVec2 window_size = ImGui::GetWindowSize();
        ImGui::SetWindowPos(ImVec2(0, resolution.y - window_size.y));
        ImGui::SetWindowSize(ImVec2(resolution.x, resolution.y * 0.22f));

        m_training_cameras_ui.image_slider->ui();
    }
    ImGui::End();
}

void MainScreen::render_training_cameras()
{
    cudaStream_t stream = m_training_cameras_ui.stream;

    ImageSlider& image_slider = *m_training_cameras_ui.image_slider;
    const int si = image_slider.start_texture_index();
    const int ei = image_slider.end_texture_index();

    // printf("[DEBUG] [MainScreen] Re-rendering training cameras from %d to %d...\n", si, ei);

    const int resolution = k_training_cameras_ui_image_resolution;

    Image4fCHW image = Image4fCHW::malloc(resolution, resolution, stream);
    image_fill(image, glm::vec4(1), stream);

    int j = 0;
    for (int i = si; i < ei; ++i) {
        // Alloc GL-mapped texture if needed
        if (j >= m_training_cameras_ui.gl_mapped_resources.size()) {
            m_training_cameras_ui.gl_mapped_resources.emplace_back(resolution, resolution);
            printf("[DEBUG] [MainScreen] Allocating GL-mapped resource for %d-th camera (%d)\n", j, i);
        }
        GLMappedResource& gl_mapped_texture = m_training_cameras_ui.gl_mapped_resources.at(j);
        // Alloc colorbuffer if needed
        if (j >= m_training_cameras_ui.color_depth.size()) {
            m_training_cameras_ui.color_depth.emplace_back(Image4fHWC::malloc(resolution, resolution, stream));
            printf("[DEBUG] [MainScreen] Allocating colorbuffer for %d-th camera (%d)\n", j, i);
        }
        Image4fHWC& color_depth = m_training_cameras_ui.color_depth.at(j);
        // Set the correct texture at the image slider slot
        m_training_cameras_ui.image_slider->texture(i) = gl_mapped_texture.texture();
        // Get training camera
        GSCamera training_camera = m_app.cameras().at(i);
        training_camera.set_resolution(resolution, resolution);
        training_camera.update(stream);
        // Render GS scene
        m_app.gs_rasterizer().forward(m_app.background_d(), m_app.scene(), false, training_camera, color_depth, stream);
        // Write to GL-mapped texture
        gl_mapped_texture.write(color_depth, stream);
        ++j;
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    m_training_cameras_ui.start_index = si;
    m_training_cameras_ui.end_index = ei;
}
