#include "App.h"

#include <optional>

#include <fmt/format.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "ScenePointcloud.h"
#include "gui/recogs_style.h"
#include "scene_io.h"
#include "selection/DiskSel3d.h"
#include "ui/DiskRasterizerScreen.h"
#include "ui/MainScreen.h"
#include "utils/image/image_misc.h"
#include "utils/image/image_save.h"
#include "utils/str_utils.h"

using namespace recogs;

namespace
{
void GLAPIENTRY MessageCallback(GLenum source,
                                GLenum type,
                                GLuint id,
                                GLenum severity,
                                GLsizei length,
                                const GLchar* message,
                                const void* userParam)
{
    if (type == GL_DEBUG_TYPE_ERROR) {
        printf("[ERROR] GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
               (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
               type,
               severity,
               message);
    }
}
} // namespace

void App::Params::validate() const
{
    CHECK_ARG(std::filesystem::is_regular_file(scene_ply), "Invalid scene PLY: {}", scene_ply.string());
    CHECK_ARG(scene_ply.extension() == ".ply", "Invalid scene PLY extension: {}", scene_ply.extension().string());
}

App::App(const Params& params)
{
    CHECK_STATE(!g_app, "Only one App can be instanced");
    g_app = this;

    params.validate();

    // Init window
    m_window = std::make_unique<Window>(Window::create(1080, 720, "ReCoGS", false));
    m_window->make_context();
    glfwSwapInterval(0);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, 0);

    m_scene_ply = params.scene_ply;
    m_scene_folder = m_scene_ply.parent_path().parent_path().parent_path();

    // Load scene/background
    m_scene = std::make_unique<Scene>(read_scene_from_ply(params.scene_ply));
    m_scene->prepare_for_training();
    std::string bytes_str = num_bytes_to_string(m_scene->num_bytes());
    printf("[INFO ] [App] Scene \"%s\" ready; Size: %s\n", m_scene_ply.filename().c_str(), bytes_str.c_str());

    float background[]{0.0f, 0.0f, 0.0f, 0.0f};
    m_scene_background.fit_data(background, std::size(background));

    m_sel3d = std::unique_ptr<Sel3d>(new DiskPcdSel3d());

    // Init ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplGlfw_InitForOpenGL(m_window->handle(), true);
    ImGui_ImplOpenGL3_Init();

    ui::apply_style(ImGui::GetStyle());

    // Init CUDA stream
    CHECK_CUDA(cudaStreamCreate(&g_stream));

    // Compute the scene min/max
    m_scene->compute_minmax(g_stream);

    // Load cameras
    m_training_cameras = read_cameras_from_json(m_scene_folder, g_stream);

    // Init screenbuffers
    glm::ivec2 resolution = m_window->framebuffer_size();
    resize_screenbuffers(resolution.x, resolution.y);

    m_draw_texture = std::make_unique<DrawTexture>();

    m_gs_rasterizer = std::make_unique<GSRasterizer>();
    m_svo_renderer = std::make_unique<SVORenderer>();
    m_disk_renderer = std::make_unique<DiskRenderer>();

    m_stereo_depth_estimator = std::make_unique<StereoDepthEstimator>(*this);

    m_window->add_key_callback([this](int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_F1) {
                printf("[DEBUG] [App] UI enabled: %d\n", ui_enabled);
                ui_enabled = !ui_enabled;
            } else if (key == GLFW_KEY_F2) {
                printf("[DEBUG] [App] Screenshot triggered\n");
                m_take_screenshot = true;
            } else if (key == GLFW_KEY_F5) {
                printf("[DEBUG] [App] Show depth: %d\n", show_depth);
                show_depth = !show_depth;
            }
        }
    });
    m_window->add_resize_callback([this](int width, int height) { resize_screenbuffers(width, height); });

    // TODO Scene pointcloud -> octree tests
    // ScenePointcloud scene_pointcloud(*this, "scenepointcloud.bin");
    // scene_pointcloud.generate(*m_scene, m_training_cameras, {1080, 720});

    set_screen(std::make_shared<MainScreen>());

    //    m_optimizer = std::make_unique<Optimizer>(*this);
    //    m_optimizer_thread = std::make_unique<std::thread>([this]() { m_optimizer->start(); });
}

App::~App()
{
    // Stop the optimizer
    m_optimizer->signal_stop();
    m_optimizer_thread->join();

    CHECK_CUDA(cudaStreamSynchronize(g_stream));
    CHECK_CUDA(cudaStreamDestroy(g_stream));

    // Shutdown ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    g_app = nullptr;
}

void App::start()
{
    std::optional<double> last_t;
    int frame_counter = 0;
    double last_fps_t = 0.0;

    while (!m_window->should_close()) {
        Window::poll_events();

        ++frame_counter;

        glm::ivec2 fb_size = m_window->framebuffer_size();

        // FPS
        double fps_t = glfwGetTime();
        double fps_dt = fps_t - last_fps_t;
        if (fps_dt > 1.0) {
            m_fps = double(frame_counter) / (fps_t - last_fps_t);
            char window_title[256];
            sprintf(window_title, "ReCoGS - %02.1f FPS", m_fps);
            m_window->set_title(window_title);
            frame_counter = 0;
            last_fps_t = fps_t;
        }

        // Inter-frame time
        float dt = 0.f;
        double t = glfwGetTime();
        if (last_t.has_value()) {
            dt = (float) (t - last_t.value());
        }
        last_t = t;

        // Update
        if (m_screen) m_screen->update(dt);

        // Render
        if (m_screen) m_screen->render(*m_color_depth);

        // Show depth
        if (show_depth) {
            // TODO depth scale factor in app args
            image_depth_to_rgb_inplace(*m_color_depth, g_stream, 0.5);
        }

        // CUDA colorbuffer -> OpenGL texture
        m_gl_mapped_resource->write(*m_color_depth, g_stream);
        CHECK_CUDA(cudaStreamSynchronize(g_stream));

        // Display OpenGL texture
        glClearColor(0, 0, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        m_draw_texture->draw(m_gl_mapped_resource->texture(), 0, 0, fb_size.x, fb_size.y);

        // ImGui!
        if (ui_enabled) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (m_screen) m_screen->ui();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        // Optionally screenshot framebuffer
        if (m_take_screenshot) {
            save_screenshot();
            m_take_screenshot = false;
        }

        // Swap buffers
        m_window->swap_buffers();

        // Execute end-of-frame jobs (e.g. screen switch)
        for (auto& job : m_end_of_frame_jobs) job();
        m_end_of_frame_jobs.clear();
    }
}

void App::save_screenshot()
{
    glm::ivec2 fb_size = m_window->framebuffer_size();
    int w = fb_size.x;
    int h = fb_size.y;
    // Read framebuffer
    std::vector<uint8_t> pixels(w * h * 3);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    // Flip Y axis
    for (int line = 0; line != h / 2; ++line) {
        std::swap_ranges(pixels.begin() + 3 * w * line,
                         pixels.begin() + 3 * w * (line + 1),
                         pixels.begin() + 3 * w * (h - line - 1));
    }
    // Save screenshot
    std::time_t time = std::time(nullptr);
    std::tm local_time = *std::localtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");
    std::filesystem::path screenshot_filepath = fmt::format("screenshot-{}.png", oss.str());
    stbi_write_png(screenshot_filepath.c_str(), w, h, 3, pixels.data(), w * 3);
}

void App::resize_screenbuffers(int width, int height)
{
    printf("[DEBUG] [App] Resizing to (%d, %d)\n", width, height);

    m_gl_mapped_resource = std::make_unique<GLTextureMapped>(GLTextureMapped::create_rgba32f(width, height));
    m_color_depth = std::make_unique<Image4fHWC>(Image4fHWC::malloc(width, height));
    if (m_screen) {
        m_screen->resize(width, height);
    }
}
