#include "App.h"

#include <optional>

#include <fmt/format.h>

#include "scene_io.h"
#include "ui/MainScreen.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_save.h"
#include "utils/str_utils.h"

using namespace gs_train;

void App::Params::validate() const
{
    CHECK_ARG(std::filesystem::is_regular_file(scene_ply), "Invalid scene PLY: %s", scene_ply.c_str());
    CHECK_ARG(scene_ply.extension() == ".ply", "Invalid scene PLY extension: %s", scene_ply.extension());
}

App::App(const Params& params)
{
    params.validate();

    m_scene_ply = params.scene_ply;
    m_scene_folder = m_scene_ply.parent_path().parent_path().parent_path();

    /* Load scene/background */
    m_scene = std::make_unique<Scene>(read_scene_from_ply(params.scene_ply));
    m_scene->prepare_for_training();
    std::string bytes_str = num_bytes_to_string(m_scene->num_bytes());
    printf("[INFO ] [App] Scene \"%s\" ready; Size: %s\n", m_scene_ply.filename().c_str(), bytes_str.c_str());

    float background[]{0.0f, 0.0f, 0.0f, 0.0f};
    m_scene_background.fit_data(background, std::size(background));

    m_selection3d = std::make_unique<Selection3d>(*this);

    /* Init window */
    m_window = std::make_unique<Window>(1080, 720, "ReCoGS", false /* resizable */);
    m_window->make_context();

    CHECK_CUDA(cudaStreamCreate(&m_stream));

    // Load cameras
    m_training_cameras = read_cameras_from_json(m_scene_folder, m_stream);

    /* Init screenbuffers */
    glm::ivec2 resolution = m_window->framebuffer_size();
    resize_screenbuffers(resolution.x, resolution.y);

    m_draw_texture = std::make_unique<DrawTexture>();

    m_gs_rasterizer = std::make_unique<GSRasterizer>();
    m_selection_renderer = std::make_unique<SelectionRenderer>();

    {
        StereoDepthEstimator::Options options{};
        m_stereo_depth_estimator = std::make_unique<StereoDepthEstimator>(*this, options);
    }

    m_window->add_key_callback([this](int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
            m_take_screenshot = true;
        }
    });
    m_window->add_resize_callback([this](int width, int height) { resize_screenbuffers(width, height); });

    set_screen(std::make_shared<MainScreen>(*this));

    m_optimizer = std::make_unique<Optimizer>(*this);
    m_optimizer_thread = std::make_unique<std::thread>([this]() { m_optimizer->start(); });
}

App::~App()
{
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaStreamDestroy(m_stream));

    // Stop the optimizer
    m_optimizer->signal_stop();
    m_optimizer_thread->join();
}

void App::start()
{
    std::optional<double> last_t;
    int frame_counter = 0;
    double last_fps_t = 0.0;

    while (!m_window->should_close()) {
        m_window->poll_events();

        ++frame_counter;

        glm::ivec2 fb_size = m_window->framebuffer_size();

        /* FPS */
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

        /* Inter-frame time */
        float dt = 0.f;
        double t = glfwGetTime();
        if (last_t.has_value()) {
            dt = (float) (t - last_t.value());
        }
        last_t = t;

        // Update
        if (m_screen) m_screen->update(dt);

        // Render
        if (m_screen) {
            auto colorbuffer_3hw =
                Image3fCHW::ref(m_colorbuffer_chw->width, m_colorbuffer_chw->height, m_colorbuffer_chw->data_d());
            m_screen->render(colorbuffer_3hw);
            // Take screenshot
            if (m_take_screenshot) {
                save_screenshot(colorbuffer_3hw);
                m_take_screenshot = false;
            }
        }

        /* Transit colorbuffer from BCHW to BHWC */
        image_cast(*m_colorbuffer_chw, *m_colorbuffer_hwc, m_stream);

        /* Display */
        m_gl_mapped_resource->write(m_colorbuffer_hwc->data_d(), m_stream);
        CHECK_CUDA(cudaStreamSynchronize(m_stream));

        // Draw OpenGL mapped texture to screen
        glClearColor(0, 0, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        m_draw_texture->draw(m_gl_mapped_resource->texture(), 0, 0, fb_size.x, fb_size.y);

        // Swap buffers
        m_window->swap_buffers();

        // Execute end-of-frame jobs (e.g. screen switch)
        for (auto& job : m_end_of_frame_jobs) job();
        m_end_of_frame_jobs.clear();
    }
}

void App::stop() { m_window->set_should_close(true); }

void App::save_screenshot(const Image3fCHW& colorbuffer)
{
    std::time_t time = std::time(nullptr);
    std::tm local_time = *std::localtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");
    std::filesystem::path screenshot_filepath = fmt::format("screenshot-{}.png", oss.str());
    image_save_png(colorbuffer, screenshot_filepath);
}

void App::resize_screenbuffers(int width, int height)
{
    printf("[DEBUG] [App] Resizing to (%d, %d)\n", width, height);

    m_gl_mapped_resource = std::make_unique<GLMappedResource>(width, height);

    m_colorbuffer_chw = std::make_unique<ColorbufferCHW>(ColorbufferCHW::malloc(width, height));
    m_colorbuffer_hwc = std::make_unique<ColorbufferHWC>(ColorbufferHWC::malloc(width, height));

    if (m_screen) m_screen->resize(width, height);
}
