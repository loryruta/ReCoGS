#include "App.h"

#include <optional>

#include "scene_io.h"
#include "ui/MainScreen.h"
#include "utils/image_layout_transition.h"

using namespace gs_train;

void App::Params::validate() const
{
    CHECK_ARG(std::filesystem::is_regular_file(scene_ply), "Invalid scene PLY: %s", scene_ply.c_str());
    CHECK_ARG(scene_ply.extension() == ".ply", "Invalid scene PLY extension: %s", scene_ply.extension());
}

App::App(const Params& params)
{
    params.validate();

    /* Load scene/background */
    m_scene = std::make_unique<Scene>(read_scene_from_ply(params.scene_ply));
    float background[]{0.0f, 0.0f, 0.0f, 0.0f};
    m_scene_background.fit_data(background, std::size(background));

    /* Init window */
    const int k_init_window_w = 1080;
    const int k_init_window_h = 720;
    m_window = std::make_unique<Window>(k_init_window_w, k_init_window_h, "RecoGS", false /* resizable */);
    m_window->make_context();

    /* Init screenbuffers */
    resize_screenbuffers(k_init_window_w, k_init_window_h);

    m_draw_texture = std::make_unique<DrawTexture>();

    m_screen = std::make_unique<MainScreen>(*this);
}

App::~App() {}

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

        /* Update */
        if (m_screen) m_screen->update(dt);

        /* Render */
        if (m_screen) m_screen->render(m_colorbuffer_b4hw.data_ptr<float>());

        /* Transit colorbuffer from BCHW to BHWC */
        transit_image_layout<ImageLayout::BCHW, ImageLayout::BHWC>(
            1,         // B
            4,         // C; Allocated is 4, written by 3DGS is 3; However, alpha is ignored when screen-quad drawing
            fb_size.y, // H
            fb_size.x, // W
            m_colorbuffer_b4hw.data_ptr<float>(),
            m_colorbuffer_bhw4.data_ptr<float>());

        /* Display */
        glClearColor(0, 0, 1, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        m_gl_mapped_resource->write(m_colorbuffer_bhw4.data_ptr<float>());
        m_draw_texture->draw(m_gl_mapped_resource->texture(), 0, 0, fb_size.x, fb_size.y);

        /* */
        m_window->swap_buffers();
    }
}

void App::stop() { m_window->set_should_close(true); }

void App::resize_screenbuffers(int width, int height)
{
    m_gl_mapped_resource = std::make_unique<GLMappedResource>(width, height);

    m_colorbuffer_b4hw.resize(width * height * 4 * sizeof(float));
    m_colorbuffer_bhw4.resize(width * height * 4 * sizeof(float));
}
