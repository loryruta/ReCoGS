#include "SelectScreen.h"

#include "App.h"
#include "ui/MainScreen.h"
#include "utils/image/image_fill.h"

using namespace gs_train;

SelectScreen::SelectScreen(App& app, GSCamera view) : m_app(app), m_view(std::move(view))
{
    Window& window = app.window();
    m_key_callback = window.add_key_callback([this](int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ENTER && action == GLFW_PRESS) {
            {
                glm::ivec2 resolution = m_app.resolution();
                // Estimate depth using stereo matching HV
                Image1fCHW depth = Image1fCHW::malloc(resolution.x, resolution.y);
                image_fill(depth, Image1fCHW::Value{INFINITY});
                m_app.stereo_depth_estimator().estimate_hv(m_view, 0.07f, depth, m_app.stream());
                // Populate the 3D selection with 2D selection unprojection
                m_selection2d->populate_selection3d(depth);
            }

            m_app.set_screen(std::make_shared<MainScreen>(m_app, m_view));
        }
    });
    printf("[DEBUG] [SelectScreen] Screen created\n");
}

SelectScreen::~SelectScreen()
{
    Window& window = m_app.window();
    CHECK_STATE(window.remove_key_callback(m_key_callback));
}

void SelectScreen::resize(int width, int height)
{
    m_depthbuffer = std::make_unique<Image1fCHW>(Image1fCHW::malloc(width, height));
    m_selection2d = std::make_unique<Selection2d>(m_app.selection3d(), m_view);
}

void SelectScreen::update(float dt)
{
    const Window& window = m_app.window();
    if (window.is_mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT) ||
        window.is_mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT)) {
        bool is_filling = window.is_mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT);
        glm::dvec2 cursor_pos = window.cursor_pos();
        if (m_last_cursor_pos) {
            m_selection2d->fill_line(*m_last_cursor_pos, cursor_pos, 25 /* radius */);
            m_last_cursor_pos = cursor_pos;
        } else {
            m_last_cursor_pos = cursor_pos;
        }
    } else if (m_last_cursor_pos) {
        m_last_cursor_pos.reset();
    }
}

void SelectScreen::render(Image3fCHW& out_colorbuffer)
{
    cudaStream_t stream = m_app.stream();
    // Render the scene
    m_app.gs_rasterizer().forward(
        m_app.background_d(), m_app.training_scene(), m_view, out_colorbuffer, *m_depthbuffer, stream);
    // Render the 2d selection (3d + current view stroke)
    m_app.selection_renderer().render(*m_selection2d, out_colorbuffer, *m_depthbuffer, stream);
}
