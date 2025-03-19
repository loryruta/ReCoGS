#include "MainScreen.h"

#include "App.h"
#include "GSRasterizer.h"
#include "ui/SelectScreen.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_misc.h"

using namespace gs_train;

MainScreen::MainScreen(App& app, std::optional<GSCamera> initial_view) : m_app(app)
{
    /* Init camera */
    if (initial_view) {
        m_camera = *initial_view;
    } else if (!app.cameras().empty()) {
        m_camera = app.cameras().at(18);
    } else {
        m_camera = GSCamera{};
    }

    /* Action listener */
    Window& window = m_app.window();
    m_key_callback = window.add_key_callback([&](int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            if (window.cursor_mode() == GLFW_CURSOR_DISABLED) {
                window.set_cursor_mode(GLFW_CURSOR_NORMAL);
                m_camera_controller.reset();
            } else if (window.cursor_mode() == GLFW_CURSOR_NORMAL) {
                m_app.stop();
            }
        }

        if (key == GLFW_KEY_F5) {
            // Rasterizer depth
            if (action == GLFW_PRESS) {
                m_view_mode = ViewMode::GSRASTERIZER_DEPTH;
            } else if (action == GLFW_RELEASE) {
                m_view_mode = ViewMode::GSRASTERIZER_COLOR;
            }
        } else if (key == GLFW_KEY_F6) {
            // Horizontal stereo
            if (action == GLFW_PRESS) {
                image_fill(*m_depthbuffer, Image1fCHW::Value{INFINITY});
                m_app.stereo_depth_estimator().estimate_single_axis(
                    m_camera, StereoDepthEstimator::Axis::H, 0.07f, *m_depthbuffer, m_app.stream());
                m_view_mode = ViewMode::STEREO_H;
            } else if (action == GLFW_RELEASE) {
                m_view_mode = ViewMode::GSRASTERIZER_COLOR;
            }
        } else if (key == GLFW_KEY_F7) {
            // Vertical stereo
            if (action == GLFW_PRESS) {
                image_fill(*m_depthbuffer, Image1fCHW::Value{INFINITY});
                m_app.stereo_depth_estimator().estimate_single_axis(
                    m_camera, StereoDepthEstimator::Axis::V, 0.07f, *m_depthbuffer, m_app.stream());
                m_view_mode = ViewMode::STEREO_V;
            } else if (action == GLFW_RELEASE) {
                m_view_mode = ViewMode::GSRASTERIZER_COLOR;
            }
        } else if (key == GLFW_KEY_F8) {
            // Horizontal/vertical stereo
            if (action == GLFW_PRESS) {
                image_fill(*m_depthbuffer, Image1fCHW::Value{INFINITY});
                m_app.stereo_depth_estimator().estimate_hv(m_camera, 0.07f, *m_depthbuffer, m_app.stream());
                m_view_mode = ViewMode::STEREO_HV;
            } else if (action == GLFW_RELEASE) {
                m_view_mode = ViewMode::GSRASTERIZER_COLOR;
            }
        }

        if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
            m_view_selection = !m_view_selection;
        }

        if (key == GLFW_KEY_H && action == GLFW_PRESS) {
            window.set_cursor_mode(GLFW_CURSOR_NORMAL);
            m_app.set_screen(std::make_shared<SelectScreen>(m_app, m_camera));
        }
    });
    m_mouse_button_callback = window.add_mouse_button_callback([&](int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            window.set_cursor_mode(GLFW_CURSOR_DISABLED);
            m_camera_controller = std::make_unique<GSCameraController>(app.window(), m_camera);
        }
    });
}

MainScreen::~MainScreen()
{
    Window& window = m_app.window();
    CHECK_STATE(window.remove_key_callback(m_key_callback));
    CHECK_STATE(window.remove_mouse_button_callback(m_mouse_button_callback));
}

void MainScreen::resize(int width, int height)
{
    printf("[DEBUG] [MainScreen] Resized window to (%d, %d)\n", width, height);

    m_camera.set_resolution(width, height);
    m_camera.update(CU_STREAM_LEGACY);

    m_depthbuffer = std::make_unique<Image1fCHW>(Image1fCHW::malloc(width, height));
}

void MainScreen::update(float dt)
{
    if (m_camera_controller) {
        if (m_view_mode == GSRASTERIZER_COLOR || m_view_mode == GSRASTERIZER_DEPTH) {
            bool updated = m_camera_controller->update(dt);
            if (updated) {
                m_camera.update(m_app.stream());
            }
        }
    }
}

void MainScreen::render(Image3fCHW& out_colorbuffer)
{
    Scene& scene = m_app.scene();
    switch (m_view_mode) {
    case GSRASTERIZER_COLOR:
        // Render the scene
        m_app.gs_rasterizer().forward( //
            m_app.background_d(),
            scene,
            true /* scene_2 */,
            m_camera,
            out_colorbuffer,
            *m_depthbuffer,
            m_app.stream());
        // Render the 3D selection
        if (m_view_selection) {
            m_app.selection_renderer().render(
                m_app.selection3d(), m_camera, out_colorbuffer, *m_depthbuffer, m_app.stream());
        }
        break;
    case GSRASTERIZER_DEPTH:
        m_app.gs_rasterizer().forward(
            m_app.background_d(), scene, true /* scene_2 */, m_camera, out_colorbuffer, *m_depthbuffer, m_app.stream());
        image_depthbuffer_to_rgb(*m_depthbuffer, out_colorbuffer, m_app.stream());
        break;
    case STEREO_H:
    case STEREO_V:
    case STEREO_HV:
        image_depthbuffer_to_rgb(*m_depthbuffer, out_colorbuffer, m_app.stream());
        break;
    }
}
