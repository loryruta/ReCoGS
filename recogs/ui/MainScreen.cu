#include "MainScreen.h"

#include "App.h"
#include "GSRasterizer.h"

using namespace gs_train;

namespace
{
/// Initialize the given camera to the first camera pose of the "train" scene from T&T
void init_camera_to_camera0_train_scene(GSCamera& camera, glm::ivec2 resolution)
{
    camera.position = {-3.0090f, -0.1109f, -3.7528f};
    camera.rotation =
        glm::quat_cast(glm::transpose(glm::mat3{{0.8761342012188561f, 0.06925962026449778f, 0.47706599800804744f},
                                                {-0.047474218398951024f, 0.9972110940209488f, -0.05758673934988211f},
                                                {-0.4797239414934442f, 0.02780537650095985f, 0.8769787916452907f}}));
    float aspect = float(resolution.y) / 1090.0f;
    camera.fx = 1159.588073303806f / (1959 * 0.5f) * aspect;
    camera.fy = 1164.6601287484507f / (1090 * 0.5f) * aspect;
    camera.width = resolution.x;
    camera.height = resolution.y;
    camera.update();
}
} // namespace

MainScreen::MainScreen(App& app) : m_app(app)
{
    /* Init camera */
    init_camera_to_camera0_train_scene(m_camera, app.resolution());

    /* Action listener */
    Window& window = m_app.window();
    m_key_callback_id = window.add_key_callback([&](int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            if (window.cursor_mode() == GLFW_CURSOR_DISABLED) {
                window.set_cursor_mode(GLFW_CURSOR_NORMAL);
                m_camera_controller.reset();
            } else if (window.cursor_mode() == GLFW_CURSOR_NORMAL) {
                m_app.stop();
            }
        }
    });
    m_mouse_button_callback_id = window.add_mouse_button_callback([&](int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            window.set_cursor_mode(GLFW_CURSOR_DISABLED);
            m_camera_controller = std::make_unique<GSCameraController>(app.window(), m_camera);
        }
    });
}

MainScreen::~MainScreen()
{
    Window& window = m_app.window();
    window.remove_key_callback(m_key_callback_id);
    window.remove_mouse_button_callback(m_mouse_button_callback_id);
}

void MainScreen::resize(int width, int height)
{
    m_camera.set_resolution(width, height);
    m_camera.update();
}

void MainScreen::update(float dt)
{
    if (m_camera_controller) m_camera_controller->update(dt);
}

void MainScreen::render(float* out_colorbuffer)
{
    m_app.gs_rasterizer().forward( //
        m_app.background_d(),
        m_app.scene(),
        m_camera,
        out_colorbuffer);
}
