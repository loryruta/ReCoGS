
#include <cstdio>
#include <filesystem>

#include "GSCamera.h"
#include "GSCameraController.h"
#include "GSFunc.h"
#include "Scene.h"
#include "scene_io.h"
#include "utils/DeviceBuffer.h"
#include "utils/image_layout_transition.h"
#include "video/DrawTexture.h"
#include "video/GLMappedResource.h"
#include "video/Window.h"

using namespace gs_train;

#define WINDOW_W 1080
#define WINDOW_H 720

int main(int argc, char* argv[])
{
    argc--;
    if (argc != 1) {
        fprintf(stderr, "Invalid syntax: %s <scene-ply>\n", argv[0]);
        return 1;
    }
    ++argv;

    std::filesystem::path scene_ply = std::filesystem::absolute(argv[0]);
    CHECK_STATE(std::filesystem::exists(scene_ply), "Invalid scene ply: %s", argv[0]);

    /* Load scene */
    Scene scene = read_scene_from_ply(scene_ply);
    printf("Scene loaded: %s\n", scene_ply.c_str());

    /* Init visualization */
    auto window = std::make_shared<Window>(WINDOW_W, WINDOW_H, "RecoGS", false /* resizable */);
    window->make_context();

    /* Init camera */
    GSCamera camera{};
    camera.position = {-3.0090f, -0.1109f, -3.7528f};
    camera.rotation =
        glm::quat_cast(glm::transpose(glm::mat3{{0.8761342012188561f, 0.06925962026449778f, 0.47706599800804744f},
                                                {-0.047474218398951024f, 0.9972110940209488f, -0.05758673934988211f},
                                                {-0.4797239414934442f, 0.02780537650095985f, 0.8769787916452907f}}));
    camera.fy = 1164.6601287484507f / (1090 * 0.5f);
    camera.fx = 1159.588073303806f / (1959 * 0.5f);
    camera.width = 1959;
    camera.height = 1090;
    // camera.set_resolution(1080, 720);
    camera.update();

    /* Interactivity */
    bool select_mode = false;
    window->add_key_callback([&](int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
            select_mode = !select_mode;
            if (select_mode) {
                window->set_cursor_mode(GLFW_CURSOR_NORMAL);
            }
        } else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            if (window->cursor_mode() == GLFW_CURSOR_DISABLED) {
                window->set_cursor_mode(GLFW_CURSOR_NORMAL);
            } else if (window->cursor_mode() == GLFW_CURSOR_NORMAL) {
                window->set_should_close(true);
            }
        }
    });
    window->add_mouse_button_callback([&](int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            if (!select_mode) {
                if (window->cursor_mode() == GLFW_CURSOR_NORMAL) {
                    window->set_cursor_mode(GLFW_CURSOR_DISABLED);
                }
            }
        }
    });

    GSCameraController camera_controller(window, camera);

    GSFunc gs_func{};

    DeviceBuffer background = DeviceBuffer::from_data<float>({0.f, 0.f, 0.f, 0.f}, "background");

    DeviceBuffer colorbuffer_bchw = DeviceBuffer::alloc<float>(WINDOW_H * WINDOW_W * 4, "colorbuffer1");
    DeviceBuffer colorbuffer_bhwc = DeviceBuffer::alloc<float>(WINDOW_H * WINDOW_W * 4, "colorbuffer2");

    GLMappedResource gl_mapped_resource(WINDOW_W, WINDOW_H);
    DrawTexture draw_texture{};

    std::optional<double> last_t;
    double fps;
    int frame_counter = 0;
    double last_fps_t = 0.0;

    while (!window->should_close()) {
        ++frame_counter;

        /* FPS */
        double fps_t = glfwGetTime();
        double fps_dt = fps_t - last_fps_t;
        if (fps_dt > 1.0) {
            fps = double(frame_counter) / (fps_t - last_fps_t);
            char window_title[256];
            sprintf(window_title, "ReCoGS - %02.1f FPS", fps);
            glfwSetWindowTitle(window->handle(), window_title);
            frame_counter = 0;
            last_fps_t = fps_t;
        }

        window->poll_events();

        /* Inter-frame time */
        float dt = 0.f;
        double t = glfwGetTime();
        if (last_t.has_value()) {
            dt = (float) (t - last_t.value());
        }
        last_t = t;

        if (dt > 0.f) {
            camera_controller.update(dt);
        }

        /* Render */
        gs_func.forward(WINDOW_W,
                        WINDOW_H,
                        background.data_ptr<float>(),
                        scene.num_vertices,
                        scene.means.data_ptr<float>(),
                        scene.shs.data_ptr<float>(),
                        scene.opacities.data_ptr<float>(),
                        scene.scales.data_ptr<float>(),
                        scene.rotations.data_ptr<float>(),
                        camera.viewmatrix_d(),
                        camera.projmatrix_d(),
                        camera.campos_d(),
                        camera.tan_fovx(),
                        camera.tan_fovy(),
                        colorbuffer_bchw.data_ptr<float>() // (3, H, W)
        );

        /* Transit colorbuffer from BCHW to BHWC */
        transit_image_layout<ImageLayout::BCHW, ImageLayout::BHWC>( //
            1,
            4, // Allocated is 4, written is 3; However, it's ignored when screen-quad drawing
            WINDOW_H,
            WINDOW_W,
            colorbuffer_bchw.data_ptr<float>(),
            colorbuffer_bhwc.data_ptr<float>());

        /* Display */
        gl_mapped_resource.write(colorbuffer_bhwc.data_ptr<float>());
        draw_texture.draw(gl_mapped_resource.texture(), 0, 0, WINDOW_W, WINDOW_H);

        window->swap_buffers();
    }

    return 0;
}
