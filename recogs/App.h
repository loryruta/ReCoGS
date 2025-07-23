#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <thread>

#include "scene_io.h"
#include "ui/Screen.h"
#include "utils/image/Image.h"
#include "video/DrawTexture.h"
#include "video/GLTextureMapped.h"
#include "video/Window.h"

BEGIN_NAMESPACE

// Forward decl
class Optimizer;
class DiskRenderer;
class TrainingCameraPool;
class Scene;
class GSRasterizer;
class Sel3d;
class SVORenderer;

struct AppParams {
    std::filesystem::path scene_ply;
    std::string app_title;

    [[nodiscard]] void validate() const;
};

class App
{
private:
    const std::filesystem::path m_scene_ply;
    const std::string m_app_title;
    std::filesystem::path m_scene_folder;

    /* Scene */
    std::unique_ptr<Scene> m_scene;
    DeviceBuffer m_scene_background{"background"};
    std::unique_ptr<Sel3d> m_sel3d;
    std::vector<CameraData> m_training_cameras;

    /* UI/Display */
    std::unique_ptr<Window> m_window;
    std::unique_ptr<GLTextureMapped> m_gl_mapped_resource{};
    std::unique_ptr<DrawTexture> m_draw_texture{}; // Lazily initialized because needs OpenGL
    std::unique_ptr<Image4fHWC> m_color_depth;
    bool m_take_screenshot = false;

    // Stats
    double m_fps = 0.0;

    // ----------------------------------------------------------------
    /* Renderers */
    // ----------------------------------------------------------------

    std::unique_ptr<GSRasterizer> m_gs_rasterizer;
    std::unique_ptr<SVORenderer> m_svo_renderer;
    std::unique_ptr<DiskRenderer> m_disk_renderer;

    std::shared_ptr<Screen> m_screen; // Would be a unique_ptr if std::move_only_function

    /// Tasks executed synchronously at the End Of the Frame.
    std::vector<std::function<void()>> m_end_of_frame_jobs;

    std::unique_ptr<Optimizer> m_optimizer;
    std::unique_ptr<std::thread> m_optimizer_thread;

public:
    bool ui_enabled = true;

    explicit App(const AppParams& params);
    ~App();

    [[nodiscard]] std::filesystem::path const& scene_ply() const { return m_scene_ply; }
    [[nodiscard]] std::filesystem::path const& scene_folder() const { return m_scene_folder; }

    [[nodiscard]] Window& window() const { return *m_window; }
    [[nodiscard]] glm::ivec2 resolution() const { return m_window->framebuffer_size(); }

    [[nodiscard]] Scene& scene() const { return *m_scene; }
    [[nodiscard]] const float* background_d() const { return m_scene_background.data_ptr<float>(); }
    [[nodiscard]] Sel3d& sel3d() const { return *m_sel3d; }
    [[nodiscard]] std::vector<CameraData> const& cameras() const { return m_training_cameras; }

    [[nodiscard]] Image4fHWC& colordepth() const { return *m_color_depth; }

    [[nodiscard]] GSRasterizer& gs_rasterizer() { return *m_gs_rasterizer; }
    [[nodiscard]] SVORenderer& svo_renderer() { return *m_svo_renderer; }
    [[nodiscard]] DiskRenderer& disk_renderer() { return *m_disk_renderer; }

    /// Display a new screen within the application.
    /// This function must only be called by the application thread (i.e. main thread).
    void set_screen(std::shared_ptr<Screen>&& new_screen)
    {
        // Changing the screen can't be executed immediately otherwise the calling screen will delete itself!
        // Therefore, the switch operation is queued at the end of the frame
        m_end_of_frame_jobs.emplace_back([this, new_screen = std::move(new_screen)]() {
            const char* from_name = m_screen ? m_screen->name() : "null";
            const char* to_name = new_screen ? new_screen->name() : "null";
            printf("[INFO ] [App] Switching screen: %s -> %s\n", from_name, to_name);
            // Delete current screen if any
            m_screen.reset();
            // Create and set the new screen
            m_screen = new_screen;
            if (new_screen) {
                glm::ivec2 resolution_ = resolution();
                m_screen->resize(resolution_.x, resolution_.y);
                printf("[DEBUG] [App] Resized %s to (%d, %d)\n", to_name, resolution_.x, resolution_.y);
            }
        });
    }

    /// Enqueue a job on the main thread.
    void enqueue_job(std::function<void()> job) { m_end_of_frame_jobs.emplace_back(job); }

    void start();

private:
    void save_screenshot();
    void resize_screenbuffers(int width, int height);
};

/*
 * Global variables
 *
 * These are variables used a lot in the code that I decided to make global instead of attributes of the App class.
 */

inline App* g_app = nullptr;
/// The stream where all UI -related CUDA operations are done.
/// It must be used instead of the default stream.
inline cudaStream_t g_stream = nullptr;

END_NAMESPACE
