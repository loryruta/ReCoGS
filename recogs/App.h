#pragma once

#include <filesystem>
#include <functional>
#include <thread>

#include "GSRasterizer.h"
#include "Scene.h"
#include "TrainingCameraPool.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "disk/DiskRenderer.h"
#include "optimizer/Optimizer.h"
#include "scene_io.h"
#include "selection/Sel3d.h"
#include "svo/SVORenderer.h"
#include "ui/Screen.h"
#include "utils/image/Image.h"
#include "video/DrawTexture.h"
#include "video/GLTextureMapped.h"
#include "video/Window.h"

namespace recogs
{
class App
{
public:
    struct Params {
        std::filesystem::path scene_ply;

        [[nodiscard]] void validate() const;
    };

private:
    std::filesystem::path m_scene_ply;
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

    std::unique_ptr<GSRasterizer> m_gs_rasterizer;
    std::unique_ptr<SVORenderer> m_svo_renderer;
    std::unique_ptr<DiskRenderer> m_disk_renderer;

    std::unique_ptr<StereoDepthEstimator> m_stereo_depth_estimator;

    std::shared_ptr<Screen> m_screen; // Would be a unique_ptr if std::move_only_function

    /// Tasks executed synchronously at the End Of the Frame.
    std::vector<std::function<void()>> m_end_of_frame_jobs;

    std::unique_ptr<Optimizer> m_optimizer;
    std::unique_ptr<std::thread> m_optimizer_thread;

public:
    bool ui_enabled = true;

    explicit App(const Params& params);
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

    [[nodiscard]] StereoDepthEstimator& stereo_depth_estimator() { return *m_stereo_depth_estimator; }

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
} // namespace recogs
