#pragma once

#include <filesystem>

#include "GSRasterizer.h"
#include "Scene.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "selection/Selection3d.h"
#include "selection/SelectionRenderer.h"
#include "ui/Screen.h"
#include "video/DrawTexture.h"
#include "video/GLMappedResource.h"
#include "video/Window.h"
#include "utils/image/Image.h"

namespace gs_train
{
class App
{
public:
    struct Params {
        std::filesystem::path scene_ply;

        [[nodiscard]] void validate() const;
    };

    /* Scene */
    std::unique_ptr<Scene> m_scene;
    DeviceBuffer m_scene_background{"background"};

    /* UI/Display */
    std::unique_ptr<Window> m_window;
    std::unique_ptr<GLMappedResource> m_gl_mapped_resource{};
    std::unique_ptr<DrawTexture> m_draw_texture{}; // Lazily initialized because needs OpenGL
    using ColorbufferCHW = Image<4, float, ImageMemoryLayout::CHW>;
    using ColorbufferHWC = Image<4, float, ImageMemoryLayout::HWC>;
    std::unique_ptr<ColorbufferCHW> m_colorbuffer_chw;
    std::unique_ptr<ColorbufferHWC> m_colorbuffer_hwc;

    /* Stats */
    double m_fps = 0.0;

    std::unique_ptr<GSRasterizer> m_gs_rasterizer;
    std::unique_ptr<SelectionRenderer> m_selection_renderer;

    std::unique_ptr<StereoDepthEstimator> m_stereo_depth_estimator;

    std::unique_ptr<Screen> m_screen;

public:
    explicit App(const Params& params);
    ~App();

    [[nodiscard]] Scene& scene() const { return *m_scene; }
    [[nodiscard]] const float* background_d() const { return m_scene_background.data_ptr<float>(); }

    [[nodiscard]] Window& window() const { return *m_window; }
    [[nodiscard]] glm::ivec2 resolution() const { return m_window->framebuffer_size(); }

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
            glm::ivec2 resolution_ = resolution();
            m_screen->resize(resolution_.x, resolution_.y);
        });
    }

    void start();
    void stop();

private:
    void resize_screenbuffers(int width, int height);
};
} // namespace gs_train
