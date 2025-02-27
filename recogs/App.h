#pragma once

#include <filesystem>

#include "GSRasterizer.h"
#include "Scene.h"
#include "depth_estimators/StereoDepthEstimator.h"
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

    GSRasterizer m_gs_rasterizer;
    std::unique_ptr<StereoDepthEstimator> m_stereo_depth_estimator;

    std::unique_ptr<Screen> m_screen;

public:
    explicit App(const Params& params);
    ~App();

    [[nodiscard]] Scene& scene() const { return *m_scene; }
    [[nodiscard]] const float* background_d() const { return m_scene_background.data_ptr<float>(); }

    [[nodiscard]] Window& window() const { return *m_window; }
    [[nodiscard]] glm::ivec2 resolution() const { return m_window->framebuffer_size(); }

    [[nodiscard]] GSRasterizer& gs_rasterizer() { return m_gs_rasterizer; }
    [[nodiscard]] StereoDepthEstimator& stereo_depth_estimator() { return *m_stereo_depth_estimator; }

    void start();
    void stop();

private:
    void resize_screenbuffers(int width, int height);
};
} // namespace gs_train
