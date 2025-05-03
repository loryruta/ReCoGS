#pragma once

#include <memory>

#include "utils/image/Image.h"
#include "video/GLMappedResource.h"
#include "video/ImageSlider.h"

namespace recogs
{
// Forward decl
class App;

namespace ui
{
class TrainingCamerasSlider
{
private:
    App& m_app;
    const int m_resolution;

    std::unique_ptr<Image4fHWC> m_colorbuffer;
    std::vector<GLMappedResource> m_gl_mapped_textures;
    std::unique_ptr<ImageSlider> m_image_slider;

public:
    /// Function called when a training camera is selected.
    std::function<void(int)> on_select;

    explicit TrainingCamerasSlider(App& app, int resolution);
    ~TrainingCamerasSlider() = default;

    void ui();
};
} // namespace ui
} // namespace recogs
