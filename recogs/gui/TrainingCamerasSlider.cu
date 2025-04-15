#include "TrainingCamerasSlider.h"

#include <imgui.h>

#include "App.h"

using namespace gs_train;
using namespace gs_train::ui;

TrainingCamerasSlider::TrainingCamerasSlider(App& app, int resolution) : m_app(app), m_resolution(resolution)
{
    size_t N = m_app.cameras().size();

    // Init colorbuffer used as a target for rendering
    m_colorbuffer = std::make_unique<Image4fHWC>(Image4fHWC::malloc(resolution, resolution));

    // Init GL-CUDA mapped textures
    m_gl_mapped_textures.reserve(N);
    for (int i = 0; i < N; ++i) {
        m_gl_mapped_textures.emplace_back(resolution, resolution);
    }

    // Init ImageSlider (imgui utility for displaying them)
    m_image_slider = std::make_unique<ImageSlider>(N);
    m_image_slider->on_image_click = [this](int i) {
        if (on_select) {
            on_select(i);
        }
    };
    m_image_slider->provide_texture = [this](int i) -> GLuint {
        // Render the scene from the camera perspective
        GSCamera camera = m_app.cameras().at(i);
        camera.set_resolution(m_resolution, m_resolution);
        camera.update(m_app.stream());
        GSRasterizer& rasterizer = m_app.gs_rasterizer();
        rasterizer.show_borders = false;
        rasterizer.forward(m_app.background_d(), m_app.scene(), false, camera, *m_colorbuffer, m_app.stream());
        // Copy the colorbuffer into the GL mapped texture
        GLMappedResource& resource = m_gl_mapped_textures.at(i);
        resource.write(*m_colorbuffer, m_app.stream());
        return resource.texture();
    };
}

void TrainingCamerasSlider::ui() { m_image_slider->ui(); }
