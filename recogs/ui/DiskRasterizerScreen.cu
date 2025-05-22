#include "DiskRasterizerScreen.h"

#include "App.h"
#include "utils/image/image_fill.h"

using namespace recogs;

DiskRasterizerScreen::DiskRasterizerScreen(App& app) : m_app(app)
{
    m_disks = create_disks_scene();
    m_camera_controller = std::make_unique<GSCameraController>(m_app.window(), m_camera);
}

void DiskRasterizerScreen::resize(int width, int height)
{
    m_camera.set_resolution(width, height);
    m_camera.update(m_app.stream());
}

void DiskRasterizerScreen::update(float dt)
{
    if (m_camera_controller) {
        bool updated = m_camera_controller->update(dt);
        if (updated) m_camera.update(m_app.stream());
        // m_camera.log_info();
    }
}

void DiskRasterizerScreen::render(Image4fHWC& color_depth)
{
    image_fill(color_depth, glm::vec4(0), m_app.stream());
    if (m_disks) {
        m_app.disk_rasterizer().forward(*m_disks, m_camera, color_depth, m_app.stream());
    }
}

void DiskRasterizerScreen::ui() {}

std::unique_ptr<Disks> DiskRasterizerScreen::create_disks_scene()
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> scales;
    std::vector<glm::vec4> rotations;

    positions.emplace_back(0, 0, 0);
    scales.emplace_back(1, 0.5f);
    rotations.emplace_back(0, 0, 0, 1);

    // GPU allocation & upload
    Disks disks{};
    disks.count = (int) positions.size();
    disks.positions.resize(positions.size());
    disks.scales.resize(positions.size());
    disks.rotations.resize(positions.size());
    thrust::copy(positions.begin(), positions.end(), disks.positions.begin());
    thrust::copy(scales.begin(), scales.end(), disks.scales.begin());
    thrust::copy(rotations.begin(), rotations.end(), disks.rotations.begin());
    return std::make_unique<Disks>(std::move(disks));
}
