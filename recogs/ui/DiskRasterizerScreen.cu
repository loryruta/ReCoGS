#include "DiskRasterizerScreen.h"

#include "App.h"
#include "disk/DiskBuffer.h"
#include "disk/DiskRenderer.h"
#include "utils/image/image_fill.h"

using namespace recogs;

DiskRasterizerScreen::DiskRasterizerScreen()
{
    init_disks_scene();
    m_camera_controller = std::make_unique<CameraController>(g_app->window(), m_camera);
}

void DiskRasterizerScreen::resize(int width, int height)
{
    m_camera.set_resolution(width, height);
    m_camera.update(g_stream);
}

void DiskRasterizerScreen::update(float dt)
{
    if (m_camera_controller) {
        bool updated = m_camera_controller->update(dt);
        if (updated) m_camera.update(g_stream);
        // m_camera.log_info();
    }
}

void DiskRasterizerScreen::render(Image4fHWC& color_depth)
{
    image_fill(color_depth, glm::vec4(0), g_stream);

    DiskRenderer_Params params{};
    params.camera = &m_camera;
    params.color_depth = &color_depth;
    params.disk_buffer = &m_disk_buffer;
    params.stream = g_stream;
    g_app->disk_renderer().render(params,
                                  [] __device__(uint32_t disk_id, float2 uv, float opacity, float* colordepth_ptr) {
                                      colordepth_ptr[0] = 1.f;
                                      colordepth_ptr[1] = 1.f;
                                      colordepth_ptr[2] = 1.f;
                                      colordepth_ptr[3] = 1.f;
                                  });
}

void DiskRasterizerScreen::ui() {}

void DiskRasterizerScreen::init_disks_scene()
{
    Disk& disk1 = m_disk_buffer.emplace_back();
    disk1.position = {0, 0, 0, 0};
    disk1.scale = {1, 0.2f};
    disk1.rotation = {0, 0, 0, 1};
    Disk& disk2 = m_disk_buffer.emplace_back();
    disk2.position = {0, 0, 5, 0};
    disk2.scale = {0.5f, 0.5f};
    disk2.rotation = {0, 0, 0, 1};
    Disk& disk3 = m_disk_buffer.emplace_back();
    disk3.position = {0, 0, -1, 0};
    disk3.scale = {0.6f, 0.8f};
    disk3.rotation = {0, 0, 0, 1};
    m_disk_buffer.upload(g_stream);
}
