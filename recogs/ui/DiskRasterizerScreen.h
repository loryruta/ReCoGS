#pragma once

#include "CameraController.h"
#include "Screen.h"
#include "disk/DiskBuffer.h"

namespace recogs
{
class DiskRasterizerScreen : public Screen
{
private:
    Camera m_camera;
    std::unique_ptr<CameraController> m_camera_controller;
    DiskBuffer m_disk_buffer;

public:
    explicit DiskRasterizerScreen();
    ~DiskRasterizerScreen() = default;

    [[nodiscard]] const char* name() const { return "DiskRasterizerScreen"; };

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image4fHWC& color_depth) override;
    void ui() override;

private:
    void init_disks_scene();
};
} // namespace recogs
