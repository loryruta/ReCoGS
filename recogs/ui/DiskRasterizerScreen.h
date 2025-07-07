#pragma once

#include "GSCameraController.h"
#include "Screen.h"
#include "triangle/DiskBuffer.h"

namespace recogs
{
class DiskRasterizerScreen : public Screen
{
private:
    GSCamera m_camera;
    std::unique_ptr<GSCameraController> m_camera_controller;
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
