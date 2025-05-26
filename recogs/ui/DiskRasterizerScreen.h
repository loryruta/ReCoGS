#pragma once

#include "GSCameraController.h"
#include "Screen.h"
#include "disk/Disks.h"

namespace recogs
{
class DiskRasterizerScreen : public Screen
{
private:
    GSCamera m_camera;
    std::unique_ptr<GSCameraController> m_camera_controller;
    std::unique_ptr<Disks> m_disks;

public:
    explicit DiskRasterizerScreen();
    ~DiskRasterizerScreen() = default;

    [[nodiscard]] const char* name() const { return "DiskRasterizerScreen"; };

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image4fHWC& color_depth) override;
    void ui() override;

private:
    static std::unique_ptr<Disks> create_disks_scene();
};
} // namespace recogs
