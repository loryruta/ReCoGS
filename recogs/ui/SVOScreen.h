#pragma once

#include "Camera.h"
#include "CameraController.h"
#include "Screen.h"
#include "svo/SVONode.h"
#include "utils/image/Image.h"

namespace recogs
{
// Forward decl
class App;

/// A screen for visualizing a Sparse Voxel Octree (SVO). Created for debugging the tracer.
class SVOScreen : public Screen
{
private:
    App& m_app;
    Camera m_camera;
    std::unique_ptr<CameraController> m_camera_controller;
    std::shared_ptr<SVO> m_svo; ///< The SVO being visualized

public:
    explicit SVOScreen(App& app, std::shared_ptr<SVO> svo);
    ~SVOScreen() = default;

    [[nodiscard]] const char* name() const { return "SVOScreen"; };

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image4fHWC& out_color_depth) override;
    void ui() override;

private:
    static std::shared_ptr<SVO> create_simple_svo();
};
} // namespace recogs
