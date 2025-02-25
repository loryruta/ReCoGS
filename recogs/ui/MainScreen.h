#pragma once

#include "GSCamera.h"
#include "GSCameraController.h"
#include "Screen.h"
#include "depth_estimators/EstimateDepth.h"
#include "utils/image/Image.h"

namespace gs_train
{
// Forward decl
class App;

/// Main screen when the app opens:
/// allow the user to navigate the 3DGS scene and visualize the selection pointcloud if any
class MainScreen : public Screen
{
private:
    App& m_app;

    GSCamera m_camera{};
    std::unique_ptr<GSCameraController> m_camera_controller;

    std::unique_ptr<Image1fCHW> m_depthbuffer;

    std::unique_ptr<EstimateDepth> m_stereo;

    int m_key_callback_id = -1;
    int m_mouse_button_callback_id = -1;

    enum ViewMode {
        GSRASTERIZER_COLOR, ///< Colorbuffer output of the GS rasterizer
        GSRASTERIZER_DEPTH, ///< Depthbuffer output of the GS rasterizer (depth estimated from gaussians)
        STEREO_H,           ///< Horizontal stereo matching
        STEREO_V,           ///< Vertical stereo matching
        STEREO_HV           ///< Aggregated horizontal/vertical stereo matching (typically with `min`)
    } m_view_mode = GSRASTERIZER_COLOR;

public:
    explicit MainScreen(App& app);
    ~MainScreen();

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image3fCHW& out_colorbuffer) override;
};
} // namespace gs_train