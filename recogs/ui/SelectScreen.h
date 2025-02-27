#pragma once

#include "GSCamera.h"
#include "GSCameraController.h"
#include "Screen.h"
#include "depth_estimators/StereoDepthEstimator.h"
#include "selection/Selection2d.h"
#include "utils/image/Image.h"

namespace gs_train
{
// Forward decl
class App;

class SelectScreen : public Screen
{
private:
    App& m_app;
    const GSCamera m_view;

    int m_key_callback;

    std::unique_ptr<Image1fCHW> m_depthbuffer;
    std::unique_ptr<Selection2d> m_selection2d;

    std::optional<glm::dvec2> m_last_cursor_pos;

public:
    explicit SelectScreen(App& app, GSCamera camera);
    ~SelectScreen() override;

    [[nodiscard]] const char* name() const override { return "SelectScreen"; }

    void resize(int width, int height) override;
    void update(float dt) override;
    void render(Image3fCHW& out_colorbuffer) override;
};
} // namespace gs_train