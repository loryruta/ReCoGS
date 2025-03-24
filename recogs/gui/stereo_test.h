#pragma once

#include <imgui.h>

#include "utils/imgui_utils.h"

namespace gs_train
{
namespace ui
{
struct StereoTest {
    enum { Capture_NONE = 0, Capture_HORIZONTAL = 1, Capture_VERTICAL = 2, Capture_HV = 3 };

    /// Which type of stereo test has to be captured the next frame.
    int capture = Capture_NONE;
    /// The currently displayed capture.
    int current_capture = Capture_NONE;

    void ui()
    {
        if (ImGui::Begin("Stereo Test")) {
            RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_HORIZONTAL, {
                if (ImGui::Button("Horizontal Stereo##stereo_test")) {
                    capture = Capture_HORIZONTAL;
                }
            });
            RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_VERTICAL, {
                if (ImGui::Button("Vertical Stereo##stereo_test")) {
                    capture = Capture_VERTICAL;
                }
            });
            RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_HV, {
                if (ImGui::Button("H/V Stereo##stereo_test")) {
                    capture = Capture_HV;
                }
            });
        }
        ImGui::End();
    }
};
} // namespace ui

} // namespace gs_train
