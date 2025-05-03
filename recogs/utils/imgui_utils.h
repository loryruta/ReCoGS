#pragma once

#include <imgui.h>

#define RCGS_IMGUI_DISABLE_BUTTON(disabled_, code_)                                                                    \
    do {                                                                                                               \
        bool condition_ = !!(disabled_);                                                                               \
        if (condition_) ImGui::BeginDisabled();                                                                        \
        code_;                                                                                                         \
        if (condition_) ImGui::EndDisabled();                                                                          \
    } while (false)

namespace recogs
{
/// Whether the user is interacting with ImGui UIs (e.g. hovering a window).
/// In such a case, the interaction event shouldn't be forwarded to, for example, the camera controller.
inline bool imgui_want_ui_interaction()
{
    ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureKeyboard || io.WantCaptureMouse;
}

/// Disable any user interaction about to happen in the current ImGui frame.
inline void imgui_disable_ui_interaction()
{
    ImGuiIO& io = ImGui::GetIO();
    io.WantCaptureKeyboard = false;
    io.WantCaptureMouse = false;
}
} // namespace recogs