#include "StereoTest.h"

using namespace gs_train::ui;

StereoTest::StereoTest()
{
    m_ht_stereo_icon = load_texture("assets/ui_stereo_ht_icon.png", 4, GL_LINEAR);
    m_vt_stereo_icon = load_texture("assets/ui_stereo_vt_icon.png", 4, GL_LINEAR);
    m_hv_stereo_icon = load_texture("assets/ui_stereo_hv_icon.png", 4, GL_LINEAR);
}

StereoTest::~StereoTest()
{
    glDeleteTextures(1, &m_ht_stereo_icon);
    glDeleteTextures(1, &m_vt_stereo_icon);
    glDeleteTextures(1, &m_hv_stereo_icon);
}

void StereoTest::ui()
{
    RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_HORIZONTAL, {
        if (ImGui::ImageButton((ImTextureID) (intptr_t) m_ht_stereo_icon, ImVec2(button_size, button_size))) {
            capture = Capture_HORIZONTAL;
        }
    });
    ImGui::SameLine();
    RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_VERTICAL, {
        if (ImGui::ImageButton((ImTextureID) (intptr_t) m_vt_stereo_icon, ImVec2(button_size, button_size))) {
            capture = Capture_VERTICAL;
        }
    });
    ImGui::SameLine();
    RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_HV, {
        if (ImGui::ImageButton((ImTextureID) (intptr_t) m_hv_stereo_icon, ImVec2(button_size, button_size))) {
            capture = Capture_HV;
        }
    });
}
