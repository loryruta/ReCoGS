#include "StereoTest.h"

using namespace recogs::ui;

StereoTest::StereoTest()
{
    m_ht_stereo_icon = load_texture("ui_stereo_ht_icon.png", 4, GL_LINEAR);
    m_vt_stereo_icon = load_texture("ui_stereo_vt_icon.png", 4, GL_LINEAR);
    m_hv_stereo_icon = load_texture("ui_stereo_hv_icon.png", 4, GL_LINEAR);
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
        if (ImGui::ImageButton(
                "##HStereoButton", (ImTextureID) (intptr_t) m_ht_stereo_icon, ImVec2(button_size, button_size))) {
            capture = Capture_HORIZONTAL;
        }
    });
    ImGui::SameLine();
    RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_VERTICAL, {
        if (ImGui::ImageButton(
                "##VStereoButton", (ImTextureID) (intptr_t) m_vt_stereo_icon, ImVec2(button_size, button_size))) {
            capture = Capture_VERTICAL;
        }
    });
    ImGui::SameLine();
    RCGS_IMGUI_DISABLE_BUTTON(current_capture == Capture_HV, {
        if (ImGui::ImageButton(
                "##HVStereoButton", (ImTextureID) (intptr_t) m_hv_stereo_icon, ImVec2(button_size, button_size))) {
            capture = Capture_HV;
        }
    });
    ImGui::SliderFloat("Scene depth epsilon", &scene_depth_epsilon, -0.4f, 0.4f, "%.3f");

    ImGui::Text("Render transform:");
    ImGui::RadioButton("Color", (int*) &render_transform, int(RenderTransform::COLOR));
    ImGui::SameLine();
    ImGui::RadioButton("Depthmap", (int*) &render_transform, int(RenderTransform::DEPTHMAP));
    ImGui::SameLine();
    ImGui::RadioButton("Normal map", (int*) &render_transform, int(RenderTransform::NORMAL_MAP));
}
