#include "ImageSlider.h"

#include <cstdio>
#include <string>

#include <imgui.h>

using namespace gs_train;

ImageSlider::ImageSlider(int N, int resolution_xy) : m_N(N), image_resolution_xy(resolution_xy)
{
    m_textures.resize(N, 0); // Initialize all textures to invalid handles
}

void ImageSlider::ui()
{
    ImGuiStyle& style = ImGui::GetStyle();

    ImVec2 size = ImGui::GetContentRegionAvail();

    // Left button
    if (ImGui::Button("<", ImVec2(button_width, size.y))) {
        m_start_texture_idx = std::max(m_start_texture_idx - 1, 0);
    }
    ImGui::SameLine();

    // Images
    char caption[128];
    int i;
    for (i = m_start_texture_idx;; ++i) {
        // Textures are finished!
        if (i >= m_N) break;

        sprintf(caption, "%d", i);
        ImVec2 caption_size = ImGui::CalcTextSize(caption);
        float image_h = size.y - caption_size.y - style.ItemSpacing.y;
        float image_w = image_h;

        // If the remaining space wouldn't be enough for placing the "next" button, skip
        float remaining_space = ImGui::GetContentRegionAvail().x - style.ItemSpacing.x - image_w;
        if (remaining_space < style.ItemSpacing.x + button_width) break;
        ImVec2 cur_pos = ImGui::GetCursorPos();
        // Write image (button)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.f, 0.f, 0.f, 0.f));
        if (ImGui::ImageButton((ImTextureID) (intptr_t) m_textures.at(i), ImVec2(image_w, image_h))) {
            on_image_click(i);
        }
        ImGui::PopStyleColor(3);
        // Write caption
        ImGui::SetCursorPosX(cur_pos.x + image_w / 2 - caption_size.x / 2);
        ImGui::SetCursorPosY(cur_pos.y + image_h + style.ItemSpacing.y);
        ImGui::Text("%s", caption);
        // Reset to next position
        ImGui::SetCursorPosX(cur_pos.x + image_w + style.ItemSpacing.x);
        ImGui::SetCursorPosY(cur_pos.y);
    }
    m_end_texture_idx = i;

    // Right button
    float right_button_width = button_width;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - right_button_width);
    if (ImGui::Button(">", ImVec2(button_width, size.y))) {
        m_start_texture_idx = std::min(m_start_texture_idx + 1, m_N - 1);
    }
}
