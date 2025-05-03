#pragma once

#include <functional>

namespace recogs::ui
{
class Header
{
private:
    static constexpr int k_max_sections = 10;

    std::array<std::tuple<std::string, std::function<void()>>, k_max_sections> m_sections;
    int m_num_sections = 0;

public:
    float height = 100;

    explicit Header() = default;
    ~Header() = default;

    void section(const std::string& title, const std::function<void()>& child_ui)
    {
        CHECK_STATE(m_num_sections < k_max_sections);
        m_sections[m_num_sections] = std::make_tuple(title, child_ui);
        ++m_num_sections;
    }

    void ui()
    {
        ImVec2 resolution = ImGui::GetIO().DisplaySize;

        if (ImGui::Begin("##header",
                         nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoScrollbar)) {
            ImGui::SetWindowSize(ImVec2(resolution.x, 0));
            ImGui::SetWindowPos(ImVec2(0, 0));
            // Sections content
            float column_x[k_max_sections];
            float column_widths[k_max_sections];
            float column_y = 0;
            ImGui::BeginTable("##header_table", m_num_sections, ImGuiTableFlags_BordersV);
            ImGui::TableNextRow();
            for (int i = 0; i < m_num_sections; ++i) {
                ImGui::TableNextColumn();
                // Section UI
                column_x[i] = ImGui::GetCursorScreenPos().x;
                column_widths[i] = ImGui::GetColumnWidth();
                const std::function<void()>& section_ui = std::get<1>(m_sections.at(i));
                section_ui();
                column_y = glm::max(ImGui::GetCursorScreenPos().y, column_y);
                //ImGui::Text(""); // Placeholder
            }
            ImGui::EndTable();
            // Footer
            for (int i = 0; i < m_num_sections; ++i) {
                const std::string& title = std::get<0>(m_sections.at(i));
                float text_width = ImGui::CalcTextSize(title.c_str()).x;
                ImVec2 cursor_pos;
                cursor_pos.x = column_x[i] + column_widths[i] / 2 - text_width / 2;
                cursor_pos.y = column_y + ImGui::GetStyle().ItemSpacing.y * 3;
                ImGui::SetCursorScreenPos(cursor_pos);
                //ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                ImGui::Text("%s", title.c_str());
                //ImGui::PopStyleColor();
            }
        }
        ImGui::End();
    }
};
} // namespace recogs::ui
