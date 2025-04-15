#include "SelectScreen.h"

#include <imgui.h>

#include "App.h"
#include "ui/MainScreen.h"
#include "utils/image/image_cast.h"
#include "utils/image/image_fill.h"
#include "utils/image/image_visit_transform.h"
#include "utils/imgui_utils.h"
#include "video/gl_utils.h"

using namespace gs_train;

SelectScreen_Toolbar::SelectScreen_Toolbar(SelectScreen& parent) : m_parent(parent)
{
    m_select_texture = load_texture("selectscreen_select.png", 4, GL_LINEAR);
    m_paintbrush_texture = load_texture("selectscreen_paintbrush.png", 4, GL_LINEAR);
    m_eraser_texture = load_texture("selectscreen_eraser.png", 4, GL_LINEAR);
}

SelectScreen_Toolbar::~SelectScreen_Toolbar()
{
    glDeleteTextures(1, &m_select_texture);
    glDeleteTextures(1, &m_paintbrush_texture);
    glDeleteTextures(1, &m_eraser_texture);
}

void SelectScreen_Toolbar::ui()
{
    ImVec2 window_size = ImGui::GetContentRegionAvail();

    ImGuiStyle& style = ImGui::GetStyle();
    float button_w = window_size.x - style.FramePadding.x * 2;
    ImVec2 button_size = ImVec2(button_w, button_w);

    // Brush
    RCGS_IMGUI_DISABLE_BUTTON(m_parent.m_mode == SelectScreen::Mode::BRUSH, {
        if (ImGui::ImageButton("Brush", (ImTextureID) (intptr_t) m_paintbrush_texture, button_size)) {
            m_parent.m_mode = SelectScreen::Mode::BRUSH;
        }
    });
    // Eraser
    RCGS_IMGUI_DISABLE_BUTTON(m_parent.m_mode == SelectScreen::Mode::ERASE, {
        if (ImGui::ImageButton("Eraser", (ImTextureID) (intptr_t) m_eraser_texture, button_size)) {
            m_parent.m_mode = SelectScreen::Mode::ERASE;
        }
    });
}

SelectScreen::SelectScreen(App& app, GSCamera camera) : m_app(app), m_camera(std::move(camera))
{
    Window& window = app.window();
    m_key_callback = window.add_key_callback([this](int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) {
            if (key == GLFW_KEY_ENTER) {
                {
                    glm::ivec2 resolution = m_app.resolution();
                    // Estimate depth using stereo matching HV
                    Image4fHWC depth = Image4fHWC::malloc(resolution.x, resolution.y);
                    image_fill(depth, Image4fHWC::Value{INFINITY}, m_app.stream());
                    m_app.stereo_depth_estimator().estimate_hv(m_camera, 0.07f, depth, m_app.stream());
                    // Populate the 3D selection with 2D selection unprojection
                    m_selection2d->populate_selection3d(depth);
                }
                m_app.set_screen(std::make_shared<MainScreen>(m_app, m_camera));
            } else if (key == GLFW_KEY_ESCAPE) {
                m_app.set_screen(std::make_shared<MainScreen>(m_app, m_camera));
            }
        }
    });
    printf("[DEBUG] [SelectScreen] Screen created\n");

    m_scroll_callback =
        window.add_scroll_callback([&](double xoffset, double yoffset) { on_scroll(xoffset, yoffset); });

    m_toolbar = std::make_unique<SelectScreen_Toolbar>(*this);
}

SelectScreen::~SelectScreen()
{
    Window& window = m_app.window();
    CHECK_STATE(window.remove_key_callback(m_key_callback));
    CHECK_STATE(window.remove_scroll_callback(m_scroll_callback));
}

void SelectScreen::resize(int width, int height)
{
    m_color_depth = std::make_unique<Image4fHWC>(Image4fHWC::malloc(width, height));
    m_camera_texture = std::make_unique<CudaTexture>(width, height);
    // TODO unsupported yet! Update camera resolution also!
    m_depthbuffer = std::make_unique<Image1fCHW>(Image1fCHW::malloc(width, height));
    m_selection2d = std::make_unique<Selection2d>(m_app.selection3d(), m_camera);
}

CudaTexture& SelectScreen::render_camera_texture()
{
    cudaStream_t stream = m_app.stream();
    // Render GS scene
    m_app.gs_rasterizer().show_borders = false;
    m_app.gs_rasterizer().forward(
        m_app.background_d(), m_app.scene(), true /* scene_2 */, m_camera, *m_color_depth, stream);
    // Render 2D and 3D selection (already projected to 2D)
    m_app.selection_renderer().render(*m_selection2d, *m_color_depth, stream);
    // Blit colorbuffer onto texture
    m_camera_texture->write(*m_color_depth, stream);
    return *m_camera_texture;
}

void SelectScreen::on_scroll(double xoffset, double yoffset)
{
    float dscale = -float(yoffset) * k_select_zoom_speed;
    m_camera_scale += dscale;
}

void SelectScreen::update(float dt)
{
    if (imgui_want_ui_interaction()) return;

    Window& window = m_app.window();
    glm::dvec2 cursor_pos = window.cursor_pos();

    bool mouse_pressed_l = window.is_mouse_button_pressed(GLFW_MOUSE_BUTTON_LEFT);
    bool mouse_pressed_r = window.is_mouse_button_pressed(GLFW_MOUSE_BUTTON_RIGHT);

    if (!mouse_pressed_l && m_last_cursor_pos_l) m_last_cursor_pos_l.reset();
    if (!mouse_pressed_r && m_last_cursor_pos_r) m_last_cursor_pos_r.reset();

    glm::vec2 dcursor_pos_r{};
    if (mouse_pressed_r) {
        if (m_last_cursor_pos_r) {
            dcursor_pos_r = cursor_pos - *m_last_cursor_pos_r;
        }
    } else if (m_last_cursor_pos_r) {
        m_last_cursor_pos_r.reset();
    }

    if (mouse_pressed_l && m_last_cursor_pos_l) {
        // Brush
        if (m_mode == Mode::BRUSH) {
            m_selection2d->fill_line(*m_last_cursor_pos_l,
                                     cursor_pos,
                                     25, // radius
                                     m_camera_offset,
                                     m_camera_scale,
                                     m_app.stream());
        }
        // Eraser
        else if (m_mode == Mode::ERASE) {
            m_selection2d->clear_line(*m_last_cursor_pos_l,
                                      cursor_pos,
                                      25, // radius
                                      m_camera_offset,
                                      m_camera_scale,
                                      m_app.stream());
        }
    }

    // Drag
    if (mouse_pressed_r && dcursor_pos_r != glm::vec2{}) {
        m_camera_offset -= dcursor_pos_r * k_select_drag_speed * m_camera_scale * dt;
    }

    if (mouse_pressed_l) m_last_cursor_pos_l = cursor_pos;
    if (mouse_pressed_r) m_last_cursor_pos_r = cursor_pos;
}

void SelectScreen::render(Image4fHWC& out_color_depth)
{
    cudaStream_t stream = m_app.stream();

    CudaTexture& camera_texture = render_camera_texture();

    image_visit(
        out_color_depth,
        [offset = m_camera_offset,
         scale = m_camera_scale,
         camera_textureobject = camera_texture.texture_object()] __device__(Image4fHWC & color_depth, int x, int y) {
            float w = (float) color_depth.width;
            float h = (float) color_depth.height;
            // Normalize (x, y) in [-1, 1] range
            glm::vec2 norm_coords;
            norm_coords.x = float(x) / w - 0.5f;
            norm_coords.y = float(y) / h - 0.5f;
            norm_coords = norm_coords * scale + offset;
            glm::vec2 uv = norm_coords + 0.5f;
            float4 color = tex2D<float4>(camera_textureobject, uv.x, uv.y);
            color_depth.set_value(x, y, glm::vec4(color.x, color.y, color.z, 0 /* depth */));
            return 0; // TODO temporary
        },
        stream);
}

void SelectScreen::ui()
{
    constexpr int k_selection_toolbar_width = 60;

    glm::vec2 resolution = m_app.window().framebuffer_size();

    if (ImGui::Begin(
            "Toolbar", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar)) {
        ImGui::SetWindowPos({0, 0});
        ImGui::SetWindowSize({k_selection_toolbar_width, resolution.y});

        m_toolbar->ui();
    }
    ImGui::End();
}
