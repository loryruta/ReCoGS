#pragma once

#include <functional>
#include <list>
#include <string>

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <glm/glm.hpp>

namespace gs_train
{
/// A wrapper of the GLFW window
class Window
{
private:
    GLFWwindow* m_handle;

    using ResizeCallback = std::function<void(int width, int height)>;
    using KeyCallback = std::function<void(int key, int scancode, int action, int mods)>;
    using MouseButtonCallback = std::function<void(int button, int action, int mods)>;
    using ScrollCallback = std::function<void(double xoffset, double yoffset)>;

    // Callbacks
    std::unordered_map<int, ResizeCallback> m_resize_callbacks;
    int m_next_resize_callback_id = 0;
    std::unordered_map<int, KeyCallback> m_key_callbacks;
    int m_next_key_callback_id = 0;
    std::unordered_map<int, MouseButtonCallback> m_mouse_button_callbacks;
    int m_next_mouse_button_callback_id = 0;
    std::unordered_map<int, ScrollCallback> m_scroll_callbacks;
    int m_next_scroll_callback_id = 0;

public:
    Window(const Window&) = delete;
    Window(Window&& other) noexcept;
    ~Window();

    [[nodiscard]] GLFWwindow* handle() const { return m_handle; };

    void set_title(const std::string& title) const { glfwSetWindowTitle(m_handle, title.c_str()); }

    [[nodiscard]] glm::ivec2 framebuffer_size() const;
    /// Alias to framebuffer_size()
    [[nodiscard]] glm::ivec2 resolution() const { return framebuffer_size(); }

    void make_context(); // TODO dependent on OpenGL code

    [[nodiscard]] bool should_close() const;
    void set_should_close(bool flag);

    [[nodiscard]] glm::dvec2 cursor_pos() const;

    [[nodiscard]] int cursor_mode() { return glfwGetInputMode(m_handle, GLFW_CURSOR); };
    void set_cursor_mode(int value) { glfwSetInputMode(m_handle, GLFW_CURSOR, value); }

    /// \return \c true if the key is pressed
    [[nodiscard]] bool is_key_pressed(int key) const { return glfwGetKey(m_handle, key) == GLFW_PRESS; }

    /// \return \c true if the mouse button is pressed
    [[nodiscard]] bool is_mouse_button_pressed(int button) const
    {
        return glfwGetMouseButton(m_handle, button) == GLFW_PRESS;
    }

    static void poll_events();
    void swap_buffers();

    int add_resize_callback(const ResizeCallback& callback);
    bool remove_resize_callback(int id);

    int add_key_callback(const KeyCallback& key_callback);
    bool remove_key_callback(int id);
    int add_mouse_button_callback(const MouseButtonCallback& mouse_button_callback);
    bool remove_mouse_button_callback(int id);
    int add_scroll_callback(const ScrollCallback& scroll_callback);
    bool remove_scroll_callback(int id);

    static Window create(int width, int height, const std::string& title, bool resizable);
    static Window create_bordered_fullscreen(const std::string& title);

private:
    explicit Window(GLFWwindow* handle);

    static void init_glfw_if_not_init();

    static void on_resize_callback(GLFWwindow* handle, int width, int height);
    static void on_key_callback(GLFWwindow* handle, int key, int scancode, int action, int mods);
    static void on_mouse_button_callback(GLFWwindow* handle, int button, int action, int mods);
    static void on_scroll_callback(GLFWwindow* handle, double xoffset, double yoffset);
};
} // namespace gs_train
