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

    using KeyCallback = std::function<void(int key, int scancode, int action, int mods)>;
    using MouseButtonCallback = std::function<void(int button, int action, int mods)>;

    std::unordered_map<int, KeyCallback> m_key_callbacks;
    int m_next_key_callback_id = 0;

    std::unordered_map<int, MouseButtonCallback> m_mouse_button_callbacks;
    int m_next_mouse_button_callback_id = 0;

public:
    explicit Window(int width, int height, const std::string& title, bool resizable = false);
    ~Window();

    [[nodiscard]] GLFWwindow* handle() const { return m_handle; };

    [[nodiscard]] std::pair<int, int> framebuffer_size() const;

    void make_context(); // TODO dependent on OpenGL code

    [[nodiscard]] bool should_close() const;
    void set_should_close(bool flag);

    [[nodiscard]] glm::dvec2 cursor_pos() const;

    [[nodiscard]] int cursor_mode() { return glfwGetInputMode(m_handle, GLFW_CURSOR); };
    void set_cursor_mode(int value) { glfwSetInputMode(m_handle, GLFW_CURSOR, value); }

    /// \return \c true if the key is pressed
    [[nodiscard]] bool is_key_pressed(int key) const { return glfwGetKey(m_handle, key); }

    void poll_events();
    void swap_buffers();

    int add_key_callback(const KeyCallback& key_callback);
    void remove_key_callback(int id);

    int add_mouse_button_callback(const MouseButtonCallback& mouse_button_callback);
    void remove_mouse_button_callback(int id);

private:
    static void on_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void on_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
};
} // namespace gs_train
