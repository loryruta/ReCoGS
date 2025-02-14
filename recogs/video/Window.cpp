#include "Window.h"

#include "utils/misc_utils.h"

using namespace gs_train;

void Window::on_key_callback(GLFWwindow* handle, int key, int scancode, int action, int mods)
{
    Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handle));
    for (const auto& [_, callback] : window->m_key_callbacks) {
        callback(key, scancode, action, mods);
    }
}

void Window::on_mouse_button_callback(GLFWwindow* handle, int button, int action, int mods)
{
    Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handle));
    for (const auto& [_, callback] : window->m_mouse_button_callbacks) {
        callback(button, action, mods);
    }
}

size_t g_alive_window_count = 0;

Window::Window(int width, int height, const std::string& title, bool resizable)
{
    if (g_alive_window_count == 0) {
        CHECK_STATE(glfwInit(), "Can't initialize GLFW");
    }
    ++g_alive_window_count;

    glfwWindowHint(GLFW_RESIZABLE, resizable);
    m_handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    CHECK_STATE(m_handle, "Can't create GLFW window");

    glfwSetWindowUserPointer(m_handle, this);

    glfwSetKeyCallback(m_handle, on_key_callback);
    glfwSetMouseButtonCallback(m_handle, on_mouse_button_callback);
}

Window::~Window()
{
    glfwDestroyWindow(m_handle);
    --g_alive_window_count;
    if (g_alive_window_count == 0) {
        glfwTerminate();
        printf("GLFW terminated\n");
    }
}

glm::ivec2 Window::framebuffer_size() const
{
    glm::ivec2 fb_size;
    glfwGetFramebufferSize(m_handle, &fb_size.x, &fb_size.y);
    return fb_size;
}

void Window::make_context()
{
    glfwMakeContextCurrent(m_handle);

    int version = gladLoadGL();
    CHECK_STATE(version != 0, "Can't load GL");
}

[[nodiscard]] bool Window::should_close() const { return glfwWindowShouldClose(m_handle); }

void Window::set_should_close(bool flag) { glfwSetWindowShouldClose(m_handle, flag); }

[[nodiscard]] glm::dvec2 Window::cursor_pos() const
{
    glm::dvec2 cur_pos;
    glfwGetCursorPos(m_handle, &cur_pos.x, &cur_pos.y);
    return cur_pos;
}

void Window::poll_events() { glfwPollEvents(); }

void Window::swap_buffers() { glfwSwapBuffers(m_handle); }

int Window::add_key_callback(const KeyCallback& key_callback)
{
    m_key_callbacks.emplace(m_next_key_callback_id, key_callback);
    return m_next_key_callback_id++;
}

void Window::remove_key_callback(int id) { m_key_callbacks.erase(id); }

int Window::add_mouse_button_callback(const MouseButtonCallback& mouse_button_callback)
{
    m_mouse_button_callbacks.emplace(m_next_mouse_button_callback_id, mouse_button_callback);
    return m_next_mouse_button_callback_id++;
}

void Window::remove_mouse_button_callback(int id) { m_mouse_button_callbacks.erase(id); }
