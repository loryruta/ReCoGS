#include "Window.h"

#include "utils/misc_utils.h"

using namespace recogs;

void Window::on_resize_callback(GLFWwindow* handle, int width, int height)
{
    Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handle));
    for (const auto& [_, callback] : window->m_resize_callbacks) {
        callback(width, height);
    }
}

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

void Window::on_scroll_callback(GLFWwindow* handle, double xoffset, double yoffset)
{
    Window* window = static_cast<Window*>(glfwGetWindowUserPointer(handle));
    for (const auto& [_, callback] : window->m_scroll_callbacks) {
        callback(xoffset, yoffset);
    }
}

size_t g_alive_window_count = 0;

Window::Window(GLFWwindow* handle) : m_handle(handle)
{
    // Register callbacks
    glfwSetWindowUserPointer(handle, this);

    glfwSetWindowSizeCallback(handle, on_resize_callback);
    glfwSetKeyCallback(handle, on_key_callback);
    glfwSetMouseButtonCallback(handle, on_mouse_button_callback);
    glfwSetScrollCallback(handle, on_scroll_callback);
}

Window::Window(Window&& other) noexcept : m_handle(other.m_handle)
{
    glfwSetWindowUserPointer(m_handle, this);

    other.m_handle = nullptr;
}

Window::~Window()
{
    if (m_handle) {
        glfwDestroyWindow(m_handle);
        --g_alive_window_count;
        if (g_alive_window_count == 0) {
            glfwTerminate();
            printf("GLFW terminated\n");
        }
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

int Window::add_resize_callback(const ResizeCallback& callback)
{
    m_resize_callbacks.emplace(m_next_resize_callback_id, callback);
    return m_next_resize_callback_id++;
}

bool Window::remove_resize_callback(int id) { return m_resize_callbacks.erase(id); }

int Window::add_key_callback(const KeyCallback& key_callback)
{
    m_key_callbacks.emplace(m_next_key_callback_id, key_callback);
    return m_next_key_callback_id++;
}

bool Window::remove_key_callback(int id) { return m_key_callbacks.erase(id); }

int Window::add_mouse_button_callback(const MouseButtonCallback& mouse_button_callback)
{
    m_mouse_button_callbacks.emplace(m_next_mouse_button_callback_id, mouse_button_callback);
    return m_next_mouse_button_callback_id++;
}

bool Window::remove_mouse_button_callback(int id) { return m_mouse_button_callbacks.erase(id); }

int Window::add_scroll_callback(const ScrollCallback& scroll_callback)
{
    m_scroll_callbacks.emplace(m_next_scroll_callback_id, scroll_callback);
    return m_next_scroll_callback_id++;
}

bool Window::remove_scroll_callback(int id) { return m_scroll_callbacks.erase(id); }

void Window::init_glfw_if_not_init()
{
    if (g_alive_window_count == 0) {
        CHECK_STATE(glfwInit(), "Can't initialize GLFW");
    }
    ++g_alive_window_count;
}

Window Window::create(int width, int height, const std::string& title, bool resizable)
{
    init_glfw_if_not_init();

    glfwWindowHint(GLFW_RESIZABLE, resizable);
    printf("[INFO ] [Window] Creating a window of size (%d, %d)\n", width, height);
    GLFWwindow* handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    CHECK_STATE(handle, "Can't create GLFW window");
    return Window(handle);
}

Window Window::create_bordered_fullscreen(const std::string& title)
{
    init_glfw_if_not_init();
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* vidmode = glfwGetVideoMode(monitor);
    printf(
        "[INFO ] [Window] Creating a bordered fullscreen window of size (%d, %d)\n", vidmode->width, vidmode->height);
    GLFWwindow* handle = glfwCreateWindow(vidmode->width, vidmode->height, title.c_str(), nullptr, nullptr);
    glfwSetWindowPos(handle, 0, 0);
    CHECK_STATE(handle, "Can't create GLFW window");
    return Window(handle);
}
