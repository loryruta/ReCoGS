#include "GLWindow.h"

#include "utils/misc_utils.h"

using namespace gs_train;

size_t g_alive_window_count = 0;

GLWindow::GLWindow(int width, int height, const std::string& title, bool resizable)
{
    if (g_alive_window_count == 0) {
        CHECK_STATE(glfwInit(), "Can't initialize GLFW");
    }
    ++g_alive_window_count;

    glfwWindowHint(GLFW_RESIZABLE, resizable);
    m_handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    CHECK_STATE(m_handle, "Can't create GLFW window");
}

GLWindow::~GLWindow()
{
    glfwDestroyWindow(m_handle);
    --g_alive_window_count;
    if (g_alive_window_count == 0) {
        glfwTerminate();
        printf("GLFW terminated\n");
    }
}

std::pair<int, int> GLWindow::framebuffer_size() const
{
    int width, height;
    glfwGetFramebufferSize(m_handle, &width, &height);
    return {width, height};
}

void GLWindow::make_context()
{
    glfwMakeContextCurrent(m_handle);

    int version = gladLoadGL();
    CHECK_STATE(version != 0, "Can't load GL");
}

[[nodiscard]] bool GLWindow::should_close() const { return glfwWindowShouldClose(m_handle); }
void GLWindow::set_should_close(bool flag) { glfwSetWindowShouldClose(m_handle, flag); }

void GLWindow::poll_events() { glfwPollEvents(); }

void GLWindow::swap_buffers() { glfwSwapBuffers(m_handle); }
