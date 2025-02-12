#pragma once

#include <string>

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

namespace gs_train
{
class GLWindow
{
private:
    GLFWwindow* m_handle;

public:
    explicit GLWindow(int width, int height, const std::string& title, bool resizable = false);
    ~GLWindow();

    [[nodiscard]] std::pair<int, int> framebuffer_size() const;

    void make_context();

    [[nodiscard]] bool should_close() const;
    void set_should_close(bool flag);

    void poll_events();
    void swap_buffers();
};
} // namespace gs_train
