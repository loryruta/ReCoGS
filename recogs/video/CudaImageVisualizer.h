#pragma once

#include <memory>
#include <optional>
#include <thread>

#include "Window.h"
#include "utils/image/Image.h"

namespace gs_train
{
class CudaImageVisualizer
{
private:
    std::shared_ptr<Window> m_window;
    cudaStream_t m_stream;

    std::unique_ptr<Image3fCHW> m_image;

    std::unique_ptr<std::thread> m_thread;

public:
    explicit CudaImageVisualizer(std::shared_ptr<Window> window);
    ~CudaImageVisualizer();

    [[nodiscard]] auto window() const { return m_window; }

    void set_image(const Image3fCHW& image);
    void clear_image();

    void start();
    void stop();

    static std::unique_ptr<CudaImageVisualizer> create(int width, int height, const char* title);

private:
    void worker();
};
} // namespace gs_train
