#pragma once

#include <functional>
#include <memory>
#include <thread>

#include "GLWindow.h"

namespace gs_train
{
class CudaImageVisualizer
{
private:
    using AdaptImageFunc = std::function<void(int image_w, int image_h, const float* img_d, float* out_img_d)>;

    std::shared_ptr<GLWindow> m_window;

    int m_image_w = -1;
    int m_image_h = -1;
    const float* m_image_d{};
    /// A transform applied to the image before visualization.
    /// Used to adapt the image provided by \c set_image to (H, W, 4) format.
    AdaptImageFunc m_adapt_image_func{};

    std::unique_ptr<std::thread> m_thread;

public:
    explicit CudaImageVisualizer(std::shared_ptr<GLWindow> window);
    ~CudaImageVisualizer();

    [[nodiscard]] auto window() const { return m_window; }

    void set_image(int image_w, int image_h, const float* image_d, const AdaptImageFunc& adapt_image_func = {});

    void start();
    void stop();

    static std::unique_ptr<CudaImageVisualizer> create(int width, int height, const char* title);

private:
    void worker();
};
} // namespace gs_train
