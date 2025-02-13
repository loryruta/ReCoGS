#include "CudaImageVisualizer.h"

#include "DrawTexture.h"
#include "GLMappedResource.h"
#include "utils/cuda_utils.h"

using namespace gs_train;

CudaImageVisualizer::CudaImageVisualizer(std::shared_ptr<GLWindow> window) : m_window(std::move(window)) {}

CudaImageVisualizer::~CudaImageVisualizer() { stop(); }

void CudaImageVisualizer::set_image(int image_w,
                                    int image_h,
                                    const float* image_d,
                                    const AdaptImageFunc& adapt_image_func)
{
    m_image_w = image_w;
    m_image_h = image_h;
    m_image_d = image_d;
    m_adapt_image_func = adapt_image_func;
}

void CudaImageVisualizer::start()
{
    CHECK_STATE(!m_thread, "start() was already called");
    m_thread = std::make_unique<std::thread>([this]() { worker(); });
}

void CudaImageVisualizer::stop()
{
    m_window->set_should_close(true);
    if (m_thread) {
        m_thread->join();
        m_thread.reset();
    }
}

void CudaImageVisualizer::worker()
{
    m_window->make_context();

    auto [W, H] = m_window->framebuffer_size(); // TODO we're assuming the window size doesn't change
    GLMappedResource gl_mapped_resource(W, H);
    DrawTexture draw_texture{};

    float* vis_img_d = nullptr; // (H, W, 4)
    if (m_adapt_image_func) {
        CHECK_CUDA(cudaMalloc(&vis_img_d, H * W * 4 * sizeof(float)));
    }

    while (!m_window->should_close()) {
        m_window->poll_events();

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        if (m_image_d) {
            if (m_adapt_image_func) {
                m_adapt_image_func(m_image_w, m_image_h, m_image_d, vis_img_d);
            } else {
                vis_img_d = const_cast<float*>(m_image_d);
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize()); // TODO this will block all device operations, sync on stream

        gl_mapped_resource.write(vis_img_d);
        auto [fb_w, fb_h] = m_window->framebuffer_size();
        draw_texture.draw(gl_mapped_resource.texture(), 0, 0, fb_w, fb_h);

        m_window->swap_buffers();
    }
}

std::unique_ptr<CudaImageVisualizer> CudaImageVisualizer::create(int width, int height, const char* title)
{
    std::shared_ptr<GLWindow> window = std::make_shared<GLWindow>(width, height, title, false /* resizable */);
    return std::make_unique<CudaImageVisualizer>(window);
}
