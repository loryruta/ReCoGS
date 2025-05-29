#include "CudaImageVisualizer.h"

#include "DrawTexture.h"
#include "GLTextureMapped.h"
#include "utils/cuda_utils.h"
#include "utils/image/image_copy.h"
#include "utils/image/image_fill.h"

using namespace recogs;

CudaImageVisualizer::CudaImageVisualizer(std::shared_ptr<Window> window) : m_window(std::move(window))
{
    CHECK_CUDA(cudaStreamCreate(&m_stream));
}

CudaImageVisualizer::~CudaImageVisualizer()
{
    stop();
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    CHECK_CUDA(cudaStreamDestroy(m_stream));
}

void CudaImageVisualizer::set_image(const Image3fCHW& image)
{
    m_image = std::make_unique<Image3fCHW>(image.create_ref());
}

void CudaImageVisualizer::clear_image() { m_image.reset(); }

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

    // TODO Assuming window size doesn't change
    const glm::ivec2 fb_size = m_window->framebuffer_size();
    const int W = fb_size.x;
    const int H = fb_size.y;
    GLTextureMapped gl_mapped_resource = GLTextureMapped::create_rgba32f(fb_size.x, fb_size.y);
    DrawTexture draw_texture{};

    Image4fCHW colorbuffer_chw = Image4fCHW::malloc(W, H);
    Image4fHWC colorbuffer_hwc = Image4fHWC::malloc(W, H);

    while (!m_window->should_close()) {
        Window::poll_events();

        // Clear the colorbuffer
        image_fill(colorbuffer_chw, glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), m_stream);
        if (m_image) {
            Image3fCHW colorbuffer_3hw = Image3fCHW::ref(W, H, colorbuffer_chw.data_d()); // Create a 3HW view on 4HW
            image_copy(*m_image, colorbuffer_3hw, m_stream);
        }
        image_copy(colorbuffer_chw, colorbuffer_hwc, m_stream);
        gl_mapped_resource.write(colorbuffer_hwc, m_stream);
        CHECK_CUDA(cudaStreamSynchronize(m_stream));

        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        draw_texture.draw(gl_mapped_resource.texture(), 0, 0, W, H);

        m_window->swap_buffers();
    }
}

std::unique_ptr<CudaImageVisualizer> CudaImageVisualizer::create(int width, int height, const char* title)
{
    std::shared_ptr<Window> window =
        std::make_shared<Window>(Window::create(width, height, title, false /* resizable */));
    return std::make_unique<CudaImageVisualizer>(window);
}
