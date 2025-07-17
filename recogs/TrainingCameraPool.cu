#include "TrainingCameraPool.h"

#include "App.h"
#include "scene_io.h"

using namespace recogs;

TrainingCameraPool::TrainingCameraPool(const std::vector<CameraData>& cameras,
                                       glm::ivec2 initial_resolution,
                                       size_t max_size,
                                       RenderImageFunc render_image_func)
    : m_cameras(cameras), m_resolution(initial_resolution), m_max_size(max_size), m_render_image_func(render_image_func)
{
}

void TrainingCameraPool::set_resolution(glm::ivec2 resolution)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cache.clear();

    m_resolution = resolution;
    for (auto& camera : m_cameras) {
        camera.set_resolution(resolution.x, resolution.y);
    }
}

size_t TrainingCameraPool::size() const { return m_cache.size(); }

std::pair<size_t, size_t> TrainingCameraPool::bytesize() const
{
    size_t cpu_bytesize = sizeof(TrainingCameraPool);
    cpu_bytesize += m_cache.size() * sizeof(TrainingCameraPool_Entry);
    size_t gpu_bytesize = m_cache.size() * m_resolution.x * m_resolution.y * sizeof(float) * 4; // Image4fHWC
    return {cpu_bytesize, gpu_bytesize};
}

std::shared_ptr<TrainingCameraPool_Entry> TrainingCameraPool::get(int camera_idx, cudaStream_t stream)
{
    CHECK_ARG(camera_idx >= 0 && camera_idx < m_cameras.size());

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto iterator = m_cache_entry_by_key.find(camera_idx);
        if (iterator != m_cache_entry_by_key.end()) {
            // The element is cached, move it to the front
            // https://stackoverflow.com/a/14580812/7358682
            m_cache.splice(m_cache.begin(), m_cache, iterator->second);
            return *m_cache_entry_by_key[camera_idx];
        }
    }
    // Create a new entry
    std::shared_ptr<TrainingCameraPool_Entry> entry_ptr;
    {
        GSCamera camera = GSCamera(m_cameras[camera_idx]);
        camera.update(stream);
        Image4fHWC image = Image4fHWC::malloc(m_resolution.x, m_resolution.y, stream);
        m_render_image_func(camera, image, stream);

        TrainingCameraPool_Entry entry{
            .camera_index = camera_idx, .camera = std::move(camera), .image = std::move(image)};
        entry_ptr = std::make_shared<TrainingCameraPool_Entry>(std::move(entry));
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_cache.size() >= m_max_size) {
            // Remove the least recently used element if the cache is full
            auto last = m_cache.back();
            m_cache.pop_back();
            m_cache_entry_by_key.erase(last->camera_index);
        }
        // Put the element
        m_cache.push_front(entry_ptr);
        m_cache_entry_by_key[camera_idx] = m_cache.begin();
    }
    return entry_ptr;
}
