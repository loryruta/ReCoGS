#pragma once

#include <filesystem>
#include <list>
#include <queue>

#include "GSCamera.h"
#include "utils/image/Image.h"

namespace recogs
{
struct TrainingCameraPool_Entry {
    int camera_index;
    GSCamera camera;
    Image4fHWC image; // Example size: 1080 x 720 x 4 x sizeof(float) = 12MB
};

/// Thread-safe cache of training cameras and rendered color/depth images.
class TrainingCameraPool // TODO rename to cache
{
public:
    using RenderImageFunc =
        std::function<void(const GSCamera& camera, Image4fHWC& out_image, cudaStream_t stream)>;

private:
    std::vector<CameraData> m_cameras;
    glm::ivec2 m_resolution;
    size_t m_max_size;
    RenderImageFunc m_render_image_func;

    using CacheData = std::list<std::shared_ptr<TrainingCameraPool_Entry>>;
    CacheData m_cache; // LRU cache
    std::map<int, CacheData::iterator> m_cache_entry_by_key;

    std::mutex m_mutex;

public:
    explicit TrainingCameraPool(const std::vector<CameraData>& cameras,
                                glm::ivec2 initial_resolution,
                                size_t max_size,
                                RenderImageFunc render_image_func);
    ~TrainingCameraPool() = default;

    void set_resolution(glm::ivec2 resolution);

    [[nodiscard]] size_t size() const;
    /// Return a pair of host memory and device memory in bytes.
    [[nodiscard]] std::pair<size_t, size_t> bytesize() const;

    ///\ param camera_idx Index of the training camera to retrieve
    /// \param stream     CUDA stream used to initialized the image on cache miss
    std::shared_ptr<TrainingCameraPool_Entry> get(int camera_idx, cudaStream_t stream);
};
} // namespace recogs
