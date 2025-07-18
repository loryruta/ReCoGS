#pragma once

#include <utility>
#include <vector>

#include "GSCamera.h"
#include "Scene.h"

namespace recogs
{
// Forward decl
class App;

class ScenePointcloud
{
private:
    App& m_app;
    const std::filesystem::path m_output_filepath;

public:
    explicit ScenePointcloud(App& app, std::filesystem::path output_filepath)
        : m_app(app), m_output_filepath(std::move(output_filepath))
    {
    }
    ~ScenePointcloud() = default;

    /// Generate the pointcloud from all the training cameras of the scene.
    void generate(const Scene& scene, const std::vector<CameraData>& cameras, glm::ivec2 resolution);

    void export_voxels_to_pcd(const std::unordered_map<uint64_t, uint16_t>& voxels, const std::filesystem::path& out_filepath);
};
} // namespace recogs