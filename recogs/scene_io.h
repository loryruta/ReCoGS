#pragma once

#include <filesystem>

#include "GSCamera.h"
#include "Scene.h"

namespace recogs
{
Scene read_scene_from_ply(const std::filesystem::path& ply_file);

void write_scene_to_ply(const Scene& scene, const std::filesystem::path& out_ply_file);

struct CameraData {
    glm::vec3 position;
    glm::quat rotation;
    float fx;
    float fy;
    int width;
    int height;

    void set_resolution(int new_width, int new_height)
    {
        float aspect = float(new_height) / float(height);
        fx *= aspect;
        fy *= aspect;
        width = new_width;
        height = new_height;
    }
};

void read_cameras_from_json(const std::filesystem::path& scene_folder, std::vector<CameraData>& out_cameras);

/// Entry of the cameras.bin file
struct ColmapCamera {
    uint32_t camera_id;
    uint32_t model_id;
    uint64_t width, height;
    double params[12];
};

/// Entry of the images.bin file
struct ColmapImage {
    uint32_t image_id;
    double R[4];
    double t[3];
    uint32_t camera_id;
    std::string image_name;
    std::vector<std::tuple<double, double, int64_t>> point_xy_ids;
};

///
class ColmapOutputReader
{
public:
    explicit ColmapOutputReader();
    ~ColmapOutputReader() = default;

    /// Read COLMAP output files (cameras.bin, images.bin), and output a list of cameras
    std::vector<GSCamera> read_cameras(const std::filesystem::path& filepath);

    std::pair<glm::mat4, float> read_dataparser_transforms(const std::filesystem::path& dataparser_transforms_filepath);
    /// Read cameras from the cameras.bin file
    std::unordered_map<uint32_t, ColmapCamera> read_cameras_(const std::filesystem::path& camerasbin_filepath);
    /// Read images from the images.bin file
    std::vector<ColmapImage> read_images(const std::filesystem::path& imagesbin_filepath);
};

///
class NerfStudioOutputReader
{
private:
    ColmapOutputReader& m_colmap_reader;

public:
    explicit NerfStudioOutputReader(ColmapOutputReader& colmap_reader) : m_colmap_reader(colmap_reader) {}
    ~NerfStudioOutputReader() = default;

    std::vector<GSCamera> read_cameras(const std::filesystem::path& output_dir);
};

} // namespace recogs
