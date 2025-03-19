#pragma once

#include <filesystem>

#include "GSCamera.h"
#include "Scene.h"

namespace gs_train
{
Scene read_scene_from_ply(const std::filesystem::path& ply_file);

void write_scene_to_ply(const Scene& scene, const std::filesystem::path& out_ply_file);

std::vector<GSCamera> read_cameras_from_json(const std::filesystem::path& scene_folder, cudaStream_t stream);

} // namespace gs_train
