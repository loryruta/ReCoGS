#pragma once

#include <filesystem>

#include "Scene.h"

namespace gs_train
{
Scene read_scene_from_ply(const std::filesystem::path& ply_file);

void write_scene_to_ply(const Scene& scene, const std::filesystem::path& out_ply_file);
} // namespace gs_train
