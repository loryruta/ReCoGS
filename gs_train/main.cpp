
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "Scene.h"
#include "scene_io.h"

using namespace gs_train;

int main(int argc, char* argv[])
{
    argc--;
    if (argc != 2) {
        fprintf(stderr, "Invalid syntax: %s <scene-ply> <edit-dataset>\n", argv[0]);
        return 1;
    }
    ++argv;

    std::filesystem::path scene_ply = argv[0];
    if (!std::filesystem::exists(scene_ply)) {
        fprintf(stderr, "Invalid scene ply: %s\n", scene_ply.c_str());
        return 1;
    }
    std::filesystem::path edit_dataset_dir = argv[1];
    if (!std::filesystem::exists(edit_dataset_dir)) {
        fprintf(stderr, "Invalid dataset directory: %s\n", edit_dataset_dir.c_str());
        return 1;
    }

    std::filesystem::path output_ply = "output.ply"; // Output written at user's cwd

    Scene scene = read_scene_from_ply(scene_ply);
    printf("Scene loaded: %s\n", std::filesystem::absolute(scene_ply).c_str());

    write_scene_to_ply(scene, output_ply);
    printf("Output written to: %s\n", std::filesystem::absolute(output_ply).c_str());

    return 0;
}
