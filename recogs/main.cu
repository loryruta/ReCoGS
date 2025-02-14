
#include <cstdio>
#include <filesystem>

#include "App.h"

using namespace gs_train;

int main(int argc, char* argv[])
{
    argc--;
    if (argc != 1) {
        fprintf(stderr, "Invalid syntax: %s <scene-ply>\n", argv[0]);
        return 1;
    }
    ++argv;

    App::Params app_params{};
    app_params.scene_ply = std::filesystem::absolute(argv[0]);

    App app(app_params);
    app.start();
    return 0;
}
