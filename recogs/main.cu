
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

    int runtime_version;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    printf("[INFO ] CUDA runtime version: %d\n", runtime_version);

    int driver_version;
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    printf("[INFO ] CUDA driver version: %d\n", driver_version);

#ifdef CUB_DEBUG_SYNC
    printf("[INFO ] CUB_DEBUG_SYNC enabled\n");
#endif

    App::Params app_params{};
    app_params.scene_ply = std::filesystem::absolute(argv[0]);

    App app(app_params);
    app.start();
    return 0;
}
