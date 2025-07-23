
#include <cstdio>
#include <filesystem>

#include "App.h"
#include "depth_estimators/FoundationStereo.h"
#include "utils/Stopwatch.h"

USING_NAMESPACE

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

    { // TODO temporary
        printf("[INFO ] [DepthEstimator] Loading Foundation Stereo model...\n");
        Stopwatch stopwatch{};
        FoundationStereo_Options options{};
        options.onnx_filepath = "foundation_stereo_736_960.onnx";
        options.engine_filepath = "foundation_stereo_736_960.engine";
        auto foundation_stereo = std::make_unique<FoundationStereo>(options);
        foundation_stereo->build_or_load();
    }

    AppParams app_params{};
    app_params.scene_ply = std::filesystem::absolute(argv[0]);
    app_params.app_title = "RECOGS";

    App app(app_params);
    app.start();
    return 0;
}
