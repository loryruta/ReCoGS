
set(LIB_SOURCES
        depth_estimators/PCVNet.cu
        depth_estimators/PCVNet.h
        depth_estimators/EstimateDepth.cu
        depth_estimators/EstimateDepth.h

        fused_ssim/ssim.cu
        fused_ssim/ssim.h
        fused_ssim/ssim_if.cu
        fused_ssim/ssim_if.h

        rasterizer/backward.cu
        rasterizer/backward.h
        rasterizer/forward.cu
        rasterizer/forward.h
        rasterizer/rasterizer.h
        rasterizer/rasterizer_impl.cu
        rasterizer/rasterizer_impl.h

        ui/MainScreen.cu
        ui/MainScreen.h
        ui/Screen.h
#        ui/SelectionScreen.cu
#        ui/SelectionScreen.h

        utils/image/image.cu
        utils/image/Image.h
        utils/image/image_cast.h
        utils/image/image_copy.h
        utils/image/image_downsample.h
        utils/image/image_fill.h
        utils/image/image_flip.h
        utils/image/image_load.h
        utils/image/image_rotate.h
        utils/image/image_save.h
        utils/image/image_visit_transform.h
        utils/DeviceBuffer.h
        utils/cuda_utils.h
        utils/misc_utils.h
        utils/stb_image.cpp
        utils/stb_image.h
        utils/stb_image_write.h
        utils/Stopwatch.h

        adam.cu
        adam.h
        App.cu
        App.h
        GSCamera.cu
        GSCamera.h
        GSCameraController.cu
        GSCameraController.h
        GSRasterizer.cu
        GSRasterizer.h
        gs_loss.cu
        gs_loss.h
        Scene.cu
        Scene.h
        scene_io.cu
        scene_io.h
#        Selection2d.cu
#        Selection2d.h
)

set(VIDEO_SOURCES
        video/CudaImageVisualizer.cu
        video/CudaImageVisualizer.h
        video/DrawTexture.cpp
        video/DrawTexture.h
        video/gl_utils.h
        video/GLMappedResource.cu
        video/GLMappedResource.h
        video/Window.cpp
        video/Window.h
)

add_library(recogs_lib STATIC ${LIB_SOURCES} ${VIDEO_SOURCES})
target_include_directories(recogs_lib PUBLIC ${CMAKE_SOURCE_DIR}/recogs)
target_link_libraries(recogs_lib PUBLIC
        fmt::fmt
        glfw
        glm::glm
        imgui::imgui
        nlohmann_json::nlohmann_json
        onnxruntime::onnxruntime
        glad)

# main
add_executable(recogs main.cu)
target_link_libraries(recogs PRIVATE recogs_lib)

# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

add_executable(test_img_opt test/test_img_opt.cu)
target_link_libraries(test_img_opt PRIVATE recogs_lib)
