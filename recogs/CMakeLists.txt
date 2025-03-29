
set(LIB_SOURCES
        depth_estimators/PCVNetEngine.cu
        depth_estimators/PCVNetEngine.h
        depth_estimators/StereoDepthEstimator.cu
        depth_estimators/StereoDepthEstimator.h

        gui/recogs_style.h
        gui/stereo_test.h

        fused_ssim/ssim.cu
        fused_ssim/ssim.h
        fused_ssim/ssim_if.cu
        fused_ssim/ssim_if.h

        optimizer/Adam.cu
        optimizer/Adam.h
        optimizer/GSLoss.cu
        optimizer/GSLoss.h
        optimizer/Optimizer.cu
        optimizer/Optimizer.h

        rasterizer/backward.cu
        rasterizer/backward.h
        rasterizer/forward.cu
        rasterizer/forward.h
        rasterizer/rasterizer.h
        rasterizer/rasterizer_impl.cu
        rasterizer/rasterizer_impl.h

        selection/Selection2d.cu
        selection/Selection2d.h
        selection/Selection3d.cu
        selection/Selection3d.h
        selection/SelectionRenderer.cu
        selection/SelectionRenderer.h

        ui/MainScreen.cu
        ui/MainScreen.h
        ui/Screen.h
        ui/SelectScreen.cu
        ui/SelectScreen.h

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
        utils/remove_statistical_outliers.h
        utils/stb_image.cpp
        utils/stb_image.h
        utils/stb_image_write.h
        utils/Stopwatch.h

        App.cu
        App.h
        GSCamera.cu
        GSCamera.h
        GSCameraController.cu
        GSCameraController.h
        GSRasterizer.cu
        GSRasterizer.h
        Scene.cu
        Scene.h
        scene_io.cu
        scene_io.h
)

set(VIDEO_SOURCES
        video/CudaImageVisualizer.cu
        video/CudaImageVisualizer.h
        video/CudaTexture.cu
        video/CudaTexture.h
        video/DrawTexture.cpp
        video/DrawTexture.h
        video/gl_utils.h
        video/GLMappedResource.cu
        video/GLMappedResource.h
        video/ImageSlider.cpp
        video/ImageSlider.h
        video/Window.cpp
        video/Window.h
)

add_library(recogs_lib STATIC ${LIB_SOURCES} ${VIDEO_SOURCES})
target_include_directories(recogs_lib PUBLIC ${CMAKE_SOURCE_DIR}/recogs)
target_link_libraries(recogs_lib PUBLIC
        cudaKDTree
        fmt::fmt
        glfw
        glm::glm
        imgui::imgui
        miniply
        nlohmann_json::nlohmann_json
        glad
        trt::nvinfer
        trt::nvonnxparser)

target_compile_options(recogs_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# main
add_executable(recogs main.cu)
target_link_libraries(recogs PRIVATE recogs_lib)
