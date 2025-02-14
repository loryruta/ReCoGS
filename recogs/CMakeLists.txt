
set(LIB_SOURCES
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

        utils/DeviceBuffer.h
        utils/cuda_utils.h
        utils/image_layout_transition.h
        utils/misc_utils.h
        utils/stb_image.cpp
        utils/stb_image.h
        utils/stb_image_write.h

        adam.cu
        adam.h
        GSCamera.cu
        GSCamera.h
        GSCameraController.cu
        GSCameraController.h
        GSFunc.cu
        GSFunc.h
        gs_loss.cu
        gs_loss.h
        Scene.cu
        Scene.h
        scene_io.cu
        scene_io.h
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

add_library(recogs_lib STATIC ${LIB_SOURCES})
target_include_directories(recogs_lib PUBLIC ${CMAKE_SOURCE_DIR}/recogs)
target_link_libraries(recogs_lib PUBLIC glm json glad glfw)

# main
add_executable(recogs main.cu ${VIDEO_SOURCES})
target_link_libraries(recogs PRIVATE recogs_lib)

# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

add_executable(test_img_opt test/test_img_opt.cu ${VIDEO_SOURCES})
target_link_libraries(test_img_opt PRIVATE recogs_lib)
