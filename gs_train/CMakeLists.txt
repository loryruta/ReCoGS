
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

        utils/buffer.h
        utils/cuda_utils.h
        utils/image_layout_transition.h
        utils/misc_utils.h
        utils/stb_image.cpp
        utils/stb_image.h
        utils/stb_image_write.h

        adam.cu
        adam.h
        Camera.cpp
        Camera.h
        gs_func.cu
        gs_func.h
        gs_loss.cu
        gs_loss.h
        Scene.cu
        Scene.h
        scene_io.cpp
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
        video/GLWindow.cpp
        video/GLWindow.h
)

add_library(gs_train STATIC ${LIB_SOURCES})
target_include_directories(gs_train PUBLIC ${CMAKE_SOURCE_DIR}/gs_train)
target_link_libraries(gs_train PUBLIC glm json)

# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

add_executable(test_img_opt test/test_img_opt.cu ${VIDEO_SOURCES})
target_link_libraries(test_img_opt PRIVATE gs_train glad glfw)

add_executable(test_gs_opt test/test_gs_opt.cu ${VIDEO_SOURCES})
target_link_libraries(test_gs_opt PRIVATE gs_train glad glfw)
