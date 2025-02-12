
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

        main.cpp
        Scene.cu
        Scene.h
        scene_io.cpp
        scene_io.h
)

set(VIDEO_SOURCES
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