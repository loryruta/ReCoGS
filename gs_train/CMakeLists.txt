


add_executable(gs_train
        rasterizer/backward.cu
        rasterizer/backward.h
        rasterizer/forward.cu
        rasterizer/forward.h
        rasterizer/rasterizer.h
        rasterizer/rasterizer_impl.cu
        rasterizer/rasterizer_impl.h

        main.cpp
)

target_include_directories(gs_train PRIVATE
        ${CMAKE_SOURCE_DIR}/gs_train
        ${CMAKE_SOURCE_DIR}/third_party/glm
)
