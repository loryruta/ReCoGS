

add_executable(recogs_test
        depth_estimators/test_stereo.cu
        depth_estimators/test_pcvnet_trt.cu

        optimizer/test_img_opt.cu

        utils/image/test_image_copy.cu
        utils/image/test_image_downsample.cu
        utils/image/test_image_flip.cu
        #utils/image/test_image_reduce.cu
        utils/image/test_image_rotate.cu
)
target_link_libraries(recogs_test PRIVATE
        recogs_lib
        Catch2::Catch2
        Catch2::Catch2WithMain)
target_compile_definitions(recogs_test PRIVATE
        DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"
)

