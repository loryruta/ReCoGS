

add_executable(recogs_test
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

