cmake_minimum_required(VERSION 3.29)
project(recogs LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif ()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Catch2
find_package(Catch2 CONFIG REQUIRED)

# glad
add_library(glad OBJECT ${THIRD_PARTY_DIR}/glad/src/glad.c)
target_include_directories(glad PUBLIC ${THIRD_PARTY_DIR}/glad/include/)

find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(TensorRT MODULE REQUIRED)

# onnxruntime
set(onnxruntime_DIR "${THIRD_PARTY_DIR}/onnxruntime/lib64/cmake/onnxruntime")
find_package(onnxruntime REQUIRED CONFIG)

add_subdirectory(recogs)
add_subdirectory(recogs_test)
