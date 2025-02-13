cmake_minimum_required(VERSION 3.29)
project(recogs LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif ()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# glad
add_library(glad OBJECT ${THIRD_PARTY_DIR}/glad/src/glad.c)
target_include_directories(glad PUBLIC ${THIRD_PARTY_DIR}/glad/include/)

# GLFW
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_INSTALL OFF CACHE BOOL "")
add_subdirectory(${THIRD_PARTY_DIR}/glfw)

# glm
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${THIRD_PARTY_DIR}/glm)

# json
add_library(json INTERFACE)
target_include_directories(json INTERFACE ${THIRD_PARTY_DIR}/json/include)

add_subdirectory(recogs)
