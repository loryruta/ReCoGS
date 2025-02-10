cmake_minimum_required(VERSION 3.29)
project(gs_editor LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(gs_editor main.cpp)

add_subdirectory(gs_train)
