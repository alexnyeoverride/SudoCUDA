cmake_minimum_required(VERSION 3.18)
project(SudoCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -rdc=true")

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

add_executable(SudoCUDA kernel.cu)

set_target_properties(SudoCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(SudoCUDA PRIVATE cudadevrt)
