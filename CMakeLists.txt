cmake_minimum_required(VERSION 3.18)
project(SudoCUDA LANGUAGES CXX CUDA)

add_executable(SudoCUDA kernel.cu)