cmake_minimum_required(VERSION 3.18)
project(PTempProj LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(pupil_target_output_dir ${PROJECT_SOURCE_DIR}/target CACHE INTERNAL "")

set(pupil_build_example OFF CACHE BOOL "")
add_subdirectory(PupilOptixLab)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${pupil_target_output_dir}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${pupil_target_output_dir}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${pupil_target_output_dir}/lib)

add_subdirectory(wavefront)