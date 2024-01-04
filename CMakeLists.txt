cmake_minimum_required(VERSION 3.25.2)
project(PupilRay LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(pupil_target_output_dir ${PROJECT_SOURCE_DIR}/target CACHE INTERNAL "")

set(pupil_build_example ON CACHE BOOL "")
add_subdirectory(PupilOptixLab)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${pupil_target_output_dir}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${pupil_target_output_dir}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${pupil_target_output_dir}/lib)

# add_subdirectory(megakernel_pt)
# add_subdirectory(wavefront_pt)
# add_subdirectory(wavefront_pt_soa)
# add_subdirectory(test)

add_subdirectory(wavefront)