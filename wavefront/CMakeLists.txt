set(target_name wavefront_pt)

cuda_compile_and_embed(wave_ray_ptx wave_ray.cu)
cuda_compile_and_embed(wave_shadow_ray_ptx wave_shadow_ray.cu)

add_executable(${target_name}
    ${wave_ray_ptx}
    ${wave_shadow_ray_ptx}
    main.cpp
    pt_pass.h
    pt_pass.cpp
    wavefront.h
    integrator.h
    integrator.cu
    denoise_pass.h
    denoise_pass.cpp
)

set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES ${pupil_cuda_architectures})
target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})