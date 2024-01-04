set(target_name wf_pt)

cuda_compile_and_embed(wave_ray_ptx wave_ray.cu)
cuda_compile_and_embed(wave_shadow_ray_ptx wave_shadow_ray.cu)
set(embedded_ptx_code ${wave_ray_ptx} ${wave_shadow_ray_ptx})

add_executable(${target_name}
    ${embedded_ptx_code}
    main.cpp
    pt_pass.h
    pt_pass.cpp
    type.h
    integrator.h
    integrator.cu
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES ${pupil_cuda_architectures})
set_target_properties(${target_name} PROPERTIES FOLDER ${wavefront_folder})
