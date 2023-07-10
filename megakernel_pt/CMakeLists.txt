set(target_name megakernel_pt)

cuda_compile_and_embed(embedded_ptx_code main.cu)

add_executable(${target_name}
    ${embedded_ptx_code}
    main.cpp
    pt_pass.h
    pt_pass.cpp
    type.h
    denoise_pass.h
    denoise_pass.cpp
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
