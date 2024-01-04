#include <optix.h>
#include "type.h"

#include "render/util.h"
#include "render/geometry.h"
#include "render/material/bsdf/bsdf.h"

#include "type.h"

using namespace Pupil;
using namespace wf;

extern "C" {
__constant__ GlobalData optix_launch_params;
}

extern "C" __global__ void __raygen__shadow() {
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= optix_launch_params.nee_record.GetNum()) return;
    auto record = optix_launch_params.nee_record[index];

    optixTrace(optix_launch_params.handle,
               record.shadow_ray_origin(),
               record.shadow_ray_dir(),
               0.0001f,
               record.shadow_ray_t_max(),
               0.f,
               255,
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0,
               1,
               0);
}
extern "C" __global__ void __miss__shadow() {
    const unsigned int index       = optixGetLaunchIndex().x;
    auto               record      = optix_launch_params.nee_record[index];
    auto               pixel_index = record.pixel_index();

    float3 radiance = make_float3(optix_launch_params.frame_buffer[pixel_index]);
    radiance += record.radiance();
    optix_launch_params.frame_buffer[pixel_index] = make_float4(radiance, 1.f);
}
extern "C" __global__ void __closesthit__shadow() {
}