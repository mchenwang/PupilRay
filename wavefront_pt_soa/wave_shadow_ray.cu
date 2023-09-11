#include <optix.h>
#include "wavefront.h"

#include "optix/util.h"
#include "render/geometry.h"
#include "render/emitter.h"
#include "render/material/bsdf/bsdf.h"

#include "cuda/random.h"

using namespace Pupil;
using namespace wavefront;

extern "C" {
__constant__ wavefront::GlobalData optix_launch_params;
}

extern "C" __global__ void __raygen__shadow() {
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= optix_launch_params.shadow_ray_queue.GetNum()) return;
    auto shadow_ray_path = optix_launch_params.shadow_ray_queue[index];
    auto ray = shadow_ray_path.ray();

    optixTrace(optix_launch_params.handle,
               ray.origin, ray.dir,
               0.0001f, ray.t, 0.f,
               255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 1, 0);
}
extern "C" __global__ void __miss__shadow() {
    const unsigned int index = optixGetLaunchIndex().x;
    auto shadow_ray_path = optix_launch_params.shadow_ray_queue[index];
    auto pixel_index = shadow_ray_path.pixel_index();

    float3 radiance = make_float3(optix_launch_params.frame_buffer[pixel_index]);
    radiance += shadow_ray_path.radiance();
    optix_launch_params.frame_buffer[pixel_index] = make_float4(radiance, 1.f);
}
extern "C" __global__ void __closesthit__shadow() {
}