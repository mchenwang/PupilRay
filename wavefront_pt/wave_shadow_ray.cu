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
    if (index >= optix_launch_params.shadow_ray_index.GetNum()) return;
    auto pixel_index = optix_launch_params.shadow_ray_index[index];
    auto &ray = optix_launch_params.cur_bounce[pixel_index].shadow_ray;

    optixTrace(optix_launch_params.handle,
               ray.origin, ray.dir,
               0.0001f, ray.t, 0.f,
               255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 1, 0);
}
extern "C" __global__ void __miss__shadow() {
    const unsigned int index = optixGetLaunchIndex().x;
    auto pixel_index = optix_launch_params.shadow_ray_index[index];
    optix_launch_params.light_eval_index.Push(pixel_index);
}
extern "C" __global__ void __closesthit__shadow() {
}