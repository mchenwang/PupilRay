#include <optix.h>
#include "wavefront.h"

#include "optix/util.h"
#include "optix/geometry.h"
#include "optix/scene/emitter.h"
#include "material/bsdf/bsdf.h"

#include "cuda/random.h"

using namespace Pupil;
using namespace wavefront;

extern "C" {
__constant__ wavefront::GlobalData optix_launch_params;
}

extern "C" __global__ void __raygen__main() {
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= optix_launch_params.ray_index.GetNum()) return;
    auto pixel_index = optix_launch_params.ray_index[index];
    auto &ray = optix_launch_params.cur_bounce[pixel_index].ray;

    optixTrace(optix_launch_params.handle,
               ray.origin, ray.dir,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 1, 0);
}

extern "C" __global__ void __miss__default() {
    const unsigned int index = optixGetLaunchIndex().x;
    auto pixel_index = optix_launch_params.ray_index[index];

    MissRecord miss_info{};
    optix_launch_params.miss_record[pixel_index] = miss_info;
    optix_launch_params.miss_index.Push(pixel_index);
}
extern "C" __global__ void __closesthit__default() {
    const unsigned int index = optixGetLaunchIndex().x;
    auto pixel_index = optix_launch_params.ray_index[index];
    auto &ray = optix_launch_params.cur_bounce[pixel_index].ray;

    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();

    HitRecord hit_info;
    sbt_data->geo.GetHitLocalGeometry(hit_info.geo, ray.dir, sbt_data->mat.twosided);
    if (sbt_data->emitter_index_offset >= 0) {
        hit_info.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();

        optix_launch_params.bsdf_eval_index.Push(pixel_index);
    } else {
        hit_info.emitter_index = -1;
    }
    hit_info.bsdf = sbt_data->mat.GetLocalBsdf(hit_info.geo.texcoord);

    optix_launch_params.hit_record[pixel_index] = hit_info;

    optix_launch_params.hit_index.Push(pixel_index);
}