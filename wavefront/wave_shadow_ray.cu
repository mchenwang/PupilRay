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

extern "C" __global__ void __raygen__shadow() {
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= optix_launch_params.shadow_ray_index.GetNum()) return;
    auto pixel_index = optix_launch_params.shadow_ray_index[index];
    auto &ray = optix_launch_params.cur_bounce[pixel_index].shadow_ray;

    optixTrace(optix_launch_params.handle,
               ray.origin, ray.dir,
               0.0001f, ray.t, 0.f,
               255, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 2, 0);
}
extern "C" __global__ void __miss__shadow() {
    const unsigned int index = optixGetLaunchIndex().x;
    auto pixel_index = optix_launch_params.shadow_ray_index[index];
    optix_launch_params.light_eval_index.Push(pixel_index);

    // auto &emitter_sample = optix_launch_params.emitter_samples[pixel_index];
    // auto &ray = optix_launch_params.cur_bounce[pixel_index].ray;
    // auto &hit = optix_launch_params.hit_record[pixel_index];

    // optix::BsdfSamplingRecord eval_record;
    // eval_record.wi = optix::ToLocal(emitter_sample.record.wi, hit.geo.normal);
    // eval_record.wo = optix::ToLocal(-ray.dir, hit.geo.normal);

    // eval_record.sampler = &optix_launch_params.random[pixel_index];
    // hit.bsdf.Eval(eval_record);
    // float3 f = eval_record.f;
    // float pdf = eval_record.pdf;
    // if (!optix::IsZero(f * emitter_sample.record.pdf)) {
    //     float NoL = abs(dot(hit.geo.normal, emitter_sample.record.wi));
    //     float mis = emitter_sample.record.is_delta ? 1.f : optix::MISWeight(emitter_sample.record.pdf, pdf);
    //     emitter_sample.record.pdf *= emitter_sample.select_pdf;

    //     float3 radiance = make_float3(optix_launch_params.frame_buffer[pixel_index]);
    //     float throughput = optix_launch_params.frame_buffer[pixel_index].w;
    //     radiance += throughput * emitter_sample.record.radiance * f * NoL * mis / emitter_sample.record.pdf;

    //     optix_launch_params.frame_buffer[pixel_index] = make_float4(radiance, throughput);
    // }
    // optix_launch_params.frame_buffer[pixel_index].w = 1.f;
}
extern "C" __global__ void __closesthit__shadow() {
}