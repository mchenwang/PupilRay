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

extern "C" __global__ void __raygen__main() {
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= optix_launch_params.ray_queue.GetNum()) return;
    auto path = optix_launch_params.ray_queue[index];
    auto ray = path.ray();

    optixTrace(optix_launch_params.handle,
               ray.origin, ray.dir,
               0.001f, 1e16f, 0.f,
               255, OPTIX_RAY_FLAG_NONE,
               0, 1, 0);
}

extern "C" __global__ void __miss__default() {
    const unsigned int index = optixGetLaunchIndex().x;

    auto path = optix_launch_params.ray_queue[index];
    auto miss = optix_launch_params.miss_queue.Alloc();
    miss.ray(path.ray());
    miss.throughput(path.throughput());
    miss.bsdf_sample_pdf(path.bsdf_sample_pdf());
    miss.pixel_index(path.pixel_index());
}
extern "C" __global__ void __closesthit__default() {
    const unsigned int index = optixGetLaunchIndex().x;
    auto path = optix_launch_params.ray_queue[index];
    auto ray = path.ray();
    auto throughput = path.throughput();
    auto bsdf_sample_pdf = path.bsdf_sample_pdf();
    auto bsdf_sample_type = path.bsdf_sample_type();
    auto pixel_index = path.pixel_index();

    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();
    auto hit = optix_launch_params.hit_queue.Alloc();
    hit.random(path.random());
    hit.ray(ray);
    hit.throughput(throughput);
    hit.pixel_index(path.pixel_index());

    optix::LocalGeometry geo;
    sbt_data->geo.GetHitLocalGeometry(geo, ray.dir, sbt_data->mat.twosided);
    hit.geo(geo);

    auto bsdf = sbt_data->mat.GetLocalBsdf(geo.texcoord);
    hit.bsdf(bsdf);

    int emitter_index = -1;
    if (sbt_data->emitter_index_offset >= 0) {
        emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();
        auto &emitter = optix_launch_params.emitters.areas[emitter_index];
        auto emission = emitter.GetRadiance(geo.texcoord);

        auto hit_emitter = optix_launch_params.hit_emitter_queue.Alloc();
        hit_emitter.ray(ray);
        hit_emitter.geo(geo);
        hit_emitter.throughput(throughput);
        hit_emitter.bsdf_sample_pdf(bsdf_sample_pdf);
        hit_emitter.bsdf_sample_type(bsdf_sample_type);
        hit_emitter.pixel_index(pixel_index);
    }
    hit.emitter_index(emitter_index);
}