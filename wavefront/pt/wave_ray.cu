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

extern "C" __global__ void __raygen__main() {
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= optix_launch_params.ray_index.GetNum()) return;
    auto pixel_index = optix_launch_params.ray_index[index];
    auto ray_dir     = optix_launch_params.path_record[pixel_index].ray_dir;
    auto ray_origin  = optix_launch_params.path_record[pixel_index].ray_origin;

    // optix_launch_params.frame_buffer[pixel_index] = make_float4(abs(ray_dir.x), abs(ray_dir.y), abs(ray_dir.z), 1.f);

    // optix_launch_params.frame_buffer[pixel_index] = make_float4(abs(ray_origin.x), abs(ray_origin.y), abs(ray_origin.z), 1.f);

    optixTrace(optix_launch_params.handle,
               ray_origin,
               ray_dir,
               0.001f,
               1e16f,
               0.f,
               255,
               OPTIX_RAY_FLAG_NONE,
               0,
               1,
               0);
}

extern "C" __global__ void __miss__default() {
    // const unsigned int index       = optixGetLaunchIndex().x;
    // auto               pixel_index = optix_launch_params.ray_index[index];
    // MissRecord miss_info{};
    // optix_launch_params.miss_record[pixel_index] = miss_info;
    // optix_launch_params.miss_index.Push(pixel_index);
}

extern "C" __global__ void __closesthit__default() {
    const unsigned int index       = optixGetLaunchIndex().x;
    auto               pixel_index = optix_launch_params.ray_index[index];
    auto               ray_dir     = optix_launch_params.path_record[pixel_index].ray_dir;
    auto               ray_origin  = optix_launch_params.path_record[pixel_index].ray_origin;

    const HitGroupData* sbt_data = (HitGroupData*)optixGetSbtDataPointer();

    HitRecord hit_info;
    sbt_data->geo.GetHitLocalGeometry(hit_info.geo, ray_dir, sbt_data->mat.twosided);
    hit_info.mat = sbt_data->mat.GetLocalBsdf(hit_info.geo.texcoord);

    optix_launch_params.hit_record[pixel_index] = hit_info;
    optix_launch_params.hit_index.Push(pixel_index);

    if (sbt_data->emitter_index < 0) return;
    auto& emitter = optix_launch_params.emitters[sbt_data->emitter_index];

    optix::EmitEvalRecord emit_record;
    emit_record.primitive_index = optixGetPrimitiveIndex();
    emitter.Eval(emit_record, hit_info.geo, ray_origin);

    auto bsdf_sample_type = optix_launch_params.path_record[pixel_index].bsdf_sample_type;
    auto bsdf_sample_pdf  = optix_launch_params.path_record[pixel_index].bsdf_sample_pdf;

    float mis = bsdf_sample_type & optix::EBsdfLobeType::Delta ?
                    1.f :
                    optix::MISWeight(bsdf_sample_pdf, emit_record.pdf * emitter.select_probability);

    float3 radiance   = make_float3(optix_launch_params.frame_buffer[pixel_index]);
    float3 throughput = optix_launch_params.throughput[pixel_index];
    radiance += emit_record.radiance * mis * throughput;
    optix_launch_params.frame_buffer[pixel_index] = make_float4(radiance, 1.f);

    // if (sbt_data->emitter_index >= 0) {
    //     auto& emitter = optix_launch_params.emitters[sbt_data->emitter_index];
    //     record->radiance += emitter.GetRadiance(record->hit.geo.texcoord);
    // }

    // if (sbt_data->emitter_index_offset >= 0) {
    //     hit_info.emitter_index = sbt_data->emitter_index_offset + optixGetPrimitiveIndex();

    //     optix_launch_params.bsdf_eval_index.Push(pixel_index);
    // } else {
    //     hit_info.emitter_index = -1;
    // }
    // hit_info.bsdf = sbt_data->mat.GetLocalBsdf(hit_info.geo.texcoord);
}