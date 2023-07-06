#pragma once
#include <optix.h>
#include "cuda/data_view.h"
#include "cuda/vec_math.h"
#include "cuda/stream.h"
#include "cuda/util.h"

#include "optix/pass.h"
#include "optix/scene/camera.h"
#include "optix/scene/emitter.h"
#include "material/optix_material.h"

namespace wavefront {
struct HitGroupData {
    Pupil::optix::material::Material mat;
    Pupil::optix::Geometry geo;
    int emitter_index_offset = -1;
};

struct Ray {
    float3 dir;
    float3 origin;
    float t;
};

struct BounceRecord {
    Ray ray;
    Ray shadow_ray;
};

struct HitRecord {
    Pupil::optix::LocalGeometry geo;
    Pupil::optix::material::Material::LocalBsdf bsdf;
    int emitter_index;
};

struct MissRecord {
    int padding;
};

struct EmitterSample {
    Pupil::optix::EmitterSampleRecord record;
    float select_pdf;
};

struct BsdfSample {
    float pdf = 0.f;
    Pupil::optix::EBsdfLobeType sampled_type = Pupil::optix::EBsdfLobeType::Unknown;
};

struct GlobalData {
    bool accumulated_flag;
    unsigned int sample_cnt;
    unsigned int random_init_num;

    OptixTraversableHandle handle;

    Pupil::cuda::ConstDataView<Pupil::optix::Camera> camera;
    Pupil::optix::EmitterGroup emitters;

    Pupil::cuda::RWArrayView<BounceRecord> cur_bounce;
    Pupil::cuda::RWArrayView<HitRecord> hit_record;
    Pupil::cuda::RWArrayView<MissRecord> miss_record;
    Pupil::cuda::RWArrayView<EmitterSample> emitter_samples;
    Pupil::cuda::RWArrayView<BsdfSample> bsdf_samples;
    Pupil::cuda::RWArrayView<Pupil::cuda::Random> random;
    Pupil::cuda::RWArrayView<float3> throughput;

    Pupil::cuda::RWArrayView<float4> albedo_buffer;
    Pupil::cuda::RWArrayView<float4> normal_buffer;
    Pupil::cuda::RWArrayView<float4> frame_buffer;
    Pupil::cuda::RWArrayView<float4> accum_buffer;

    Pupil::cuda::DynamicArray<unsigned int> ray_index;
    Pupil::cuda::DynamicArray<unsigned int> shadow_ray_index;
    Pupil::cuda::DynamicArray<unsigned int> hit_index;
    Pupil::cuda::DynamicArray<unsigned int> miss_index;
    Pupil::cuda::DynamicArray<unsigned int> emitter_sample_index;
    Pupil::cuda::DynamicArray<unsigned int> light_eval_index;
    Pupil::cuda::DynamicArray<unsigned int> bsdf_eval_index;
};
}// namespace wavefront