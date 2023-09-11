#pragma once
#include "cuda/vec_math.h"
#include "render/geometry.h"

namespace wavefront {
struct Ray {
    float3 dir;
    float3 origin;
    float t;
};

struct PathRecord {
    Ray ray;
    float3 throughput;
    float bsdf_sample_pdf;
    Pupil::optix::EBsdfLobeType bsdf_sample_type;
    Pupil::cuda::Random random;
    unsigned int pixel_index;
};

struct HitRecord {
    Ray ray;
    Pupil::optix::LocalGeometry geo;
    float3 throughput;
    float bsdf_sample_pdf;
    int emitter_index;
    Pupil::cuda::Random random;
    unsigned int pixel_index;

    Pupil::optix::material::Material::LocalBsdf bsdf;
};

struct HitEmitterRecord {
    Ray ray;
    Pupil::optix::LocalGeometry geo;
    float3 throughput;
    int emitter_index;
    float bsdf_sample_pdf;
    Pupil::optix::EBsdfLobeType bsdf_sample_type;
    unsigned int pixel_index;
};

struct MissRecord {
    Ray ray;
    float3 throughput;
    float bsdf_sample_pdf;
    unsigned int pixel_index;
};

struct ShadowPathRecord {
    Ray ray;
    float3 radiance;
    unsigned int pixel_index;
};
}// namespace wavefront