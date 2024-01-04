#pragma once

#include "render/camera.h"
#include "render/emitter.h"
#include "render/material.h"
#include "render/geometry.h"

#include "cuda/util.h"

namespace wf {
    struct PathRecord {
        float3 ray_dir;
        float3 ray_origin;

        float                       bsdf_sample_pdf;
        Pupil::optix::EBsdfLobeType bsdf_sample_type;
    };

    struct HitRecord {
        Pupil::optix::Material::LocalBsdf mat;
        Pupil::optix::LocalGeometry       geo;
        int                               emitter_index = -1;
    };

    struct NEERecord {
        float  shadow_ray_t_max;
        float3 shadow_ray_dir;
        float3 shadow_ray_o;
        float3 radiance;
    };

    struct GlobalData {
        struct {
            unsigned int max_depth;
            bool         accumulated_flag;

            struct {
                unsigned int width;
                unsigned int height;
            } frame;
        } config;
        unsigned int random_seed;
        unsigned int sample_cnt;

        OptixTraversableHandle     handle;
        Pupil::optix::Camera       camera;
        Pupil::optix::EmitterGroup emitters;

        Pupil::cuda::RWArrayView<float4> accum_buffer;
        Pupil::cuda::RWArrayView<float4> frame_buffer;

        Pupil::cuda::RWArrayView<Pupil::cuda::Random> random;
        Pupil::cuda::RWArrayView<float3>              throughput;
        Pupil::cuda::RWArrayView<PathRecord>          path_record;
        Pupil::cuda::RWArrayView<HitRecord>           hit_record;
        Pupil::cuda::RWArrayView<NEERecord>           nee_record;

        Pupil::cuda::DynamicArray<unsigned int> ray_index;
        Pupil::cuda::DynamicArray<unsigned int> hit_index;
        Pupil::cuda::DynamicArray<unsigned int> shadow_ray_index;
    };

    struct HitGroupData {
        Pupil::optix::Material mat;
        Pupil::optix::Geometry geo;
        int                    emitter_index = -1;
    };

}// namespace wf