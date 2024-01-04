#pragma once

#include "render/camera.h"
#include "render/emitter.h"
#include "render/material.h"
#include "render/geometry.h"

#include "cuda/util.h"
#include "soa_wrap.h"

namespace wf {
    struct PathRecord {
        float3 ray_dir;
        float3 ray_origin;
        float3 throughput;

        float                       bsdf_sample_pdf;
        Pupil::optix::EBsdfLobeType bsdf_sample_type;

        unsigned int random_seed;
        unsigned int pixel_index;
    };

    struct HitRecord {
        Pupil::optix::Material::LocalBsdf mat;
        Pupil::optix::LocalGeometry       geo;
        float3                            ray_dir;
        float3                            throughput;
        int                               emitter_index = -1;
        unsigned int                      random_seed;
        unsigned int                      pixel_index;
    };

    struct NEERecord {
        float        shadow_ray_t_max;
        float3       shadow_ray_dir;
        float3       shadow_ray_origin;
        float3       radiance;
        unsigned int pixel_index;
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
        unsigned int depth;

        OptixTraversableHandle     handle;
        Pupil::optix::Camera       camera;
        Pupil::optix::EmitterGroup emitters;

        Pupil::cuda::RWArrayView<float4> accum_buffer;
        Pupil::cuda::RWArrayView<float4> frame_buffer;

        wf::soa::PathRecordArray path_record;
        wf::soa::HitRecordArray  hit_record;
        wf::soa::NEERecordArray  nee_record;
    };

    struct HitGroupData {
        Pupil::optix::Material mat;
        Pupil::optix::Geometry geo;
        int                    emitter_index = -1;
    };

}// namespace wf