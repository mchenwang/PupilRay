#pragma once
#include <optix.h>
#include "cuda/data_view.h"
#include "cuda/vec_math.h"
#include "cuda/stream.h"
#include "cuda/util.h"

#include "optix/pass.h"
#include "render/camera.h"
#include "render/emitter.h"
#include "render/material/optix_material.h"

#include "wave_context.h"
#include "soa_item.h"

namespace wavefront {
struct HitGroupData {
    Pupil::optix::material::Material mat;
    Pupil::optix::Geometry geo;
    int emitter_index_offset = -1;
};

struct GlobalData {
    bool accumulated_flag;
    unsigned int sample_cnt;
    unsigned int random_init_num;

    OptixTraversableHandle handle;

    Pupil::cuda::ConstDataView<Pupil::optix::Camera> camera;
    Pupil::optix::EmitterGroup emitters;

    SoA::PathQueue ray_queue;
    SoA::HitQueue hit_queue;
    SoA::HitEmitterQueue hit_emitter_queue;
    SoA::MissQueue miss_queue;
    SoA::ShadowPathQueue shadow_ray_queue;

    Pupil::cuda::RWArrayView<float4> albedo_buffer;
    Pupil::cuda::RWArrayView<float4> normal_buffer;
    Pupil::cuda::RWArrayView<float4> frame_buffer;
    Pupil::cuda::RWArrayView<float4> accum_buffer;
};
}// namespace wavefront