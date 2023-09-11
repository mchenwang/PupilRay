#pragma once

#include "wavefront.h"
#include "pt_pass.h"

namespace wavefront {
class Integrator {
public:
    Integrator(PTPass::OptixPass *ray_pass, PTPass::ShadowOptixPass *shadow_ray_pass) noexcept
        : m_ray_pass(ray_pass), m_shadow_ray_pass(shadow_ray_pass) {}

    void SetFrameSize(uint2 frame_size) noexcept {
        m_frame_size = frame_size;
        m_max_wave_size = frame_size.x * frame_size.y;
    }

    void SetMaxRayDepth(unsigned int max_depth) noexcept { m_max_depth = max_depth; }

    void Trace(Pupil::cuda::RWDataView<GlobalData> &g_data, Pupil::cuda::Stream *stream) noexcept;

public:
    uint2 m_frame_size;
    unsigned int m_max_wave_size;
    unsigned int m_max_depth = 2;

    PTPass::OptixPass *m_ray_pass;
    PTPass::ShadowOptixPass *m_shadow_ray_pass;
};
}// namespace wavefront