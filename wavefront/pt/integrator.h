#pragma once

#include "type.h"
#include "pt_pass.h"

namespace wf {
    class Integrator {
    public:
        Integrator(Pupil::optix::Pass* ray_pass, Pupil::optix::Pass* shadow_ray_pass) noexcept
            : m_ray_pass(ray_pass), m_shadow_ray_pass(shadow_ray_pass) {}

        void SetFrameSize(unsigned int frame_size_x, unsigned int frame_size_y) noexcept {
            m_frame_size    = make_uint2(frame_size_x, frame_size_y);
            m_max_wave_size = frame_size_x * frame_size_y;
        }

        void SetMaxRayDepth(unsigned int max_depth) noexcept { m_max_depth = max_depth; }

        void Trace(Pupil::cuda::RWDataView<GlobalData>& g_data, Pupil::cuda::Stream* stream) noexcept;

    public:
        uint2        m_frame_size;
        unsigned int m_max_wave_size;
        unsigned int m_max_depth = 2;

        Pupil::optix::Pass* m_ray_pass;
        Pupil::optix::Pass* m_shadow_ray_pass;
    };
}// namespace wf