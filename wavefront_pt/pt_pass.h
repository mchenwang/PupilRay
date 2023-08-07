#pragma once

#include "wavefront.h"

#include "system/pass.h"
#include "world/world.h"

#include "cuda/stream.h"
#include "optix/pass.h"

#include <memory>

namespace wavefront {
struct SBTTypes : public Pupil::optix::EmptySBT {
    using HitGroupDataType = HitGroupData;
};
using ShadowSBTTypes = Pupil::optix::EmptySBT;

class PTPass : public Pupil::Pass {
public:
    using OptixPass = Pupil::optix::Pass<SBTTypes, wavefront::GlobalData>;
    using ShadowOptixPass = Pupil::optix::Pass<ShadowSBTTypes, wavefront::GlobalData>;

    constexpr static std::string_view PT_RESULT_BUFFER = "pt result buffer";

    PTPass(std::string_view name = "Wavefront Path Tracing") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;

    void SetScene(Pupil::world::World *) noexcept;

private:
    std::unique_ptr<Pupil::cuda::Stream> m_stream = nullptr;
    std::unique_ptr<OptixPass> m_ray_pass = nullptr;
    std::unique_ptr<ShadowOptixPass> m_shadow_ray_pass = nullptr;
};

}// namespace wavefront