#pragma once

#include "wavefront.h"

#include "system/pass.h"
#include "system/world.h"

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

    PTPass(std::string_view name = "Wavefront Path Tracing") noexcept;
    virtual void Run() noexcept override;
    virtual void Inspector() noexcept override;

    virtual void BeforeRunning() noexcept override {}
    virtual void AfterRunning() noexcept override {}

    void SetScene(Pupil::World *) noexcept;

private:
    std::unique_ptr<Pupil::cuda::Stream> m_stream = nullptr;
    std::unique_ptr<OptixPass> m_ray_pass = nullptr;
    std::unique_ptr<ShadowOptixPass> m_shadow_ray_pass = nullptr;
};

}// namespace wavefront