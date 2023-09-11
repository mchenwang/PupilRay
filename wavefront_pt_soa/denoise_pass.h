#pragma once

#include "system/pass.h"
#include "world/world.h"
#include "optix/denoiser.h"

#include "cuda/stream.h"

#include "util/timer.h"

#include <memory>
#include <mutex>

namespace Pupil {
class DenoisePass : public Pass {
public:
    static bool s_enabled_flag;

    DenoisePass(std::string_view name = "Denoise") noexcept;
    virtual void OnRun() noexcept override;
    virtual void Inspector() noexcept override;

    void SetScene(world::World *) noexcept;

private:
    std::unique_ptr<cuda::Stream> m_stream;
    std::unique_ptr<optix::Denoiser> m_denoiser;

    Timer m_timer;
};
}// namespace Pupil