#pragma once

#include "system/pass.h"
#include "optix/pass.h"

namespace wf {
    class PTPass : public Pupil::Pass {
    public:
        PTPass(std::string_view name = "Wavefront Path Tracing") noexcept;
        ~PTPass() noexcept;
        virtual void OnRun() noexcept override;
        virtual void Console() noexcept override;
        virtual void Synchronize() noexcept override;

    private:
        void InitPipeline() noexcept;
        void BindingEventCallback() noexcept;

        struct Impl;
        Impl* m_impl = nullptr;
    };
}// namespace wf