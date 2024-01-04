#include "pt_pass.h"
#include "type.h"
#include "imgui.h"

#include "cuda/context.h"
#include "cuda/check.h"
#include "optix/context.h"
#include "optix/pipeline.h"

#include "system/system.h"
#include "system/event.h"
#include "system/buffer.h"
#include "system/world.h"
#include "system/profiler.h"
#include "system/gui/pass.h"

#include "render/camera.h"

#include "integrator.h"

#include <memory>
#include <mutex>

extern "C" char wave_ray_ptx[];
extern "C" char wave_shadow_ray_ptx[];

using namespace Pupil;

struct wf::PTPass::Impl {
    int  max_depth;
    bool accumulated_flag;

    Pupil::Scene*        scene = nullptr;
    Pupil::optix::Camera camera;

    util::CountableRef<cuda::Stream> stream;

    std::unique_ptr<Pupil::optix::Pass> ray_pass        = nullptr;
    std::unique_ptr<Pupil::optix::Pass> shadow_ray_pass = nullptr;
    std::unique_ptr<wf::Integrator>     integrator      = nullptr;

    GlobalData                       optix_launch_params;
    CUdeviceptr                      optix_launch_params_cuda_memory = 0;
    cuda::RWDataView<wf::GlobalData> global_data_view;

    size_t output_pixel_num = 0;

    CUdeviceptr dynamic_array_cnt_buffer = 0;

    std::atomic_bool dirty = true;

    Timer* timer;

    int frame_cnt = 0;
};

namespace wf {
    PTPass::PTPass(std::string_view name) noexcept
        : Pupil::Pass(name) {
        m_impl = new Impl();

        m_impl->stream          = util::Singleton<cuda::StreamManager>::instance()->Alloc(cuda::EStreamTaskType::Render, "wavefront pt");
        m_impl->ray_pass        = std::make_unique<optix::Pass>(sizeof(wf::GlobalData), m_impl->stream);
        m_impl->shadow_ray_pass = std::make_unique<optix::Pass>(sizeof(wf::GlobalData), m_impl->stream);
        m_impl->integrator      = std::make_unique<wf::Integrator>(m_impl->ray_pass.get(), m_impl->shadow_ray_pass.get());

        m_impl->timer = util::Singleton<Profiler>::instance()
                            ->AllocTimer(name, m_impl->stream, 60);

        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&m_impl->optix_launch_params_cuda_memory),
            sizeof(GlobalData),
            *m_impl->stream));

        m_impl->global_data_view.SetData(m_impl->optix_launch_params_cuda_memory);

        InitPipeline();
        BindingEventCallback();

        util::Singleton<Event::Center>::instance()
            ->Send(Gui::Event::CanvasDisplayTargetChange, {std::string{"pt result"}});
    }

    PTPass::~PTPass() noexcept {
        CUDA_FREE(m_impl->optix_launch_params_cuda_memory);
        CUDA_FREE(m_impl->dynamic_array_cnt_buffer);
        delete m_impl;
    }

    void PTPass::OnRun() noexcept {
        if (m_impl->dirty) {
            m_impl->optix_launch_params.camera                  = m_impl->camera;
            m_impl->optix_launch_params.config.max_depth        = m_impl->max_depth;
            m_impl->optix_launch_params.config.accumulated_flag = m_impl->accumulated_flag;
            m_impl->optix_launch_params.sample_cnt              = 0;
            m_impl->optix_launch_params.random_seed             = 0;
            m_impl->frame_cnt                                   = 0;
            m_impl->optix_launch_params.handle                  = m_impl->scene->GetIASHandle(1, true);
            m_impl->optix_launch_params.emitters                = m_impl->scene->GetOptixEmitters();
            m_impl->dirty                                       = false;
        }

        m_impl->timer->Start();
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(m_impl->optix_launch_params_cuda_memory),
            &m_impl->optix_launch_params,
            sizeof(GlobalData),
            cudaMemcpyHostToDevice,
            *m_impl->stream));

        m_impl->integrator->Trace(m_impl->global_data_view, m_impl->stream.Get());
        m_impl->timer->Stop();

        m_impl->frame_cnt++;

        m_impl->optix_launch_params.sample_cnt += m_impl->optix_launch_params.config.accumulated_flag;
        ++m_impl->optix_launch_params.random_seed;
    }

    void PTPass::Synchronize() noexcept {
        m_impl->stream->Synchronize();
    }

    void PTPass::Console() noexcept {
        Pupil::Pass::Console();

        util::Singleton<Profiler>::instance()->ShowPlot(name);

        ImGui::Text("sample cnt: %d", m_impl->frame_cnt);

        ImGui::InputInt("max trace depth", &m_impl->max_depth);
        m_impl->max_depth = clamp(m_impl->max_depth, 1, 128);
        if (m_impl->optix_launch_params.config.max_depth != m_impl->max_depth) {
            m_impl->dirty = true;
        }

        if (ImGui::Checkbox("accumulate radiance", &m_impl->accumulated_flag)) {
            m_impl->dirty = true;
        }

        if (ImGui::Button("reset")) {
            m_impl->dirty = true;
        }
    }

    void PTPass::BindingEventCallback() noexcept {
        auto event_center = util::Singleton<Event::Center>::instance();
        event_center->BindEvent(
            Event::DispatcherRender, Event::CameraChange, new Event::Handler0A([this]() {
                m_impl->camera.camera_to_world  = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetToWorldMatrix());
                m_impl->camera.sample_to_camera = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetSampleToCameraMatrix());

                m_impl->dirty = true;
            }));

        event_center->BindEvent(Event::DispatcherRender, Event::InstanceChange, new Event::Handler0A([this]() { m_impl->dirty = true; }));
        event_center->BindEvent(Event::DispatcherRender, Event::CameraChange, new Event::Handler0A([this]() { m_impl->frame_cnt = 0; }));

        event_center->BindEvent(
            Event::DispatcherRender, Event::SceneReset, new Event::Handler0A([this]() {
                m_impl->scene = util::Singleton<World>::instance()->GetScene();

                m_impl->camera.camera_to_world  = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetToWorldMatrix());
                m_impl->camera.sample_to_camera = cuda::MakeMat4x4(m_impl->scene->GetCamera()->GetSampleToCameraMatrix());

                m_impl->optix_launch_params.config.frame.width      = m_impl->scene->film_w;
                m_impl->optix_launch_params.config.frame.height     = m_impl->scene->film_h;
                m_impl->optix_launch_params.config.max_depth        = m_impl->scene->max_depth;
                m_impl->optix_launch_params.config.accumulated_flag = true;

                m_impl->max_depth        = m_impl->optix_launch_params.config.max_depth;
                m_impl->accumulated_flag = m_impl->optix_launch_params.config.accumulated_flag;

                m_impl->optix_launch_params.random_seed = 0;
                m_impl->optix_launch_params.sample_cnt  = 0;

                m_impl->output_pixel_num = m_impl->optix_launch_params.config.frame.width *
                                           m_impl->optix_launch_params.config.frame.height;

                m_impl->integrator->SetFrameSize(m_impl->optix_launch_params.config.frame.width, m_impl->optix_launch_params.config.frame.height);
                m_impl->integrator->SetMaxRayDepth(m_impl->optix_launch_params.config.max_depth);

                auto buf_mngr = util::Singleton<BufferManager>::instance();
                {
                    BufferDesc desc{
                        .name           = "pt accum buffer",
                        .flag           = EBufferFlag::None,
                        .width          = static_cast<uint32_t>(m_impl->scene->film_w),
                        .height         = static_cast<uint32_t>(m_impl->scene->film_h),
                        .stride_in_byte = sizeof(float) * 4};
                    m_impl->optix_launch_params.accum_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name = "pt result";
                    desc.flag = EBufferFlag::AllowDisplay;
                    m_impl->optix_launch_params.frame_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name           = "random";
                    desc.flag           = EBufferFlag::None;
                    desc.stride_in_byte = sizeof(cuda::Random);
                    m_impl->optix_launch_params.random.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name           = "throughput";
                    desc.stride_in_byte = sizeof(float3);
                    m_impl->optix_launch_params.throughput.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name           = "path_record";
                    desc.flag           = EBufferFlag::None;
                    desc.stride_in_byte = sizeof(wf::PathRecord);
                    m_impl->optix_launch_params.path_record.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name           = "hit_record";
                    desc.flag           = EBufferFlag::None;
                    desc.stride_in_byte = sizeof(wf::HitRecord);
                    m_impl->optix_launch_params.hit_record.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);

                    desc.name           = "nee_record";
                    desc.flag           = EBufferFlag::None;
                    desc.stride_in_byte = sizeof(wf::NEERecord);
                    m_impl->optix_launch_params.nee_record.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->output_pixel_num);
                }

                {

                    CUDA_CHECK(cudaMallocAsync(
                        reinterpret_cast<void**>(&m_impl->dynamic_array_cnt_buffer),
                        sizeof(unsigned int) * 3,
                        *m_impl->stream));

                    BufferDesc desc{
                        .name           = "ray index",
                        .flag           = EBufferFlag::None,
                        .width          = static_cast<uint32_t>(m_impl->scene->film_w * m_impl->scene->film_h),
                        .height         = 1,
                        .stride_in_byte = sizeof(unsigned int)};
                    m_impl->optix_launch_params.ray_index.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->dynamic_array_cnt_buffer);
                    desc.name = "hit index";
                    m_impl->optix_launch_params.hit_index.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->dynamic_array_cnt_buffer + sizeof(unsigned int));
                    desc.name = "shadow ray index";
                    m_impl->optix_launch_params.shadow_ray_index.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_impl->dynamic_array_cnt_buffer + sizeof(unsigned int) * 2);
                }

                m_impl->optix_launch_params.handle = m_impl->scene->GetIASHandle(1, true);

                {
                    auto instances    = m_impl->scene->GetInstances();
                    auto instance_num = static_cast<unsigned int>(instances.size());

                    {
                        auto sbt = m_impl->ray_pass->GetSBT();
                        sbt->SetRayGenRecord<void>();
                        sbt->SetHitgroupRecord<HitGroupData>(instance_num * 2);
                        sbt->SetMissRecord<void>(1);

                        sbt->BindData("ray gen", nullptr);
                        sbt->BindData("miss", nullptr);

                        for (auto i = 0u; i < instance_num; i++) {
                            HitGroupData hit;
                            hit.geo = instances[i].shape->GetOptixGeometry();
                            hit.mat = instances[i].material->GetOptixMaterial();

                            if (instances[i].emitter != nullptr) {
                                hit.emitter_index = m_impl->scene->GetEmitterIndex(instances[i].emitter);
                            } else
                                hit.emitter_index = -1;

                            if (hit.geo.type == optix::Geometry::EType::TriMesh) {
                                sbt->BindData("hit", &hit, i);
                            } else if (hit.geo.type == optix::Geometry::EType::Sphere) {
                                sbt->BindData("hit sphere", &hit, i);
                            } else {
                                sbt->BindData("hit curve", &hit, i);
                            }
                        }

                        sbt->Finish();
                    }

                    {
                        auto sbt = m_impl->shadow_ray_pass->GetSBT();
                        sbt->SetRayGenRecord<void>();
                        sbt->SetHitgroupRecord<void>(instance_num);
                        sbt->SetMissRecord<void>(1);

                        sbt->BindData("ray gen", nullptr);
                        sbt->BindData("miss shadow", nullptr);

                        for (auto i = 0u; i < instance_num; i++) {
                            HitGroupData hit;
                            hit.geo = instances[i].shape->GetOptixGeometry();
                            hit.mat = instances[i].material->GetOptixMaterial();

                            // if (instances[i].emitter != nullptr) {
                            //     hit.emitter_index = m_impl->scene->GetEmitterIndex(instances[i].emitter);
                            // } else
                            //     hit.emitter_index = -1;

                            if (hit.geo.type == optix::Geometry::EType::TriMesh) {
                                sbt->BindData("hit shadow", nullptr, i);
                            } else if (hit.geo.type == optix::Geometry::EType::Sphere) {
                                sbt->BindData("hit shadow sphere", nullptr, i);
                            } else {
                                sbt->BindData("hit shadow curve", nullptr, i);
                            }
                        }
                        sbt->Finish();
                    }
                }

                m_impl->dirty = true;
            }));
    }

    void PTPass::InitPipeline() noexcept {
        {
            auto pipeline = m_impl->ray_pass->GetPipeline();

            pipeline->SetPipelineLaunchParamsVariableName("optix_launch_params");
            pipeline->EnalbePrimitiveType(optix::Pipeline::EPrimitiveType::Sphere);
            pipeline->EnalbePrimitiveType(optix::Pipeline::EPrimitiveType::Curve);

            auto pt_module     = pipeline->CreateModule(optix::EModuleType::UserDefined, wave_ray_ptx);
            auto sphere_module = pipeline->CreateModule(optix::EModuleType::BuiltinSphereIS);
            auto curve_module  = pipeline->CreateModule(optix::EModuleType::BuiltinCurveIS);

            pipeline->CreateRayGen("ray gen").SetModule(pt_module).SetEntry("__raygen__main");
            pipeline->CreateMiss("miss").SetModule(pt_module).SetEntry("__miss__default");
            pipeline->CreateHitgroup("hit").SetCHModule(pt_module).SetCHEntry("__closesthit__default");
            pipeline->CreateHitgroup("hit sphere").SetCHModule(pt_module).SetCHEntry("__closesthit__default").SetISModule(sphere_module);
            pipeline->CreateHitgroup("hit curve").SetCHModule(pt_module).SetCHEntry("__closesthit__default").SetISModule(curve_module);

            pipeline->Finish();
        }
        {
            auto pipeline = m_impl->shadow_ray_pass->GetPipeline();

            pipeline->SetPipelineLaunchParamsVariableName("optix_launch_params");
            pipeline->EnalbePrimitiveType(optix::Pipeline::EPrimitiveType::Sphere);
            pipeline->EnalbePrimitiveType(optix::Pipeline::EPrimitiveType::Curve);

            auto pt_module     = pipeline->CreateModule(optix::EModuleType::UserDefined, wave_shadow_ray_ptx);
            auto sphere_module = pipeline->CreateModule(optix::EModuleType::BuiltinSphereIS);
            auto curve_module  = pipeline->CreateModule(optix::EModuleType::BuiltinCurveIS);

            pipeline->CreateRayGen("ray gen").SetModule(pt_module).SetEntry("__raygen__shadow");
            pipeline->CreateMiss("miss shadow").SetModule(pt_module).SetEntry("__miss__shadow");
            pipeline->CreateHitgroup("hit shadow").SetCHModule(pt_module).SetCHEntry("__closesthit__shadow");
            pipeline->CreateHitgroup("hit shadow sphere").SetCHModule(pt_module).SetCHEntry("__closesthit__shadow").SetISModule(sphere_module);
            pipeline->CreateHitgroup("hit shadow curve").SetCHModule(pt_module).SetCHEntry("__closesthit__shadow").SetISModule(curve_module);

            pipeline->Finish();
        }
    }

}// namespace wf