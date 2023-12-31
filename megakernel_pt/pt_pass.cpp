#include "pt_pass.h"
#include "imgui.h"

#include "denoise_pass.h"

#include "cuda/context.h"
#include "optix/context.h"
#include "optix/module.h"

#include "util/event.h"
#include "system/system.h"
#include "system/gui/gui.h"
#include "world/world.h"
#include "world/render_object.h"

extern "C" char embedded_ptx_code[];

namespace {
int m_max_depth;
bool m_accumulated_flag;

size_t m_frame_cnt = 0;

Pupil::world::World *m_world = nullptr;
bool m_allow_animation = false;
Pupil::world::RenderObject *m_s1 = nullptr;
Pupil::world::RenderObject *m_s2 = nullptr;

double m_time_cost = 0.;
}// namespace

namespace Pupil::pt {
PTPass::PTPass(std::string_view name) noexcept
    : Pass(name) {
    auto optix_ctx = util::Singleton<optix::Context>::instance();
    auto cuda_ctx = util::Singleton<cuda::Context>::instance();
    m_stream = std::make_unique<cuda::Stream>();
    m_optix_pass = std::make_unique<optix::Pass<SBTTypes, OptixLaunchParams>>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline();
    BindingEventCallback();
}

void PTPass::OnRun() noexcept {
    m_timer.Start();
    {
        if (m_allow_animation && (m_s1 || m_s2)) {
            m_optix_launch_params.sample_cnt = 0;
            m_optix_launch_params.random_seed = 0;
            if (m_s1) {
                util::Transform transform;
                transform.Translate(0.f, 0.5f + 0.5f * std::sinf((float)m_frame_cnt / (10 * M_PIf)), 0.f);
                m_s1->UpdateTransform(transform);
            }
            if (m_s2) {
                util::Transform transform;
                transform.Translate(std::cosf((float)m_frame_cnt / (10 * M_PIf)) * 0.5f, 0.5f - std::sinf((float)m_frame_cnt / (10 * M_PIf)), 0.f);
                m_s2->UpdateTransform(transform);
            }
            m_frame_cnt++;
            m_dirty = true;
        }

        if (m_dirty) {
            m_optix_launch_params.camera.SetData(m_world_camera->GetCudaMemory());
            m_optix_launch_params.config.max_depth = m_max_depth;
            m_optix_launch_params.config.accumulated_flag = m_accumulated_flag;
            m_optix_launch_params.sample_cnt = 0;
            m_dirty = false;
            m_optix_launch_params.handle = m_world->GetIASHandle(2, true);
            m_optix_launch_params.emitters = m_world->emitters->GetEmitterGroup();
        }

        m_optix_pass->Run(m_optix_launch_params, m_optix_launch_params.config.frame.width,
                          m_optix_launch_params.config.frame.height);
        m_optix_pass->Synchronize();

        m_optix_launch_params.sample_cnt += m_optix_launch_params.config.accumulated_flag;
        ++m_optix_launch_params.random_seed;
    }
    m_timer.Stop();
    m_time_cost = m_timer.ElapsedMilliseconds();
}

void PTPass::InitOptixPipeline() noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto pt_module = module_mngr->GetModule(embedded_ptx_code);

    optix::PipelineDesc pipeline_desc;
    {
        // for mesh(triangle) geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .ray_gen_entry = "__raygen__main",
            .miss_entry = "__miss__default",
            .hit_group = { .ch_entry = "__closesthit__default" },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .miss_entry = "__miss__shadow",
            .hit_group = { .ch_entry = "__closesthit__shadow" },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }

    {
        // for sphere geo
        optix::RayTraceProgramDesc forward_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__default",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(forward_ray_desc);
        optix::RayTraceProgramDesc shadow_ray_desc{
            .module_ptr = pt_module,
            .hit_group = { .ch_entry = "__closesthit__shadow",
                           .intersect_module = sphere_module },
        };
        pipeline_desc.ray_trace_programs.push_back(shadow_ray_desc);
    }
    {
        auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
        pipeline_desc.callable_programs.insert(
            pipeline_desc.callable_programs.end(),
            mat_programs.begin(), mat_programs.end());
    }
    m_optix_pass->InitPipeline(pipeline_desc);
}

void PTPass::SetScene(world::World *world) noexcept {
    m_world_camera = world->camera.get();

    m_world = world;

    m_optix_launch_params.config.frame.width = world->scene->sensor.film.w;
    m_optix_launch_params.config.frame.height = world->scene->sensor.film.h;
    m_optix_launch_params.config.max_depth = world->scene->integrator.max_depth;
    m_optix_launch_params.config.accumulated_flag = true;

    m_max_depth = m_optix_launch_params.config.max_depth;
    m_accumulated_flag = m_optix_launch_params.config.accumulated_flag;

    m_optix_launch_params.random_seed = 0;
    m_optix_launch_params.sample_cnt = 0;

    m_output_pixel_num = m_optix_launch_params.config.frame.width *
                         m_optix_launch_params.config.frame.height;
    auto buf_mngr = util::Singleton<BufferManager>::instance();
    {
        BufferDesc desc{
            .name = "pt result buffer",
            .flag = EBufferFlag::AllowDisplay,
            .width = static_cast<uint32_t>(world->scene->sensor.film.w),
            .height = static_cast<uint32_t>(world->scene->sensor.film.h),
            .stride_in_byte = sizeof(float) * 4
        };
        m_optix_launch_params.frame_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);

        desc.name = "albedo";
        m_optix_launch_params.albedo.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);

        desc.name = "normal";
        m_optix_launch_params.normal.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);

        desc.name = "motion vector";
        m_optix_launch_params.motion_vector.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);

        desc.name = "pt accum buffer";
        desc.flag = EBufferFlag::None;
        m_optix_launch_params.accum_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, m_output_pixel_num);
    }
    m_optix_launch_params.handle = world->GetIASHandle(2, true);
    m_optix_launch_params.emitters = world->emitters->GetEmitterGroup();

    {
        optix::SBTDesc<SBTTypes> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__main"
        };
        {
            int emitter_index_offset = 0;
            using HitGroupDataRecord = optix::ProgDataDescPair<SBTTypes::HitGroupDataType>;
            for (auto &&ro : world->GetRenderobjects()) {
                HitGroupDataRecord hit_default_data{};
                hit_default_data.program = "__closesthit__default";
                hit_default_data.data.mat = ro->mat;
                hit_default_data.data.geo = ro->geo;
                if (ro->is_emitter) {
                    hit_default_data.data.emitter_index_offset = emitter_index_offset;
                    emitter_index_offset += ro->sub_emitters_num;
                }

                desc.hit_datas.push_back(hit_default_data);

                HitGroupDataRecord hit_shadow_data{};
                hit_shadow_data.program = "__closesthit__shadow";
                hit_shadow_data.data.mat.type = ro->mat.type;
                desc.hit_datas.push_back(hit_shadow_data);
            }
        }
        {
            optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
                .program = "__miss__default"
            };
            desc.miss_datas.push_back(miss_data);
            optix::ProgDataDescPair<SBTTypes::MissDataType> miss_shadow_data = {
                .program = "__miss__shadow"
            };
            desc.miss_datas.push_back(miss_shadow_data);
        }
        {
            auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
            for (auto &mat_prog : mat_programs) {
                if (mat_prog.cc_entry) {
                    optix::ProgDataDescPair<SBTTypes::CallablesDataType> cc_data = {
                        .program = mat_prog.cc_entry
                    };
                    desc.callables_datas.push_back(cc_data);
                }
                if (mat_prog.dc_entry) {
                    optix::ProgDataDescPair<SBTTypes::CallablesDataType> dc_data = {
                        .program = mat_prog.dc_entry
                    };
                    desc.callables_datas.push_back(dc_data);
                }
            }
        }
        m_optix_pass->InitSBT(desc);
    }

    m_s1 = world->GetRenderObject("movable_s1");
    m_s2 = world->GetRenderObject("movable_s2");

    m_dirty = true;
}

void PTPass::BindingEventCallback() noexcept {
    EventBinder<EWorldEvent::CameraChange>([this](void *) {
        m_dirty = true;
    });

    EventBinder<EWorldEvent::RenderInstanceUpdate>([this](void *) {
        m_dirty = true;
    });

    EventBinder<ESystemEvent::SceneLoad>([this](void *p) {
        SetScene((world::World *)p);
    });
}

void PTPass::Inspector() noexcept {
    ImGui::Text("time cost: %.3lf", m_time_cost);
    ImGui::Text("sample count: %d", m_optix_launch_params.sample_cnt + 1);
    ImGui::InputInt("max trace depth", &m_max_depth);
    m_max_depth = clamp(m_max_depth, 1, 128);
    if (m_optix_launch_params.config.max_depth != m_max_depth) {
        m_dirty = true;
    }

    if (ImGui::Checkbox("accumulate radiance", &m_accumulated_flag)) {
        m_dirty = true;
    }

    ImGui::Checkbox("allow sphere animation", &m_allow_animation);
}
}// namespace Pupil::pt