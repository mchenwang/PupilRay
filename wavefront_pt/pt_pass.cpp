#include "pt_pass.h"
#include "denoise_pass.h"

#include "wavefront.h"

#include "system/system.h"
#include "system/gui/gui.h"
#include "world/world.h"
#include "world/render_object.h"

#include "optix/context.h"

#include "cuda/util.h"

#include "util/event.h"
#include "util/timer.h"

#include "imgui.h"

#include "integrator.h"

using namespace Pupil;

extern "C" char wave_ray_ptx[];
extern "C" char wave_shadow_ray_ptx[];

namespace {
int m_max_depth_gui = 2;
bool m_accumulated_flag = true;

std::atomic_bool m_dirty = false;
uint2 m_frame_size;
unsigned int m_max_wave_num = 0;

Pupil::Timer m_timer;
double m_time_cost_pre_frame = 0.;

unsigned int m_max_depth = 2;
wavefront::GlobalData m_global_data;
CUdeviceptr m_global_data_cuda_memory = 0;
Pupil::cuda::RWDataView<wavefront::GlobalData> m_global_data_view;

std::unique_ptr<wavefront::Integrator> m_integrator = nullptr;

Pupil::world::World *m_world = nullptr;
Pupil::world::CameraHelper *m_world_camera = nullptr;

void InitOptixPipeline(wavefront::PTPass::OptixPass *ray_pass, wavefront::PTPass::ShadowOptixPass *shadow_ray_pass) noexcept {
    auto module_mngr = util::Singleton<optix::ModuleManager>::instance();

    auto sphere_module = module_mngr->GetModule(optix::EModuleBuiltinType::SpherePrimitive);
    auto ray_module = module_mngr->GetModule(wave_ray_ptx);
    {
        optix::PipelineDesc pipeline_desc;
        {
            optix::RayTraceProgramDesc mesh_ray_desc{
                .module_ptr = ray_module,
                .ray_gen_entry = "__raygen__main",
                .miss_entry = "__miss__default",
                .hit_group = { .ch_entry = "__closesthit__default" },
            };
            pipeline_desc.ray_trace_programs.push_back(mesh_ray_desc);
            optix::RayTraceProgramDesc sphere_ray_desc{
                .module_ptr = ray_module,
                .hit_group = { .ch_entry = "__closesthit__default",
                               .intersect_module = sphere_module },
            };
            pipeline_desc.ray_trace_programs.push_back(sphere_ray_desc);
            auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
            pipeline_desc.callable_programs.insert(pipeline_desc.callable_programs.end(), mat_programs.begin(), mat_programs.end());
        }
        ray_pass->InitPipeline(pipeline_desc);
    }
    auto shadow_ray_module = module_mngr->GetModule(wave_shadow_ray_ptx);
    {
        optix::PipelineDesc pipeline_desc;
        {
            optix::RayTraceProgramDesc mesh_ray_desc{
                .module_ptr = shadow_ray_module,
                .ray_gen_entry = "__raygen__shadow",
                .miss_entry = "__miss__shadow",
                .hit_group = { .ch_entry = "__closesthit__shadow" },
            };
            pipeline_desc.ray_trace_programs.push_back(mesh_ray_desc);
            optix::RayTraceProgramDesc sphere_ray_desc{
                .module_ptr = shadow_ray_module,
                .hit_group = { .ch_entry = "__closesthit__shadow",
                               .intersect_module = sphere_module },
            };
            pipeline_desc.ray_trace_programs.push_back(sphere_ray_desc);
            auto mat_programs = Pupil::resource::GetMaterialProgramDesc();
            pipeline_desc.callable_programs.insert(pipeline_desc.callable_programs.end(), mat_programs.begin(), mat_programs.end());
        }
        shadow_ray_pass->InitPipeline(pipeline_desc);
    }
}
}// namespace

namespace wavefront {
PTPass::PTPass(std::string_view name) noexcept
    : Pass(name) {
    m_stream = std::make_unique<Pupil::cuda::Stream>();

    m_global_data_cuda_memory = Pupil::cuda::CudaMemcpyToDevice(&m_global_data, sizeof(m_global_data));
    m_global_data_view.SetData(m_global_data_cuda_memory);

    auto optix_ctx = util::Singleton<optix::Context>::instance();
    m_ray_pass = std::make_unique<OptixPass>(optix_ctx->context, m_stream->GetStream());
    m_shadow_ray_pass = std::make_unique<ShadowOptixPass>(optix_ctx->context, m_stream->GetStream());
    InitOptixPipeline(m_ray_pass.get(), m_shadow_ray_pass.get());

    m_integrator = std::make_unique<Integrator>(m_ray_pass.get(), m_shadow_ray_pass.get());

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

void PTPass::OnRun() noexcept {
    if (!util::Singleton<System>::instance()->render_flag) return;

    m_timer.Start();
    {
        if (m_dirty) {
            m_dirty = false;
            m_global_data.camera.SetData(m_world_camera->GetCudaMemory());
            m_global_data.accumulated_flag = m_accumulated_flag;
            m_max_depth = m_max_depth_gui;
            m_integrator->SetMaxRayDepth(m_max_depth);
            m_global_data.random_init_num = 0;
            m_global_data.sample_cnt = 0;
            m_global_data.handle = m_world->GetIASHandle(1, true);
        }

        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void *>(m_global_data_cuda_memory),
            &m_global_data, sizeof(m_global_data),
            cudaMemcpyHostToDevice, *m_stream.get()));

        m_integrator->Trace(m_global_data_view, m_stream.get());

        m_stream->Synchronize();
        ++m_global_data.random_init_num;
        m_global_data.sample_cnt += m_global_data.accumulated_flag;
    }
    m_timer.Stop();
    m_time_cost_pre_frame = m_timer.ElapsedMilliseconds();
}

void PTPass::SetScene(Pupil::world::World *world) noexcept {
    m_world = world;
    m_world_camera = world->camera.get();
    m_frame_size.x = static_cast<unsigned int>(world->scene->sensor.film.w);
    m_frame_size.y = static_cast<unsigned int>(world->scene->sensor.film.h);

    m_max_depth_gui = world->scene->integrator.max_depth;
    m_accumulated_flag = true;

    m_global_data.accumulated_flag = m_accumulated_flag;
    m_max_depth = m_max_depth_gui;

    auto output_pixel_num = m_frame_size.x * m_frame_size.y;

    m_max_wave_num = output_pixel_num;

    m_integrator->SetFrameSize(m_frame_size);

    auto buf_mngr = util::Singleton<BufferManager>::instance();
    auto dynamic_array_buf_mngr = util::Singleton<cuda::DynamicArrayManager>::instance();
    {
        BufferDesc desc{
            .name = "bounce buffer",
            .flag = EBufferFlag::None,
            .width = static_cast<uint32_t>(world->scene->sensor.film.w),
            .height = static_cast<uint32_t>(world->scene->sensor.film.h),
            .stride_in_byte = sizeof(BounceRecord)
        };
        m_global_data.cur_bounce.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "hit buffer";
        desc.stride_in_byte = sizeof(HitRecord);
        m_global_data.hit_record.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "miss buffer";
        desc.stride_in_byte = sizeof(MissRecord);
        m_global_data.miss_record.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "emitter samples buffer";
        desc.stride_in_byte = sizeof(EmitterSample);
        m_global_data.emitter_samples.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "bsdf samples buffer";
        desc.stride_in_byte = sizeof(BsdfSample);
        m_global_data.bsdf_samples.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "pixel random buffer";
        desc.stride_in_byte = sizeof(Pupil::cuda::Random);
        m_global_data.random.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "throughput buffer";
        desc.stride_in_byte = sizeof(float3);
        m_global_data.throughput.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "accum buffer";
        desc.stride_in_byte = sizeof(float4);
        m_global_data.accum_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "albedo buffer";
        desc.flag = EBufferFlag::AllowDisplay;
        desc.stride_in_byte = sizeof(float4);
        m_global_data.albedo_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "normal buffer";
        desc.flag = EBufferFlag::AllowDisplay;
        desc.stride_in_byte = sizeof(float4);
        m_global_data.normal_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);

        desc.name = "pt result buffer";
        desc.flag = EBufferFlag::AllowDisplay;
        desc.stride_in_byte = sizeof(float4);
        m_global_data.frame_buffer.SetData(buf_mngr->AllocBuffer(desc)->cuda_ptr, output_pixel_num);
    }
    {
        BufferDesc desc{
            .name = "ray index",
            .flag = EBufferFlag::None,
            .width = static_cast<uint32_t>(world->scene->sensor.film.w * world->scene->sensor.film.h),
            .height = 1,
            .stride_in_byte = sizeof(unsigned int)
        };
        m_global_data.ray_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
        desc.name = "shadow ray index";
        m_global_data.shadow_ray_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
        desc.name = "hit index";
        m_global_data.hit_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
        desc.name = "miss index";
        m_global_data.miss_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
        desc.name = "emitter sampler index";
        m_global_data.emitter_sample_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
        desc.name = "light eval index";
        m_global_data.light_eval_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
        desc.name = "bsdf eval index";
        m_global_data.bsdf_eval_index = dynamic_array_buf_mngr->GetDynamicArray<unsigned int>(buf_mngr->AllocBuffer(desc)->cuda_ptr, 0);
    }

    m_global_data.emitters = world->emitters->GetEmitterGroup();
    m_global_data.handle = world->GetIASHandle(1, true);

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
            }
        }
        {
            optix::ProgDataDescPair<SBTTypes::MissDataType> miss_data = {
                .program = "__miss__default"
            };
            desc.miss_datas.push_back(miss_data);
        }
        m_ray_pass->InitSBT(desc);
    }
    {
        optix::SBTDesc<ShadowSBTTypes> desc{};
        desc.ray_gen_data = {
            .program = "__raygen__shadow"
        };
        {
            int emitter_index_offset = 0;
            using HitGroupDataRecord = optix::ProgDataDescPair<ShadowSBTTypes::HitGroupDataType>;
            for (auto &&ro : world->GetRenderobjects()) {
                HitGroupDataRecord hit_default_data{};
                hit_default_data.program = "__closesthit__shadow";
                desc.hit_datas.push_back(hit_default_data);
            }
        }
        {
            optix::ProgDataDescPair<ShadowSBTTypes::MissDataType> miss_data = {
                .program = "__miss__shadow"
            };
            desc.miss_datas.push_back(miss_data);
        }
        m_shadow_ray_pass->InitSBT(desc);
    }

    m_dirty = true;
}

void PTPass::Inspector() noexcept {
    ImGui::Text("time cost: %.3lf ms", m_time_cost_pre_frame);
    ImGui::Text("sample count: %d", m_global_data.sample_cnt + 1);
    ImGui::InputInt("max trace depth", &m_max_depth_gui);
    m_max_depth_gui = clamp(m_max_depth_gui, 1, 128);
    if (m_max_depth != m_max_depth_gui) {
        m_dirty = true;
    }

    if (ImGui::Checkbox("accumulate radiance", &m_accumulated_flag)) {
        m_dirty = true;
    }
}
}// namespace wavefront