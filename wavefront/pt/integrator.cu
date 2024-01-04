#include "integrator.h"
#include "cuda/kernel.h"
#include "cuda/util.h"

using namespace Pupil;

namespace wf {
    void InitialPath(uint2 launch_size, cuda::RWDataView<GlobalData>& g_data, cuda::Stream* stream) noexcept;
    void ScatterRays(unsigned int launch_size, cuda::RWDataView<GlobalData>& g_data, cuda::Stream* stream) noexcept;
    void AccumulateRadiance(unsigned int launch_size, cuda::RWDataView<GlobalData>& g_data, cuda::Stream* stream) noexcept;
}// namespace wf

namespace wf {
    void Integrator::Trace(cuda::RWDataView<GlobalData>& g_data, Pupil::cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel([g_data] __device__() { g_data->ray_index.Clear(); }, stream);
        InitialPath(m_frame_size, g_data, stream);
        m_max_depth = 64;

        for (int depth = 0; depth < m_max_depth; ++depth) {
            Pupil::cuda::LaunchKernel([g_data] __device__() { g_data->hit_index.Clear(); }, stream);

            m_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);
            if (depth == m_max_depth - 1) break;

            Pupil::cuda::LaunchKernel([g_data] __device__() { g_data->ray_index.Clear(); g_data->shadow_ray_index.Clear(); }, stream);

            ScatterRays(m_max_wave_size, g_data, stream);
            m_shadow_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);
        }
        AccumulateRadiance(m_max_wave_size, g_data, stream);
    }

    void InitialPath(uint2 launch_size, Pupil::cuda::RWDataView<GlobalData>& g_data, Pupil::cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel2D(
            launch_size, [g_data] __device__(uint2 index, uint2 size) {
                const unsigned int pixel_index = index.y * size.x + index.x;
                auto&              camera      = g_data->camera;
                g_data->random[pixel_index].Init(4, pixel_index, g_data->random_seed);

                auto&        random          = g_data->random[pixel_index];
                const float2 subpixel_jitter = random.Next2();

                const float2 subpixel =
                    make_float2(
                        (static_cast<float>(index.x) + subpixel_jitter.x) / static_cast<float>(size.x),
                        (static_cast<float>(index.y) + subpixel_jitter.y) / static_cast<float>(size.y));
                // const float2 subpixel = make_float2((static_cast<float>(index.x)) / w, (static_cast<float>(index.y)) / h);
                const float4 point_on_film = make_float4(subpixel, 0.f, 1.f);

                float4 d = camera.sample_to_camera * point_on_film;

                d /= d.w;
                d.w = 0.f;
                d   = normalize(d);

                float3 ray_direction = normalize(make_float3(camera.camera_to_world * d));

                float3 ray_origin = make_float3(
                    camera.camera_to_world.r0.w,
                    camera.camera_to_world.r1.w,
                    camera.camera_to_world.r2.w);

                g_data->path_record[pixel_index].ray_dir    = ray_direction;
                g_data->path_record[pixel_index].ray_origin = ray_origin;
                g_data->ray_index.Push(pixel_index);

                g_data->frame_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
                g_data->throughput[pixel_index]   = make_float3(1.f, 1.f, 1.f);
            },
            stream);
    }

    void ScatterRays(unsigned int launch_size, cuda::RWDataView<GlobalData>& g_data, cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel1D(
            launch_size, [g_data] __device__(unsigned int launch_index, unsigned int size) {
                if (launch_index >= g_data->hit_index.GetNum()) return;
                const auto pixel_index = g_data->hit_index[launch_index];
                auto&      random      = g_data->random[pixel_index];
                auto&      hit         = g_data->hit_record[pixel_index];
                auto       ray_dir     = g_data->path_record[pixel_index].ray_dir;
                auto       throughput  = g_data->throughput[pixel_index];

                auto& geo  = hit.geo;
                auto& bsdf = hit.mat;

                // direct lighting
                {
                    auto emitter = g_data->emitters.SelectOneEmiiter(random.Next());

                    Pupil::optix::EmitterSampleRecord emitter_sample_record;
                    emitter->SampleDirect(emitter_sample_record, geo, random.Next2());

                    optix::BsdfSamplingRecord eval_record;
                    eval_record.wi      = optix::ToLocal(emitter_sample_record.wi, geo.normal);
                    eval_record.wo      = optix::ToLocal(-ray_dir, geo.normal);
                    eval_record.sampler = &random;
                    bsdf.Eval(eval_record);

                    float3 bsdf_eval_f   = eval_record.f;
                    float  bsdf_eval_pdf = eval_record.pdf;

                    float emit_pdf = emitter_sample_record.pdf * emitter->select_probability;
                    if (optix::IsValid(emit_pdf)) {
                        NEERecord nee;
                        nee.shadow_ray_dir   = emitter_sample_record.wi;
                        nee.shadow_ray_o     = geo.position;
                        nee.shadow_ray_t_max = emitter_sample_record.distance - 0.0001f;

                        float NoL = abs(dot(geo.normal, emitter_sample_record.wi));
                        float mis = emitter_sample_record.is_delta ? 1.f : optix::MISWeight(emitter_sample_record.pdf, bsdf_eval_pdf);

                        nee.radiance = emitter_sample_record.radiance * throughput * bsdf_eval_f * NoL * mis / emit_pdf;

                        g_data->nee_record[pixel_index] = nee;
                        g_data->shadow_ray_index.Push(pixel_index);
                    }
                }

                // bsdf sampling
                {
                    optix::BsdfSamplingRecord bsdf_sample_record;
                    bsdf_sample_record.wo      = optix::ToLocal(-ray_dir, geo.normal);
                    bsdf_sample_record.sampler = &random;
                    bsdf.Sample(bsdf_sample_record);

                    if (optix::IsValid(bsdf_sample_record.pdf)) {
                        PathRecord record;
                        record.bsdf_sample_pdf  = bsdf_sample_record.pdf;
                        record.bsdf_sample_type = bsdf_sample_record.sampled_type;
                        throughput *= bsdf_sample_record.f * abs(bsdf_sample_record.wi.z) / bsdf_sample_record.pdf;

                        record.ray_dir    = optix::ToWorld(bsdf_sample_record.wi, geo.normal);
                        record.ray_origin = geo.position;

                        g_data->path_record[pixel_index] = record;
                        g_data->ray_index.Push(pixel_index);
                    }
                }

                g_data->throughput[pixel_index] = throughput;
            },
            stream);
    }

    void AccumulateRadiance(unsigned int launch_size, cuda::RWDataView<GlobalData>& g_data, cuda::Stream* stream) noexcept {
        Pupil::cuda::LaunchKernel1D(
            launch_size, [g_data] __device__(unsigned int pixel_index, unsigned int size) {
                auto radiance = make_float3(g_data->frame_buffer[pixel_index]);

                if (g_data->config.accumulated_flag && g_data->sample_cnt > 0) {
                    const float  t   = 1.f / (g_data->sample_cnt + 1.f);
                    const float3 pre = make_float3(g_data->accum_buffer[pixel_index]);
                    radiance         = lerp(pre, radiance, t);
                }
                g_data->accum_buffer[pixel_index] = make_float4(radiance, 1.f);
                g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
            },
            stream);
    }
}// namespace wf