#include "integrator.h"
#include "cuda/kernel.h"

namespace wavefront {
using namespace Pupil;

void InitialPath(uint2 launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void HandleFirstHitEmitter(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void HandleMiss(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, bool is_background, cuda::Stream *stream) noexcept;
void DirectLightSampling(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept;
void EvalLight(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void BsdfSampling(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void EvalBsdf(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void AccumulateRadiance(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
}// namespace wavefront

namespace wavefront {
void Integrator::Trace(cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel(
        [g_data] __device__() {
            g_data->ray_index.Clear();
            g_data->shadow_ray_index.Clear();
            g_data->hit_index.Clear();
            g_data->miss_index.Clear();
            g_data->emitter_sample_index.Clear();
            g_data->light_eval_index.Clear();
            g_data->bsdf_eval_index.Clear();
        },
        stream);

    InitialPath(m_frame_size, g_data, stream);

    m_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);

    HandleFirstHitEmitter(m_max_wave_size, g_data, stream);
    HandleMiss(m_max_wave_size, g_data, true, stream);

    for (auto depth = 1u; depth < m_max_depth; ++depth) {
        DirectLightSampling(m_max_wave_size, g_data, depth, stream);

        m_shadow_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);
        EvalLight(m_max_wave_size, g_data, stream);
        Pupil::cuda::LaunchKernel(
            [g_data] __device__() {
                g_data->ray_index.Clear();
                g_data->shadow_ray_index.Clear();
                g_data->miss_index.Clear();
                g_data->emitter_sample_index.Clear();
                g_data->light_eval_index.Clear();
                g_data->bsdf_eval_index.Clear();
            },
            stream);
        BsdfSampling(m_max_wave_size, g_data, stream);
        Pupil::cuda::LaunchKernel([g_data] __device__() { g_data->hit_index.Clear(); }, stream);

        m_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);

        EvalBsdf(m_max_wave_size, g_data, stream);
        HandleMiss(m_max_wave_size, g_data, false, stream);
    }

    AccumulateRadiance(m_max_wave_size, g_data, stream);
}

void InitialPath(uint2 launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel2D(
        launch_size, [g_data] __device__(uint2 index, uint2 size) {
            const unsigned int pixel_index = index.y * size.x + index.x;
            auto &camera = *g_data->camera.GetDataPtr();
            g_data->random[pixel_index].Init(4, pixel_index, g_data->random_init_num);

            auto &random = g_data->random[pixel_index];
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
            d = normalize(d);

            float3 ray_direction = normalize(make_float3(camera.camera_to_world * d));

            float3 ray_origin = make_float3(
                camera.camera_to_world.r0.w,
                camera.camera_to_world.r1.w,
                camera.camera_to_world.r2.w);

            g_data->cur_bounce[pixel_index].ray.dir = ray_direction;
            g_data->cur_bounce[pixel_index].ray.origin = ray_origin;
            g_data->cur_bounce[pixel_index].ray.t = 1e16f;
            g_data->ray_index.Push(pixel_index);

            g_data->frame_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
            g_data->normal_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
            g_data->albedo_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
            g_data->throughput[pixel_index] = make_float3(1.f, 1.f, 1.f);
        },
        stream);
}

void HandleFirstHitEmitter(unsigned int launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->hit_index.GetNum()) return;
            const auto pixel_index = g_data->hit_index[index];
            const auto &hit = g_data->hit_record[pixel_index];

            if (hit.emitter_index >= 0) {
                auto &emitter = g_data->emitters.areas[hit.emitter_index];
                auto emission = emitter.GetRadiance(hit.geo.texcoord);
                g_data->frame_buffer[pixel_index] = make_float4(emission, 1.f);
            }

            g_data->normal_buffer[pixel_index] = make_float4(hit.geo.normal, 1.f);
            g_data->albedo_buffer[pixel_index] = make_float4(hit.bsdf.GetAlbedo(), 1.f);
        },
        stream);
}

void HandleMiss(unsigned int launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, bool is_background, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data, is_background] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->miss_index.GetNum() || !g_data->emitters.env) return;

            auto pixel_index = g_data->miss_index[index];
            auto &ray = g_data->cur_bounce[pixel_index].ray;

            auto &env = *g_data->emitters.env.GetDataPtr();

            optix::LocalGeometry env_local;
            env_local.position = ray.origin + ray.dir;
            optix::EmitEvalRecord emit_record;
            env.Eval(emit_record, env_local, ray.origin);

            float3 radiance = make_float3(g_data->frame_buffer[pixel_index]);

            if (is_background) {
                radiance += emit_record.radiance;
            } else {
                float3 throughput = g_data->throughput[pixel_index];
                auto &bsdf_sample = g_data->bsdf_samples[pixel_index];

                float mis = optix::MISWeight(bsdf_sample.pdf, emit_record.pdf);
                radiance += throughput * emit_record.radiance * mis;
            }

            g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
        },
        stream);
}

void DirectLightSampling(unsigned int launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, unsigned int depth, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data, depth] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->hit_index.GetNum()) return;

            auto pixel_index = g_data->hit_index[index];

            float rr = depth > 2 ? 0.95 : 1.0;
            if (g_data->random[pixel_index].Next() > rr)
                return;

            g_data->throughput[pixel_index] /= rr;

            auto &hit = g_data->hit_record[pixel_index];
            auto &random = g_data->random[pixel_index];
            auto &emitter = g_data->emitters.SelectOneEmiiter(random.Next());
            Pupil::optix::EmitterSampleRecord emitter_sample_record;
            emitter.SampleDirect(emitter_sample_record, hit.geo, random.Next2());

            g_data->cur_bounce[pixel_index].shadow_ray.dir = emitter_sample_record.wi;
            g_data->cur_bounce[pixel_index].shadow_ray.origin = hit.geo.position;
            g_data->cur_bounce[pixel_index].shadow_ray.t = emitter_sample_record.distance - 0.0001f;
            g_data->shadow_ray_index.Push(pixel_index);

            g_data->emitter_samples[pixel_index].record = emitter_sample_record;
            g_data->emitter_samples[pixel_index].select_pdf = emitter.select_probability;
            g_data->emitter_sample_index.Push(pixel_index);
        },
        stream);
}

void EvalLight(unsigned int launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->light_eval_index.GetNum()) return;

            auto pixel_index = g_data->light_eval_index[index];
            auto &emitter_sample = g_data->emitter_samples[pixel_index];
            auto &ray = g_data->cur_bounce[pixel_index].ray;
            auto &hit = g_data->hit_record[pixel_index];

            optix::BsdfSamplingRecord eval_record;
            eval_record.wi = optix::ToLocal(emitter_sample.record.wi, hit.geo.normal);
            eval_record.wo = optix::ToLocal(-ray.dir, hit.geo.normal);

            eval_record.sampler = &g_data->random[pixel_index];
            hit.bsdf.Eval(eval_record);
            float3 f = eval_record.f;
            float pdf = eval_record.pdf;
            if (!optix::IsZero(f * emitter_sample.record.pdf)) {
                float NoL = abs(dot(hit.geo.normal, emitter_sample.record.wi));
                // if (NoL > 0.f) {
                float mis = emitter_sample.record.is_delta ? 1.f : optix::MISWeight(emitter_sample.record.pdf, pdf);
                emitter_sample.record.pdf *= emitter_sample.select_pdf;

                float3 radiance = make_float3(g_data->frame_buffer[pixel_index]);
                float3 throughput = g_data->throughput[pixel_index];
                radiance += throughput * emitter_sample.record.radiance * f * NoL * mis / emitter_sample.record.pdf;

                g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
                // }
            }
        },
        stream);
}

void BsdfSampling(unsigned int launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->hit_index.GetNum()) return;
            auto pixel_index = g_data->hit_index[index];
            auto &ray = g_data->cur_bounce[pixel_index].ray;
            auto &hit = g_data->hit_record[pixel_index];

            float3 wo = optix::ToLocal(-ray.dir, hit.geo.normal);
            optix::BsdfSamplingRecord bsdf_sample_record;
            bsdf_sample_record.wo = wo;
            bsdf_sample_record.sampler = &g_data->random[pixel_index];
            hit.bsdf.Sample(bsdf_sample_record);

            if (optix::IsZero(bsdf_sample_record.f * abs(bsdf_sample_record.wi.z)) || optix::IsZero(bsdf_sample_record.pdf))
                return;

            g_data->throughput[pixel_index] *= bsdf_sample_record.f * abs(bsdf_sample_record.wi.z) / bsdf_sample_record.pdf;

            g_data->cur_bounce[pixel_index].ray.dir = optix::ToWorld(bsdf_sample_record.wi, hit.geo.normal);
            g_data->cur_bounce[pixel_index].ray.origin = hit.geo.position;
            g_data->cur_bounce[pixel_index].ray.t = 1e16f;

            g_data->bsdf_samples[pixel_index].pdf = bsdf_sample_record.pdf;
            g_data->bsdf_samples[pixel_index].sampled_type = bsdf_sample_record.sampled_type;

            g_data->ray_index.Push(pixel_index);
        },
        stream);
}

void EvalBsdf(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->bsdf_eval_index.GetNum()) return;
            auto pixel_index = g_data->bsdf_eval_index[index];
            auto &hit = g_data->hit_record[pixel_index];
            auto &ray = g_data->cur_bounce[pixel_index].ray;
            auto &bsdf_sample = g_data->bsdf_samples[pixel_index];

            auto &emitter = g_data->emitters.areas[hit.emitter_index];
            optix::EmitEvalRecord emit_record;
            emitter.Eval(emit_record, hit.geo, ray.origin);

            if (!optix::IsZero(emit_record.pdf)) {
                float mis = bsdf_sample.sampled_type & optix::EBsdfLobeType::Delta ?
                                1.f :
                                optix::MISWeight(bsdf_sample.pdf, emit_record.pdf * emitter.select_probability);

                auto radiance = make_float3(g_data->frame_buffer[pixel_index]);
                auto throughput = g_data->throughput[pixel_index];
                radiance += throughput * emit_record.radiance * mis;

                g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
            }
        },
        stream);
}

void AccumulateRadiance(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data] __device__(unsigned int pixel_index, unsigned int size) {
            auto radiance = make_float3(g_data->frame_buffer[pixel_index]);

            if (g_data->accumulated_flag && g_data->sample_cnt > 0) {
                const float t = 1.f / (g_data->sample_cnt + 1.f);
                const float3 pre = make_float3(g_data->accum_buffer[pixel_index]);
                radiance = lerp(pre, radiance, t);
            }
            g_data->accum_buffer[pixel_index] = make_float4(radiance, 1.f);
            g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
        },
        stream);
}
}// namespace wavefront