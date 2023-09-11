#include "integrator.h"
#include "cuda/kernel.h"

namespace wavefront {
using namespace Pupil;

void InitialPath(uint2 launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void HandleHit(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept;
void HandleMiss(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept;
void ScatterRays(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept;
// void HandleFirstHitEmitter(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
// void DirectLightSampling(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept;
// void EvalLight(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
// void BsdfSampling(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
// void EvalBsdf(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
void AccumulateRadiance(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept;
}// namespace wavefront

namespace wavefront {
void Integrator::Trace(cuda::RWDataView<GlobalData> &g_data, cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel(
        [g_data] __device__() {
            g_data->ray_queue.Clear();
            g_data->hit_queue.Clear();
            g_data->hit_emitter_queue.Clear();
            g_data->miss_queue.Clear();
            g_data->shadow_ray_queue.Clear();
        },
        stream);

    InitialPath(m_frame_size, g_data, stream);
    // for (auto depth = 0u; depth <= 0; ++depth) {
    for (auto depth = 0u; depth <= m_max_depth; ++depth) {
        m_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);
        HandleHit(m_max_wave_size, g_data, depth, stream);
        HandleMiss(m_max_wave_size, g_data, depth, stream);
        Pupil::cuda::LaunchKernel([g_data] __device__() { g_data->ray_queue.Clear(); }, stream);

        if (depth == m_max_depth) break;

        ScatterRays(m_max_wave_size, g_data, depth, stream);
        m_shadow_ray_pass->Run(reinterpret_cast<CUdeviceptr>(g_data.GetDataPtr()), m_max_wave_size, 1);
        Pupil::cuda::LaunchKernel(
            [g_data] __device__() {
                g_data->hit_queue.Clear();
                g_data->hit_emitter_queue.Clear();
                g_data->miss_queue.Clear();
                g_data->shadow_ray_queue.Clear();
            },
            stream);
    }

    AccumulateRadiance(m_max_wave_size, g_data, stream);
}

void InitialPath(uint2 launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel2D(
        launch_size, [g_data] __device__(uint2 index, uint2 size) {
            const unsigned int pixel_index = index.y * size.x + index.x;
            auto &camera = *g_data->camera.GetDataPtr();

            cuda::Random random;
            random.Init(4, pixel_index, g_data->random_init_num);
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

            auto new_camera_ray_path = g_data->ray_queue.Alloc();

            Ray camera_ray;
            camera_ray.dir = normalize(make_float3(camera.camera_to_world * d));
            camera_ray.origin = make_float3(camera.camera_to_world.r0.w, camera.camera_to_world.r1.w, camera.camera_to_world.r2.w);
            camera_ray.t = 1e16f;
            new_camera_ray_path.ray(camera_ray);
            new_camera_ray_path.pixel_index(pixel_index);
            new_camera_ray_path.random(random);
            new_camera_ray_path.throughput(make_float3(1.f));

            g_data->frame_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
            g_data->normal_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
            g_data->albedo_buffer[pixel_index] = make_float4(0.f, 0.f, 0.f, 1.f);
        },
        stream);
}

void HandleHit(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data, depth] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->hit_emitter_queue.GetNum()) return;

            auto hit = g_data->hit_emitter_queue[index];
            const auto pixel_index = hit.pixel_index();
            auto geo = hit.geo();
            auto ray = hit.ray();

            auto &emitter = g_data->emitters.areas[hit.emitter_index()];
            float3 radiance = make_float3(g_data->frame_buffer[pixel_index]);
            if (depth > 0) {
                auto bsdf_sample_type = hit.bsdf_sample_type();
                auto bsdf_sample_pdf = hit.bsdf_sample_pdf();
                optix::EmitEvalRecord emit_record;
                emitter.Eval(emit_record, geo, ray.origin);

                float mis = bsdf_sample_type & optix::EBsdfLobeType::Delta ?
                                1.f :
                                optix::MISWeight(bsdf_sample_pdf, emit_record.pdf * emitter.select_probability);

                auto throughput = hit.throughput();
                radiance += emit_record.radiance * mis * throughput;
            } else {
                radiance = emitter.GetRadiance(geo.texcoord);
            }
            g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
        },
        stream);
}

void HandleMiss(unsigned int launch_size, Pupil::cuda::RWDataView<GlobalData> &g_data, unsigned int depth, Pupil::cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data, depth] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->miss_queue.GetNum() || !g_data->emitters.env) return;

            const auto miss = g_data->miss_queue[index];
            const auto pixel_index = miss.pixel_index();
            const auto ray = miss.ray();

            auto &env = *g_data->emitters.env.GetDataPtr();

            optix::LocalGeometry env_local;
            env_local.position = ray.origin + ray.dir;
            optix::EmitEvalRecord emit_record;
            env.Eval(emit_record, env_local, ray.origin);

            float3 radiance = make_float3(g_data->frame_buffer[pixel_index]);

            if (depth > 0) {
                const auto bsdf_sample_pdf = miss.bsdf_sample_pdf();
                const auto throughput = miss.throughput();
                float mis = optix::MISWeight(bsdf_sample_pdf, emit_record.pdf);
                radiance += throughput * emit_record.radiance * mis;
            } else {
                radiance += emit_record.radiance;
            }

            g_data->frame_buffer[pixel_index] = make_float4(radiance, 1.f);
        },
        stream);
}

void ScatterRays(unsigned int launch_size, cuda::RWDataView<GlobalData> &g_data, unsigned int depth, cuda::Stream *stream) noexcept {
    Pupil::cuda::LaunchKernel1D(
        launch_size, [g_data, depth] __device__(unsigned int index, unsigned int size) {
            if (index >= g_data->hit_queue.GetNum()) return;
            auto hit = g_data->hit_queue[index];
            auto ray = hit.ray();
            auto geo = hit.geo();
            auto throughput = hit.throughput();
            auto bsdf_sample_pdf = hit.bsdf_sample_pdf();
            auto random = hit.random();
            auto pixel_index = hit.pixel_index();

            float rr = depth > 2 ? 0.95 : 1.0;
            if (random.Next() > rr)
                return;

            auto bsdf = hit.bsdf();

            // direct lighting sampling
            {
                auto &emitter = g_data->emitters.SelectOneEmiiter(random.Next());
                Pupil::optix::EmitterSampleRecord emitter_sample_record;
                emitter.SampleDirect(emitter_sample_record, geo, random.Next2());

                optix::BsdfSamplingRecord eval_record;
                eval_record.wi = optix::ToLocal(emitter_sample_record.wi, geo.normal);
                eval_record.wo = optix::ToLocal(-ray.dir, geo.normal);

                eval_record.sampler = &random;

                bsdf.Eval(eval_record);
                float3 bsdf_eval_f = eval_record.f;
                float bsdf_eval_pdf = eval_record.pdf;

                auto emit_pdf = emitter_sample_record.pdf * emitter.select_probability;
                if (!optix::IsZero(emit_pdf * bsdf_eval_f)) {
                    Ray shadow_ray;
                    shadow_ray.dir = emitter_sample_record.wi;
                    shadow_ray.origin = geo.position;
                    shadow_ray.t = emitter_sample_record.distance - 0.0001f;

                    float NoL = abs(dot(geo.normal, emitter_sample_record.wi));
                    float mis = emitter_sample_record.is_delta ? 1.f : optix::MISWeight(emitter_sample_record.pdf, bsdf_eval_pdf);

                    auto shadow_ray_path = g_data->shadow_ray_queue.Alloc();
                    shadow_ray_path.ray(shadow_ray);
                    float3 radiance = emitter_sample_record.radiance * throughput * bsdf_eval_f * NoL * mis / emit_pdf;
                    shadow_ray_path.radiance(radiance);
                    shadow_ray_path.pixel_index(pixel_index);
                }
            }

            // bsdf sampling
            {
                float3 wo = optix::ToLocal(-ray.dir, geo.normal);
                optix::BsdfSamplingRecord bsdf_sample_record;
                bsdf_sample_record.wo = wo;
                bsdf_sample_record.sampler = &random;
                bsdf.Sample(bsdf_sample_record);

                if (optix::IsZero(bsdf_sample_record.f * bsdf_sample_record.wi.z * bsdf_sample_record.pdf))
                    return;

                auto path = g_data->ray_queue.Alloc();
                Ray scatter_ray;
                scatter_ray.dir = optix::ToWorld(bsdf_sample_record.wi, geo.normal);
                scatter_ray.origin = geo.position;
                scatter_ray.t = 1e16f;
                path.ray(scatter_ray);
                path.throughput(throughput * bsdf_sample_record.f * abs(bsdf_sample_record.wi.z) / bsdf_sample_record.pdf);
                path.bsdf_sample_pdf(bsdf_sample_record.pdf);
                path.bsdf_sample_type(bsdf_sample_record.sampled_type);
                path.random(random);
                path.pixel_index(pixel_index);
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