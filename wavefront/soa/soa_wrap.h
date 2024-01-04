#pragma once

#include "cuda/util.h"
#include "render/material.h"
#include "render/geometry.h"

namespace wf::soa {
    class DynamicArrayStruct {
    protected:
        CUdeviceptr m_data CONST_STATIC_INIT(0);
        size_t m_slice_num CONST_STATIC_INIT(0);
        CUdeviceptr m_num  CONST_STATIC_INIT(0);

    public:
        CUDA_HOSTDEVICE DynamicArrayStruct() noexcept {}

        CUDA_HOST void SetData(CUdeviceptr cuda_data, size_t slice_num, CUdeviceptr num_data) noexcept {
            m_data      = cuda_data;
            m_slice_num = slice_num;
            m_num       = num_data;
        }
        CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }

        CUDA_DEVICE void         Clear() noexcept { *reinterpret_cast<unsigned int*>(m_num) = 0; }
        CUDA_DEVICE unsigned int GetNum() const noexcept { return *reinterpret_cast<unsigned int*>(m_num); }

#ifndef PUPIL_CPP
        CUDA_DEVICE unsigned int Alloc() noexcept {
            unsigned int* num = reinterpret_cast<unsigned int*>(m_num);
            return atomicAdd(num, 1);
        }
#endif

        template<typename T>
        CUDA_DEVICE T& Get(unsigned long long stride_offset, unsigned int index) {
            return *reinterpret_cast<T*>(m_data + stride_offset * m_slice_num + index * sizeof(T));
        }
    };

    class PathRecordArray : public DynamicArrayStruct {
    public:
        struct Accessor {
        private:
            unsigned int     m_i;
            PathRecordArray* m_data;

        public:
            CUDA_DEVICE Accessor(PathRecordArray* data, unsigned int index) noexcept : m_i(index), m_data(data) {}

            CUDA_DEVICE float3 ray_dir() const noexcept {
                float3 item;
                auto   offset = 0ull;
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void ray_dir(float3 item) const noexcept {
                auto offset                                         = 0ull;
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE float3 ray_origin() const noexcept {
                float3 item;
                auto   offset = sizeof(float3);
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void ray_origin(float3 item) const noexcept {
                auto offset                                         = sizeof(float3);
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE float3 throughput() const noexcept {
                float3 item;
                auto   offset = sizeof(float3) * 2;
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void throughput(float3 item) const noexcept {
                auto offset                                         = sizeof(float3) * 2;
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE float bsdf_sample_pdf() const noexcept {
                auto offset = sizeof(float3) * 3;
                return m_data->Get<float>(offset, m_i);
            }
            CUDA_DEVICE void bsdf_sample_pdf(float pdf) noexcept {
                auto offset                     = sizeof(float3) * 3;
                m_data->Get<float>(offset, m_i) = pdf;
            }

            CUDA_DEVICE Pupil::optix::EBsdfLobeType bsdf_sample_type() const noexcept {
                auto offset = sizeof(float3) * 3 + sizeof(float);
                return m_data->Get<Pupil::optix::EBsdfLobeType>(offset, m_i);
            }
            CUDA_DEVICE void bsdf_sample_type(const Pupil::optix::EBsdfLobeType& type) const noexcept {
                auto offset                                           = sizeof(float3) * 3 + sizeof(float);
                m_data->Get<Pupil::optix::EBsdfLobeType>(offset, m_i) = type;
            }

            CUDA_DEVICE unsigned int random_seed() const noexcept {
                auto offset = sizeof(float3) * 3 + sizeof(float) + sizeof(Pupil::optix::EBsdfLobeType);
                return m_data->Get<unsigned int>(offset, m_i);
            }
            CUDA_DEVICE void random_seed(unsigned int seed) const noexcept {
                auto offset                            = sizeof(float3) * 3 + sizeof(float) + sizeof(Pupil::optix::EBsdfLobeType);
                m_data->Get<unsigned int>(offset, m_i) = seed;
            }

            CUDA_DEVICE unsigned int pixel_index() const noexcept {
                auto offset = sizeof(float3) * 3 + sizeof(float) + sizeof(Pupil::optix::EBsdfLobeType) + sizeof(unsigned int);
                return m_data->Get<unsigned int>(offset, m_i);
            }
            CUDA_DEVICE void pixel_index(unsigned int index) const noexcept {
                auto offset                            = sizeof(float3) * 3 + sizeof(float) + sizeof(Pupil::optix::EBsdfLobeType) + sizeof(unsigned int);
                m_data->Get<unsigned int>(offset, m_i) = index;
            }
        };

#ifndef PUPIL_CPP
        CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArrayStruct::Alloc()); }
#endif

        CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
            return Accessor(this, index);
        }
    };

    class HitRecordArray : public DynamicArrayStruct {
    public:
        struct Accessor {
        private:
            unsigned int    m_i;
            HitRecordArray* m_data;

        public:
            CUDA_DEVICE Accessor(HitRecordArray* data, unsigned int index) noexcept : m_i(index), m_data(data) {}

            CUDA_DEVICE Pupil::optix::Material::LocalBsdf mat() const noexcept {
                Pupil::optix::Material::LocalBsdf item;
                auto                              offset = 0ull;
                item.type                                = m_data->Get<decltype(item.type)>(offset, m_i);
                item.diffuse.reflectance                 = m_data->Get<decltype(item.diffuse.reflectance)>(offset + sizeof(item.type), m_i);
                return item;
            }
            CUDA_DEVICE void mat(const Pupil::optix::Material::LocalBsdf& item) const noexcept {
                auto offset                                                                      = 0ull;
                m_data->Get<decltype(item.type)>(offset, m_i)                                    = item.type;
                m_data->Get<decltype(item.diffuse.reflectance)>(offset + sizeof(item.type), m_i) = item.diffuse.reflectance;
            }

            CUDA_DEVICE Pupil::optix::LocalGeometry geo() const noexcept {
                Pupil::optix::LocalGeometry item;
                auto                        offset = sizeof(Pupil::optix::Material::LocalBsdf);
                item.position.x                    = m_data->Get<float>(offset, m_i);
                item.position.y                    = m_data->Get<float>(offset + sizeof(float), m_i);
                item.position.z                    = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                offset                             = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(float3);
                item.normal.x                      = m_data->Get<float>(offset, m_i);
                item.normal.y                      = m_data->Get<float>(offset + sizeof(float), m_i);
                item.normal.z                      = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                offset                             = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(float3) * 2;
                item.texcoord.x                    = m_data->Get<float>(offset, m_i);
                item.texcoord.y                    = m_data->Get<float>(offset + sizeof(float), m_i);
                return item;
            }
            CUDA_DEVICE void geo(const Pupil::optix::LocalGeometry& item) const noexcept {
                auto offset                                         = sizeof(Pupil::optix::Material::LocalBsdf);
                m_data->Get<float>(offset, m_i)                     = item.position.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.position.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.position.z;
                offset                                              = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(float3);
                m_data->Get<float>(offset, m_i)                     = item.normal.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.normal.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.normal.z;
                offset                                              = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(float3) * 2;
                m_data->Get<float>(offset, m_i)                     = item.texcoord.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.texcoord.y;
            }

            CUDA_DEVICE float3 throughput() const noexcept {
                float3 item;
                auto   offset = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry);
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void throughput(float3 item) const noexcept {
                auto offset                                         = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry);
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE float3 ray_dir() const noexcept {
                float3 item;
                auto   offset = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3);
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void ray_dir(float3 item) const noexcept {
                auto offset                                         = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3);
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE int emitter_index() const noexcept {
                auto offset = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3) * 2;
                return m_data->Get<float>(offset, m_i);
            }
            CUDA_DEVICE void emitter_index(int index) noexcept {
                auto offset                     = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3) * 2;
                m_data->Get<float>(offset, m_i) = index;
            }

            CUDA_DEVICE unsigned int random_seed() const noexcept {
                auto offset = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3) * 2 + sizeof(int);
                return m_data->Get<unsigned int>(offset, m_i);
            }
            CUDA_DEVICE void random_seed(unsigned int seed) const noexcept {
                auto offset                            = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3) * 2 + sizeof(int);
                m_data->Get<unsigned int>(offset, m_i) = seed;
            }

            CUDA_DEVICE unsigned int pixel_index() const noexcept {
                auto offset = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3) * 2 + sizeof(unsigned int) + sizeof(int);
                return m_data->Get<unsigned int>(offset, m_i);
            }
            CUDA_DEVICE void pixel_index(unsigned int index) const noexcept {
                auto offset                            = sizeof(Pupil::optix::Material::LocalBsdf) + sizeof(Pupil::optix::LocalGeometry) + sizeof(float3) * 2 + sizeof(unsigned int) + sizeof(int);
                m_data->Get<unsigned int>(offset, m_i) = index;
            }
        };

#ifndef PUPIL_CPP
        CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArrayStruct::Alloc()); }
#endif

        CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
            return Accessor(this, index);
        }
    };

    class NEERecordArray : public DynamicArrayStruct {
    public:
        struct Accessor {
        private:
            unsigned int    m_i;
            NEERecordArray* m_data;

        public:
            CUDA_DEVICE Accessor(NEERecordArray* data, unsigned int index) noexcept : m_i(index), m_data(data) {}

            CUDA_DEVICE float shadow_ray_t_max() const noexcept {
                float item;
                auto  offset = 0ull;
                item         = m_data->Get<decltype(item)>(offset, m_i);
                return item;
            }
            CUDA_DEVICE void shadow_ray_t_max(float item) const noexcept {
                auto offset                              = 0ull;
                m_data->Get<decltype(item)>(offset, m_i) = item;
            }

            CUDA_DEVICE float3 shadow_ray_dir() const noexcept {
                float3 item;
                auto   offset = sizeof(float);
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void shadow_ray_dir(float3 item) const noexcept {
                auto offset                                         = sizeof(float);
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE float3 shadow_ray_origin() const noexcept {
                float3 item;
                auto   offset = sizeof(float3) + sizeof(float);
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void shadow_ray_origin(float3 item) const noexcept {
                auto offset                                         = sizeof(float3) + sizeof(float);
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE float3 radiance() const noexcept {
                float3 item;
                auto   offset = sizeof(float3) * 2 + sizeof(float);
                item.x        = m_data->Get<float>(offset, m_i);
                item.y        = m_data->Get<float>(offset + sizeof(float), m_i);
                item.z        = m_data->Get<float>(offset + sizeof(float) * 2, m_i);
                return item;
            }
            CUDA_DEVICE void radiance(float3 item) const noexcept {
                auto offset                                         = sizeof(float3) * 2 + sizeof(float);
                m_data->Get<float>(offset, m_i)                     = item.x;
                m_data->Get<float>(offset + sizeof(float), m_i)     = item.y;
                m_data->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
            }

            CUDA_DEVICE unsigned int pixel_index() const noexcept {
                auto offset = sizeof(float3) * 3 + sizeof(float);
                return m_data->Get<unsigned int>(offset, m_i);
            }
            CUDA_DEVICE void pixel_index(unsigned int index) const noexcept {
                auto offset                            = sizeof(float3) * 3 + sizeof(float);
                m_data->Get<unsigned int>(offset, m_i) = index;
            }
        };

#ifndef PUPIL_CPP
        CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArrayStruct::Alloc()); }
#endif

        CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
            return Accessor(this, index);
        }
    };
}// namespace wf::soa