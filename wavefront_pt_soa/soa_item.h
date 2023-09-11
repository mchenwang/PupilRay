#pragma once
#include "cuda/data_view.h"
#include "cuda/vec_math.h"
#include "cuda/stream.h"
#include "cuda/util.h"
#include "wave_context.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace wavefront::SoA {
template<typename T>
class Array {
private:
    CUdeviceptr m_data CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE Array() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data) noexcept { m_data = cuda_data; }
    CUDA_HOSTDEVICE T *GetDataPtr() const noexcept { return reinterpret_cast<T *>(m_data); }

    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }
    CUDA_HOSTDEVICE T &operator[](unsigned int index) const noexcept {
        return *reinterpret_cast<T *>(m_data + index * sizeof(T));
    }
};

using Float = Array<float>;
using Int = Array<int>;
using Bool = Array<bool>;

struct Float2 {
    Float x;
    Float y;
};
struct Float3 {
    Float x;
    Float y;
    Float z;
};
struct Float4 {
    Float x;
    Float y;
    Float z;
    Float w;
};

class DynamicArray {
protected:
    CUdeviceptr m_data CONST_STATIC_INIT(0);
    size_t m_slice_size CONST_STATIC_INIT(0);
    CUdeviceptr m_num CONST_STATIC_INIT(0);

public:
    CUDA_HOSTDEVICE DynamicArray() noexcept {}

    CUDA_HOST void SetData(CUdeviceptr cuda_data, CUdeviceptr num_data, size_t slice_size) noexcept {
        m_data = cuda_data;
        m_slice_size = slice_size;
        m_num = num_data;
    }
    CUDA_HOSTDEVICE operator bool() const noexcept { return m_data != 0; }

    CUDA_DEVICE void Clear() noexcept { *reinterpret_cast<unsigned int *>(m_num) = 0; }
    CUDA_DEVICE unsigned int GetNum() const noexcept { return *reinterpret_cast<unsigned int *>(m_num); }

#ifndef PUPIL_CPP
    CUDA_DEVICE unsigned int Alloc() noexcept {
        unsigned int *num = reinterpret_cast<unsigned int *>(m_num);
        return atomicAdd(num, 1);
    }
#endif

    template<typename T>
    CUDA_DEVICE T &Get(unsigned long long stride_offset, unsigned int index) {
        return *reinterpret_cast<T *>(m_data + stride_offset * m_slice_size + index * sizeof(T));
    }

    template<typename T>
    inline const T &Get(unsigned long long stride_offset, unsigned int index) const {
        return *reinterpret_cast<T *>(m_data + stride_offset * m_slice_size + index * sizeof(T));
    }
};

struct PathQueue : public DynamicArray {
    struct Accessor {
    private:
        unsigned int m_i;
        PathQueue *m_q;

    public:
        CUDA_DEVICE Accessor(PathQueue *path_q, unsigned int index) noexcept : m_i(index), m_q(path_q) {}

        CUDA_DEVICE Ray ray() const noexcept {
            Ray item;
            constexpr unsigned long long offset = 0;
            item.dir.x = m_q->Get<float>(offset, m_i);
            item.dir.y = m_q->Get<float>(offset + sizeof(float), m_i);
            item.dir.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.origin.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.origin.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.origin.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.t = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            return item;
        }

        CUDA_DEVICE void ray(const Ray &item) noexcept {
            constexpr unsigned long long offset = 0;
            m_q->Get<float>(offset + 0, m_i) = item.dir.x;
            m_q->Get<float>(offset + sizeof(float), m_i) = item.dir.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.dir.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.origin.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.origin.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.origin.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.t;
        }

        CUDA_DEVICE float3 throughput() const noexcept {
            float3 item;
            constexpr unsigned long long offset = sizeof(Ray);
            item.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            return item;
        }

        CUDA_DEVICE void throughput(const float3 &item) noexcept {
            constexpr unsigned long long offset = sizeof(Ray);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
        }

        CUDA_DEVICE float bsdf_sample_pdf() const noexcept {
            constexpr unsigned long long offset = sizeof(Ray) + sizeof(float3);
            return m_q->Get<float>(offset, m_i);
        }
        CUDA_DEVICE void bsdf_sample_pdf(float pdf) noexcept {
            constexpr unsigned long long offset = sizeof(Ray) + sizeof(float3);
            m_q->Get<float>(offset, m_i) = pdf;
        }

        CUDA_DEVICE Pupil::optix::EBsdfLobeType bsdf_sample_type() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11;
            return m_q->Get<Pupil::optix::EBsdfLobeType>(offset, m_i);
        }
        CUDA_DEVICE void bsdf_sample_type(const Pupil::optix::EBsdfLobeType &type) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11;
            m_q->Get<Pupil::optix::EBsdfLobeType>(offset, m_i) = type;
        }

        CUDA_DEVICE Pupil::cuda::Random random() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11 + sizeof(Pupil::optix::EBsdfLobeType);
            return m_q->Get<Pupil::cuda::Random>(offset, m_i);
        }
        CUDA_DEVICE void random(const Pupil::cuda::Random &r) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11 + sizeof(Pupil::optix::EBsdfLobeType);
            m_q->Get<Pupil::cuda::Random>(offset, m_i) = r;
        }

        CUDA_DEVICE unsigned int pixel_index() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11 + sizeof(Pupil::optix::EBsdfLobeType) + sizeof(Pupil::cuda::Random);
            return m_q->Get<unsigned int>(offset, m_i);
        }
        CUDA_DEVICE void pixel_index(unsigned int pi) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11 + sizeof(Pupil::optix::EBsdfLobeType) + sizeof(Pupil::cuda::Random);
            m_q->Get<unsigned int>(offset, m_i) = pi;
        }
    };

#ifndef PUPIL_CPP
    CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArray::Alloc()); }
#endif

    CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
        return Accessor(this, index);
    }
};

struct HitQueue : public DynamicArray {
    struct Accessor {
    private:
        unsigned int m_i;
        HitQueue *m_q;

    public:
        CUDA_DEVICE Accessor(HitQueue *path_q, unsigned int index) noexcept : m_i(index), m_q(path_q) {}

        CUDA_DEVICE Ray ray() const noexcept {
            Ray item;
            constexpr unsigned long long offset = 0;
            item.dir.x = m_q->Get<float>(offset, m_i);
            item.dir.y = m_q->Get<float>(offset + sizeof(float), m_i);
            item.dir.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.origin.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.origin.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.origin.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.t = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            return item;
        }

        CUDA_DEVICE void ray(const Ray &item) noexcept {
            constexpr unsigned long long offset = 0;
            m_q->Get<float>(offset + 0, m_i) = item.dir.x;
            m_q->Get<float>(offset + sizeof(float), m_i) = item.dir.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.dir.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.origin.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.origin.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.origin.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.t;
        }

        CUDA_DEVICE Pupil::optix::LocalGeometry geo() const noexcept {
            Pupil::optix::LocalGeometry item;
            constexpr unsigned long long offset = sizeof(Ray);
            item.position.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.position.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.position.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.normal.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.normal.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.normal.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.texcoord.x = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            item.texcoord.y = m_q->Get<float>(offset + sizeof(float) * 7, m_i);
            return item;
        }

        CUDA_DEVICE void geo(const Pupil::optix::LocalGeometry &item) const noexcept {
            constexpr unsigned long long offset = sizeof(Ray);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.position.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.position.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.position.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.normal.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.normal.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.normal.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.texcoord.x;
            m_q->Get<float>(offset + sizeof(float) * 7, m_i) = item.texcoord.y;
        }

        CUDA_DEVICE float3 throughput() const noexcept {
            float3 item;
            constexpr unsigned long long offset = sizeof(Ray) + sizeof(Pupil::optix::LocalGeometry);
            item.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            return item;
        }

        CUDA_DEVICE void throughput(const float3 &item) noexcept {
            constexpr unsigned long long offset = sizeof(Ray) + sizeof(Pupil::optix::LocalGeometry);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
        }

        CUDA_DEVICE float bsdf_sample_pdf() const noexcept {
            return m_q->Get<float>(sizeof(float) * 18, m_i);
        }
        CUDA_DEVICE void bsdf_sample_pdf(float pdf) noexcept {
            m_q->Get<float>(sizeof(float) * 18, m_i) = pdf;
        }

        CUDA_DEVICE int emitter_index() const noexcept {
            return m_q->Get<float>(sizeof(float) * 19, m_i);
        }
        CUDA_DEVICE void emitter_index(int idx) noexcept {
            m_q->Get<float>(sizeof(float) * 19, m_i) = idx;
        }

        CUDA_DEVICE Pupil::cuda::Random random() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int);
            return m_q->Get<Pupil::cuda::Random>(offset, m_i);
        }
        CUDA_DEVICE void random(const Pupil::cuda::Random &r) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int);
            m_q->Get<Pupil::cuda::Random>(offset, m_i) = r;
        }

        CUDA_DEVICE unsigned int pixel_index() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int) + sizeof(Pupil::cuda::Random);
            return m_q->Get<unsigned int>(offset, m_i);
        }
        CUDA_DEVICE void pixel_index(unsigned int pi) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int) + sizeof(Pupil::cuda::Random);
            m_q->Get<unsigned int>(offset, m_i) = pi;
        }

        CUDA_DEVICE Pupil::optix::material::Material::LocalBsdf bsdf() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int) + sizeof(Pupil::cuda::Random) + sizeof(unsigned int);
            return m_q->Get<Pupil::optix::material::Material::LocalBsdf>(offset, m_i);
        }

        CUDA_DEVICE void bsdf(const Pupil::optix::material::Material::LocalBsdf &bsdf) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int) + sizeof(Pupil::cuda::Random) + sizeof(unsigned int);
            m_q->Get<Pupil::optix::material::Material::LocalBsdf>(offset, m_i) = bsdf;
        }
    };

#ifndef PUPIL_CPP
    CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArray::Alloc()); }
#endif

    CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
        return Accessor(this, index);
    }
};

struct HitEmitterQueue : public DynamicArray {
    struct Accessor {
    private:
        unsigned int m_i;
        HitEmitterQueue *m_q;

    public:
        CUDA_DEVICE Accessor(HitEmitterQueue *path_q, unsigned int index) noexcept : m_i(index), m_q(path_q) {}

        CUDA_DEVICE Ray ray() const noexcept {
            Ray item;
            constexpr unsigned long long offset = 0;
            item.dir.x = m_q->Get<float>(offset, m_i);
            item.dir.y = m_q->Get<float>(offset + sizeof(float), m_i);
            item.dir.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.origin.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.origin.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.origin.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.t = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            return item;
        }

        CUDA_DEVICE void ray(const Ray &item) noexcept {
            constexpr unsigned long long offset = 0;
            m_q->Get<float>(offset + 0, m_i) = item.dir.x;
            m_q->Get<float>(offset + sizeof(float), m_i) = item.dir.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.dir.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.origin.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.origin.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.origin.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.t;
        }

        CUDA_DEVICE Pupil::optix::LocalGeometry geo() const noexcept {
            Pupil::optix::LocalGeometry item;
            constexpr unsigned long long offset = sizeof(Ray);
            item.position.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.position.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.position.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.normal.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.normal.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.normal.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.texcoord.x = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            item.texcoord.y = m_q->Get<float>(offset + sizeof(float) * 7, m_i);
            return item;
        }

        CUDA_DEVICE void geo(const Pupil::optix::LocalGeometry &item) const noexcept {
            constexpr unsigned long long offset = sizeof(Ray);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.position.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.position.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.position.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.normal.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.normal.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.normal.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.texcoord.x;
            m_q->Get<float>(offset + sizeof(float) * 7, m_i) = item.texcoord.y;
        }

        CUDA_DEVICE float3 throughput() const noexcept {
            float3 item;
            constexpr unsigned long long offset = sizeof(Ray) + sizeof(Pupil::optix::LocalGeometry);
            item.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            return item;
        }

        CUDA_DEVICE void throughput(const float3 &item) noexcept {
            constexpr unsigned long long offset = sizeof(Ray) + sizeof(Pupil::optix::LocalGeometry);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
        }

        CUDA_DEVICE int emitter_index() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 18;
            return m_q->Get<int>(offset, m_i);
        }

        CUDA_DEVICE void emitter_index(int item) noexcept {
            constexpr unsigned long long offset = sizeof(float) * 18;
            m_q->Get<int>(offset, m_i) = item;
        }

        CUDA_DEVICE float bsdf_sample_pdf() const noexcept {
            return m_q->Get<float>(sizeof(float) * 18 + sizeof(int), m_i);
        }
        CUDA_DEVICE void bsdf_sample_pdf(float pdf) noexcept {
            m_q->Get<float>(sizeof(float) * 18 + sizeof(int), m_i) = pdf;
        }

        CUDA_DEVICE Pupil::optix::EBsdfLobeType bsdf_sample_type() const noexcept {
            return m_q->Get<Pupil::optix::EBsdfLobeType>(sizeof(float) * 19 + sizeof(int), m_i);
        }
        CUDA_DEVICE void bsdf_sample_type(const Pupil::optix::EBsdfLobeType &type) noexcept {
            m_q->Get<Pupil::optix::EBsdfLobeType>(sizeof(float) * 19 + sizeof(int), m_i) = type;
        }

        CUDA_DEVICE unsigned int pixel_index() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int) + sizeof(Pupil::optix::EBsdfLobeType);
            return m_q->Get<unsigned int>(offset, m_i);
        }
        CUDA_DEVICE void pixel_index(unsigned int pi) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 19 + sizeof(int) + sizeof(Pupil::optix::EBsdfLobeType);
            m_q->Get<unsigned int>(offset, m_i) = pi;
        }
    };

#ifndef PUPIL_CPP
    CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArray::Alloc()); }
#endif

    CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
        return Accessor(this, index);
    }
};

struct MissQueue : public DynamicArray {
    struct Accessor {
    private:
        unsigned int m_i;
        MissQueue *m_q;

    public:
        CUDA_DEVICE Accessor(MissQueue *path_q, unsigned int index) noexcept : m_i(index), m_q(path_q) {}

        CUDA_DEVICE Ray ray() const noexcept {
            Ray item;
            constexpr unsigned long long offset = 0;
            item.dir.x = m_q->Get<float>(offset, m_i);
            item.dir.y = m_q->Get<float>(offset + sizeof(float), m_i);
            item.dir.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.origin.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.origin.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.origin.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.t = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            return item;
        }

        CUDA_DEVICE void ray(const Ray &item) noexcept {
            constexpr unsigned long long offset = 0;
            m_q->Get<float>(offset + 0, m_i) = item.dir.x;
            m_q->Get<float>(offset + sizeof(float), m_i) = item.dir.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.dir.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.origin.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.origin.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.origin.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.t;
        }

        CUDA_DEVICE float3 throughput() const noexcept {
            float3 item;
            constexpr unsigned long long offset = sizeof(Ray);
            item.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            return item;
        }

        CUDA_DEVICE void throughput(const float3 &item) noexcept {
            constexpr unsigned long long offset = sizeof(Ray);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
        }

        CUDA_DEVICE float bsdf_sample_pdf() const noexcept {
            return m_q->Get<float>(sizeof(float) * 10, m_i);
        }
        CUDA_DEVICE void bsdf_sample_pdf(float pdf) noexcept {
            m_q->Get<float>(sizeof(float) * 10, m_i) = pdf;
        }

        CUDA_DEVICE unsigned int pixel_index() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11;
            return m_q->Get<unsigned int>(offset, m_i);
        }
        CUDA_DEVICE void pixel_index(unsigned int pi) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 11;
            m_q->Get<unsigned int>(offset, m_i) = pi;
        }
    };

#ifndef PUPIL_CPP
    CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArray::Alloc()); }
#endif

    CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
        return Accessor(this, index);
    }
};

struct ShadowPathQueue : public DynamicArray {
    struct Accessor {
    private:
        unsigned int m_i;
        ShadowPathQueue *m_q;

    public:
        CUDA_DEVICE Accessor(ShadowPathQueue *path_q, unsigned int index) noexcept : m_i(index), m_q(path_q) {}

        CUDA_DEVICE Ray ray() const noexcept {
            Ray item;
            constexpr unsigned long long offset = 0;
            item.dir.x = m_q->Get<float>(offset, m_i);
            item.dir.y = m_q->Get<float>(offset + sizeof(float), m_i);
            item.dir.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            item.origin.x = m_q->Get<float>(offset + sizeof(float) * 3, m_i);
            item.origin.y = m_q->Get<float>(offset + sizeof(float) * 4, m_i);
            item.origin.z = m_q->Get<float>(offset + sizeof(float) * 5, m_i);
            item.t = m_q->Get<float>(offset + sizeof(float) * 6, m_i);
            return item;
        }

        CUDA_DEVICE void ray(const Ray &item) noexcept {
            constexpr unsigned long long offset = 0;
            m_q->Get<float>(offset + 0, m_i) = item.dir.x;
            m_q->Get<float>(offset + sizeof(float), m_i) = item.dir.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.dir.z;
            m_q->Get<float>(offset + sizeof(float) * 3, m_i) = item.origin.x;
            m_q->Get<float>(offset + sizeof(float) * 4, m_i) = item.origin.y;
            m_q->Get<float>(offset + sizeof(float) * 5, m_i) = item.origin.z;
            m_q->Get<float>(offset + sizeof(float) * 6, m_i) = item.t;
        }

        CUDA_DEVICE float3 radiance() const noexcept {
            float3 item;
            constexpr unsigned long long offset = sizeof(Ray);
            item.x = m_q->Get<float>(offset + sizeof(float) * 0, m_i);
            item.y = m_q->Get<float>(offset + sizeof(float) * 1, m_i);
            item.z = m_q->Get<float>(offset + sizeof(float) * 2, m_i);
            return item;
        }

        CUDA_DEVICE void radiance(const float3 &item) noexcept {
            constexpr unsigned long long offset = sizeof(Ray);
            m_q->Get<float>(offset + sizeof(float) * 0, m_i) = item.x;
            m_q->Get<float>(offset + sizeof(float) * 1, m_i) = item.y;
            m_q->Get<float>(offset + sizeof(float) * 2, m_i) = item.z;
        }

        CUDA_DEVICE unsigned int pixel_index() const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 10;
            return m_q->Get<unsigned int>(offset, m_i);
        }
        CUDA_DEVICE void pixel_index(unsigned int pi) const noexcept {
            constexpr unsigned long long offset = sizeof(float) * 10;
            m_q->Get<unsigned int>(offset, m_i) = pi;
        }
    };

#ifndef PUPIL_CPP
    CUDA_DEVICE Accessor Alloc() noexcept { return Accessor(this, DynamicArray::Alloc()); }
#endif

    CUDA_DEVICE Accessor operator[](unsigned int index) noexcept {
        return Accessor(this, index);
    }
};

}// namespace wavefront::SoA