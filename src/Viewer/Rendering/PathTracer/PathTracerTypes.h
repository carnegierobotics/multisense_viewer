//
// Created by magnus on 5/3/25.
//

#ifndef PATHTRACERTYPES_H
#define PATHTRACERTYPES_H

#include <float.h>

#include <sycl/sycl.hpp>


// --- Generic M×N matrix wrapping sycl::vec<float,N> rows ---

// --- Generic M×N matrix wrapping sycl::vec<float,N> rows ---
template <size_t M, size_t N>
struct Matrix {
    static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
    using RowType    = sycl::vec<float, N>;
    using value_type = float;
    std::array<RowType, M> row;

    // default constructor
    Matrix() = default;

    // cast constructor: drop extra cols/rows if converting from larger matrix
    template <size_t P, size_t Q,
              typename = std::enable_if_t<(P>=M && Q>=N)>>
    explicit Matrix(Matrix<P, Q> const &other) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                row[i][j] = other.row[i][j];
    }
};

// --- Matrix × Matrix multiplication ---
template <size_t M, size_t N, size_t P>
Matrix<M, P> operator*(Matrix<M, N> const &A,
                       Matrix<N, P> const &B)
{
    Matrix<M, P> C{};
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < N; ++k)
                sum += A.row[i][k] * B.row[k][j];
            C.row[i][j] = sum;
        }
    }
    return C;
}

// --- Matrix × Vector (MxN × N) → M-vector ---
template <size_t M, size_t N>
sycl::vec<float, M> operator*(Matrix<M, N> const &A,
                              sycl::vec<float, N> const &v)
{
    sycl::vec<float, M> r{};
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j)
            sum += A.row[i][j] * v[j];
        r[i] = sum;
    }
    return r;
}

// --- Vector × Matrix (M × MxN) → N-vector ---
template <size_t M, size_t N>
sycl::vec<float, N> operator*(sycl::vec<float, M> const &v,
                              Matrix<M, N> const &A)
{
    sycl::vec<float, N> r{};
    for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < M; ++i)
            sum += v[i] * A.row[i][j];
        r[j] = sum;
    }
    return r;
}

// --- Scalar × Matrix multiplication ---
template <size_t M, size_t N>
Matrix<M,N> operator*(Matrix<M,N> const &A, float s) {
    Matrix<M,N> R{};
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            R.row[i][j] = A.row[i][j] * s;
    return R;
}

template <size_t M, size_t N>
Matrix<M,N> operator*(float s, Matrix<M,N> const &A) {
    return A * s;
}


// --- Transpose ---
template <size_t M, size_t N>
Matrix<N, M> transpose(Matrix<M, N> const &m) {
    Matrix<N, M> t{};
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            t.row[j][i] = m.row[i][j];
    return t;
}

// --- Inverse for 3×3 ---
inline Matrix<3,3> inverse(Matrix<3,3> const &m) {
    auto &r = m.row;
    float a00 = r[0][0], a01 = r[0][1], a02 = r[0][2];
    float a10 = r[1][0], a11 = r[1][1], a12 = r[1][2];
    float a20 = r[2][0], a21 = r[2][1], a22 = r[2][2];
    float co0 =  a11*a22 - a12*a21;
    float co1 = -a10*a22 + a12*a20;
    float co2 =  a10*a21 - a11*a20;
    float det = a00*co0 + a01*co1 + a02*co2;
    Matrix<3,3> inv{};
    inv.row[0] = sycl::vec<float,3>( co0, (-a01*a22 + a02*a21),  (a01*a12 - a02*a11));
    inv.row[1] = sycl::vec<float,3>( co1, ( a00*a22 - a02*a20), (-a00*a12 + a02*a10));
    inv.row[2] = sycl::vec<float,3>( co2, (-a00*a21 + a01*a20),  (a00*a11 - a01*a10));
    return inv * (1.0f/det);
}


// --- Normalize vector ---
template <size_t N>
sycl::vec<float,N> normalize(sycl::vec<float,N> const &v) {
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i)
        sum += v[i] * v[i];
    float invLen = sycl::rsqrt(sum);
    return v * invLen;
}

// --- Convenient vector types ---
using float2 = sycl::vec<float,2>;

// --- float3 wrapper with conversions ---
struct float3 : public sycl::vec<float,3> {
    using base = sycl::vec<float,3>;
    using base::vec;

    float3(base const &v)
      : base(v) {}

    float3(sycl::vec<float,4> const &v)
      : base(v.x(), v.y(), v.z()) {}
};

// --- float4 wrapper with homogeneous support ---
struct float4 : public sycl::vec<float,4> {
    using base = sycl::vec<float,4>;
    using base::vec;

    // from float3 + w
    float4(::float3 const &v, float w)
      : base(v.x(), v.y(), v.z(), w) {}

    // convert from base type
    float4(base const &v)
      : base(v) {}

};

// --- Redirect sycl::float3/float4 to our wrappers ---
namespace sycl {
    using float3 = ::float3;
    using float4 = ::float4;
}

// --- Convenient aliases ---
using float2 = sycl::vec<float,2>;
using float2x2 = Matrix<2,2>;
using float3x3 = Matrix<3,3>;
using float4x4 = Matrix<4,4>;



// --- Wrapper overloads to route through base operator* ---
template <size_t M>
sycl::vec<float, M> operator*(Matrix<M, 3> const &A, float3 const &v) {
    return ::operator*<M,3>(A, static_cast<sycl::vec<float,3>>(v));

}

template <size_t M>
sycl::vec<float, M> operator*(Matrix<M, 4> const &A, float4 const &v) {
    // explicitly invoke the Matrix×Vector template with N=4
    return ::operator*<M,4>(A, static_cast<sycl::vec<float,4>>(v));
}


namespace VkRender::PathTracer {

    // ─────────────────────────────────────────────────────────────────────────────
    // GPU-friendly Ray using float4 (w used for homogeneous coords)
    // ─────────────────────────────────────────────────────────────────────────────
    struct alignas(16) Ray {
        float3 origin; // (x,y,z, 1.0)
        float3 direction; // (dx,dy,dz,0.0)
    };

    static_assert(alignof(Ray) == 16);

    inline Ray makeRay(float3 o, // w should be 1.0f
                       float3 d) // w should be 0.0f
    {
        Ray r;
        r.origin = o;
        r.direction = d;
        // pad is uninitialized—no need to set
        return r;
    }


    // ─────────────────────────────────────────────────────────────────────────────
    // GPU-friendly Hit: 32 bytes, 16-byte aligned
    // ─────────────────────────────────────────────────────────────────────────────
    struct alignas(16) Hit {
        // first 16 bytes
        float t = FLT_MAX; //  4 B  ray parameter
        float u = 0, v = 0; //  8 B  barycentrics

        float3 hitPoint;

        // second 16 bytes
        uint32_t primIdx = INT32_MAX; //  4 B
        uint32_t instIdx = INT32_MAX; //  4 B
    };

    static_assert(alignof(Hit) == 16);
}

#endif //PATHTRACERTYPES_H
