//
// Created by magnus on 5/3/25.
//

#ifndef PATHTRACERTYPES_H
#define PATHTRACERTYPES_H

#include <float.h>

#include <sycl/sycl.hpp>


// --- Generic M×N matrix wrapping sycl::vec<float,N> rows ---

// --- Generic M×N matrix wrapping sycl::vec<float,N> rows ---
template<size_t M, size_t N>
struct Matrix {
    static_assert(M > 0 && N > 0, "Matrix dimensions must be positive");
    using RowType = sycl::vec<float, N>;
    using value_type = float;
    std::array<RowType, M> row;

    // default constructor
    Matrix() = default;

    // cast constructor: drop extra cols/rows if converting from larger matrix
    template<size_t P, size_t Q,
        typename = std::enable_if_t<(P >= M && Q >= N)> >
    explicit Matrix(Matrix<P, Q> const &other) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j)
                row[i][j] = other.row[i][j];
    }
};

// --- Matrix × Matrix multiplication ---
template<size_t M, size_t N, size_t P>
Matrix<M, P> operator*(Matrix<M, N> const &A,
                       Matrix<N, P> const &B) {
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
template<size_t M, size_t N>
sycl::vec<float, M> operator*(Matrix<M, N> const &A,
                              sycl::vec<float, N> const &v) {
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
template<size_t M, size_t N>
sycl::vec<float, N> operator*(sycl::vec<float, M> const &v,
                              Matrix<M, N> const &A) {
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
template<size_t M, size_t N>
Matrix<M, N> operator*(Matrix<M, N> const &A, float s) {
    Matrix<M, N> R{};
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            R.row[i][j] = A.row[i][j] * s;
    return R;
}

template<size_t M, size_t N>
Matrix<M, N> operator*(float s, Matrix<M, N> const &A) {
    return A * s;
}


// --- Transpose ---
template<size_t M, size_t N>
Matrix<N, M> transpose(Matrix<M, N> const &m) {
    Matrix<N, M> t{};
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            t.row[j][i] = m.row[i][j];
    return t;
}

// --- Inverse for 3×3 ---
inline Matrix<3, 3> inverse(Matrix<3, 3> const &m) {
    auto &r = m.row;
    float a00 = r[0][0], a01 = r[0][1], a02 = r[0][2];
    float a10 = r[1][0], a11 = r[1][1], a12 = r[1][2];
    float a20 = r[2][0], a21 = r[2][1], a22 = r[2][2];
    float co0 = a11 * a22 - a12 * a21;
    float co1 = -a10 * a22 + a12 * a20;
    float co2 = a10 * a21 - a11 * a20;
    float det = a00 * co0 + a01 * co1 + a02 * co2;
    Matrix<3, 3> inv{};
    inv.row[0] = sycl::vec<float, 3>(co0, (-a01 * a22 + a02 * a21), (a01 * a12 - a02 * a11));
    inv.row[1] = sycl::vec<float, 3>(co1, (a00 * a22 - a02 * a20), (-a00 * a12 + a02 * a10));
    inv.row[2] = sycl::vec<float, 3>(co2, (-a00 * a21 + a01 * a20), (a00 * a11 - a01 * a10));
    return inv * (1.0f / det);
}




/* ---------- 1. POD wrapper, no inheritance, no namespace injection -------- */
struct  alignas(16) float3 {
    sycl::vec<float,3>  v  __attribute__((aligned(16)));

    /* implicit from base */
    float3(sycl::vec<float, 3> const &b = {0, 0, 0}) : v(b) {
    }

    float3(float x, float y, float z) : v{x, y, z} {
    }

    explicit float3(float x) : v{x, x, x} {
    }

    explicit float3(sycl::vec<float,4> const& b) : v{b.x(), b.y(), b.z()} {}


    /* implicit to base */
    operator sycl::vec<float, 3>() const { return v; }

    /* ---------- subscript operator ----------------------------------- */
    float& operator[](std::size_t i)             { return v[i]; }   // l-value
    float  operator[](std::size_t i) const       { return v[i]; }   // r-value

    /* ---------- unary operators ------------------------------------ */
    float3  operator-() const { return float3{ -v }; }
    float3  operator+() const { return *this; }          // optional

    /* helpers identical to sycl::vec API */
    float x() const { return v.x(); }
    float &x() { return v.x(); }
    float y() const { return v.y(); }
    float &y() { return v.y(); }
    float z() const { return v.z(); }
    float &z() { return v.z(); }

};

/* float4, same idea -------------------------------------------------------- */
struct float4 {
    sycl::vec<float, 4> v;
    /* implicit */
    /* NOLINTNEXTLINE(google-explicit-constructor) */
    float4(sycl::vec<float, 4> const &b = {0, 0, 0, 0}) : v(b) {
    }

    float4(float x, float y, float z, float w) : v{x, y, z, w} {
    }

    float4(float3 const &p, float w) : v{p.x(), p.y(), p.z(), w} {
    }

    operator sycl::vec<float, 4>() const { return v; }

    /* ---------- subscript operator ----------------------------------- */
    float& operator[](std::size_t i)             { return v[i]; }   // l-value
    float  operator[](std::size_t i) const       { return v[i]; }   // r-value

    float x() const { return v.x(); }
    float &x() { return v.x(); }
    float y() const { return v.y(); }
    float &y() { return v.y(); }
    float z() const { return v.z(); }
    float &z() { return v.z(); }
    float w() const { return v.w(); }
    float &w() { return v.w(); }
};

/* ---------- 2. arithmetic that delegates to sycl::vec --------------------- */
inline float3 operator+(float3 a, float3 b)
{
    return { a.x() + b.x(),
             a.y() + b.y(),
             a.z() + b.z() };
}

inline float3 operator-(float3 a, float3 b)
{
    return { a.x() - b.x(),
             a.y() - b.y(),
             a.z() - b.z() };
}

/* component-wise product */
inline float3 operator*(float3 a, float3 b)
{
    return { a.x() * b.x(),
             a.y() * b.y(),
             a.z() * b.z() };
}

/* scalar products */
inline float3 operator*(float3 a, float s)
{
    return { a.x()*s, a.y()*s, a.z()*s };
}
inline float3 operator*(float s, float3 a) { return a * s; }

inline float3 operator/(float3 a, float s)
{
    float inv = 1.f / s;
    return { a.x()*inv, a.y()*inv, a.z()*inv };
}

inline float3 min(float3 a, float3 b) { return sycl::min(a.v, b.v); }
inline float3 max(float3 a, float3 b) { return sycl::max(a.v, b.v); }
inline float dot(float3 a, float3 b) { return sycl::dot(a.v, b.v); }


/*
// --- Normalize vector ---
inline float4 normalize(float4 &v) {
    float sum = 0.0f;
    for (size_t i = 0; i < 4; ++i)
        sum += v[i] * v[i];
    float invLen = sycl::rsqrt(sum);
    return v * invLen;
}
*/
inline float3 normalize(const float3 &v) {
    float sum = 0.0f;
    for (size_t i = 0; i < 3; ++i)
        sum += v[i] * v[i];
    float invLen = sycl::rsqrt(sum);
    return v * invLen;
}

/* keep float2 as single alias once */
using float2 = sycl::float2;

// --- Convenient aliases ---
using float2x2 = Matrix<2, 2>;
using float3x3 = Matrix<3, 3>;
using float4x4 = Matrix<4, 4>;


// --- Wrapper overloads to route through base operator* ---
template<size_t M>
sycl::vec<float, M> operator*(Matrix<M, 3> const &A, float3 const &v) {
    return ::operator*<M, 3>(A, static_cast<sycl::vec<float, 3>>(v));
}

template<size_t M>
sycl::vec<float, M> operator*(Matrix<M, 4> const &A, float4 const &v) {
    // explicitly invoke the Matrix×Vector template with N=4
    return ::operator*<M, 4>(A, static_cast<sycl::vec<float, 4>>(v));
}

static_assert(sizeof(float3) == 16, "float3 must be 16 bytes");
static_assert(alignof(float3) == 16, "float3 must be 16-byte aligned");

#endif //PATHTRACERTYPES_H
