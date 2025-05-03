//
// Created by magnus on 5/3/25.
//

#ifndef PATHTRACERTYPES_H
#define PATHTRACERTYPES_H

#include <sycl/sycl.hpp>

// RayTraing in a weekend container for rays

namespace VkRender::PathTracer {
    using float2 = sycl::float2;
    using float3 = sycl::float3;
    using float4 = sycl::float4;


    // ─────────────────────────────────────────────────────────────────────────────
    // GPU-friendly Ray using float4 (w used for homogeneous coords)
    // ─────────────────────────────────────────────────────────────────────────────
    struct alignas(16) Ray {
        sycl::float3 origin; // (x,y,z, 1.0)
        float pad0 = 0.0f;
        sycl::float3 direction; // (dx,dy,dz,0.0)
        float pad1 = 0.0f;
    };

    static_assert(alignof(Ray) == 16);

    inline Ray makeRay(sycl::float3 o, // w should be 1.0f
                       sycl::float3 d) // w should be 0.0f
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
        float t; //  4 B  ray parameter
        float u, v; //  8 B  barycentrics
        float pad0; //  4 B  (pad to 16)

        sycl::float3 hitPoint;

        // second 16 bytes
        uint32_t primIdx; //  4 B
        uint32_t instIdx; //  4 B
        uint32_t pad1; //  4 B
        uint32_t pad2; //  4 B
    };

    static_assert(alignof(Hit) == 16);

    inline float3 reflect(const float3& v, const float3& n) { return v - 2 * sycl::dot(v, n) * n; }



}
#endif //PATHTRACERTYPES_H
