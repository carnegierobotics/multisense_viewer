//
// Created by magnus on 5/13/25.
//


#include "Viewer/Rendering/PathTracer/Device/KernelHelpers.h"

namespace VkRender::PathTracer {

    SYCL_EXTERNAL bool intersectTriangle(const Ray &ray, const float3 v0, const float3 v1, const float3 v2, float &outT, float &outU,
    float &outV, float tMin, bool cullBF)     {
        constexpr float EPS   = 5.f * std::numeric_limits<float>::epsilon(); // 6e‑7
        constexpr float EDGE_EPS = 1e-5f;   // <── tighter than tMin, loose enough for fp error

        /* 1.  Edges and normal */
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;

        float3 p  = sycl::cross(ray.direction, e2);
        float  det = sycl::dot(e1, p);

        if (cullBF) { if (det <= EPS) return false; }
        else        { if (sycl::fabs(det) <= EPS) return false; }

        float  invDet = 1.f / det;
        float3 tvec   = ray.origin - v0;

        /* 2.  u */
        float u = sycl::dot(tvec, p) * invDet;
        if (u < -EDGE_EPS || u > 1.f + EDGE_EPS) return false;

        /* 3.  v */
        float3 q = sycl::cross(tvec, e1);
        float  v = sycl::dot(ray.direction, q) * invDet;
        if (v < -EDGE_EPS || u + v > 1.f + EDGE_EPS) return false;

        /* 4.  t */
        float t = sycl::dot(e2, q) * invDet;
        if (t <= tMin) return false;

        outT = t;
        outU = u;
        outV = v;
        return true;
    }
}
