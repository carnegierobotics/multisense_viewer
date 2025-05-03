//
// Created by magnus on 5/3/25.
//

#ifndef KERNELHELPERS_H
#define KERNELHELPERS_H

namespace VkRender::PathTracer{

 float rnd(uint32_t seed) {
        // simple LCG or whatever you have
        seed = 1664525u * seed + 1013904223u;
        return (seed & 0x00FFFFFF) / float(0x01000000);
    }

    //------------------------------------------------------------------------------
    /// Performs a ray–AABB intersection test using the slab method.
    /// \param ray     The ray to test.
    /// \param node    The BVH node containing bboxMin/bboxMax.
    /// \param invDir  Precomputed component-wise 1/dir of the ray.
    /// \param tMax    Current closest hit distance (to cull far nodes).
    /// \param tEntry  Output earliest intersection distance.
    /// \returns True if the ray hits the AABB before tMax.
    //------------------------------------------------------------------------------
    static inline bool slabIntersectAABB(
        const Ray &ray,
        const BVHNode &node,
        const float3 &invDir,
        float tMax,
        float &tEntry) {
        float3 t0 = (node.bboxMin - ray.origin) * invDir;
        float3 t1 = (node.bboxMax - ray.origin) * invDir;
        float3 tmin3 = sycl::min(t0, t1);
        float3 tmax3 = sycl::max(t0, t1);

        float tmin = sycl::fmax(sycl::fmax(tmin3.x(), tmin3.y()), tmin3.z());
        float tmax = sycl::fmin(sycl::fmin(tmax3.x(), tmax3.y()), tmax3.z());

        if (tmax < sycl::fmax(tmin, 0.f) || tmin > tMax)
            return false;

        tEntry = tmin;
        return true;
    }

    //------------------------------------------------------------------------------
    /// Möller–Trumbore ray–triangle intersection.
    /// \param ray   The ray in world space.
    /// \param v0,v1,v2  Triangle vertex positions.
    /// \param outT   Output distance along ray.
    /// \param outU,outV  Output barycentric coords.
    /// \returns True if the ray hits the triangle.
    //------------------------------------------------------------------------------
    static inline bool intersectTriangle(
        const Ray &ray,
        const float3 &v0,
        const float3 &v1,
        const float3 &v2,
        float &outT,
        float &outU,
        float &outV) {
        const float3 e1 = v1 - v0;
        const float3 e2 = v2 - v0;
        const float3 p = sycl::cross(ray.direction, e2);
        float det = sycl::dot(e1, p);
        if (sycl::fabs(det) < 1e-8f)
            return false;
        float invDet = 1.f / det;

        const float3 tvec = ray.origin - v0;
        float u = sycl::dot(tvec, p) * invDet;
        if (u < 0.f || u > 1.f)
            return false;

        const float3 q = sycl::cross(tvec, e1);
        float v = sycl::dot(ray.direction, q) * invDet;
        if (v < 0.f || u + v > 1.f)
            return false;

        float t = sycl::dot(e2, q) * invDet;
        if (t <= 1e-4f)
            return false;

        outT = t;
        outU = u;
        outV = v;
        return true;
    }


    //------------------------------------------------------------------------------
    /// Samples a point and normal on an area light uniformly.
    /// \param light  The area light.
    /// \param seed   RNG seed.
    /// \param outPos Output world-space position.
    /// \param outN   Output surface normal.
    /// \param outPdf Output PDF of the sample.
    //------------------------------------------------------------------------------
    static inline void sampleAreaLight(
        const AreaLight &light,
        uint32_t seed,
        float3 &outPos,
        float3 &outN,
        float &outPdf) {
        float u = rnd(seed);
        float v = rnd(seed ^ 0x9ABC);
        outPos = light.origin + u * light.edgeU + v * light.edgeV;
        outN = normalize(cross(light.edgeU, light.edgeV));
        outPdf = 1.f / light.area;
    }

    //------------------------------------------------------------------------------
    /// Samples a cosine-weighted direction around a normal.
    /// \param N      The surface normal.
    /// \param seed1  RNG seed #1.
    /// \param seed2  RNG seed #2.
    /// \returns A unit-length direction.
    //------------------------------------------------------------------------------
    static inline float3 sampleCosineHemisphere(
        const float3 &N,
        uint32_t seed1,
        uint32_t seed2) {
        float r1 = rnd(seed1);
        float r2 = rnd(seed2);
        float phi = 2.f * M_PIf * r1;
        float cosT = sqrt(1.f - r2);
        float sinT = sqrt(r2);
        float3 local = {cos(phi) * sinT, sin(phi) * sinT, cosT};

        float3 up = fabs(N.z()) < 0.99f ? float3{0, 0, 1} : float3{1, 0, 0};
        float3 tangent = normalize(cross(up, N));
        float3 bitan = cross(N, tangent);
        return normalize(
            local.x() * tangent +
            local.y() * bitan +
            local.z() * N);
    }

    //------------------------------------------------------------------------------
    /// Spawns the next photon bounce ray.
    /// \param hit    Intersection record.
    /// \param N      Surface normal at hit.
    /// \param seed1  RNG seed #1.
    /// \param seed2  RNG seed #2.
    /// \returns A new Ray starting just above the surface.
    //------------------------------------------------------------------------------
    static inline Ray spawnNextRay(
        const Hit &hit,
        const float3 &N,
        uint32_t seed1,
        uint32_t seed2) {
        float3 dir = sampleCosineHemisphere(N, seed1, seed2);
        float3 origin = hit.hitPoint + N * 1e-4f;
        return makeRay(origin, dir);
    }



}

#endif //KERNELHELPERS_H
