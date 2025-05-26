//
// Created by magnus on 5/3/25.
//

#ifndef KERNELHELPERS_H
#define KERNELHELPERS_H
#include <glm/glm.hpp>

#include "Viewer/Rendering/PathTracer/GPUDataTypes.h"

#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif


namespace VkRender::PathTracer {
    struct PCG32 {
        // 64-bit state, 64-bit stream (“increment”), both must be odd
        uint64_t state;
        uint64_t inc;

        // Seed with an arbitrary 64-bit seed and stream identifier
        // stream must be odd
        void seed(uint64_t init_state, uint64_t init_seq = 1u) {
            state = 0u;
            inc = (init_seq << 1u) | 1u;
            nextUInt();
            state += init_state;
            nextUInt();
        }

        // Advance generator and return 32-bit uniformly random integer
        inline uint32_t nextUInt() {
            uint64_t oldstate = state;
            // advance internal state
            state = oldstate * 6364136223846793005ULL + inc;
            // calculate output function (XSH RR), uses oldstate
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

        // Return float in [0,1)
        inline float nextFloat() {
            // use top 24 bits for a 24-bit mantissa
            return (nextUInt() & 0x00FFFFFF) / float(0x01000000);
        }
    };

    inline float3 glm2sycl(const glm::vec3 &v) {
        return float3{v.x, v.y, v.z};
    }

    inline float2 glm2sycl(const glm::vec2 &v) {
        return float2{v.x, v.y};
    }

    inline float4 glm2sycl(const glm::vec4 &v) {
        return float4{v.x, v.y, v.z, v.w};
    }


    inline glm::vec3 sycl2glm(const float3 &v) {
        return glm::vec3{v.x(), v.y(), v.z()};
    }

    inline glm::vec4 sycl2glm(const float4 &v) {
        return {v.x(), v.y(), v.z(), v.w()};
    }

    inline float4x4 glm2sycl(const glm::mat4 &m) {
        float4x4 out;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                out.row[r][c] = m[c][r];
        return out;
    }

    inline float3x3 glm2sycl(const glm::mat3 &m) {
        float3x3 out;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                out.row[r][c] = m[c][r];
        return out;
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
    inline bool slabIntersectAABB(const Ray &ray,
                                  const TLASNode &node,
                                  const float3 &invDir,
                                  float tMaxLimit,
                                  float &tEntry) {
        float3 t0 = (node.aabbMin - ray.origin) * invDir;
        float3 t1 = (node.aabbMax - ray.origin) * invDir;

        float3 tmin3 = sycl::min(t0, t1);
        float3 tmax3 = sycl::max(t0, t1);

        float tmin = sycl::fmax(sycl::fmax(tmin3.x(), tmin3.y()), tmin3.z());
        float tmax = sycl::fmin(sycl::fmin(tmax3.x(), tmax3.y()), tmax3.z());

        /* 1.  Origin outside slabs AND entry after exit  ➜  miss          */
        if (tmin > tmax) return false;

        /* 2.  Whole box lies behind the ray                                  */
        if (tmax < 0.0f) return false;

        /* 3.  Already found a closer hit in the SAME SPACE                   */
        if (tmin > tMaxLimit) return false;

        tEntry = sycl::fmax(tmin, 0.0f); // clamp if origin is inside
        return true;
    }


    inline bool slabIntersectAABB(const Ray &ray,
                                  const BVHNode &node,
                                  const float3 &invDir,
                                  float tMaxLimit,
                                  float &tEntry) {
        float3 t0 = (node.aabbMin - ray.origin) * invDir;
        float3 t1 = (node.aabbMax - ray.origin) * invDir;

        float3 tmin3 = sycl::min(t0, t1);
        float3 tmax3 = sycl::max(t0, t1);

        float tmin = sycl::fmax(sycl::fmax(tmin3.x(), tmin3.y()), tmin3.z());
        float tmax = sycl::fmin(sycl::fmin(tmax3.x(), tmax3.y()), tmax3.z());

        /* 1.  Origin outside slabs AND entry after exit  ➜  miss          */
        if (tmin > tmax) {
            return false;
        }
        /* 2.  Whole box lies behind the ray                                  */
        if (tmax < 0.0f) return false;

        /* 3.  Already found a closer hit in the SAME SPACE                   */
        if (tmin > tMaxLimit) return false;

        tEntry = sycl::fmax(tmin, 0.0f); // clamp if origin is inside
        return true;
    }


    inline float3 safeInvDir(const float3 &dir) {
        constexpr float EPS = 1e-8f; // treat anything smaller as “zero”
        constexpr float HUGE = 1e30f; // 2^100 ≃ 1.27e30 still fits in float

        float3 inv;

        inv.x() = (sycl::fabs(dir.x()) < EPS) ? HUGE : 1.f / dir.x();
        inv.y() = (sycl::fabs(dir.y()) < EPS) ? HUGE : 1.f / dir.y();
        inv.z() = (sycl::fabs(dir.z()) < EPS) ? HUGE : 1.f / dir.z();

        return inv;
    }

    //------------------------------------------------------------------------------
    /// Möller–Trumbore ray–triangle intersection.
    /// \param ray   The ray in world space.
    /// \param v0,v1,v2  Triangle vertex positions.
    /// \param outT   Output distance along ray.
    /// \param outU,outV  Output barycentric coords.
    /// \returns True if the ray hits the triangle.
    //------------------------------------------------------------------------------
    SYCL_EXTERNAL bool intersectTriangle(const Ray &ray,
                                         const float3 v0,
                                         const float3 v1,
                                         const float3 v2,
                                         float &outT,
                                         float &outU,
                                         float &outV,
                                         float tMin = 1e-4f,
                                         bool cullBF = false);

    SYCL_EXTERNAL bool intersectPatch(const Ray &ray,
                                      const OrientedPoint &P,
                                      float &tHit,
                                      float &kHit,

                                      bool &relfective);


    inline void sampleMeshLight(
        const MeshLight &light,
        PCG32 &rng,
        float3 &outPos,
        float3 &outN,
        float &outPdf) {
        // --- 1) Pick a triangle index by area-weighted CDF -------------
        float uTri = rng.nextFloat();
        // binary search in the CDF array
        int lo = 0, hi = int(light.triangleCount) - 1;;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (uTri <= light.cdf[mid]) hi = mid;
            else lo = mid + 1;
        }
        int tri = lo;

        // --- 2) Uniformly sample a point on that triangle -------------
        // generate two more randoms (mix the seed for decorrelation)
        float u = rng.nextFloat();
        float v = rng.nextFloat();
        // fold back into the triangle if outside
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        // object-space position
        float3 pObj =
                light.v0[tri]
                + u * light.edge1[tri]
                + v * light.edge2[tri];

        // 3) Transform to world‐space
        //   assume objectToWorld is a sycl::float4x4
        float4 pH = light.transform.objectToWorld * float4{pObj, 1.0f};
        outPos = float3{pH.x(), pH.y(), pH.z()};

        float3x3 rotation = float3x3(light.transform.objectToWorld); // drop translation
        float3x3 normalMat = transpose(inverse(rotation)); // inverse‐transpose


        // --- 3) Return the surface normal and the PDF for position ----
        outN = normalize(normalMat * light.normal[tri]);
        // PDF = 1 / total emissive area (uniform over mesh surface)
        outPdf = 1.0f / light.totalArea;
    }

    //------------------------------------------------------------------------------
    /// Samples a cosine-weighted direction around a normal.
    /// \param N      The surface normal.
    /// \param seed1  RNG seed #1.
    /// \param seed2  RNG seed #2.
    /// \returns A unit-length direction.
    //------------------------------------------------------------------------------
    SYCL_EXTERNAL inline void sampleCosineHemisphere(
        PCG32 &rng, const float3 &n,
        float3 &outDir, float &outPdf) {
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float r = sycl::sqrt(u1);
        float phi = 2.f * M_PIf * u2;

        float x = r * sycl::cos(phi);
        float y = r * sycl::sin(phi);
        float z = sycl::sqrt(1.f - u1);

        // build an ONB around n
        float3 up = sycl::fabs(n.z()) < .999f ? float3{0, 0, 1} : float3{1, 0, 0};
        float3 tang = ::normalize(sycl::cross(up, n));
        float3 bit = sycl::cross(n, tang);

        outDir = normalize(x * tang + y * bit + z * n);
        outPdf = sycl::max(0.f, sycl::dot(outDir, n)) / M_PIf; // cosθ/π
    }

    SYCL_EXTERNAL inline float3 reflect(const float3 &v, const float3 &n) {
        return v - 2.f * sycl::dot(n, v) * n;
    }

    /*
    // specular Blinn‑Phong lobe sampling – returns dir and pdf
    SYCL_EXTERNAL inline void sampleBlinnPhongSpecular(
        PCG32 &rng, const float3 &n, const float3 &omegaIn,
        float shininess,
        float3 &outDir, float &outPdf) {
        // 1. sample half‑vector
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();

        float cosThetaH = sycl::pow(u1, 1.f / (shininess + 1.f));
        float sinThetaH = sycl::sqrt(1.f - cosThetaH * cosThetaH);
        float phi = 2.f * M_PIf * u2;

        float3 up = sycl::fabs(n.z()) < .999f ? float3{0, 0, 1} : float3{1, 0, 0};
        float3 tang = ::normalize(sycl::cross(up, n));
        float3 bit = sycl::cross(n, tang);

        float3 h = ::normalize(sinThetaH * sycl::cos(phi) * tang +
                             sinThetaH * sycl::sin(phi) * bit +
                             cosThetaH * n);

        // 2. reflect incoming direction about h
        outDir = ::normalize(reflect(-omegaIn, h));

        // 3. pdf  p(ωo) = (s+2)/(2π) (n·h)^s  (n·h)/(4 |h·ωi|)
        float nh = sycl::max(0.f, sycl::dot(n, h));
        float hi = sycl::max(0.f, sycl::dot(h, -omegaIn));
        outPdf = ((shininess + 2.f) * nh * sycl::pow(nh, shininess)) /
                 (2.f * M_PIf * 4.f * hi + 1e-7f); // 1e-7 to avoid /0
    }

*/

    //──────────────── world → object and back ────────────────────────────────
    inline Ray toObjectSpace(const Ray &rayW, const Transform &xf) {
        Ray r;
        /* 1.  Transform origin – w = 1                                      */
        float4 hO = xf.worldToObject * float4{rayW.origin, 1.f};
        r.origin = float3{hO.x(), hO.y(), hO.z()} / hO.w(); // <- perspective divide

        /* 2.  Transform direction – w = 0  (no translation component)       */
        float4 hD = xf.worldToObject * float4{rayW.direction, 0.f};
        r.direction = normalize(float3{hD.x(), hD.y(), hD.z()}); // w is already 0
        return r;
    }

    inline float3 toWorldPoint(const float3 &pO, const Transform &xf) {
        float4 hp = xf.objectToWorld * float4{pO, 1.f};
        return float3{hp.x(), hp.y(), hp.z()} / hp.w();
    }
}

#endif //KERNELHELPERS_H
