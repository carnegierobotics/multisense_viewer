//
// Created by magnus on 5/3/25.
//

#ifndef KERNELHELPERS_H
#define KERNELHELPERS_H
#include <glm/glm.hpp>

#include "GPUDataTypes.h"

namespace VkRender::PathTracer {

    struct PCG32
    {
        // 64-bit state, 64-bit stream (“increment”), both must be odd
        uint64_t state;
        uint64_t inc;

        // Seed with an arbitrary 64-bit seed and stream identifier
        // stream must be odd
        void seed(uint64_t init_state, uint64_t init_seq = 1u) {
            state = 0u;
            inc   = (init_seq << 1u) | 1u;
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
            uint32_t rot        = static_cast<uint32_t>(oldstate >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

        // Return float in [0,1)
        inline float nextFloat() {
            // use top 24 bits for a 24-bit mantissa
            return (nextUInt() & 0x00FFFFFF) / float(0x01000000);
        }
    };

    inline sycl::float3 glm2sycl(const glm::vec3 &v) {
        return sycl::float3{ v.x, v.y, v.z };
    }

    inline sycl::float4 glm2sycl(const glm::vec4 &v) {
        return sycl::float4{ v.x, v.y, v.z, v.w };
    }

    inline float4x4 glm2sycl(const glm::mat4 &m) {
        float4x4 out;
        for(int r=0;r<4;++r)
            for(int c=0;c<4;++c)
                out.row[r][c] = m[c][r];
        return out;
    }

    inline float3x3 glm2sycl(const glm::mat3 &m) {
        float3x3 out;
        for(int r=0;r<3;++r)
            for(int c=0;c<3;++c)
                out.row[r][c] = m[c][r];
        return out;
    }


     /*
    // SYCL → GLM vec3
    inline glm::vec3 sycl2glm(const sycl::float3 &v) {
        return glm::vec3{ v.x(), v.y(), v.z() };
    }

    // SYCL → GLM vec4
    inline glm::vec4 sycl2glm(const sycl::float4 &v) {
        return glm::vec4{ v.x(), v.y(), v.z(), v.w() };
    }

    // matrix conversions -----------------------------------------------

    // SYCL marray< float4, 4 > is row‑major: m[row][col]
    // GLM mat4  is column‑major: m[col][row]
    inline glm::mat4 sycl2glm(const float4x4 &M) {
        glm::mat4 out(1.0f);
        for(int row = 0; row < 4; ++row) {
            for(int col = 0; col < 4; ++col) {
                out[col][row] = M[row][col];
            }
        }
        return out;
    }

    // SYCL marray< float3, 3 > is row‑major: m[row][col]
    // GLM mat3  is column‑major: m[col][row]
    inline glm::mat3 sycl2glm(const float3x3 &M) {
        glm::mat3 out(1.0f);
        for(int row = 0; row < 3; ++row) {
            for(int col = 0; col < 3; ++col) {
                out[col][row] = M[row][col];
            }
        }
        return out;
    }

*/

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
        const Ray& ray,
        const BVHNode& node,
        const float3& invDir,
        float tMax,
        float& tEntry) {
        /*
        float3 tmp = node.bboxMin - ray.origin;
        float3 t0 = tmp * invDir;
        float3 tmp2 = node.bboxMax - ray.origin;
        float3 t1 = tmp2 * invDir;
        float3 tmin3 = sycl::min(t0, t1);
        float3 tmax3 = sycl::max(t0, t1);

        float tmin = sycl::fmax(sycl::fmax(tmin3.x(), tmin3.y()), tmin3.z());
        float tmax = sycl::fmin(sycl::fmin(tmax3.x(), tmax3.y()), tmax3.z());

        bool beyondClosestGlobalHit = tmin > tMax;
        bool noOverlap = tmin > tmax;
        bool boxBehindRay = tmax < 0.f;
        if (beyondClosestGlobalHit || noOverlap || boxBehindRay)
            return false;

        tEntry = tmin;
        */
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
        const Ray& ray,
        const float3& v0,
        const float3& v1,
        const float3& v2,
        float& outT,
        float& outU,
        float& outV) {
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
    /*
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
    */

    static inline void sampleMeshLight(
        const MeshLight& light,
         PCG32& rng,
        sycl::float3& outPos,
        sycl::float3& outN,
        float& outPdf) {
        // --- 1) Pick a triangle index by area-weighted CDF -------------
        float uTri = rng.nextFloat();
        // binary search in the CDF array
        int lo = 0, hi = int(light.cdf.size()) - 1;;
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
        sycl::float3 pObj =
            light.v0[tri]
          + u * light.edge1[tri]
          + v * light.edge2[tri];

        // 3) Transform to world‐space
        //   assume objectToWorld is a sycl::float4x4
        sycl::float4 pH = light.transform.objectToWorld * sycl::float4{pObj, 1.0f};
        outPos = sycl::float3{ pH.x(), pH.y(), pH.z() };

        float3x3 rotation = float3x3(light.transform.objectToWorld);               // drop translation
        float3x3 normalMat = transpose( inverse( rotation ) );          // inverse‐transpose


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
    static inline float3 sampleCosineHemisphere(
        const float3& N,
        PCG32& rng) {
        float r1 = rng.nextFloat();
        float r2 = rng.nextFloat();
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
        const Hit& hit,
        const float3& N,
        PCG32& rng) {
        float3 dir = sampleCosineHemisphere(N, rng);
        float3 origin = hit.hitPoint + N * 1e-4f;
        return makeRay(origin, dir);
    }
}

#endif //KERNELHELPERS_H
