//
// Created by magnus on 5/13/25.
//


#include "Viewer/Rendering/PathTracer/Device/KernelHelpers.h"

namespace VkRender::PathTracer {

    //---------------------------------------------------------------------
    //  Quadric patch ∩ ray   (object space, z-up)
    //---------------------------------------------------------------------
    SYCL_EXTERNAL bool intersectPatch(const Ray&  ray,
                                      const OrientedPoint& P,
                                      float& tHit,
                                      float&               kHit,
                                      bool &reflective)          // ‹NEW›

    {
        /* ------------------------------------------------------------------ *
         * 1) intersect supporting plane  z = 0                                *
         * ------------------------------------------------------------------ */
        const float denom = ray.direction.z();
        if (sycl::fabs(denom) < 1.0e-6f) return false;           // ray ‖ plane

        const float t = -ray.origin.z() / denom;
        if (t < 0.0f) return false;                              // behind origin

        // 2) point of intersection in XY
        const float3 pObj = ray.origin + t * ray.direction;

        /* ------------------------------------------------------------------ *
            * 3) quick reject outside axis-aligned support rectangle              *
            * ------------------------------------------------------------------ */
        /*
        if (pObj.x() < P.minSupport.x() || pObj.x() > P.maxSupport.x() ||
            pObj.y() < P.minSupport.y() || pObj.y() > P.maxSupport.y())
            return false;
        */

        /* ------------------------------------------------------------------ *
            * 4) evaluate β-kernel k(r)                                           *
            *    - infer patch radius R  from max(|minSupport|, |maxSupport|)     *
            * ------------------------------------------------------------------ */
        /*
        const float2 sMin = P.minSupport;
        const float2 sMax = P.maxSupport;

        const float rMin = sycl::length(sMin);
        const float rMax = sycl::length(sMax);

        const float R = sycl::max(rMin, rMax);           // isotropic radius
        const float r2 = (pObj.x()*pObj.x() + pObj.y()*pObj.y()) / (R*R);
        if (r2 > 1.0f) return false;                             // safety
        const float p  = 4.0f * sycl::exp(P.beta);               // exponent
        const float k  = sycl::pow(1.0f - r2, p);                // kernel value
        */

        const float σx      = P.covX;
        const float σy      = P.covY;

        // Anisotropic Gaussian kernel
        auto gaussianKernel = [&](float x, float y) {
            float r2 = (x*x)/(σx*σx) + (y*y)/(σy*σy);
            return std::exp(-0.5f * r2);
        };

        float g = gaussianKernel(pObj.x(), pObj.y());

        float kernelValue = g * P.opacity;

        /* ------------------------------------------------------------------ *
         * 5) decide material response                                         *
         * ------------------------------------------------------------------ */
        reflective = (kernelValue >= P.threshold);      // ≥ threshold: opaque / mirror
        //  < threshold: transparent
        kHit       = kernelValue;
        tHit = t;
        return true;
    }

    // -- helper suggested by Wächter & Binder, HPG 2019 (“Watertight RT”)
    inline float scaledEps(float a, float b)
    {
        return 1.e-8f * sycl::fabs(a + b);
    }

    SYCL_EXTERNAL bool intersectTriangle(const Ray &ray, const float3 v0, const float3 v1, const float3 v2, float &outT, float &outU,
    float &outV, float tMin)     {
        const float3 e1 = v1 - v0;
        const float3 e2 = v2 - v0;

        const float3 h  = cross(ray.direction, e2);
        const float  a  = dot(e1, h);

        // 1. Parallel?
        if (sycl::fabs(a) < 1.0e-4f) return false;

        const float  f  = 1.0f / a;
        const float3 s  = ray.origin - v0;
        const float  u  = f * dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;

        const float3 q  = cross(s, e1);
        const float  v  = f * dot(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f) return false;

        const float  t  = f * dot(e2, q);
        if (t <= tMin) return false;  // behind the ray or farther than a previous hit

        outT = t;
        outU = u;
        outV = v;

        return true;
    }
}
