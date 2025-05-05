//
// Created by magnus on 3/7/25.
//

#ifndef PATHTRACERKERNELCOMMON_H
#define PATHTRACERKERNELCOMMON_H

namespace VkRender::PathTracer {
    inline float calculateGeodesic(const glm::vec3 hitLocal, const QuadricInputAssembly &quadric, float alphaX,
                                   float alphaY, GPUDataOutput::QuadraticInfo *quadInfo = nullptr) {

        /*
        float rho = sqrtf(std::pow(hitLocal.x, 2.0f) + std::pow(hitLocal.y, 2.0f));
        float theta = atan2f(hitLocal.y, hitLocal.x);


        // Compute a(theta) = c * ( (alphaX * cos²(theta))/(a²) + (alphaY * sin²(theta))/(b²) )
        float a_theta = quadric.c * ((alphaX * std::cos(theta) * std::cos(theta)) / (quadric.a * quadric.a)
                                     + (alphaY * std::sin(theta) * std::sin(theta)) / (quadric.b * quadric.b));

        // Compute the geodesic distance l(ρ) along the surface
        const float epsilon = 1e-6f;


        float geodesicDist = FLT_MAX;
        if (std::fabs(a_theta) > epsilon) {
            // l(ρ) = (ρ/2)*sqrt(1+4a(θ)²ρ²) + asinh(2a(θ)ρ)/(4a(θ))
            float term1 = 0.5f * rho * std::sqrt(1.0f + 4.0f * a_theta * a_theta * rho * rho);
            float term2 = std::asinh(2.0f * a_theta * rho) / (4.0f * a_theta);
            geodesicDist = term1 + term2;
        } else {
            // When a(θ) is nearly zero, use Euclidean distance.
            geodesicDist = rho;
        }



        // Square distance function
        /*
                float rho_x = std::fabs(hitLocal.x);
                float rho_y = std::fabs(hitLocal.y);

                float geodesic_x = 0.0f;
                float geodesic_y = 0.0f;

                if (std::fabs(a_theta) > epsilon) {
                    auto computeGeodesic = [&](float rho_val) -> float {
                        float term1 = 0.5f * rho_val * std::sqrt(1.0f + 4.0f * a_theta * a_theta * rho_val * rho_val);
                        float term2 = std::asinh(2.0f * a_theta * rho_val) / (4.0f * a_theta);
                        return term1 + term2;
                    };

                    geodesic_x = computeGeodesic(rho_x);
                    geodesic_y = computeGeodesic(rho_y);
                } else {
                    // When a(θ) is nearly zero, use Euclidean distance.
                    geodesic_x = rho_x;
                    geodesic_y = rho_y;
                }
                float geodesicDist = std::max(geodesic_x, geodesic_y);

                */

        float geodesicDist =   sqrtf(std::pow(hitLocal.x, 2.0f) + std::pow(hitLocal.y, 2.0f) + std::pow(hitLocal.z, 2.0f));

        if (quadInfo) {
            quadInfo->geodesic = geodesicDist;
        }


        return geodesicDist;
    }

    // Helper function: ray-AABB intersection (using the slab method)
    // Returns true if the ray (origin, dir) hits the AABB between t=0 and t_max.
    bool rayAABBIntersect(const glm::vec3 &origin, const glm::vec3 &dir,
                          const glm::vec3 &bboxMin, const glm::vec3 &bboxMax,
                          float t_max) {
        float tmin = 0.0f;
        float tmax = t_max;
        for (int i = 0; i < 3; i++) {
            float invD = 1.0f / dir[i];
            float t0 = (bboxMin[i] - origin[i]) * invD;
            float t1 = (bboxMax[i] - origin[i]) * invD;
            if (invD < 0.0f)
                std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    // Helper function: performs detailed intersection test for one quadric.
    // Returns true if the ray (origin, dir) intersects the given quadric.
    // Fills tCandidate, hitWorld, hitNormal, and beta (the kernel value).
    inline bool intersectQuadricLeaf(const glm::vec3 &origin, const glm::vec3 &dir,
                                     const QuadricInputAssembly &quadric,
                                     float &tCandidate,
                                     glm::vec3 &hitWorld,
                                     glm::vec3 &hitNormal,
                                     float &beta,
                                     GPUDataOutput::QuadraticInfo &quadraticInfo) {
        // ---------------------------------------------------------------------------
        // Ray–plane intersection – plane’s local normal is (0,0,1) and passes through
        // the local origin (so dPlane = 0)
        // ---------------------------------------------------------------------------
        glm::mat4 transform = quadric.transform.getTransform();
        float det = glm::determinant(transform);
        if (fabs(det) < 1e-6f)
            return false;                                   // non-invertible transform

        glm::mat4 localFromWorld = glm::inverse(transform);

        // local-space ray -----------------------------------------------------------
        glm::vec3 o = glm::vec3(localFromWorld * glm::vec4(origin, 1.0f)); // q_o
        glm::vec3 d = glm::vec3(localFromWorld * glm::vec4(dir,    0.0f)); // q_dir
        quadraticInfo.localRayOrigin    = o;
        quadraticInfo.localRayDirection = d;

        // plane data ----------------------------------------------------------------
        const glm::vec3 n(0.0f, 0.0f, 1.0f);   // unit normal  (local space)
        const float     dPlane = 0.0f;         // plane offset (n·x + d = 0 ⇒ z = 0)

        const float eps = 1e-6f;
        float denom = glm::dot(n, d);          // n · d
        if (fabs(denom) < eps)                 // ray parallel to plane
            return false;

        float numer = -(glm::dot(n, o) + dPlane);
        float t     = numer / denom;           // t = -(n·o + d)/ (n·d)
        if (t <= eps)                          // hit lies behind the origin
            return false;

        // success -------------------------------------------------------------------
        tCandidate                    = t;
        quadraticInfo.hitLocal       = o + t * d;         // q_hit,l


        // Compute the local hit point.
        glm::vec3 hitLocal = o + d * tCandidate;
        // Transform back to world space.
        glm::vec4 hitW4 = quadric.transform.getTransform() * glm::vec4(hitLocal, 1.0f);
        hitWorld = glm::vec3(hitW4) / hitW4.w;

        quadraticInfo.hitLocal = hitLocal;

        // Check if hitLocal is within valid (x,y) bounds.
        if (hitLocal.x < quadric.min.x || hitLocal.x > quadric.max.x)
            return false;
        if (hitLocal.y < quadric.min.y || hitLocal.y > quadric.max.y)
            return false;

        // Evaluate the beta kernel.
        float geodesicDist = calculateGeodesic(hitLocal, quadric, 0, 0, &quadraticInfo);

        float r = geodesicDist;

        if (r > 1.0f) {
            return false; // Not within threshold
        }
        auto betaKernel = [&](float r, float bExp) -> float {
            return std::pow(1.0f - (r * r), 4.0f * std::exp(bExp));
        };
        float bkValue = betaKernel(r, quadric.b_beta);
        if (bkValue < quadric.threshold)
            return false; // Not within threshold

        quadraticInfo.betaContribution = bkValue;
        // Compute the local normal via the gradient.
        glm::vec3 nLocal(
            2.0f * quadric.a * quadric.a * hitLocal.x,
            2.0f * quadric.a * quadric.a * hitLocal.y,
           -2.0f * hitLocal.z
        );
        nLocal = glm::normalize(nLocal);

        glm::mat3 mat = glm::mat3(quadric.transform.getTransform());
        glm::mat3 worldNormalMat = glm::transpose(glm::inverse(mat));
        glm::vec3 nWorld = glm::normalize(worldNormalMat * nLocal);

        if (glm::dot(nWorld, -dir) < 0.0f)
            nWorld = -nWorld;

        hitNormal = n;
        beta = bkValue;
        return true;
    }

    static bool checkContributionCollision(const glm::vec3 &e_o, const glm::vec3 &e_d,
                                           const QuadricInputAssembly &quadric, glm::vec3 &hit) {

        // ---------------------------------------------------------------------------
        // Ray–plane intersection – plane’s local normal is (0,0,1) and passes through
        // the local origin (so dPlane = 0)
        // ---------------------------------------------------------------------------
        glm::mat4 transform = quadric.transform.getTransform();
        float det = glm::determinant(transform);
        if (fabs(det) < 1e-6f)
            return false;                                   // non-invertible transform

        glm::mat4 localFromWorld = glm::inverse(transform);

        // local-space ray -----------------------------------------------------------
        glm::vec3 o = glm::vec3(localFromWorld * glm::vec4(e_o, 1.0f)); // q_o
        glm::vec3 d = glm::vec3(localFromWorld * glm::vec4(e_d,    0.0f)); // q_dir


        // plane data ----------------------------------------------------------------
        const glm::vec3 n(0.0f, 0.0f, 1.0f);   // unit normal  (local space)
        const float     dPlane = 0.0f;         // plane offset (n·x + d = 0 ⇒ z = 0)

        const float eps = 1e-6f;
        float denom = glm::dot(n, d);          // n · d
        if (fabs(denom) < eps)                 // ray parallel to plane
            return false;

        float numer = -(glm::dot(n, o) + dPlane);
        float t     = numer / denom;           // t = -(n·o + d)/ (n·d)
        if (t <= eps)                          // hit lies behind the origin
            return false;

        // success -------------------------------------------------------------------

        // Compute the local hit point.
        glm::vec3 hitLocal = o + d * t;
        // Transform back to world space.

        // Check if hitLocal is within valid (x,y) bounds.
        if (hitLocal.x < quadric.min.x || hitLocal.x > quadric.max.x)
            return false;
        if (hitLocal.y < quadric.min.y || hitLocal.y > quadric.max.y)
            return false;

        // Evaluate the beta kernel.
        float alphaX = std::tanh(quadric.t_x);
        float alphaY = std::tanh(quadric.t_y);
        GPUDataOutput::QuadraticInfo quadInfo;
        float geodesicDist = calculateGeodesic(hitLocal, quadric, alphaX, alphaY, &quadInfo);

        float r = geodesicDist;

        if (r > 1.0f) {
            return false; // Not within threshold
        }
        auto betaKernel = [&](float r, float bExp) -> float {
            return std::pow(1.0f - (r * r), 4.0f * std::exp(bExp));
        };
        float bkValue = betaKernel(r, quadric.b_beta);
        if (bkValue < quadric.threshold)
            return false; // Not within threshold

        return true;
    }
}
#endif //PATHTRACERKERNELCOMMON_H
