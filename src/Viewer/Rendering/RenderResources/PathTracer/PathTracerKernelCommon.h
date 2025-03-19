//
// Created by magnus on 3/7/25.
//

#ifndef PATHTRACERKERNELCOMMON_H
#define PATHTRACERKERNELCOMMON_H

namespace VkRender::PathTracer {

    inline float calculateGeodesic(const glm::vec3 hitLocal, const QuadricInputAssembly& quadric, float alphaX, float alphaY) {
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
        return geodesicDist;

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
        return geodesicDist;
        */
    }
    // Helper function: ray-AABB intersection (using the slab method)
    // Returns true if the ray (origin, dir) hits the AABB between t=0 and t_max.
    bool rayAABBIntersect(const glm::vec3& origin, const glm::vec3& dir,
                          const glm::vec3& bboxMin, const glm::vec3& bboxMax,
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
    inline bool intersectQuadricLeaf(const glm::vec3& origin, const glm::vec3& dir,
                                     const QuadricInputAssembly& quadric,
                                     float& tCandidate,
                                     glm::vec3& hitWorld,
                                     glm::vec3& hitNormal,
                                     float& beta,
                                     GPUDataOutput::QuadraticInfo& quadraticInfo) {
        // Transform the ray into local space.
        glm::mat4 transform = quadric.transform.getTransform();
        float det = glm::determinant(transform);
        if (fabs(det) < 1e-6f)
            return false; // Invalid transform

        glm::mat4 localFromWorld = glm::inverse(transform);
        glm::vec4 o4 = localFromWorld * glm::vec4(origin, 1.0f);
        glm::vec4 d4 = localFromWorld * glm::vec4(dir, 0.0f);
        glm::vec3 o = glm::vec3(o4);
        glm::vec3 d = glm::vec3(d4);

        quadraticInfo.localRayOrigin = o;
        quadraticInfo.localRayDirection = d;

        // Shorthand parameters.
        float alphaX = std::tanh(quadric.t_x);
        float alphaY = std::tanh(quadric.t_y);

        // Solve quadratic: A*t^2 + B*t + C = 0.
        float Ax = d.x;
        float Ay = d.y;
        float Az = d.z;
        float Ox = o.x;
        float Oy = o.y;
        float Oz = o.z;

        float A = quadric.c * (
            alphaX * (Ax * Ax) / (quadric.a * quadric.a) +
            alphaY * (Ay * Ay) / (quadric.b * quadric.b)
        );
        float B = quadric.c * (
            2.0f * alphaX * Ox * Ax / (quadric.a * quadric.a) +
            2.0f * alphaY * Oy * Ay / (quadric.b * quadric.b)
        ) - Az;
        float C = quadric.c * (
            alphaX * (Ox * Ox) / (quadric.a * quadric.a) +
            alphaY * (Oy * Oy) / (quadric.b * quadric.b)
        ) - Oz;

        float eps = 1e-5f;
        tCandidate = std::numeric_limits<float>::max();
        if (fabs(A) < eps) {
            if (fabs(B) < eps)
                return false; // No solution
            float tLin = -C / B;
            if (tLin > eps) {
                tCandidate = tLin;
                quadraticInfo.B = B;
                quadraticInfo.C = C;
            }
            else
                return false;
        }
        else {
            float disc = (B * B) - 4.0f * A * C;
            if (disc < 0.0f)
                return false; // No real roots

            quadraticInfo.A = A;
            quadraticInfo.B = B;
            quadraticInfo.C = C;
            quadraticInfo.discriminant = disc;

            float sqrtDisc = std::sqrt(disc);
            float t1 = (-B - sqrtDisc) / (2.0f * A);
            float t2 = (-B + sqrtDisc) / (2.0f * A);
            float tMin = std::numeric_limits<float>::max();
            if (t1 > eps && t1 < tMin) {
                tMin = t1;
                quadraticInfo.rootIndex = 1;
            }
            if (t2 > eps && t2 < tMin) {
                tMin = t2;
                quadraticInfo.rootIndex = 2;
            }
            if (tMin == std::numeric_limits<float>::max())
                return false; // No valid solution
            tCandidate = tMin;
        }

        // Compute the local hit point.
        glm::vec3 hitLocal = o + d * tCandidate;
        // Transform back to world space.
        glm::vec4 hitW4 = quadric.transform.getTransform() * glm::vec4(hitLocal, 1.0f);
        hitWorld = glm::vec3(hitW4) / hitW4.w;


        // Check if hitLocal is within valid (x,y) bounds.
        if (hitLocal.x < quadric.min.x || hitLocal.x > quadric.max.x)
            return false;
        if (hitLocal.y < quadric.min.y || hitLocal.y > quadric.max.y)
            return false;


        // Evaluate the beta kernel.
        float geodesicDist = calculateGeodesic(hitLocal, quadric, alphaX, alphaY);

        quadraticInfo.geodesic = geodesicDist;
        float r = geodesicDist / quadric.kernelScale;
        auto betaKernel = [&](float r, float bExp) -> float {
            if (r > 1.0f)
                r = 1.0f;
            return std::pow(1.0f - r * r, 4.0f * std::exp(bExp));
        };
        float bkValue = betaKernel(r, quadric.b_beta);
        if (bkValue < quadric.threshold)
            return false; // Not within threshold

        quadraticInfo.betaContribution = bkValue;
        // Compute the local normal via the gradient.
        glm::vec3 gradLocal(
            2.0f * quadric.c * alphaX * hitLocal.x / (quadric.a * quadric.a),
            2.0f * quadric.c * alphaY * hitLocal.y / (quadric.b * quadric.b),
            -1.0f
        );
        glm::mat3 mat = glm::mat3(quadric.transform.getTransform());
        float det2 = glm::determinant(mat);
        if (fabs(det2) < 1e-6f)
            return false;
        glm::mat3 invT = glm::inverseTranspose(mat);
        glm::vec3 normalW = glm::normalize(invT * gradLocal);
        if (glm::dot(normalW, -dir) < 0.0f)
            normalW = -normalW;

        hitNormal = normalW;
        beta = bkValue;
        return true;
    }

    static bool checkContributionCollision(const glm::vec3& e_o, const glm::vec3& e_d,
                                           const QuadricInputAssembly& quadric, glm::vec3& hit) {
        float eps = 1.0f * 1e-5f;

        // Transform the ray into the quadric’s local space.
        glm::mat4 transform = quadric.transform.getTransform();

        glm::mat4 localFromWorld = glm::inverse(transform);
        glm::vec4 o4 = localFromWorld * glm::vec4(e_o, 1.0f);
        glm::vec4 d4 = localFromWorld * glm::vec4(e_d, 0.0f);
        glm::vec3 o = glm::vec3(o4);
        glm::vec3 d = glm::vec3(d4);

        // Shorthand for the quadric parameters.
        float alphaX = std::tanh(quadric.t_x);
        float alphaY = std::tanh(quadric.t_y);

        // Set up the quadratic equation coefficients.
        float Ax = d.x, Ay = d.y, Az = d.z;
        float Ox = o.x, Oy = o.y, Oz = o.z;
        float A = quadric.c * (alphaX * (Ax * Ax) / (quadric.a * quadric.a) +
            alphaY * (Ay * Ay) / (quadric.b * quadric.b));
        float B = quadric.c * (2.0f * alphaX * Ox * Ax / (quadric.a * quadric.a) +
            2.0f * alphaY * Oy * Ay / (quadric.b * quadric.b)) - Az;
        float C = quadric.c * (alphaX * (Ox * Ox) / (quadric.a * quadric.a) +
            alphaY * (Oy * Oy) / (quadric.b * quadric.b)) - Oz;


        // Helper lambda: Given a ray parameter t, compute the intersection, normal, and beta.
        auto computeIntersection = [&](float t) -> bool {
            // Compute local hit point.
            glm::vec3 hitLocal = o + d * t;
            glm::vec4 hitW4 = transform * glm::vec4(hitLocal, 1.0f);
            hit = glm::vec3(hitW4) / hitW4.w;

            // Check if hitLocal is within valid (x,y) bounds.
            if (hitLocal.x < quadric.min.x || hitLocal.x > quadric.max.x)
                return false;
            if (hitLocal.y < quadric.min.y || hitLocal.y > quadric.max.y)
                return false;

            float geodesicDist = calculateGeodesic(hitLocal, quadric, alphaX, alphaY);
            float r = geodesicDist / quadric.kernelScale;
            if (r > 1.0f)
                r = 1.0f;
            float beta = std::pow(1.0f - r * r, 4.0f * std::exp(quadric.b_beta));

            // Compute the local gradient and transform it to world space.
            glm::vec3 gradLocal(
                2.0f * quadric.c * alphaX * hitLocal.x / (quadric.a * quadric.a),
                2.0f * quadric.c * alphaY * hitLocal.y / (quadric.b * quadric.b),
                -1.0f
            );
            glm::mat3 mat = glm::mat3(transform);
            glm::mat3 invT = glm::inverseTranspose(mat);
            glm::vec3 normal = glm::normalize(invT * gradLocal);
            if (glm::dot(normal, -e_d) < 0.0f)
                normal = -normal;
            // For non-contribution rays, enforce the beta kernel threshold.
            if (beta < quadric.threshold)
                return false;

            return true;
        };

        // Handle the degenerate (linear) case.
        if (fabs(A) < eps) {
            if (fabs(B) < eps)
                return false; // No solution.
            float tLin = -C / B;
            if (tLin <= eps)
                return false;
            return computeIntersection(tLin);
        }

        // Solve the quadratic equation.
        float discriminant = (B * B) - 4.0f * A * C;
        if (discriminant < eps)
            return false; // No real roots exist.

        float sqrtDiscriminant = std::sqrt(discriminant);
        float root1 = (-B - sqrtDiscriminant) / (2.0f * A);
        float root2 = (-B + sqrtDiscriminant) / (2.0f * A);

        // Suppose we define a small scene-friendly epsilon
        float sceneEps = eps;

        // Solve for root1, root2
        if (root1 > sceneEps && computeIntersection(root1)) return true;
        if (root2 > sceneEps && computeIntersection(root2)) return true;

        return false;
    }
}
#endif //PATHTRACERKERNELCOMMON_H
