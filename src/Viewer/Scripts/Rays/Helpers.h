//
// Created by magnus on 3/11/25.
//

#ifndef HELPERS_H
#define HELPERS_H

namespace RayHelpers {



    /*

            static bool checkCameraPlaneIntersection(
            const glm::vec3 &rayOriginWorld,
            const glm::vec3 &rayDirWorld,
            glm::vec3 &hitPointCam, // out: intersection in camera space
            float &tIntersect, // out: parameter t
            float &contributionScore,
            const glm::mat4& entityTransform,
            const VkRender::PinholeParameters& parameters// out: parameter contributionScore
        ) {
            // 1) Transform to camera space
            glm::mat3 entityRotation = glm::mat3(entityTransform);
            // Camera plane normal in world space
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(entityRotation * glm::vec3(0.0f, 0.0f, -1.0f));

            glm::vec4 cameraPlanePointWorld4 = entityTransform * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

            glm::vec3 cameraPlanePointWorld = glm::vec3(cameraPlanePointWorld4 / cameraPlanePointWorld4.w);
            // Ray-plane intersection

            // Ray-plane intersection calculation
            float denom = glm::dot(cameraPlaneNormalWorld, rayDirWorld);
            if (std::abs(denom) < 1e-6) {
                return false; // Ray is parallel to the plane
            }
            glm::vec3 p0l0 = cameraPlanePointWorld - rayOriginWorld;
            float t = glm::dot(p0l0, cameraPlaneNormalWorld) / denom;
            if (t < 1e-6) {
                return false; // Intersection is behind the ray origin
            }
            glm::vec3 intersectionPoint = rayOriginWorld + t * rayDirWorld;

            float det = glm::determinant(entityTransform);
            if (fabs(det) < 1e-6f) {
                return false;
            };

            glm::mat4 view = glm::inverse(entityTransform);
            glm::vec4 intersectionPointCamera = view * glm::vec4(intersectionPoint, 1.0f);
            glm::vec3 intersectionCamSpace = glm::vec3(intersectionPointCamera) / intersectionPointCamera.w;

            // Sensor plane bounds in camera space
            float halfW = (parameters.width * 0.5f) / parameters.fx;
            float halfH = (parameters.height * 0.5f) / parameters.fy;

            // Check bounds
            if (intersectionCamSpace.x < -halfW || intersectionCamSpace.x > halfW ||
                intersectionCamSpace.y < -halfH || intersectionCamSpace.y > halfH) {
                return false; // Outside sensor bounds
            }

            float cosAngle = std::abs(glm::dot(cameraPlaneNormalWorld, rayDirWorld)); // Ensure positive cosine
            contributionScore = cosAngle; // Higher cosine means closer to perpendicular


            // If we reach here, the ray intersects the camera plane within bounds
            hitPointCam = intersectionPoint; // Intersection point in camera space
            tIntersect = t; // Distance along the ray to the intersection
            return true;
        }


    static bool checkContributionCollision(const glm::vec3& e_o, const glm::vec3& e_d,
                                                 const VkRender::PathTracer::QuadricInputAssembly& quadric,
                                                 glm::vec3& hit,
                                                 glm::vec3& normal,
                                                 float& beta) {
        // Transform the ray into the quadric’s local space.
        glm::mat4 transform = quadric.transform.getTransform();
        float det = glm::determinant(transform);
        if (fabs(det) < 1e-6f)
            return false; // Invalid transform.

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

        float eps = std::numeric_limits<float>::epsilon();

        // Helper lambda: Given a ray parameter t, compute the intersection, normal, and beta.
        auto computeIntersection = [&](float t) -> bool {
            // Compute local hit point.
            glm::vec3 hitLocal = o + d * t;
            glm::vec4 hitW4 = transform * glm::vec4(hitLocal, 1.0f);
            hit = glm::vec3(hitW4) / hitW4.w;

            // Evaluate the beta kernel.
            float R_general = std::sqrt(
                std::fabs(alphaX) * (hitLocal.x * hitLocal.x) / (quadric.a * quadric.a) +
                std::fabs(alphaY) * (hitLocal.y * hitLocal.y) / (quadric.b * quadric.b)
            );
            float r = R_general / quadric.kernelScale;
            if (r > 1.0f)
                r = 1.0f;
            beta = std::pow(1.0f - r * r, 4.0f * std::exp(quadric.b_beta));

            // Compute the local gradient and transform it to world space.
            glm::vec3 gradLocal(
                2.0f * quadric.c * alphaX * hitLocal.x / (quadric.a * quadric.a),
                2.0f * quadric.c * alphaY * hitLocal.y / (quadric.b * quadric.b),
                -1.0f
            );
            glm::mat3 mat = glm::mat3(transform);
            float det2 = glm::determinant(mat);
            if (fabs(det2) < 1e-6f)
                return false;
            glm::mat3 invT = glm::inverseTranspose(mat);
            normal = glm::normalize(invT * gradLocal);
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
        else {
            // Solve the quadratic equation.
            float discriminant = (B * B) - 4.0f * A * C;
            if (discriminant < 0.0f)
                return false; // No real roots exist.

            float sqrtDiscriminant = std::sqrt(discriminant);
            float root1 = (-B - sqrtDiscriminant) / (2.0f * A);
            float root2 = (-B + sqrtDiscriminant) / (2.0f * A);

            // For contribution rays, test both roots.
            if (root1 > eps && computeIntersection(root1))
                return true;
            if (root2 > eps && computeIntersection(root2))
                return true;
            return false;
        }
    }


    // Updated computeWorldHitPoint using the newer logic.
    // Note: This version now assumes that the quadric’s parameters and transform
    // are packaged into a QuadricInputAssembly struct.
    static bool computeWorldHitPoint(const glm::vec3& e_o, const glm::vec3& e_d,
                                     const VkRender::PathTracer::QuadricInputAssembly& quadric,
                                     glm::vec3& hit,
                                     glm::vec3& normal,
                                     float& beta, bool isContributionRay = false) {
        // Transform the ray into the quadric’s local space.
        glm::mat4 transform = quadric.transform.getTransform();
        float det = glm::determinant(transform);
        if (fabs(det) < 1e-6f)
            return false; // Invalid transform

        glm::mat4 localFromWorld = glm::inverse(transform);
        glm::vec4 o4 = localFromWorld * glm::vec4(e_o, 1.0f);
        glm::vec4 d4 = localFromWorld * glm::vec4(e_d, 0.0f);
        glm::vec3 o = glm::vec3(o4);
        glm::vec3 d = glm::vec3(d4);

        // Shorthand for the parameters.
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

        float eps = 1e-5f;
        float tCandidate = std::numeric_limits<float>::max();

        // Solve the quadratic (or linear) equation.
        if (fabs(A) < eps) {
            if (fabs(B) < eps)
                return false; // No solution (degenerate case).
            float tLin = -C / B;
            if (tLin > eps)
                tCandidate = tLin;
            else
                return false;
        }
        else {
            // Calculate the discriminant of the quadratic equation.
            float discriminant = (B * B) - 4.0f * A * C;
            if (discriminant < 0.0f) {
                // No real roots exist.
                return false;
            }

            // Compute the square root of the discriminant.
            float sqrtDiscriminant = std::sqrt(discriminant);

            // Calculate the two possible roots.
            float root1 = (-B - sqrtDiscriminant) / (2.0f * A);
            float root2 = (-B + sqrtDiscriminant) / (2.0f * A);

            // Initialize the minimum positive solution to the maximum possible float value.
            float tMin = std::numeric_limits<float>::max();

            // Choose the smallest positive root greater than eps.
            if (root1 > eps && root1 < tMin) {
                tMin = root1;
            }
            if (root2 > eps && root2 < tMin) {
                tMin = root2;
            }

            // If no valid positive root is found, return false.
            if (tMin == std::numeric_limits<float>::max()) {
                return false;
            }
            // Set the candidate value.
            tCandidate = tMin;
        }

        // Compute the local hit point.
        glm::vec3 hitLocal = o + d * tCandidate;
        // Transform the local hit point back to world space.
        glm::vec4 hitW4 = transform * glm::vec4(hitLocal, 1.0f);
        hit = glm::vec3(hitW4) / hitW4.w;

        // Evaluate the beta kernel.
        float R_general = std::sqrt(
            std::fabs(alphaX) * (hitLocal.x * hitLocal.x) / (quadric.a * quadric.a) +
            std::fabs(alphaY) * (hitLocal.y * hitLocal.y) / (quadric.b * quadric.b)
        );
        float r = R_general / quadric.kernelScale;
        auto betaKernel = [&](float r, float bExp) -> float {
            if (r > 1.0f)
                r = 1.0f;
            return std::pow(1.0f - r * r, 4.0f * std::exp(bExp));
        };
        float bkValue = betaKernel(r, quadric.b_beta);

        // Compute the local normal via the gradient.
        glm::vec3 gradLocal(
            2.0f * quadric.c * alphaX * hitLocal.x / (quadric.a * quadric.a),
            2.0f * quadric.c * alphaY * hitLocal.y / (quadric.b * quadric.b),
            -1.0f
        );

        // Transform the normal to world space.
        glm::mat3 mat = glm::mat3(transform);
        float det2 = glm::determinant(mat);
        if (fabs(det2) < 1e-6f)
            return false;
        glm::mat3 invT = glm::inverseTranspose(mat);
        glm::vec3 normalW = glm::normalize(invT * gradLocal);
        if (glm::dot(normalW, -e_d) < 0.0f)
            normalW = -normalW;

        normal = normalW;
        beta = bkValue;

        if (bkValue < quadric.threshold)
            return false; // The hit does not meet the kernel threshold.


        return true;
    }

    static bool computeWorldHitPoint2(const glm::vec3& e_o, const glm::vec3& e_d,
                                      const glm::vec3& g_c,
                                      float a, float b, float c,
                                      float tx, float ty,
                                      glm::vec3& normal, glm::vec3& hit) {
        float alpha_x = tanh(tx);
        float alpha_y = tanh(ty);
        // Since R_w2g is the identity, the local coordinates are simply:
        //   local direction: e_d,l,gt = e_d
        //   local origin:    e_o,l,gt = e_o - g_c
        glm::vec3 e_d_l = e_d;
        glm::vec3 e_o_l = e_o - g_c;

        // Compute the intersection parameters.
        // Note: In your provided printout, A turns out to be zero because the x,y components of e_d are zero.
        float A = c * (alpha_x * (e_d_l.x * e_d_l.x) / (a * a) +
            alpha_y * (e_d_l.y * e_d_l.y) / (b * b));

        float B = c * (2.0f * alpha_x * (e_o_l.x * e_d_l.x) / (a * a) +
                2.0f * alpha_y * (e_o_l.y * e_d_l.y) / (b * b))
            - e_d_l.z;

        float C = c * (alpha_x * (e_o_l.x * e_o_l.x) / (a * a) +
                alpha_y * (e_o_l.y * e_o_l.y) / (b * b))
            - e_o_l.z;

        // In the provided code, the quadratic term A is zero (or negligible)
        // so the intersection parameter is computed as:
        float disc = (B * B) - 4 * A * C;

        // Solve for the smallest positive t (g_tmin)
        float g_tmin = 0.0f;

        if (std::fabs(A) <= std::numeric_limits<float>::epsilon()) {
            g_tmin = -C / B;
        }
        else {
            g_tmin = (-B + std::sqrt(disc)) / (2.0f * A);
        }

        // Compute the local hit point: g_hit,l,gt = e_d,l,gt * t + e_o,l,gt
        glm::vec3 g_hit_l = e_d_l * g_tmin + e_o_l;

        // World hit point is then given by (R_g2w * local_point + g_c).
        // Since R_g2w is the identity, we simply add g_c.
        glm::vec3 g_hit = g_hit_l + g_c;

        // Compute the local normal via the gradient.
        glm::vec3 gradLocal(
            2.0f * c * alpha_x * g_hit_l.x / (a * a),
            2.0f * c * alpha_y * g_hit_l.y / (b * b),
            -1.0f
        );

        if (glm::dot(gradLocal, -e_d) < 0.0f)
            gradLocal = -gradLocal;

        normal = gradLocal;
        hit = g_hit;
        return true;
    }
    */
}

#endif //HELPERS_H
