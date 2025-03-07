//
// Created by magnus on 3/7/25.
//

#ifndef PATHTRACERKERNELCOMMON_H
#define PATHTRACERKERNELCOMMON_H
namespace VkRender::PathTracer{


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
                                 float &beta)  {
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
        if (tLin > eps)
            tCandidate = tLin;
        else
            return false;
    } else {
        float disc = (B * B) - 4.0f * A * C;
        if (disc < 0.0f)
            return false; // No real roots

        float sqrtDisc = std::sqrt(disc);
        float t1 = (-B - sqrtDisc) / (2.0f * A);
        float t2 = (-B + sqrtDisc) / (2.0f * A);
        float tMin = std::numeric_limits<float>::max();
        if (t1 > eps && t1 < tMin)
            tMin = t1;
        if (t2 > eps && t2 < tMin)
            tMin = t2;
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
    if (bkValue < quadric.threshold)
        return false; // Not within threshold

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
    if (glm::dot(normalW, dir) > 0.0f)
        normalW = -normalW;

    hitNormal = normalW;
    beta = bkValue;
    return true;
}



}
#endif //PATHTRACERKERNELCOMMON_H
