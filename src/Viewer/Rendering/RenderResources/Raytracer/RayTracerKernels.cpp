//
// Created by magnus on 11/27/24.
//

#include "RayTracerKernels.h"


namespace VkRender::RT::Kernels {

    inline bool rayTriangleIntersect(
            const glm::vec3& orig,
            const glm::vec3& dir,
            const glm::vec3& v0,
            const glm::vec3& v1,
            const glm::vec3& v2,
            float& t, float& u, float& v) {

        const float EPSILON = 1e-8f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(dir, edge2);
        float a = glm::dot(edge1, h);

        if (fabs(a) < EPSILON)
            return false;

        float f = 1.0f / a;
        glm::vec3 s = orig - v0;
        u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return false;

        glm::vec3 q = glm::cross(s, edge1);
        v = f * glm::dot(dir, q);
        if (v < 0.0f || u + v > 1.0f)
            return false;

        t = f * glm::dot(edge2, q);
        return t > EPSILON;
    }

    void rayTracingKernel(
            sycl::handler& cgh,
            const sycl::accessor<char, 1, sycl::access::mode::write>& imageAcc,
            const sycl::accessor<Vertex, 1, sycl::access::mode::read>& vertexAcc,
            const sycl::accessor<uint32_t, 1, sycl::access::mode::read>& indexAcc,
            const sycl::accessor<MeshInfo, 1, sycl::access::mode::read>& meshInfoAcc,
            size_t numMeshes,
            uint32_t width,
            uint32_t height) {

        cgh.parallel_for<class RayTracingKernel>(
                sycl::range<2>(height, width), [=](sycl::item<2> item) {
                    uint32_t x = item.get_id(1);
                    uint32_t y = item.get_id(0);

                    // Normalize pixel coordinates to [-1,1]
                    float u = (2.0f * x) / width - 1.0f;
                    float v = (2.0f * y) / height - 1.0f;

                    // Camera setup
                    glm::vec3 rayOrigin = {0.0f, 0.0f, 0.0f};
                    glm::vec3 rayDir = glm::normalize(glm::vec3(u, v, -1.0f));

                    float t_min = 0.001f;
                    float t_max = 1000.0f;
                    float t_closest = t_max;
                    glm::vec3 hitColor = {0.0f, 0.0f, 0.0f};
                    bool hit = false;

                    // Ray-triangle intersection
                    for (size_t m = 0; m < numMeshes; ++m) {
                        const MeshInfo& meshInfo = meshInfoAcc[m];
                        uint32_t indexOffset = meshInfo.indexOffset;
                        uint32_t indexCount = meshInfo.indexCount;

                        // For simplicity, we are not applying transforms here
                        // To apply transforms, you would need to implement matrix multiplication

                        for (uint32_t i = indexOffset; i < indexOffset + indexCount; i += 3) {
                            uint32_t idx0 = indexAcc[i];
                            uint32_t idx1 = indexAcc[i + 1];
                            uint32_t idx2 = indexAcc[i + 2];

                            const Vertex& v0 = vertexAcc[idx0];
                            const Vertex& v1 = vertexAcc[idx1];
                            const Vertex& v2 = vertexAcc[idx2];

                            float t, u_param, v_param;
                            if (rayTriangleIntersect(rayOrigin, rayDir, v0.pos, v1.pos, v2.pos, t, u_param, v_param)) {
                                if (t < t_closest && t > t_min) {
                                    t_closest = t;
                                    hitColor = glm::abs(glm::cross(v1.pos - v0.pos, v2.pos - v0.pos));
                                    hit = true;
                                }
                            }
                        }
                    }

                    uint8_t r, g, b, a = 255;

                    if (hit) {
                        // Simple shading using the hit normal as color
                        r = static_cast<uint8_t>(glm::clamp(hitColor.x * 255.0f, 0.0f, 255.0f));
                        g = static_cast<uint8_t>(glm::clamp(hitColor.y * 255.0f, 0.0f, 255.0f));
                        b = static_cast<uint8_t>(glm::clamp(hitColor.z * 255.0f, 0.0f, 255.0f));
                    } else {
                        // Background color
                        r = g = b = 0;
                    }

                    uint32_t index = (y * width + x) * 4;
                    imageAcc[index + 0] = r;
                    imageAcc[index + 1] = g;
                    imageAcc[index + 2] = b;
                    imageAcc[index + 3] = a;
                });
    }

}