//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACERKERNELS_H
#define MULTISENSE_VIEWER_RAYTRACERKERNELS_H

#include <sycl/sycl.hpp>
#include <Viewer/Rendering/Components/CameraComponent.h>
#include <Viewer/Rendering/Components/TransformComponent.h>

#include "Definitions.h"


namespace VkRender::RT::Kernels {
    static bool rayTriangleIntersect(
            glm::vec3 &ray_origin,
            glm::vec3 &ray_dir,
            const glm::vec3 &a,
            const glm::vec3 &b,
            const glm::vec3 &c,
            glm::vec3 &out_intersection) {
        float epsilon = 1e-7f;

        glm::vec3 edge1 = b - a;
        glm::vec3 edge2 = c - a;
        glm::vec3 h = glm::cross(ray_dir, edge2);
        float det = glm::dot(edge1, h);

        // If det is near zero, ray is parallel to triangle
        if (det > -epsilon && det < epsilon)
            return false;

        float inv_det = 1.0f / det;
        glm::vec3 s = ray_origin - a;
        float u = inv_det * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return false;

        glm::vec3 q = glm::cross(s, edge1);
        float v = inv_det * glm::dot(ray_dir, q);
        if (v < 0.0f || (u + v) > 1.0f)
            return false;

        float t = inv_det * glm::dot(edge2, q);
        if (t > epsilon) {
            out_intersection = ray_origin + ray_dir * t;
            return true;
        }

        return false;
    }

    static bool
    intersectPlane(const glm::vec3 &n, const glm::vec3 &p0, const glm::vec3 &l0, const glm::vec3 &l, float &t) {
        // Assuming vectors are normalized
        float denom = glm::dot(n, l);
        if (std::abs(denom) > 1e-6) { // Avoid near-parallel cases
            glm::vec3 p0l0 = p0 - l0;
            t = glm::dot(p0l0, n) / denom;
            return (t >= 0); // Only return true for intersections in front of the ray
        }
        return false;
    }

    class RenderKernel {
    public:
        RenderKernel(GPUData gpuData, uint32_t width, uint32_t height, uint32_t size, TransformComponent cameraPose,
                     PinholeCamera camera)
                : m_gpuData(gpuData), m_width(width), m_height(height), m_size(size), m_cameraTransform(cameraPose),
                  m_camera(camera) {
        }

        void operator()(sycl::nd_item<2> item) const {
            uint32_t x = item.get_global_id(1);
            uint32_t y = item.get_global_id(0);

            uint32_t pixelIndex = (y * m_width + x) * 4;
            if (pixelIndex >= m_size)
                return;

            float fx = m_camera.m_fx;
            float fy = m_camera.m_fy;
            float cx = m_camera.m_cx;
            float cy = m_camera.m_cy;
            float Z_plane = -1.0f;

            auto mapPixelTo3D = [&](float u, float v) {
                float X = -(u - cx) * Z_plane / fx;
                float Y = -(v - cy) * Z_plane / fy; // Notice the minus sign before (v - cy)
                float Z = Z_plane;
                return glm::vec3(X, Y, Z);
            };
            glm::vec3 direction = mapPixelTo3D(static_cast<float>(x), static_cast<float>(y));
            glm::vec3 rayOrigin = m_cameraTransform.translation;
            glm::vec3 worldRayDir = glm::normalize(glm::mat3(m_cameraTransform.rotation) * glm::normalize(direction));
            // Primary ray intersection
            float closest_t = FLT_MAX;
            bool hit = false;
            size_t hitIdx = 0; // Index of the Gaussian that is hit

            for (size_t idx = 0; idx < m_gpuData.numGaussians; ++idx) {
                glm::vec3 &pos = m_gpuData.gaussianInputAssembly[idx].position;
                glm::vec3 &normal = m_gpuData.gaussianInputAssembly[idx].normal;
                glm::vec2 &scale = m_gpuData.gaussianInputAssembly[idx].scale;
                float intensity = m_gpuData.gaussianInputAssembly[idx].intensity;

                float t = FLT_MAX;
                if (intersectPlane(normal, pos, rayOrigin, worldRayDir, t) && t < closest_t) {
                    closest_t = t;
                    hit = true;
                    hitIdx = idx;

                }
            }


            float pixelIntensity = 0.0f;

            if (hit) {
                // Retrieve hit Gaussian parameters
                glm::vec3 &pos = m_gpuData.gaussianInputAssembly[hitIdx].position;
                glm::vec3 &normal = m_gpuData.gaussianInputAssembly[hitIdx].normal;
                glm::vec2 &scale = m_gpuData.gaussianInputAssembly[hitIdx].scale;
                float intensity = m_gpuData.gaussianInputAssembly[hitIdx].intensity;

                // Compute intersection point
                glm::vec3 intersectionPoint = rayOrigin + closest_t * worldRayDir;

                glm::vec3 u = pos - intersectionPoint;
                glm::vec3 v = glm::cross(u, normal);

                float alpha = glm::length(u);
                float beta = glm::length(v);

                if ((alpha < scale.x) * 3 && (beta * 3) < scale.y) {
                    // Evaluate the Gaussian:
                    // G(alpha, beta) = intensity * exp(-0.5 * ((alpha^2 / scale.x^2) + (beta^2 / scale.y^2)))
                    float gaussVal = intensity * std::exp(-0.5f * ((alpha * alpha) / (scale.x * scale.x)
                                                                   + (beta * beta) / (scale.y * scale.y)));

                    pixelIntensity = gaussVal * 255;

                }
            }


            if (hit) {
                m_gpuData.imageMemory[pixelIndex + 0] = pixelIntensity; // R
                m_gpuData.imageMemory[pixelIndex + 1] = pixelIntensity; // G
                m_gpuData.imageMemory[pixelIndex + 2] = pixelIntensity; // B
                m_gpuData.imageMemory[pixelIndex + 3] = 255; // A}
            } else {
                // No intersection: clear pixel to some background color.
                m_gpuData.imageMemory[pixelIndex + 0] = 0; // R
                m_gpuData.imageMemory[pixelIndex + 1] = 0; // G
                m_gpuData.imageMemory[pixelIndex + 2] = 0; // B
                m_gpuData.imageMemory[pixelIndex + 3] = 255; // A
            }
        }

    private:
        GPUData m_gpuData;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_size;

        TransformComponent m_cameraTransform;
        PinholeCamera m_camera;
    };
}


#endif //MULTISENSE_VIEWER_RAYTRACERKERNELS_H
