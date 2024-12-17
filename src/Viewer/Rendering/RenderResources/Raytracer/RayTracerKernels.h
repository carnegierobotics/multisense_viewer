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
        glm::vec3& ray_origin,
        glm::vec3& ray_dir,
        const glm::vec3& a,
        const glm::vec3& b,
        const glm::vec3& c,
        glm::vec3& out_intersection) {
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
        glm::vec3 worldRayDir = glm::mat3(m_cameraTransform.rotation) * glm::normalize(direction);
        // Primary ray intersection
        float closest_t = FLT_MAX;
        bool hit = false;
        glm::vec3 hitPoint(0.0f);
        size_t triangleCount = m_gpuData.numIndices / 3;
        for (size_t tri = 0; tri < triangleCount; ++tri) {
            uint32_t i0 = m_gpuData.indices[tri * 3 + 0];
            uint32_t i1 = m_gpuData.indices[tri * 3 + 1];
            uint32_t i2 = m_gpuData.indices[tri * 3 + 2];
            const glm::vec3& a = m_gpuData.vertices[i0].position;
            const glm::vec3& b = m_gpuData.vertices[i1].position;
            const glm::vec3& c = m_gpuData.vertices[i2].position;
            glm::vec3 intersectionPoint;
            if (rayTriangleIntersect(rayOrigin, worldRayDir, a, b, c, intersectionPoint)) {
                float dist = glm::distance(rayOrigin, intersectionPoint);
                if (dist < closest_t) {
                    closest_t = dist;
                    hit = true;
                    hitPoint = intersectionPoint;
                }
            }
        }

        if (hit) {
            // If we hit a triangle, color the pixel accordingly.
            // For a simple visualization, let’s map intersection distance to grayscale.
            uint8_t intensity = static_cast<uint8_t>(glm::clamp(closest_t, 0.0f, 255.0f));
            m_gpuData.imageMemory[pixelIndex + 0] = 255; // R
            m_gpuData.imageMemory[pixelIndex + 1] = 255; // G
            m_gpuData.imageMemory[pixelIndex + 2] = 255; // B
            m_gpuData.imageMemory[pixelIndex + 3] = 255; // A
        }
        else {
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
