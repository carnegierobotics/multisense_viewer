//
// Created by magnus-desktop on 10/22/24.
//

#ifndef RASTERIZERUTILS_H
#define RASTERIZERUTILS_H
#include <Viewer/VkRender/Core/Camera.h>

namespace VkRender{
    struct GaussianPoint
    {
        // Input data
        glm::vec3 position;
        glm::vec3 scale;
        glm::quat rotation;
        float opacity;

        // After preprocessing
        glm::vec3 conic{};
        glm::vec3 screenPos{};
        glm::vec3 color{};
        glm::ivec2 bbMin{};
        glm::ivec2 bbMax{};
        float depth = 0.0f;
        int radius = 0;
        uint32_t tilesTouched = 0;
    };

    // TODO remove and replace with something more intuitive
    struct CameraParams {
        float tanFovX = 0;
        float tanFovY = 0;
        float focalX = 0;
        float focalY = 0;
    };

    struct PreProcessSettings
    {
        glm::vec3 tileGrid{};
        uint32_t numTiles{};
        uint32_t shDim = 0;
        uint32_t shDegree = 1;
        CameraParams params;
        uint32_t numPoints = 0;
    };

    struct PreProcessData {
        Camera camera;
        PreProcessSettings preProcessSettings;
    };

    static glm::mat3 computeCov3D(const glm::vec3 &scale, const glm::quat &q) {

        glm::mat3 S(0.f);
        S[0][0] = scale.x;
        S[1][1] = scale.y;
        S[2][2] = scale.z;
        glm::mat3 R = glm::mat3_cast(q);
        glm::mat3 St = glm::transpose(S);
        glm::mat3 Rt = glm::transpose(R);
        glm::mat3 Sigma = R * S * St * Rt;
        return Sigma;

    }

    static glm::vec3 computeCov2D(const glm::vec4 &pView,
                                  const glm::mat3 &cov3D, const glm::mat4 &viewMat,
                                  const CameraParams &camera,
                                  bool debug = false) {
        glm::vec4 t = pView;
        const float limx = 1.3f * camera.tanFovX;
        const float limy = 1.3f * camera.tanFovY;
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
        t.y = std::min(limy, std::max(-limy, tytz)) * t.z;

        float l = glm::length(pView);
        glm::mat3 J = glm::mat3(camera.focalY / t.z, 0.0f, 0.0f,
                                0.0f, camera.focalY / t.z, 0.0f,
                                -(camera.focalY * t.x) / (t.z * t.z), -(camera.focalY * t.y) / (t.z * t.z), 0.0f);

        auto W = glm::mat3(viewMat);
        glm::mat3 T = J * W;
        glm::mat3 cov = T * cov3D * glm::transpose(T);

        cov[0][0] += 0.3f;
        cov[1][1] += 0.3f;
        return {cov[0][0], cov[1][0], cov[1][1]};
    }

    static void getRect(const glm::vec2 p, int max_radius, glm::ivec2 &rect_min, glm::ivec2 &rect_max,
                        glm::vec3 grid = glm::vec3(0.0f), float BLOCK_X = 16, float BLOCK_Y = 16) {
        rect_min = {
                std::min(static_cast<int>(grid.x), std::max(0, static_cast<int>(((p.x - max_radius) / BLOCK_X)))),
                std::min(static_cast<int>(grid.y), std::max(0, static_cast<int>(((p.y - max_radius) / BLOCK_Y))))
        };
        rect_max = glm::vec2(
                std::min(static_cast<int>(grid.x),
                         std::max(0, static_cast<int>(((p.x + max_radius + BLOCK_X - 1.0f) / BLOCK_X)))),
                std::min(static_cast<int>(grid.y),
                         std::max(0, static_cast<int>(((p.y + max_radius + BLOCK_Y - 1.0f) / BLOCK_Y))))
        );
    }


    static CameraParams getHtanfovxyFocal(float fovy, float h, float w) {
        float htany = std::tan(glm::radians(fovy) / 2.0f);
        float htanx = htany / h * w;
        float focal_y = h / (2.0f * htany);
        float focal_x = focal_y * (w / h); // Ensure aspect ratio is maintained

        return {htanx, htany, focal_x, focal_y};
    }


    }
#endif //RASTERIZERUTILS_H
