//
// Created by magnus on 7/3/24.
//

#ifndef MULTISENSE_VIEWER_RASTERIZERUTILS_H
#define MULTISENSE_VIEWER_RASTERIZERUTILS_H

#include <glm/glm.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdint>

namespace Rasterizer {


    struct GaussianPoint {
        float opacity{};
        glm::vec3 color{};
        glm::vec3 conic{};
        glm::vec3 screenPos{};
        uint32_t tileArea = 0;
        uint32_t tileInclusiveSum = 0;
        glm::ivec2 bbMin, bbMax;
        float depth = 0.0f;
        int radius = 0;
    };

    struct CameraParams {
        float tanFovX = 0;
        float tanFovY = 0;
        float focalX = 0;
        float focalY = 0;
    };

    struct PreprocessInfo{
        glm::mat4 viewMatrix{};
        glm::mat4 projectionMatrix{};
        uint32_t width = 0, height = 0;
        CameraParams params{};
        glm::vec3 tileGrid{};
        glm::vec3 camPos{};
        uint32_t shDim = 0;
        uint32_t shDegree = 1;
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

    static glm::vec3 computeColorFromSH(uint32_t idx, glm::vec3 pos, glm::vec3 camPos, uint32_t deg, uint32_t maxCoeffs, float* shs){
        const float SH_C0 = 0.28209479177387814f;
        const float SH_C1 = 0.4886025119029199f;

        const float SH_C2[] = {
                1.0925484305920792f,
                -1.0925484305920792f,
                0.31539156525252005f,
                -1.0925484305920792f,
                0.5462742152960396f
        };
         const float SH_C3[] = {
                -0.5900435899266435f,
                2.890611442640554f,
                -0.4570457994644658f,
                0.3731763325901154f,
                -0.4570457994644658f,
                1.445305721320277f,
                -0.5900435899266435f
        };


        glm::vec3 dir = glm::normalize(pos - camPos);
        glm::vec3* sh = ((glm::vec3*)shs) + idx * maxCoeffs;
        glm::vec3 result = SH_C0 * sh[0];

        if (deg > 0) {
            float x = dir.x;
            float y = dir.y;
            float z = dir.z;
            result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

            if (deg > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                result = result +
                         SH_C2[0] * xy * sh[4] +
                         SH_C2[1] * yz * sh[5] +
                         SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                         SH_C2[3] * xz * sh[7] +
                         SH_C2[4] * (xx - yy) * sh[8];

                if (deg > 2)
                {
                    result = result +
                             SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                             SH_C3[1] * xy * z * sh[10] +
                             SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                             SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                             SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                             SH_C3[5] * z * (xx - yy) * sh[14] +
                             SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
                }
            }
        }
        result += 0.5f;

        return glm::max(result, 0.0f);
    }
}

#endif //MULTISENSE_VIEWER_RASTERIZERUTILS_H
