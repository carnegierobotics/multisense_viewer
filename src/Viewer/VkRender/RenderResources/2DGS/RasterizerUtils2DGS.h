//
// Created by magnus-desktop on 10/22/24.
//

#ifndef RASTERIZERUTILS2D_H
#define RASTERIZERUTILS2D_H
#include <Viewer/VkRender/Core/Camera.h>

namespace VkRender::Rasterizer2DUtils {
    struct GaussianPoint {
        // Input data
        glm::vec3 position;
        glm::vec3 scale;
        glm::quat rotation;
        float opacity;
        glm::vec3 color{};
        std::array<std::array<float, 15>, 3> shCoeffs;
        // After preprocessing
        glm::vec3 conic{};
        glm::vec3 screenPos{};
        glm::vec3 computedColor{};
        glm::ivec2 bbMin{};
        glm::ivec2 bbMax{};
        float depth = 0.0f;
        int radius = 0;
        uint32_t tilesTouched = 0;
        glm::mat3 T;
    };

    // TODO remove and replace with something more intuitive
    struct CameraParams {
        float tanFovX = 0;
        float tanFovY = 0;
        float focalX = 0;
        float focalY = 0;
    };

    struct PreProcessSettings {
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


    static bool compute_aabb(
            glm::mat3 T,
            float cutoff,
            glm::vec2& point_image,
            glm::vec2& extent
    ) {
        glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
        float d = glm::dot(t, T[2] * T[2]);
        if (d == 0.0) return false;
        glm::vec3 f = (1 / d) * t;

        glm::vec2 p = glm::vec2(
                glm::dot(f, T[0] * T[2]),
                glm::dot(f, T[1] * T[2])
        );

        glm::vec2 h0 = p * p -
                       glm::vec2(
                               glm::dot(f, T[0] * T[0]),
                               glm::dot(f, T[1] * T[1])
                       );

        glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
        point_image = {p.x, p.y};
        extent = {h.x, h.y};
        return true;
    }

    static void getRect(const glm::vec2 p, int max_radius, glm::ivec2& rect_min, glm::ivec2& rect_max,
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

    static glm::vec3 computeColorFromSH(const std::array<std::array<float, 15>, 3>& sh_coeffs,
                                        const glm::vec3& color_dc,
                                        const glm::vec3& pos,
                                        const glm::vec3& camPos) {
        // SH constants
        const float SH_C0 = 0.28209479177387814f; // sqrt(1/(4*pi))
        const float SH_C1 = 0.4886025119029199f; // sqrt(3/(4*pi))
        const float SH_C2[] = {
            1.0925484305920792f, // sqrt(15/(4*pi))
            -1.0925484305920792f, // -sqrt(15/(4*pi))
            0.31539156525252005f, // sqrt(5/(16*pi))
            -1.0925484305920792f, // -sqrt(15/(4*pi))
            0.5462742152960396f // sqrt(15/(16*pi))
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

        // Start with the DC component
        glm::vec3 result = SH_C0 * color_dc;

        // Check if SH coefficients are available
        if (!sh_coeffs.empty() && !sh_coeffs[0].empty()) {
            int num_coeffs_per_channel = static_cast<int>(sh_coeffs[0].size());
            int deg = static_cast<int>(std::sqrt(num_coeffs_per_channel + 1)) - 1;

            // Compute direction vector
            glm::vec3 dir = glm::normalize(pos - camPos);
            float x = dir.x;
            float y = dir.y;
            float z = dir.z;

            glm::vec3 sh_sum(0.0f);

            // Degree 1 SH
            if (deg >= 1 && num_coeffs_per_channel >= 3) {
                for (int c = 0; c < 3; ++c) {
                    float sh_l1_mneg1 = sh_coeffs[c][0];
                    float sh_l1_m0 = sh_coeffs[c][1];
                    float sh_l1_m1 = sh_coeffs[c][2];

                    sh_sum[c] += -SH_C1 * y * sh_l1_mneg1;
                    sh_sum[c] += SH_C1 * z * sh_l1_m0;
                    sh_sum[c] += -SH_C1 * x * sh_l1_m1;
                }
            }

            // Degree 2 SH
            if (deg >= 2 && num_coeffs_per_channel >= 8) {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;

                for (int c = 0; c < 3; ++c) {
                    float sh_l2_mneg2 = sh_coeffs[c][3];
                    float sh_l2_mneg1 = sh_coeffs[c][4];
                    float sh_l2_m0 = sh_coeffs[c][5];
                    float sh_l2_m1 = sh_coeffs[c][6];
                    float sh_l2_m2 = sh_coeffs[c][7];

                    sh_sum[c] += SH_C2[0] * xy * sh_l2_mneg2;
                    sh_sum[c] += SH_C2[1] * yz * sh_l2_mneg1;
                    sh_sum[c] += SH_C2[2] * (2.0f * zz - xx - yy) * sh_l2_m0;
                    sh_sum[c] += SH_C2[3] * xz * sh_l2_m1;
                    sh_sum[c] += SH_C2[4] * (xx - yy) * sh_l2_m2;
                }
            }

            // Degree 3 SH
            if (deg >= 3 && num_coeffs_per_channel >= 15) {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;

                for (int c = 0; c < 3; ++c) {
                    float sh_l3_mneg3 = sh_coeffs[c][8];
                    float sh_l3_mneg2 = sh_coeffs[c][9];
                    float sh_l3_mneg1 = sh_coeffs[c][10];
                    float sh_l3_m0 = sh_coeffs[c][11];
                    float sh_l3_m1 = sh_coeffs[c][12];
                    float sh_l3_m2 = sh_coeffs[c][13];
                    float sh_l3_m3 = sh_coeffs[c][14];

                    sh_sum[c] += SH_C3[0] * y * (3.0f * xx - yy) * sh_l3_mneg3;
                    sh_sum[c] += SH_C3[1] * xy * z * sh_l3_mneg2;
                    sh_sum[c] += SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_l3_mneg1;
                    sh_sum[c] += SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_l3_m0;
                    sh_sum[c] += SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_l3_m1;
                    sh_sum[c] += SH_C3[5] * z * (xx - yy) * sh_l3_m2;
                    sh_sum[c] += SH_C3[6] * x * (xx - 3.0f * yy) * sh_l3_m3;
                }
            }

            // Add the SH contributions
            result += sh_sum;
        }

        // Offset and clamp the result
        result += 0.5f;
        result = glm::clamp(result, 0.0f, 1.0f);

        return result;
    }

    static glm::vec3
    computeColorFromSH(uint32_t idx, glm::vec3 pos, glm::vec3 camPos, uint32_t deg, uint32_t maxCoeffs, float* shs) {
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


        float* sh = &shs[idx * 3];
        glm::vec3 result = SH_C0 * glm::vec3(sh[0], sh[1], sh[2]);

        if (deg > 0) {
            glm::vec3 dir = glm::normalize(pos - camPos);

            float x = dir.x;
            float y = dir.y;
            float z = dir.z;
            result = result - SH_C1 * y * glm::vec3(sh[3], sh[4], sh[5]) + SH_C1 * z * glm::vec3(sh[6], sh[7], sh[8]) -
                SH_C1 * x * glm::vec3(sh[9], sh[10], sh[11]);

            if (deg > 1) {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                result = result +
                    SH_C2[0] * xy * sh[4] +
                    SH_C2[1] * yz * sh[5] +
                    SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                    SH_C2[3] * xz * sh[7] +
                    SH_C2[4] * (xx - yy) * sh[8];

                if (deg > 2) {
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

        return glm::max(glm::min(result, 1.0f), 0.0f);
    }
}
#endif //RASTERIZERUTILS_H
