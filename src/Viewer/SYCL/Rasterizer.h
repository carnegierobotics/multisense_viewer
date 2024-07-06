//
// Created by magnus on 7/3/24.
//

#ifndef MULTISENSE_VIEWER_RASTERIZER_H
#define MULTISENSE_VIEWER_RASTERIZER_H

#include <sycl/sycl.hpp>

#include "RasterizerUtils.h"


namespace Rasterizer {

    class Preprocess {
    public:
        Preprocess(glm::vec3 *positions, glm::vec3 *scales, glm::quat *quats, float *opacities, float *shs,
                   uint32_t *tilesTouched, GaussianPoint *pointsBuffer, const PreprocessInfo *info) : m_positions(
                positions), m_scales(scales),
                                                                                                      m_quaternions(
                                                                                                              quats),
                                                                                                      m_opacities(
                                                                                                              opacities),
                                                                                                      m_sphericalHarmonics(
                                                                                                              shs),
                                                                                                      m_numTilesTouched(
                                                                                                              tilesTouched),
                                                                                                      m_points(
                                                                                                              pointsBuffer),
                                                                                                      m_scene(info) {
        }


        void operator()(sycl::id<1> idx) const {
            size_t i = idx.get(0);
            m_numTilesTouched[i] = 0;

            glm::vec3 scale = m_scales[i];
            glm::quat q = m_quaternions[i];
            glm::vec3 position = m_positions[i];

            glm::vec4 posView = m_scene->viewMatrix * glm::vec4(position, 1.0f);
            glm::vec4 posClip = m_scene->projectionMatrix * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;

            //sycl::ext::oneapi::experimental::printf("idx: %d, depth: %f. Pos (%f, %f, %f) \n", i, posView.z, position.x,position.y, position.z);

            if (posView.z >= -0.3f) {
                //sycl::ext::oneapi::experimental::printf("Culled: %d\n", i);
                return;
            }

            float pixPosX = ((posNDC.x + 1.0f) * m_scene->width - 1.0f) * 0.5f;
            float pixPosY = ((posNDC.y + 1.0f) * m_scene->height - 1.0f) * 0.5f;
            auto screenPosPoint = glm::vec3(pixPosX, pixPosY, posNDC.z);

            glm::mat3 cov3D = computeCov3D(scale, q);
            glm::vec3 cov2D = computeCov2D(posView, cov3D, m_scene->viewMatrix, m_scene->params, false);


            // Invert covariance (EWA)
            float determinant = cov2D.x * cov2D.z - (cov2D.y * cov2D.y);
            if (determinant != 0) {
                float invDeterminant = 1 / determinant;

                // Compute extent in screen space (by finding eigenvalues of
                // 2D covariance matrix). Use extent to compute a bounding rectangle
                // of screen-space tiles that this Gaussian overlaps with. Quit if
                // rectangle covers 0 tiles.

                float mid = 0.5f * (cov2D.x + cov2D.z);
                float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - determinant));
                float lambda2 = mid - std::sqrt(std::max(0.1f, mid * mid - determinant));
                float my_radius = ceilf(2.0f * std::sqrt(std::max(lambda1, lambda2)));
                glm::ivec2 rect_min, rect_max;
                getRect(screenPosPoint, my_radius, rect_min, rect_max, m_scene->tileGrid);
                if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
                    return;
                glm::vec3 dir = glm::normalize(position - m_scene->camPos);
                glm::vec3 color =
                        SH_C0 * glm::vec3(m_sphericalHarmonics[idx * m_scene->shDim + 0],
                                          m_sphericalHarmonics[idx * m_scene->shDim + 1],
                                          m_sphericalHarmonics[idx * m_scene->shDim + 2]);
                color += 0.5f;

                auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant,
                                       cov2D.x * invDeterminant);


                m_points[idx].depth = posNDC.z;
                m_points[idx].radius = my_radius;
                m_points[idx].conic = conic;
                m_points[idx].screenPos = screenPosPoint;
                m_points[idx].color = color;
                m_points[idx].opacityBuffer = m_opacities[idx];
                // How many tiles we access
                // rect_min/max are in tile space

                m_numTilesTouched[idx] = static_cast<int>((rect_max.y - rect_min.y) *
                                                          (rect_max.x - rect_min.x));

            }
        }

    private:
        glm::vec3 *m_positions;
        glm::vec3 *m_scales = nullptr;
        glm::quat *m_quaternions = nullptr;
        float *m_opacities = nullptr;
        float *m_sphericalHarmonics = nullptr;
        uint32_t *m_numTilesTouched{0};
        GaussianPoint *m_points{0};
        const PreprocessInfo *m_scene;
    };


    class InclusiveSum {
    public:

        void operator()(sycl::id<1> idx) const {
            for (int i = 0; i < numPoints; ++i) {
                m_pointOffsets[i] = m_numTilesTouched[i];
                if (i > 0) {
                    m_pointOffsets[i] += m_pointOffsets[i - 1];
                }
            }
        }

        InclusiveSum(uint32_t *numTilesTouched, uint32_t *hostPtr, uint32_t pts) : m_numTilesTouched(numTilesTouched),
                                                                                   m_pointOffsets(hostPtr),
                                                                                   numPoints(pts) {}

    private:
        uint32_t *m_numTilesTouched;
        uint32_t *m_pointOffsets;
        uint32_t numPoints = 0;
    };


    class DuplicateGaussians {

        static uint as_uint(const float x) {
            return *(uint *) &x;
        }

        static ushort float_to_half(
                const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
            const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
            const uint e = (b & 0x7F800000) >> 23; // exponent
            const uint m = b &
                           0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
            return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
                   ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                   (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
        }


    public:
        DuplicateGaussians(GaussianPoint *gaussianPoints,
                           uint32_t *pointsOffsets,
                           uint32_t *keys,
                           uint32_t *values,
                           uint32_t numRendered,
                           glm::vec3 tileGrid)
                : m_gaussianPoints(gaussianPoints),
                  m_pointsOffsets(pointsOffsets),
                  m_keys(keys),
                  m_values(values),
                  m_numRendered(numRendered),
                  m_tileGrid(tileGrid) {}

        void operator()(sycl::id<1> idx) const {
            uint32_t i = idx.get(0);
            const auto &gaussian = m_gaussianPoints[i];

            if (gaussian.radius > 0) {
                uint32_t off = (i == 0) ? 0 : m_pointsOffsets[i - 1];
                glm::ivec2 rect_min, rect_max;
                getRect(gaussian.screenPos, gaussian.radius, rect_min, rect_max, m_tileGrid);
                for (int y = rect_min.y; y < rect_max.y; ++y) {
                    for (int x = rect_min.x; x < rect_max.x; ++x) {
                        if (off >= m_numRendered) {
                            break;
                        }
                        uint32_t key = static_cast<uint16_t>(y) * static_cast<uint16_t>(m_tileGrid.x) + x;
                        key <<= 16;
                        uint16_t half = float_to_half(gaussian.depth);
                        key |= half;
                        m_keys[off] = key;
                        m_values[off] = i;
                        ++off;
                    }
                }
            }
        }

    private:
        GaussianPoint *m_gaussianPoints;
        uint32_t *m_pointsOffsets;
        uint32_t *m_keys;
        uint32_t *m_values;
        uint32_t m_numRendered = 0;
        glm::vec3 m_tileGrid;

    };
}

#endif //MULTISENSE_VIEWER_RASTERIZER_H
