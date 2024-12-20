//
// Created by magnus-desktop on 10/22/24.
//

#ifndef RASTERIZER3D_H
#define RASTERIZER3D_H

#include <cmath>
#include <sycl/sycl.hpp>

#include "RasterizerUtils3DGS.h"


namespace VkRender::Rasterizer {
    class Preprocess {
    public:
        Preprocess(GaussianPoint *points, PreProcessData *preProcessData)
                : m_points(points),
                  m_camera(preProcessData->camera),
                  m_settings(preProcessData->preProcessSettings), m_width(preProcessData->width), m_height(preProcessData->height) {
        }


        void operator()(sycl::id<1> idx) const {
            uint32_t i = idx.get(0);

            m_points[i].tilesTouched = 0;
            m_points[i].radius = 0;
            m_points[i].depth = 0;
            m_points[i].screenPos = glm::vec3(0);

            glm::vec3 scale = m_points[i].scale;
            glm::quat q = m_points[i].rotation;
            glm::vec3 position = m_points[i].position;
            glm::vec4 posView = m_camera.matrices.view * glm::vec4(position, 1.0f);
            glm::vec4 posClip = m_camera.matrices.projection * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;
            int x = position.x, y = position.y, z = position.z;
            //int zview = posView.z;
            if (posView.z >= -0.32f) {
                return;
            }

            float pixPosX = ((posNDC.x + 1.0f) * static_cast<float>(m_width) - 1.0f) * 0.5f;
            float pixPosY = ((posNDC.y + 1.0f) * static_cast<float>(m_height) - 1.0f) * 0.5f;
            auto screenPosPoint = glm::vec3(pixPosX, pixPosY, posNDC.z);

            glm::mat3 cov3D = computeCov3D(scale, q);
            glm::vec3 cov2D = computeCov2D(posView, cov3D, m_camera.matrices.view, m_settings.params, false);

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
                float my_radius = ceilf(3.0f * std::sqrt(std::max(lambda1, lambda2)));
                glm::ivec2 rect_min, rect_max;

                getRect(screenPosPoint, static_cast<int>(my_radius), rect_min, rect_max, m_settings.tileGrid);

                if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
                    return;

                std::array<std::array<float, 15>, 3> sh = m_points[i].shCoeffs;
                glm::vec3 color = m_points[i].color;
                glm::vec3 cameraPos = m_camera.matrices.position;
                glm::vec3 gaussianPos = m_points[i].position;

                glm::vec3 result = computeColorFromSH(m_points[i].shCoeffs, color, gaussianPos, cameraPos);
                //glm::vec3 result = color;
                m_points[i].computedColor = result;

                auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant,
                                       cov2D.x * invDeterminant);

                m_points[i].depth = posNDC.z;
                m_points[i].radius = my_radius;
                m_points[i].conic = conic;
                m_points[i].screenPos = screenPosPoint;

                // How many tiles we access
                // rect_min/max are in tile space
                auto numTilesTouched = static_cast<uint32_t>((rect_max.y - rect_min.y) *
                                                             (rect_max.x - rect_min.x));
                m_points[i].tilesTouched = numTilesTouched;
            }
        }

        GaussianPoint *m_points;
        PinholeCamera m_camera;
        PreProcessSettings m_settings;
        uint32_t m_width, m_height;
    };

    class InclusiveSum {
    public:
        InclusiveSum(GaussianPoint *points, uint32_t *pointOffsets, uint32_t numPoints,
                     sycl::local_accessor<uint32_t> localMem, uint32_t *groupTotals)
                : m_points(points), m_output(pointOffsets), m_numPoints(numPoints), m_localMem(localMem) , m_groupTotals(groupTotals) {
        }

        void operator()(sycl::nd_item<1> item) const {
            size_t localIdx = item.get_local_id(0);
            size_t globalIdx = item.get_global_id(0);
            size_t localRange = item.get_local_range(0);
            size_t groupLinearId = item.get_group_linear_id();
            auto subGroup = item.get_sub_group();
            subGroup.get_group_range();

            if (globalIdx < m_numPoints) {
                m_localMem[localIdx] = m_points[globalIdx].tilesTouched;
            } else {
                m_localMem[localIdx] = 0;
            }
            sycl::group_barrier(item.get_group());

            // Perform the inclusive scan
            for (size_t offset = 1; offset < localRange; offset *= 2) {
                uint32_t val = 0;
                if (localIdx >= offset)
                    val = m_localMem[localIdx - offset];
                sycl::group_barrier(item.get_group());
                m_localMem[localIdx] += val;
                sycl::group_barrier(item.get_group());
            }
            sycl::group_barrier(item.get_group());

            if (globalIdx < m_numPoints) {
                m_output[globalIdx] = m_localMem[localIdx];
            }

            if (localIdx == localRange - 1) {
                m_groupTotals[item.get_group_linear_id()] = m_localMem[localIdx];
            }

        }

        sycl::local_accessor<uint32_t> m_localMem;
        GaussianPoint *m_points;
        uint32_t *m_output;
        uint32_t *m_groupTotals;
        uint32_t m_numPoints;
    };

    class InclusiveSum2 {
    public:
        InclusiveSum2(GaussianPoint *points, uint32_t *pointOffsets, uint32_t numPoints,
                      sycl::local_accessor<uint32_t> localMem) : m_points(points),
                                                                 m_output(pointOffsets), m_numPoints(numPoints),
                                                                 m_localMem(localMem) {

        }


        void operator()(sycl::nd_item<1> item) const {
            size_t globalIdx = item.get_global_id(0);
            size_t localIdx = item.get_local_id(0);
            size_t localRange = item.get_local_range(0);
            if (globalIdx == 0){
                m_output[0] = m_points[0].tilesTouched;
                for (size_t i = 1; i < m_numPoints; ++i){
                    m_output[i] =  m_output[i - 1] + m_points[i].tilesTouched;
                }
            }

        }

        sycl::local_accessor<uint32_t> m_localMem;

        GaussianPoint *m_points;
        uint32_t *m_output;
        uint32_t m_numPoints;
    };


    class DuplicateGaussians {
        static uint as_uint(const float x) {
            return *(uint *) &x;
        }

        static ushort float_to_half(
                const float x) {
            // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
            const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
            const uint e = (b & 0x7F800000) >> 23; // exponent
            const uint m = b &
                           0x007FFFFF;
            // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
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
                           glm::vec3 &tileGrid)
                : m_gaussianPoints(gaussianPoints),
                  m_pointsOffsets(pointsOffsets),
                  m_keys(keys),
                  m_values(values),
                  m_numRendered(numRendered),
                  m_tileGrid(tileGrid) {
        }

    private:
        GaussianPoint *m_gaussianPoints;
        uint32_t *m_pointsOffsets;
        uint32_t *m_keys;
        uint32_t *m_values;
        uint32_t m_numRendered;
        glm::vec3 m_tileGrid;

    public:
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
                        auto depth = static_cast<uint16_t>(gaussian.depth * 65535);
                        key |= depth;
                        m_keys[off] = key;
                        m_values[off] = i;
                        ++off;
                    }
                }
            }
        }
    };

    class IdentifyTileRanges {
    public:
        IdentifyTileRanges(glm::ivec2 *ranges, uint32_t *keys, size_t size)
                : m_ranges(ranges), m_keys(keys), m_size(size) {
        }

    private:
        glm::ivec2 *m_ranges;
        uint32_t *m_keys;
        uint32_t m_size;

    public:
        void operator()(sycl::id<1> id) const {
            int idx = id[0];

            if (idx > m_size)
                return;

            uint32_t key = m_keys[idx];
            uint16_t currentTile = key >> 16;
            if (idx == 0) {
                m_ranges[currentTile].x = 0;
            } else {
                uint16_t prevTile = m_keys[idx - 1] >> 16;
                if (currentTile != prevTile) {
                    m_ranges[prevTile].y = idx;
                    m_ranges[currentTile].x = idx;
                }
            }

            if (idx == m_size - 1) {
                m_ranges[currentTile].y = static_cast<int>(m_size);
            }
        };
    };

    class RasterizeGaussians {
    public:
        RasterizeGaussians(glm::ivec2 *ranges, uint32_t *values, GaussianPoint *gaussianPoints,
                           uint8_t *imageBuffer, size_t size, uint32_t m_imageWidth, uint32_t imageHeight)
                : m_ranges(ranges), m_values(values), m_gaussianPoints(gaussianPoints),
                  m_imageBuffer(imageBuffer), m_size(size),
                  m_imageWidth(m_imageWidth), m_imageHeight(imageHeight) {
        }

    private:
        uint32_t *m_values;
        glm::ivec2 *m_ranges;
        GaussianPoint *m_gaussianPoints;
        uint8_t *m_imageBuffer;

        uint32_t m_size;
        uint32_t m_imageWidth;
        uint32_t m_imageHeight;

    public:
        void operator()(sycl::nd_item<2> item) const {
            uint32_t gridDim = item.get_group_range(0);
            uint32_t global_id_x = item.get_global_id(1); // x-coordinate (column)
            uint32_t global_id_y = item.get_global_id(0); // y-coordinate (row)
            uint32_t group_id_x = item.get_group(1);
            uint32_t group_id_y = item.get_group(0);
            uint32_t group_id_x_max = item.get_group_range(1);

            // Calculate the global pixel row and column
            uint32_t row = global_id_y; // Calculate the flipped row
            uint32_t col = global_id_x;
            uint32_t global_linear_id = (row * (m_imageWidth) + col) * 4;


            if (row < m_imageHeight && col < m_imageWidth) {
                uint32_t tileId = group_id_y * group_id_x_max + group_id_x;

                glm::ivec2 range = m_ranges[tileId];
                uint32_t BLOCK_SIZE = 16 * 16;
                const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
                int toDo = range.y - range.x;

                // Initialize helper variables
                float T = 1.0f;
                float C[3] = {0};
                {
                    if (range.x >= 0 && range.y >= 0) {
                        for (int listIndex = range.x; listIndex < range.y; ++listIndex) {
                            uint32_t index = m_values[listIndex];
                            if (index > m_size || listIndex > m_size) {
                                continue;
                            }
                            const GaussianPoint &point = m_gaussianPoints[index];
                            // Perform processing on the point and update the image
                            glm::vec2 pos = point.screenPos;
                            // Calculate the exponent term
                            glm::vec2 diff = glm::vec2(col, row) - pos;
                            glm::vec3 c = point.conic;
                            glm::mat2 V(c.x, c.y, c.y, c.z);
                            float power = -0.5f * glm::dot(diff, V * diff);
                            if (power > 0.0f) {
                                continue;
                            }
                            float alpha = std::min(0.99f, point.opacity * expf(power));
                            if (alpha < 1.0f / 255.0f)
                                continue;
                            float test_T = T * (1 - alpha);
                            if (test_T < 0.0001f) {
                                continue;
                            }
                            // Eq. (3) from 3D Gaussian splatting paper.
                            for (int ch = 0; ch < 3; ch++) {
                                C[ch] += point.computedColor[ch] * alpha * T;
                            }
                            T = test_T;
                        }
                    }
                }
                m_imageBuffer[global_linear_id] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
                m_imageBuffer[global_linear_id + 1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
                m_imageBuffer[global_linear_id + 2] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
                m_imageBuffer[global_linear_id +
                              3] = static_cast<uint8_t>(255.0f); // Assuming full alpha for simplicity
            } // endif
        }
    };
}
#endif //RASTERIZER_H
