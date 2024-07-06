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
                                                                                                      m_scene(*info) {
        }


        void operator()(sycl::id<1> idx) const {
            uint32_t i = idx.get(0);
            m_numTilesTouched[i] = 0;

            glm::vec3 scale = m_scales[i];
            glm::quat q = m_quaternions[i];
            glm::vec3 position = m_positions[i];

            glm::vec4 posView = m_scene.viewMatrix * glm::vec4(position, 1.0f);
            glm::vec4 posClip = m_scene.projectionMatrix * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;

            //sycl::ext::oneapi::experimental::printf("i: %d, depth: %f. Pos (%f, %f, %f) \n", i, posView.z, position.x,position.y, position.z);

            if (posView.z >= -0.3f) {
                //sycl::ext::oneapi::experimental::printf("Culled: %d\n", i);
                return;
            }

            float pixPosX = ((posNDC.x + 1.0f) * static_cast<float>(m_scene.width ) - 1.0f) * 0.5f;
            float pixPosY = ((posNDC.y + 1.0f) * static_cast<float>(m_scene.height) - 1.0f) * 0.5f;
            auto screenPosPoint = glm::vec3(pixPosX, pixPosY, posNDC.z);

            glm::mat3 cov3D = computeCov3D(scale, q);
            glm::vec3 cov2D = computeCov2D(posView, cov3D, m_scene.viewMatrix, m_scene.params, false);


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

                getRect(screenPosPoint, static_cast<int>(my_radius), rect_min, rect_max, m_scene.tileGrid);

                if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
                    return;

                glm::vec3 dir = glm::normalize(position - m_scene.camPos);
                float SH_C0 = 0.28209479177387814f;

                glm::vec3 color =
                        SH_C0 * glm::vec3(m_sphericalHarmonics[i * m_scene.shDim + 0],
                                          m_sphericalHarmonics[i * m_scene.shDim + 1],
                                          m_sphericalHarmonics[i * m_scene.shDim + 2]);

                color += 0.5f;


                auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant,
                                       cov2D.x * invDeterminant);


                m_points[i].depth = posNDC.z;
                m_points[i].radius = my_radius;
                m_points[i].conic = conic;
                m_points[i].screenPos = screenPosPoint;
                m_points[i].color = color;
                m_points[i].opacityBuffer = m_opacities[i];
                // How many tiles we access
                // rect_min/max are in tile space

                m_numTilesTouched[i] = static_cast<int>((rect_max.y - rect_min.y) *
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
        PreprocessInfo m_scene;
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

    private:
        GaussianPoint *m_gaussianPoints;
        uint32_t *m_pointsOffsets;
        uint32_t *m_keys;
        uint32_t *m_values;
        uint32_t m_numRendered = 0;
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
                            sycl::ext::oneapi::experimental::printf(
                                    "DuplicateGaussians: m_numRendered is less than calculated offset: %u/%u\n", off,
                                    m_numRendered);
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


    };

    class IdentifyTileRangesInit {
    public:
        explicit IdentifyTileRangesInit(glm::ivec2 *ranges) : m_ranges(ranges) {}

        void operator()(sycl::id<1> id) const {
            auto index = id[0];
            // Ensure index is within valid bounds to prevent out-of-range access
            if (index >= 3600)
                return;

            m_ranges[index] = glm::ivec2(-1, -1);
        };

    private:
        glm::ivec2 *m_ranges;
    };

    class IdentifyTileRanges {
    public:
        IdentifyTileRanges(glm::ivec2 *ranges, uint32_t *keys, size_t size)
                : m_ranges(ranges), m_keys(keys), m_size(size) {}

    private:
        glm::ivec2 *m_ranges;
        uint32_t *m_keys;
        uint32_t m_size;

    public:
        void operator()(sycl::id<1> id) const {

            int idx = id[0];

            uint32_t key = m_keys[idx];
            uint16_t currentTile = key >> 16;
            if (currentTile < 3600) {


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
                    m_ranges[currentTile].y = m_size;
                }

            }
        };
    };

    class RasterizeGaussians {
    public:
        RasterizeGaussians(glm::ivec2 *ranges, uint32_t *keys, uint32_t *values, GaussianPoint *gaussianPoints,
                           uint8_t *imageBuffer, size_t size, uint32_t m_imageWidth, uint32_t imageHeight,
                           uint32_t horizontalBlocks, uint32_t numTiles)
                : m_ranges(ranges), m_keys(keys), m_values(values), m_gaussianPoints(gaussianPoints),
                  m_imageBuffer(imageBuffer), m_size(size),
                  m_imageWidth(m_imageWidth), m_imageHeight(imageHeight), m_horizontalBlocks(horizontalBlocks),
                  m_numTiles(numTiles) {}

    private:
        uint32_t *m_keys;
        uint32_t *m_values;
        glm::ivec2 *m_ranges;
        GaussianPoint *m_gaussianPoints;
        uint8_t *m_imageBuffer;

        uint32_t m_size;
        uint32_t m_imageWidth;
        uint32_t m_imageHeight;
        uint32_t m_horizontalBlocks;
        uint32_t m_numTiles;
    public:
        void operator()(sycl::nd_item<2> item) const {
            auto subGroup = item.get_sub_group();
            uint32_t blockIdx = item.get_group(0);
            uint32_t blockDim = item.get_local_range(0);
            uint32_t gridDim = item.get_group_range(0);
            uint32_t threadIdx = item.get_local_id(0);
            uint32_t warpIndex = subGroup.get_local_linear_id();
            uint32_t warpSize = subGroup.get_max_local_range()[0];

            auto globalID = item.get_global_id(); // Get global indices of the work item
            uint32_t row = globalID[0];
            uint32_t col = globalID[1];
            if (row < m_imageHeight && col < m_imageWidth) {
                uint32_t groupRow = row / 16;
                uint32_t groupCol = col / 16;
                uint32_t tileId = groupRow * m_horizontalBlocks + groupCol;
                // Ensure tileId is within bounds
                if (tileId >= m_numTiles) {
                    sycl::ext::oneapi::experimental::printf(
                            "TileId %u out of bounds (max %u ). groupRow %u, groupCol %u, m_horizontalBlocks %u, m_imageWidth %u \n",
                            tileId, static_cast<uint32_t>(m_numTiles - 1), groupRow, groupCol,
                            m_horizontalBlocks, m_imageWidth);
                    return;
                }
                //size_t tileId = group.get_group_id(1) * m_horizontalBlocks + group.get_group_id(0);
                glm::ivec2 range = m_ranges[tileId];
                // Initialize helper variables
                float T = 1.0f;
                float C[3] = {0};
                if (range.x >= 0 && range.y >= 0) {

                    for (int listIndex = range.x; listIndex < range.y; ++listIndex) {
                        uint32_t index = m_values[listIndex];
                        const GaussianPoint &point = m_gaussianPoints[index];
                        if (index >= m_size || listIndex >= 3600)
                            continue;
                        //sycl::ext::oneapi::experimental::printf("ListIndex: %u index: %u\n", listIndex, index);
                        // Perform processing on the point and update the image
                        // Example: Set the pixel to a specific value
                        glm::vec2 pos = point.screenPos;
                        // Calculate the exponent term
                        glm::vec2 diff = glm::vec2(col, row) - pos;
                        glm::vec3 c = point.conic;
                        glm::mat2 V(c.x, c.y, c.y, c.z);
                        float power = -0.5f * glm::dot(diff, V * diff);
                        if (power > 0.0f) {
                            continue;
                        }
                        float alpha = std::min(0.99f, point.opacityBuffer * expf(power));
                        if (alpha < 1.0f / 255.0f)
                            continue;
                        float test_T = T * (1 - alpha);
                        if (test_T < 0.0001f) {
                            continue;
                        }
                        // Eq. (3) from 3D Gaussian splatting paper.
                        for (int ch = 0; ch < 3; ch++) {
                            C[ch] += point.color[ch] * alpha * T;
                        }
                        T = test_T;
                    }
                }

                uint32_t baseIndex = (row * m_imageWidth + col) * 4;

                m_imageBuffer[baseIndex] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
                m_imageBuffer[baseIndex + 1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
                m_imageBuffer[baseIndex + 2] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
                m_imageBuffer[baseIndex + 3] = static_cast<uint8_t>(255.0f); // Assuming full alpha for simplicity
            } // endif
        }
    };
}

#endif //MULTISENSE_VIEWER_RASTERIZER_H
