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
        Preprocess(glm::vec3 *positions, glm::vec3 *normals, glm::vec3 *scales, glm::quat *quats, float *opacities,
                   float *shs,
                   uint32_t *tilesTouched, GaussianPoint *pointsBuffer, const PreprocessInfo *info) : m_positions(
                positions), m_normals(normals), m_scales(scales),
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
            m_points[i].radius = 0;
            m_points[i].depth = 0;
            m_points[i].screenPos = glm::vec3(0);

            glm::vec3 scale = m_scales[i];
            glm::quat q = m_quaternions[i];
            glm::vec3 position = m_positions[i];

            glm::vec4 posView = m_scene.viewMatrix * glm::vec4(position, 1.0f);
            glm::vec4 posClip = m_scene.projectionMatrix * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;

            //sycl::ext::oneapi::experimental::printf("i: %d, depth: %f. Pos (%f, %f, %f) \n", i, posView.z, position.x,position.y, position.z);

            if (posView.z >= -0.2f) {
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
                float my_radius = ceilf(3.0f * std::sqrt(std::max(lambda1, lambda2)));
                glm::ivec2 rect_min, rect_max;

                getRect(screenPosPoint, static_cast<int>(my_radius), rect_min, rect_max, m_scene.tileGrid);

                if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
                    return;

                glm::vec3 color;
                if (m_scene.colorMethod == 1) {
                    glm::mat3 R = glm::mat3_cast(q);
                    glm::vec4 n_view   = glm::vec4{R[0][2], R[1][2], R[2][2], 1.0f} * m_scene.viewMatrix;

                    color = (glm::normalize(glm::vec3(n_view))  + glm::vec3(1.0f)) * 0.5f;
                } else if (m_scene.colorMethod == 0){
                    color = computeColorFromSH(idx, position, m_scene.camPos, m_scene.shDegree, m_scene.shDim, m_sphericalHarmonics);
                } else if (m_scene.colorMethod == 2){

                    color = 1.0f - glm::vec3(posNDC.z);
                }


                /*
                float SH_C0 = 0.28209479177387814f;
                glm::vec3 color =
                        SH_C0 * glm::vec3(m_sphericalHarmonics[i * m_scene.shDim + 0],
                                          m_sphericalHarmonics[i * m_scene.shDim + 1],
                                          m_sphericalHarmonics[i * m_scene.shDim + 2]);

                color += 0.5f;
                */


                auto conic = glm::vec3(cov2D.z * invDeterminant, -cov2D.y * invDeterminant,
                                       cov2D.x * invDeterminant);


                m_points[i].depth = posNDC.z;
                m_points[i].radius = my_radius;
                m_points[i].conic = conic;
                m_points[i].screenPos = screenPosPoint;
                m_points[i].color = color;
                m_points[i].opacity = m_opacities[i];
                // How many tiles we access
                // rect_min/max are in tile space
                auto numTilesTouched = static_cast<uint32_t>((rect_max.y - rect_min.y) *
                                                             (rect_max.x - rect_min.x));
                m_numTilesTouched[i] = numTilesTouched;

                /*
                sycl::ext::oneapi::experimental::printf("NumTilesTouched for %u: %u\n", i, numTilesTouched);
                sycl::ext::oneapi::experimental::printf("Rect for %u: min (%u, %u), max: (%u, %u)\n", i, rect_min.x, rect_min.y, rect_max.x, rect_max.y);
                */

            }
        }

    private:
        glm::vec3 *m_positions;
        glm::vec3 *m_normals;
        glm::vec3 *m_scales = nullptr;
        glm::quat *m_quaternions = nullptr;
        float *m_opacities = nullptr;
        float *m_sphericalHarmonics = nullptr;
        uint32_t *m_numTilesTouched{0};
        GaussianPoint *m_points{0};
        PreprocessInfo m_scene;
        uint32_t maxDepth = 0;
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
                           float* depthValues,
                           uint32_t numRendered,
                           glm::vec3 tileGrid)
                : m_gaussianPoints(gaussianPoints),
                  m_pointsOffsets(pointsOffsets),
                  m_keys(keys),
                  m_values(values),
                  m_depthValues(depthValues),
                  m_numRendered(numRendered),
                  m_tileGrid(tileGrid) {}

    private:
        GaussianPoint *m_gaussianPoints;
        uint32_t *m_pointsOffsets;
        uint32_t *m_keys;
        uint32_t *m_values;
        uint32_t m_numRendered = 0;
        glm::vec3 m_tileGrid;
        float* m_depthValues;

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
                        auto depth = static_cast<uint16_t>(gaussian.depth * 65535);
                        key |= depth;
                        m_keys[off] = key;
                        m_values[off] = i;
                        m_depthValues[i] = gaussian.depth;
                        ++off;
                    }
                }
            }
        }


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
        RasterizeGaussians(glm::ivec2 *ranges, uint32_t *keys, uint32_t *values, GaussianPoint *gaussianPoints,
                           uint8_t *imageBuffer, size_t size, uint32_t m_imageWidth, uint32_t imageHeight,
                           uint32_t horizontalBlocks, uint32_t numTiles, float maxDepthValue = -1, float minDepthValue = -1)
                : m_ranges(ranges), m_keys(keys), m_values(values), m_gaussianPoints(gaussianPoints),
                  m_imageBuffer(imageBuffer), m_size(size),
                  m_imageWidth(m_imageWidth), m_imageHeight(imageHeight), m_horizontalBlocks(horizontalBlocks),
                  m_numTiles(numTiles), m_maxDepthValue(maxDepthValue) , m_minDepthValue(minDepthValue) {}

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
        float m_maxDepthValue = -1;
        float m_minDepthValue = -1;
    public:
        void operator()(sycl::nd_item<2> item) const {
            uint32_t gridDim = item.get_group_range(0);
            auto block = item.get_group();
            // Get the global IDs
            uint32_t global_id_x = item.get_global_id(1); // x-coordinate (column)
            uint32_t global_id_y = item.get_global_id(0); // y-coordinate (row)
            // Get the local IDs within the tile
            uint32_t local_id_x = item.get_local_id(1);
            uint32_t local_id_y = item.get_local_id(0);
            // Get the group (tile) IDs
            uint32_t group_id_x = item.get_group(1);
            uint32_t group_id_y = item.get_group(0);

            uint32_t group_id_x_max = item.get_group_range(1);
            uint32_t group_id_y_max = item.get_group_range(0);

            uint32_t global_linear_id = (global_id_y * (m_imageWidth) + global_id_x) * 4;
            // Calculate the global pixel row and column
            uint32_t row = global_id_y;
            uint32_t col = global_id_x;

            //    if (group_id_x != 40 || group_id_y != 22){
            //      return;
            //   }

            if (row < m_imageHeight && col < m_imageWidth) {
                uint32_t tileId = group_id_y * group_id_x_max + group_id_x;


                glm::ivec2 range = m_ranges[tileId];
                uint32_t BLOCK_SIZE = 16 * 16;
                const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
                int toDo = range.y - range.x;


                // Initialize helper variables
                float T = 1.0f;
                float C[3] = {0};
                //if (group_id_x == 40 && group_id_y == 22)
                {
                    if (range.x >= 0 && range.y >= 0) {

                        for (int listIndex = range.x; listIndex < range.y; ++listIndex) {
                            uint32_t index = m_values[listIndex];
                            if (index > m_size || listIndex > m_size) {
                                continue;
                            }
                            const GaussianPoint &point = m_gaussianPoints[index];
                            //sycl::ext::oneapi::experimental::printf("ListIndex: %u index: %u\n", listIndex, index);
                            // Perform processing on the point and update the image
                            // Example: Set the pixel to a specific value
                            glm::vec2 pos = point.screenPos;
                            // Calculate the exponent term
                            //glm::vec2 d = glm::vec2(col, row) - pos;
                            //glm::vec3 c = point.conic;
                            //float power = -0.5f * (c.x * d.x * d.x + c.z * d.y * d.y) - c.y * d.x * d.y;

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
                                if (m_maxDepthValue != -1){
                                    float col  = (point.color[ch] - m_minDepthValue) / (m_maxDepthValue - m_minDepthValue);
                                    //col = std::min(col, 1.0f);
                                    C[ch] += col * alpha * T;
                                } else {
                                    C[ch] += point.color[ch] * alpha * T;
                                }
                            }
                            T = test_T;
                        }
                    }
                }
                m_imageBuffer[global_linear_id] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
                m_imageBuffer[global_linear_id + 1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
                m_imageBuffer[global_linear_id + 2] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
                m_imageBuffer[global_linear_id +
                              3] = static_cast<uint8_t>(255.0f); // Assuming full alpha for simplicity

            } // endif
        }
    };

    class RasterizeGaussiansPerPixel {
    public:
        RasterizeGaussiansPerPixel(glm::ivec2 *ranges, uint32_t *keys, uint32_t *values,
                                   GaussianPoint *gaussianPoints,
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
        void operator()(sycl::id<1> item) const {

            uint32_t index = item.get(0) * 4;
            uint32_t row = index / (m_imageWidth * 4);
            uint32_t col = (index % (m_imageWidth * 4)) / 4;

            if (row < m_imageHeight && col < m_imageWidth) {
                //size_t tileId = group.get_group_id(1) * m_horizontalBlocks + group.get_group_id(0);
                // Initialize helper variables
                float T = 1.0f;
                float C[3] = {0};

                for (uint32_t gaussianIndex = 0; gaussianIndex < m_size; ++gaussianIndex) {
                    const GaussianPoint &point = m_gaussianPoints[gaussianIndex];
                    // Perform processing on the point and update the image
                    // Example: Set the pixel to a specific value
                    glm::vec2 pos = point.screenPos;
                    // Calculate the exponent term
                    glm::vec2 diff = pos - glm::vec2(col, row);
                    glm::vec3 c = point.conic;
                    glm::mat2 V(c.x, c.y, c.y, c.z);
                    float power = -0.5f * glm::dot(diff, V * diff);
                    if (power > 0.0f) {
                        continue;
                    }
                    float alpha = point.opacity * expf(power);

                    if (alpha < 1.0f / 255.0f)
                        continue;

                    float test_T = T * (1 - alpha);
                    if (test_T < 0.0001f) {
                        continue;
                    }

                    // Eq. (3) from 3D Gaussian splatting paper.
                    for (int ch = 0; ch < 3; ch++) {
                        C[ch] += point.color[ch] * alpha;
                    }
                    T = test_T;
                }
                //uint32_t baseIndex = (row * m_imageWidth + col) * 4;

                m_imageBuffer[index] = static_cast<uint8_t>((C[0] + T * 0.0f) * 255.0f);
                m_imageBuffer[index + 1] = static_cast<uint8_t>((C[1] + T * 0.0f) * 255.0f);
                m_imageBuffer[index + 2] = static_cast<uint8_t>((C[2] + T * 0.0f) * 255.0f);
                m_imageBuffer[index + 3] = static_cast<uint8_t>(255.0f); // Assuming full alpha for simplicity
            }
        }
    };
}

#endif //MULTISENSE_VIEWER_RASTERIZER_H