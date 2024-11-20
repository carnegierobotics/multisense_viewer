//
// Created by magnus-desktop on 10/22/24.
//

#ifndef RASTERIZER_H
#define RASTERIZER_H

#include <cmath>
#include <sycl/sycl.hpp>
#include <glm/gtc/matrix_transform.hpp> // For glm::scale and other matrix transformations

#include "RasterizerUtils2DGS.h"


namespace VkRender::Rasterizer2D {
    class Preprocess {
    public:
        Preprocess(Rasterizer2DUtils::GaussianPoint *points, Rasterizer2DUtils::PreProcessData *preProcessData)
                : m_points(points),
                  m_camera(preProcessData->camera),
                  m_settings(preProcessData->preProcessSettings) {
        }


        void operator()(sycl::id<1> idx) const {
            uint32_t i = idx.get(0);

            m_points[i].tilesTouched = 0;
            m_points[i].radius = 0;
            m_points[i].depth = 0;
            m_points[i].screenPos = glm::vec3(0);

            glm::vec3 scale = m_points[i].scale;
            glm::quat quat = m_points[i].rotation;
            glm::vec3 position = m_points[i].position;

            glm::vec4 posView = m_camera.matrices.view * glm::vec4(position, 1.0f);
            glm::vec4 posClip = m_camera.matrices.perspective * posView;
            glm::vec3 posNDC = glm::vec3(posClip) / posClip.w;
            int x = position.x, y = position.y, z = position.z;
            //int zview = posView.z;
            if (posView.z >= -0.32f) {
                return;
            }

            glm::mat4 H = glm::mat4(
                    glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
                    glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
                    glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
                    glm::vec4(x, y, z, 1.0f)
            );
            glm::vec4 point_world = glm::vec4(H[3]);

            glm::mat4 projection = m_camera.matrices.perspective;
            glm::mat4 view = m_camera.matrices.view;
            std::cout << "projection: " << glm::to_string(projection) << std::endl;


            glm::vec4 point_ndc = projection * view * point_world;
            glm::vec3 point_screen = glm::vec3(point_ndc) / point_ndc.w;
            glm::vec3 point_viewport;
            point_viewport.x = (point_screen.x + 1.0f) * 0.5f * m_camera.width();
            point_viewport.y = (1.0f - (point_screen.y + 1.0f) * 0.5f) * m_camera.height();
            point_viewport.z = point_screen.z; // Depth remains unchanged


            std::cout << "camera: " << glm::to_string(glm::inverse(view)[3]) << std::endl;
            std::cout << "point_world: " << glm::to_string(point_world) << std::endl;
            std::cout << "point_ndc: " << glm::to_string(point_ndc) << std::endl;
            std::cout << "point_screen: " << glm::to_string(point_screen) << std::endl;
            std::cout << "point_viewport: " << glm::to_string(point_viewport) << std::endl;

            glm::mat4 points_ndc = projection * view * H;
            // Perform perspective division for each column
            glm::mat4 points_ndc_norm;
            for (int i = 0; i < 4; ++i) {
                glm::vec4 column = points_ndc[i];  // Get column as vec4
                if (column.w != 0.0f) {
                    column /= column.w;  // Perspective division
                } else {
                    std::cerr << "Warning: Division by zero for column " << i << "!" << std::endl;
                }
                points_ndc_norm[i] = glm::vec4(column);
            }
            std::cout << "points_ndc: " << glm::to_string(points_ndc) << std::endl;
            std::cout << "points_ndc_norm: " << glm::to_string(points_ndc_norm) << std::endl;

            glm::mat4 ndc2pix = glm::transpose(glm::mat4(
                glm::vec4(float(m_camera.width()) / 2.0, 0.0, 0.0, float(m_camera.width()-1) / 2.0),
                glm::vec4(0.0, float(m_camera.height()) / 2.0, 0.0, float(m_camera.height()-1) / 2.0),
                glm::vec4(0.0, 0.0, 1.0, 0.0),
                glm::vec4(0.0, 0.0, 0.0, 1.0)
            ));

            glm::mat4 T = points_ndc_norm * ndc2pix;
            std::cout << "T: " << glm::to_string(T) << std::endl;


            // Compute center and radius
            // compute_aabb
            float filterSize = 2.0f;
            float cutoff = 3.0f;
            float radius = 0;
            glm::vec2 pointImage;
            glm::vec2 extent;
            if (Rasterizer2DUtils::compute_aabb(T, cutoff, pointImage, extent)) {
                radius = ceil(std::max(std::max(extent.x, extent.y), cutoff * filterSize));

            }

            glm::ivec2 rect_min, rect_max;
            Rasterizer2DUtils::getRect(pointImage, radius, rect_min, rect_max, m_settings.tileGrid);
            if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
                return;
            std::array<std::array<float, 15>, 3> sh = m_points[i].shCoeffs;
            m_points[i].depth = posNDC.z;
            m_points[i].radius = radius;
            m_points[i].screenPos = glm::vec3(pointImage, 0.0f);
            // How many tiles we access
            // rect_min/max are in tile space
            auto numTilesTouched = static_cast<uint32_t>((rect_max.y - rect_min.y) *
                                                         (rect_max.x - rect_min.x));
            m_points[i].tilesTouched = numTilesTouched;
            m_points[i].T = T;

            glm::vec3 result = m_points[i].color;
            m_points[i].computedColor = result;

            //glm::vec3 cameraPos = m_camera.pose.pos;
            //glm::vec3 gaussianPos = m_points[i].position;
            //glm::vec3 result = Rasterizer2DUtils::computeColorFromSH(m_points[i].shCoeffs, color, gaussianPos, cameraPos);

            /*
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
                glm::vec3 cameraPos = m_camera.pose.pos;
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
            */
        }

        Rasterizer2DUtils::GaussianPoint *m_points;
        Camera m_camera;
        Rasterizer2DUtils::PreProcessSettings m_settings;
    };

    class InclusiveSum {
    public:
        InclusiveSum(Rasterizer2DUtils::GaussianPoint *points, uint32_t *pointOffsets, uint32_t numPoints,
                     sycl::local_accessor<uint32_t> localMem, uint32_t *groupTotals)
                : m_points(points), m_output(pointOffsets), m_numPoints(numPoints), m_localMem(localMem),
                  m_groupTotals(groupTotals) {
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
        Rasterizer2DUtils::GaussianPoint *m_points;
        uint32_t *m_output;
        uint32_t *m_groupTotals;
        uint32_t m_numPoints;
    };

    class InclusiveSum2 {
    public:
        InclusiveSum2(Rasterizer2DUtils::GaussianPoint *points, uint32_t *pointOffsets, uint32_t numPoints,
                      sycl::local_accessor<uint32_t> localMem) : m_points(points),
                                                                 m_output(pointOffsets), m_numPoints(numPoints),
                                                                 m_localMem(localMem) {

        }


        void operator()(sycl::nd_item<1> item) const {
            size_t globalIdx = item.get_global_id(0);
            size_t localIdx = item.get_local_id(0);
            size_t localRange = item.get_local_range(0);
            if (globalIdx == 0) {
                m_output[0] = m_points[0].tilesTouched;
                for (size_t i = 1; i < m_numPoints; ++i) {
                    m_output[i] = m_output[i - 1] + m_points[i].tilesTouched;
                }
            }

        }

        sycl::local_accessor<uint32_t> m_localMem;

        Rasterizer2DUtils::GaussianPoint *m_points;
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
        DuplicateGaussians(Rasterizer2DUtils::GaussianPoint *gaussianPoints,
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
        Rasterizer2DUtils::GaussianPoint *m_gaussianPoints;
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
                Rasterizer2DUtils::getRect(gaussian.screenPos, gaussian.radius, rect_min, rect_max, m_tileGrid);
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
        RasterizeGaussians(glm::ivec2 *ranges, uint32_t *values, Rasterizer2DUtils::GaussianPoint *gaussianPoints,
                           uint8_t *imageBuffer, size_t size, uint32_t m_imageWidth, uint32_t imageHeight)
                : m_ranges(ranges), m_values(values), m_gaussianPoints(gaussianPoints),
                  m_imageBuffer(imageBuffer), m_size(size),
                  m_imageWidth(m_imageWidth), m_imageHeight(imageHeight) {
        }

    private:
        uint32_t *m_values;
        glm::ivec2 *m_ranges;
        Rasterizer2DUtils::GaussianPoint *m_gaussianPoints;
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

            float filterInvSquare = 2.0f;

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
                            const Rasterizer2DUtils::GaussianPoint &point = m_gaussianPoints[index];
                            const auto &TM = point.T;
                            glm::vec2 xy = point.screenPos;
                            glm::vec2 pix = glm::vec2(static_cast<float>(col),
                                                      static_cast<float>(row));

                            glm::vec3 Tu = glm::vec3(TM[0][0], TM[0][1], TM[0][2]);
                            glm::vec3 Tv = glm::vec3(TM[1][0], TM[1][1], TM[1][2]);
                            glm::vec3 Tw = glm::vec3(TM[2][0], TM[2][1], TM[2][2]);

                            glm::vec3 k = pix.x * Tw - Tu;
                            glm::vec3 l = pix.y * Tw - Tv;
                            glm::vec3 p = glm::cross(l, k);
                            if (col == m_imageHeight / 2 && row == m_imageHeight / 2) {
                                std::cout << "l: " << glm::to_string(l) << std::endl;
                                std::cout << "k: " << glm::to_string(k) << std::endl;
                                std::cout << "p: " << glm::to_string(p) << std::endl;
                                Tu = glm::cross(k, p);
                            }
                            if (p.z == 0) return;

                            glm::vec2 s = {p.x / p.z, p.y / p.z};
                            float rho3d = (s.x * s.x + s.y * s.y);
                            glm::vec2 d = {xy.x - pix.x, xy.y - pix.y};
                            float rho2d = filterInvSquare * (d.x * d.x + d.y * d.y);

                            // compute intersection and depth
                            float rho = std::min(rho3d, rho2d);
                            float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
                            if (depth < 0.2) continue;

                            float power = -0.5f * rho;
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
