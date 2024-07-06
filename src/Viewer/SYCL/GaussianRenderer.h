//
// Created by magnus on 5/17/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANRENDERER_H
#define MULTISENSE_VIEWER_GAUSSIANRENDERER_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <glm/ext/quaternion_float.hpp>
#include <sycl/sycl.hpp>
#include <filesystem>
#include "Viewer/Core/Camera.h"
#include "AbstractRenderer.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "RasterizerUtils.h"

namespace VkRender {


    class GaussianRenderer {
    public:

        struct GaussianPoints {
            std::vector<glm::vec3> positions;
            std::vector<glm::quat> quats;
            std::vector<glm::vec3> scales;
            std::vector<float> opacities;
            std::vector<float> sphericalHarmonics;  // Add this line
            uint32_t shDim = 3; // default rgb
            [[nodiscard]] uint32_t getSize() const {
                return positions.size();
            }

            uint32_t getShDim() const {
                return shDim;
            }
        };

        GaussianRenderer() = default;

        ~GaussianRenderer() {
            clearBuffers();
        }

        void setup(const VkRender::AbstractRenderer::InitializeInfo &initInfo);

        void render(const AbstractRenderer::RenderInfo &info, const VkRender::RenderUtils *pUtils);

        uint8_t *getImage();

        uint32_t getImageSize();

        GaussianRenderer::GaussianPoints gs;
    public:
        static GaussianPoints loadNaive();

        static GaussianPoints loadFromFile(std::filesystem::path path, int i);

        bool loadedPly = false;

        void setupBuffers(const VkRender::Camera *camera);

        void singleOneSweep();

    private:
        VkRender::AbstractRenderer::InitializeInfo m_initInfo;
        uint8_t *m_image = nullptr;

        sycl::queue queue{};
        glm::vec3 *positionBuffer = nullptr;
        glm::vec3 *scalesBuffer = nullptr;
        glm::quat *quaternionBuffer = nullptr;
        float *opacityBuffer = nullptr;
        float *sphericalHarmonicsBuffer = nullptr;

        Rasterizer::GaussianPoint *pointsBuffer = nullptr;
        uint32_t *numTilesTouchedBuffer = nullptr;
        uint32_t *pointOffsets = nullptr;

        uint32_t* keysBuffer{0};
        uint32_t* valuesBuffer{0};

        sycl::buffer<glm::vec3, 1> covariance2DBuffer{0};
        sycl::buffer<glm::vec3, 1> conicBuffer{0};
        sycl::buffer<glm::vec3, 1> screenPosBuffer{0};
        sycl::buffer<glm::mat3, 1> covarianceBuffer{0};
        sycl::buffer<glm::vec3, 1> colorOutputBuffer{0};


        sycl::buffer<glm::ivec2, 1> rangesBuffer{0};


        sycl::buffer<uint32_t, 1> numTilesTouchedInclusiveSumBuffer{0};
        // Optimization buffers
        sycl::buffer<bool, 1> activeGSBuffer{0};

        sycl::buffer<uint8_t, 3> pngImageBuffer{sycl::range<3>()};
        sycl::buffer<uint8_t, 3> imageBuffer{sycl::range<3>()};
        sycl::buffer<uint8_t, 3> imageBuffer2{sycl::range<3>()};
        uint32_t width{}, height{};
        std::vector<uint8_t> flattenedImage;

        std::chrono::duration<double>
        preprocess(glm::mat4 viewMatrix, glm::mat4 projectionMatrix, uint32_t imageWidth, uint32_t imageHeight,
                   glm::vec3 camPos, glm::vec3 tileGrid, Rasterizer::CameraParams params);

        std::chrono::duration<double> inclusiveSum(uint32_t *numRendered);

        std::chrono::duration<double>
        duplicateGaussians(uint32_t numRendered, const glm::vec3 &tileGrid, uint32_t gridSize);

        std::chrono::duration<double> sortGaussians(uint32_t numRendered);

        std::chrono::duration<double> rasterizeGaussians();


        std::chrono::duration<double> identifyTileRanges(uint32_t numTiles, uint32_t numRendered);

        void clearBuffers();
    };

}

#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
