//
// Created by magnus on 5/17/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANRENDERER_H
#define MULTISENSE_VIEWER_GAUSSIANRENDERER_H

#include <cmath>
#include <cstdint>
#include <vector>
#include <glm/ext/quaternion_float.hpp>
#include <filesystem>
#include <sycl/sycl.hpp>

#include "Viewer/Core/Camera.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/SYCL/RasterizerUtils.h"
#include "Viewer/SYCL/AbstractRenderer.h"

namespace VkRender {
    class Sorter;

    class GaussianRenderer {
    public:

        struct GaussianPoints {
            std::vector<glm::vec3> positions;
            std::vector<glm::quat> quats;
            std::vector<glm::vec3> scales;
            std::vector<float> opacities;
            std::vector<float> sphericalHarmonics;  // Add this line
            uint32_t shDim = 1; // default rgb
            [[nodiscard]] uint32_t getSize() const {
                return positions.size();
            }

            uint32_t getShDim() const {
                return shDim; // TODO implement
            }
        };

        GaussianRenderer();

        ~GaussianRenderer();

        void setup(const VkRender::AbstractRenderer::InitializeInfo &initInfo, bool useCPU = false);
        void render(const AbstractRenderer::RenderInfo &info, const VkRender::RenderUtils *pUtils);
        uint8_t *getImage();
        uint32_t getImageSize();
        GaussianRenderer::GaussianPoints gs;
    public:

        static GaussianPoints loadFromFile(std::filesystem::path path, int i);

        void setupBuffers(const VkRender::Camera *camera);

        void singleOneSweep();

    private:
        VkRender::AbstractRenderer::InitializeInfo m_initInfo;
        uint8_t *m_image = nullptr;

        glm::vec3 *positionBuffer = nullptr;
        glm::vec3 *scalesBuffer = nullptr;
        glm::quat *quaternionBuffer = nullptr;
        float *opacityBuffer = nullptr;
        float *sphericalHarmonicsBuffer = nullptr;

        Rasterizer::GaussianPoint *pointsBuffer = nullptr;
        uint32_t *numTilesTouchedBuffer = nullptr;
        uint32_t *pointOffsets = nullptr;
        uint32_t* keysBuffer = nullptr;
        uint32_t* valuesBuffer = nullptr;

        glm::ivec2* rangesBuffer = nullptr;
        uint8_t* imageBuffer = nullptr;
        std::unique_ptr<Sorter> sorter;

        uint32_t width{}, height{};
        sycl::event renderEvent;
        sycl::queue queue{};

        void clearBuffers();

        void logTimes(std::chrono::duration<double, std::milli> t1, std::chrono::duration<double, std::milli> t2,
                      std::chrono::duration<double, std::milli> t3, std::chrono::duration<double, std::milli> t4,
                      std::chrono::duration<double, std::milli> t5, std::chrono::duration<double, std::milli> t6,
                      std::chrono::duration<double, std::milli> t7, std::chrono::duration<double, std::milli> t8,
                      std::chrono::duration<double, std::milli> t9, bool error);
    };

}

#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
