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
#include "Viewer/SYCL/radixsort/RadixSorter.h"

namespace VkRender {


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

        GaussianRenderer() = default;

        ~GaussianRenderer() {
            clearBuffers();
        }

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

        sycl::queue queue{};
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
        void clearBuffers();
    };

}

#endif //MULTISENSE_VIEWER_GAUSSIANRENDERER_H
