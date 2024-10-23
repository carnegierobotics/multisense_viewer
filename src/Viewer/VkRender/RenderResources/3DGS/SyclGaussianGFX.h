//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_SYCLGAUSSIANGFX_H
#define MULTISENSE_VIEWER_SYCLGAUSSIANGFX_H

#include <glm/glm.hpp>
#include <sycl/sycl.hpp>

#include "Viewer/VkRender/Components/GaussianComponent.h"
#include "Viewer/VkRender/RenderResources/3DGS/RasterizerUtils.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/VkRender/RenderResources/3DGS/radixsort/RadixSorter.h"

namespace VkRender {


    class SyclGaussianGFX {
    public:
        explicit SyclGaussianGFX(sycl::queue &queue)
                : m_queue(queue) {
            uint32_t initialSize = 1;

            m_gaussianPointsPtr = sycl::malloc_shared<GaussianPoint>(initialSize, queue);
            m_preProcessDataPtr = sycl::malloc_shared<PreProcessData>(1, queue);
            if (!m_gaussianPointsPtr) {
                throw std::runtime_error("Failed to allocate Gaussian points buffer");
            }
            m_numGaussians = 0;

            m_sorter = std::make_unique<Sorter>(queue, 1 << 24);

        }

        ~SyclGaussianGFX() {
            // Clean up manually allocated memory
            if (m_gaussianPointsPtr) {
                sycl::free(m_gaussianPointsPtr, m_queue);
            }
        }

        void render(std::shared_ptr<Scene> &scene, std::shared_ptr<VulkanTexture2D> &outputTexture, Camera &camera);

        void updateGaussianPoints(const std::vector<GaussianPoint> &newPoints);

        void preProcessGaussians(uint8_t *imageMemory);

    private:
        sycl::queue &m_queue;
        PreProcessData m_preProcessData;

        void rasterizeGaussians(const GaussianComponent &gaussianComp);

        uint32_t m_numGaussians = 1;
        GaussianPoint *m_gaussianPointsPtr;
        PreProcessData *m_preProcessDataPtr;

        std::unique_ptr<Sorter> m_sorter;
    };
}


#endif //MULTISENSE_VIEWER_SYCLGAUSSIANGFX_H
