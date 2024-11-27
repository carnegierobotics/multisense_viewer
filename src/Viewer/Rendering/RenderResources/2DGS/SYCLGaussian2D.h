//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_SYCLGAUSSIAN2D_H
#define MULTISENSE_VIEWER_SYCLGAUSSIAN2D_H

#include <sycl/sycl.hpp>

#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Rendering/RenderResources/2DGS/RasterizerUtils2DGS.h"
#include "Viewer/Scenes/Scene.h"

namespace VkRender {


    class SYCLGaussian2D {
    public:
        explicit SYCLGaussian2D(sycl::queue &queue)
                : m_queue(queue) {
            uint32_t initialSize = 1;

            m_gaussianPointsPtr = sycl::malloc_shared<Rasterizer2DUtils::GaussianPoint>(initialSize, queue);
            m_preProcessDataPtr = sycl::malloc_shared<Rasterizer2DUtils::PreProcessData>(1, queue);
            if (!m_gaussianPointsPtr) {
                throw std::runtime_error("Failed to allocate Gaussian points buffer");
            }
            m_numGaussians = 0;
        }

        ~SYCLGaussian2D() {
            // Clean up manually allocated memory
            if (m_gaussianPointsPtr) {
                sycl::free(m_gaussianPointsPtr, m_queue);
            }
        }

        void render(std::shared_ptr<Scene> &scene, std::shared_ptr<VulkanTexture2D> &outputTexture);

        void updateGaussianPoints(const std::vector<Rasterizer2DUtils::GaussianPoint> &newPoints);

        void preProcessGaussians(uint8_t *imageMemory);

        void setActiveCamera(const std::shared_ptr<Camera>& cameraPtr){
            m_activeCamera = cameraPtr;
        }
        std::shared_ptr<Camera> getActiveCamera() const {
            return m_activeCamera.lock();
        }

    private:

        std::weak_ptr<Camera> m_activeCamera;

        sycl::queue &m_queue;
        Rasterizer2DUtils::PreProcessData m_preProcessData;

        void rasterizeGaussians(const GaussianComponent &gaussianComp);

        uint32_t m_numGaussians = 1;
        Rasterizer2DUtils::GaussianPoint *m_gaussianPointsPtr;
        Rasterizer2DUtils::PreProcessData *m_preProcessDataPtr;

        void renderGaussiansWithProfiling(uint8_t *imageMemory, bool enable_profiling);
        std::vector<unsigned long> m_durations;

        void copyAndSortKeysAndValues(sycl::queue &m_queue, uint32_t *keysBuffer, uint32_t *valuesBuffer,
                                      size_t numRendered);
    };
}


#endif //MULTISENSE_VIEWER_SYCLGAUSSIAN2D_H
