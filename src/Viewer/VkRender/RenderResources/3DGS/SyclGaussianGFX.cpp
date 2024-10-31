//
// Created by magnus on 10/21/24.
//

#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>
#include <execution>
#include <utility>

#include "Viewer/VkRender/RenderResources/3DGS/SyclGaussianGFX.h"
#include "Viewer/VkRender/RenderResources/3DGS/Rasterizer.h"

namespace VkRender {
    void SyclGaussianGFX::render(std::shared_ptr<Scene> &scene, std::shared_ptr<VulkanTexture2D> &outputTexture) {


        auto *imageMemory = static_cast<uint8_t *>(malloc(outputTexture->getSize()));
        auto &registry = scene->getRegistry();
        // Find all entities with GaussianComponent
        registry.view<GaussianComponent>().each([&](auto entity, GaussianComponent &gaussianComp) {
            if (gaussianComp.addToRenderer) {
                std::vector<GaussianPoint> newPoints;

                for (size_t i = 0; i < gaussianComp.size(); ++i) {
                    GaussianPoint point{};
                    point.position = gaussianComp.means[i];
                    point.scale = gaussianComp.scales[i];
                    point.rotation = gaussianComp.rotations[i];
                    point.opacity = gaussianComp.opacities[i];
                    point.color = gaussianComp.colors[i];
                    if (!gaussianComp.shCoeffs.empty())
                        point.shCoeffs = gaussianComp.shCoeffs[i];

                    newPoints.push_back(point);
                }
                if (!newPoints.empty())
                    updateGaussianPoints(newPoints);
            }
        });

        if (m_numGaussians < 1) {
            free(imageMemory);
            return;
        }

        auto activeCameraPtr = m_activeCamera.lock(); // Lock to get shared_ptr

        uint32_t BLOCK_X = 16, BLOCK_Y = 16;
        glm::vec3 tileGrid((activeCameraPtr->width() + BLOCK_X - 1) / BLOCK_X,
                           (activeCameraPtr->height() + BLOCK_Y - 1) / BLOCK_Y, 1);
        uint32_t numTiles = tileGrid.x * tileGrid.y;
        auto params = getHtanfovxyFocal(activeCameraPtr->m_Fov, activeCameraPtr->height(), activeCameraPtr->width());

        m_preProcessData.camera = *activeCameraPtr;
        //m_preProcessData.camera.matrices.perspective[1] = -m_preProcessData.camera.matrices.perspective[1];
        m_preProcessData.preProcessSettings.tileGrid = tileGrid;
        m_preProcessData.preProcessSettings.numTiles = numTiles;
        m_preProcessData.preProcessSettings.numPoints = m_numGaussians;
        m_preProcessData.preProcessSettings.params = params;


        m_queue.memcpy(m_preProcessDataPtr, &m_preProcessData, sizeof(PreProcessData)).wait();

        preProcessGaussians(imageMemory);
        // Start rendering
        // Copy output to texture
        outputTexture->loadImage(imageMemory, outputTexture->getSize());

        // Free the allocated memory
        free(imageMemory);
    }

    void SyclGaussianGFX::updateGaussianPoints(const std::vector<GaussianPoint> &newPoints) {
        sycl::free(m_gaussianPointsPtr, m_queue);
        m_numGaussians = newPoints.size();
        m_gaussianPointsPtr = sycl::malloc_shared<GaussianPoint>(m_numGaussians, m_queue);


        m_queue.memcpy(m_gaussianPointsPtr, newPoints.data(), newPoints.size() * sizeof(GaussianPoint)).wait();
    }

    void copyAndSortKeysAndValues(sycl::queue &m_queue, uint32_t *keysBuffer, uint32_t *valuesBuffer, size_t numRendered) {
        // Step 1: Allocate Unified Shared Memory (USM) for key-value pairs
        using KeyValue = std::pair<uint32_t, uint32_t>;

        // Allocate USM for the key-value pairs directly
        KeyValue* keyValuePairs = sycl::malloc_shared<KeyValue>(numRendered, m_queue);

        // Step 2: Initialize the key-value pairs from the provided buffers
        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(numRendered), [=](sycl::id<1> idx) {
                keyValuePairs[idx[0]].first = keysBuffer[idx[0]];
                keyValuePairs[idx[0]].second = valuesBuffer[idx[0]];
            });
        }).wait();

        // Step 3: Sort the key-value pairs directly using the C++ standard algorithm with offloading
        std::sort(std::execution::par_unseq, keyValuePairs, keyValuePairs + numRendered, [](const KeyValue &a, const KeyValue &b) {
            return a.first < b.first;
        });

        // Step 4: Write the sorted keys and values back to the provided buffers
        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(numRendered), [=](sycl::id<1> idx) {
                keysBuffer[idx[0]] = keyValuePairs[idx[0]].first;
                valuesBuffer[idx[0]] = keyValuePairs[idx[0]].second;
            });
        }).wait();

        // Free the USM memory
        sycl::free(keyValuePairs, m_queue);
    }

    void SyclGaussianGFX::preProcessGaussians(uint8_t *imageMemory) {
        uint32_t imageWidth = m_preProcessData.camera.width();
        uint32_t imageHeight = m_preProcessData.camera.height();

        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(m_numGaussians),
                           Rasterizer::Preprocess(m_gaussianPointsPtr, m_preProcessDataPtr));
        }).wait();


        auto *pointOffsets = sycl::malloc_device<uint32_t>(m_numGaussians, m_queue);
        m_queue.memset(pointOffsets, static_cast<uint32_t>(0x00), m_numGaussians * sizeof(uint32_t)).wait();

        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(1),
                           Rasterizer::InclusiveSum(m_gaussianPointsPtr, pointOffsets,
                                                    m_preProcessDataPtr->preProcessSettings.numPoints));
        }).wait();


        uint32_t numRendered = 0;
        m_queue.memcpy(&numRendered, pointOffsets + (m_numGaussians - 1), sizeof(uint32_t)).wait();
        // Inclusive sum complete

        Log::Logger::getInstance()->traceWithFrequency("3dgsrendering", 60, "3DGS Rendering, Gaussians: {}",
                                                       numRendered);
        if (numRendered > 0) {

            auto *keysBuffer = sycl::malloc_device<uint32_t>(numRendered, m_queue);
            auto *valuesBuffer = sycl::malloc_device<uint32_t>(numRendered, m_queue);
            m_queue.wait();

            m_queue.submit([&](sycl::handler &h) {
                h.parallel_for<class duplicates>(sycl::range<1>(m_numGaussians),
                                                 Rasterizer::DuplicateGaussians(m_gaussianPointsPtr, pointOffsets,
                                                                                keysBuffer,
                                                                                valuesBuffer,
                                                                                numRendered,
                                                                                m_preProcessDataPtr->preProcessSettings.tileGrid));
            }).wait();


            //m_sorter->performOneSweep(keysBuffer, valuesBuffer, numRendered);
            //m_sorter->verifySort(keysBuffer, numRendered);
            //m_sorter->resetMemory();


            copyAndSortKeysAndValues(m_queue, keysBuffer, valuesBuffer, numRendered);
            //m_sorter->verifySort(keysBuffer, numRendered);

            auto *rangesBuffer = sycl::malloc_device<glm::ivec2>(m_preProcessData.preProcessSettings.numTiles, m_queue);
            m_queue.memset(rangesBuffer, static_cast<int>(0x00),
                           m_preProcessData.preProcessSettings.numTiles * sizeof(int) * 2).wait();


            m_queue.submit([&](sycl::handler &h) {
                h.parallel_for<class IdentifyTileRanges>(numRendered,
                                                         Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer,
                                                                                        numRendered));
            }).wait();

            // Rasterize Gaussians
            const uint32_t tileWidth = 16;
            const uint32_t tileHeight = 16;

            sycl::range<2> localWorkSize(tileHeight, tileWidth);
            size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
            size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
            sycl::range<2> globalWorkSize(globalHeight, globalWidth);

            auto *imageBuffer = sycl::malloc_device<uint8_t>(imageWidth * imageHeight * 4, m_queue);
            m_queue.fill(imageBuffer, static_cast<uint8_t>(0x00), imageWidth * imageHeight * 4).wait();

            m_queue.submit([&](sycl::handler &h) {
                auto range = sycl::nd_range<2>(globalWorkSize, localWorkSize);
                h.parallel_for<class RenderGaussians>(range,
                                                      Rasterizer::RasterizeGaussians(rangesBuffer,
                                                                                     valuesBuffer, m_gaussianPointsPtr,
                                                                                     imageBuffer, numRendered,
                                                                                     imageWidth, imageHeight));
            }).wait();


            m_queue.memcpy(imageMemory, imageBuffer, imageWidth * imageHeight * 4);
            m_queue.wait();

            sycl::free(imageBuffer, m_queue);
            sycl::free(rangesBuffer, m_queue);
            sycl::free(pointOffsets, m_queue);
            sycl::free(keysBuffer, m_queue);
            sycl::free(valuesBuffer, m_queue);
        }
    }


    void SyclGaussianGFX::rasterizeGaussians(const GaussianComponent &gaussianComp) {
    }
}
