//
// Created by magnus on 8/15/24.
//
#include <glm/ext/quaternion_float.hpp>

#include "GaussianModelGraphicsPipeline.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/RenderPipelines/3DGS/RasterizerUtils.h"
#include "Rasterizer.h"

namespace VkRender {


    GaussianModelGraphicsPipeline::GaussianModelGraphicsPipeline(VulkanDevice& vulkanDevice) : m_vulkanDevice(vulkanDevice) {
        try {
            // Create a queue using the CPU device selector
            auto gpuSelector = [](const sycl::device &dev) {
                if (dev.is_gpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };

            auto cpuSelector = [](const sycl::device &dev) {
                if (dev.is_cpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };    // Define a callable device selector using a lambda

            queue = sycl::queue(gpuSelector);
            // Use the queue for your computation
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("GPU device not found");
            Log::Logger::getInstance()->info("Falling back to default device selector");
            // Fallback to default device selector
            queue = sycl::queue(sycl::property::queue::in_order());
        }


        Log::Logger::getInstance()->info("Selected Device {}",
                                         queue.get_device().get_info<sycl::info::device::name>().c_str());
    }


    template<>
    void
    GaussianModelGraphicsPipeline::bind<GaussianModelComponent>(
            GaussianModelComponent &modelComponent, Camera &camera) {

        auto &gs = modelComponent.getGaussians();
        Log::Logger::getInstance()->info("Loaded {} Gaussians", gs.getSize());
        try {
            uint32_t numPoints = gs.getSize();
            m_numPoints = numPoints;
            m_shDim = gs.getShDim();

            positionBuffer = sycl::malloc_device<glm::vec3>(numPoints, queue);
            scalesBuffer = sycl::malloc_device<glm::vec3>(numPoints, queue);
            quaternionBuffer = sycl::malloc_device<glm::quat>(numPoints, queue);
            opacityBuffer = sycl::malloc_device<float>(numPoints, queue);
            sphericalHarmonicsBuffer = sycl::malloc_device<float>(gs.sphericalHarmonics.size(), queue);

            queue.memcpy(positionBuffer, gs.positions.data(), numPoints * sizeof(glm::vec3));
            queue.memcpy(scalesBuffer, gs.scales.data(), numPoints * sizeof(glm::vec3));
            queue.memcpy(quaternionBuffer, gs.quats.data(), numPoints * sizeof(glm::quat));
            queue.memcpy(opacityBuffer, gs.opacities.data(), numPoints * sizeof(float));
            queue.memcpy(sphericalHarmonicsBuffer, gs.sphericalHarmonics.data(),
                         gs.sphericalHarmonics.size() * sizeof(float));
            queue.wait_and_throw();

            numTilesTouchedBuffer = sycl::malloc_device<uint32_t>(numPoints, queue);
            pointOffsets = sycl::malloc_device<uint32_t>(numPoints, queue);
            pointsBuffer = sycl::malloc_device<Rasterizer::GaussianPoint>(numPoints, queue);

            uint32_t sortBufferSize = (1 << 25);
            keysBuffer = sycl::malloc_device<uint32_t>(sortBufferSize, queue);
            valuesBuffer = sycl::malloc_device<uint32_t>(sortBufferSize, queue);

            width = camera.m_width;
            height = camera.m_height;
            m_imageSize = width * height * 4;
            m_image = reinterpret_cast<uint8_t *>(std::malloc(width * height * 4));
            sorter = std::make_unique<Sorter>(queue, sortBufferSize);

            imageBuffer = sycl::malloc_device<uint8_t>(width * height * 4, queue);
            rangesBuffer = sycl::malloc_device<glm::ivec2>((width / 16) * (height / 16), queue);
            m_boundBuffers = true;

            m_textureVideo = std::make_shared<TextureVideo>(width, height, &m_vulkanDevice, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FORMAT_R8G8B8A8_UNORM);

        } catch (sycl::exception &e) {
            std::cerr << "Caught a SYCL exception: " << e.what() << std::endl;
            return;
        } catch (std::exception &e) {
            std::cerr << "Caught a standard exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr << "Caught an unknown exception." << std::endl;
            return;
        }


        queue.wait_and_throw();

    }


    void GaussianModelGraphicsPipeline::draw(CommandBuffer &cmdBuffers) {
        // Copy image we rendered to a texture

        // Draw the texture
    }

    GaussianModelGraphicsPipeline::~GaussianModelGraphicsPipeline() {
        if (m_boundBuffers) {
            sycl::free(positionBuffer, queue);
            sycl::free(scalesBuffer, queue);
            sycl::free(quaternionBuffer, queue);
            sycl::free(opacityBuffer, queue);
            sycl::free(sphericalHarmonicsBuffer, queue);
            sycl::free(numTilesTouchedBuffer, queue);
            sycl::free(pointsBuffer, queue);
            sycl::free(imageBuffer, queue);
            sycl::free(rangesBuffer, queue);

            sycl::free(keysBuffer, queue);
            sycl::free(valuesBuffer, queue);
            free(m_image);
        }
    }

    uint8_t *GaussianModelGraphicsPipeline::getImage() {
        return m_image;
    }

    uint32_t GaussianModelGraphicsPipeline::getImageSize() {
        return m_imageSize;
    }


    void GaussianModelGraphicsPipeline::generateImage(Camera& camera) {
        auto params = Rasterizer::getHtanfovxyFocal(camera.m_Fov, camera.m_height, camera.m_width);
        glm::mat4 viewMatrix = camera.matrices.view;
        glm::mat4 projectionMatrix = camera.matrices.perspective;
        glm::vec3 camPos = camera.pose.pos;
        // Flip the second row of the projection matrix
        projectionMatrix[1] = -projectionMatrix[1];
        uint32_t BLOCK_X = 16, BLOCK_Y = 16;
        const uint32_t imageWidth = width;
        const uint32_t imageHeight = height;
        const uint32_t tileWidth = 16;
        const uint32_t tileHeight = 16;
        glm::vec3 tileGrid((imageWidth + BLOCK_X - 1) / BLOCK_X, (imageHeight + BLOCK_Y - 1) / BLOCK_Y, 1);
        uint32_t numTiles = tileGrid.x * tileGrid.y;

        // Start timing
        // Preprocess
        try {
            auto startAll = std::chrono::high_resolution_clock::now();

            auto startWaitForQueue = std::chrono::high_resolution_clock::now();
            queue.wait();
            std::chrono::duration<double, std::milli> waitForQueueDuration =
                    std::chrono::high_resolution_clock::now() - startWaitForQueue;

            queue.fill(imageBuffer, static_cast<uint8_t>(0x00), width * height * 4);

            size_t numPoints = m_numPoints;
            Rasterizer::PreprocessInfo scene{};
            scene.projectionMatrix = projectionMatrix;
            scene.viewMatrix = viewMatrix;
            scene.params = params;
            scene.camPos = camPos;
            scene.tileGrid = tileGrid;
            scene.height = height;
            scene.width = width;
            scene.shDim = m_shDim;
            scene.shDegree = 1.0;

            auto startPreprocess = std::chrono::high_resolution_clock::now();
            queue.fill(numTilesTouchedBuffer, 0x00, numPoints).wait();

            queue.submit([&](sycl::handler &h) {

                h.parallel_for(sycl::range<1>(numPoints), Rasterizer::Preprocess(positionBuffer, scalesBuffer,
                                                                                 quaternionBuffer, opacityBuffer,
                                                                                 sphericalHarmonicsBuffer,
                                                                                 numTilesTouchedBuffer,
                                                                                 pointsBuffer, &scene));
            }).wait();

            auto endPreprocess = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> preprocessDuration = endPreprocess - startPreprocess;
            auto startInclusiveSum = std::chrono::high_resolution_clock::now();
            //queue.fill(pointOffsets, 0x00, numPoints).wait();

            queue.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<1>(1),
                               Rasterizer::InclusiveSum(numTilesTouchedBuffer, pointOffsets, numPoints));
            }).wait();


            uint32_t numRendered = 1;
            queue.memcpy(&numRendered, pointOffsets + (numPoints - 1), sizeof(uint32_t)).wait();
            //printf("3DGS Rendering: Total Gaussians: %u. %uM\n", numRendered, numRendered / 1e6);

            Log::Logger::getInstance()->info("3DGS Rendering: Total Gaussians: {}. {:.3f}M", numRendered,
                                              numRendered / 1e6);
            if (numRendered > 0) {


                std::chrono::duration<double, std::milli> inclusiveSumDuration =
                        std::chrono::high_resolution_clock::now() - startInclusiveSum;

                auto startDuplicateGaussians = std::chrono::high_resolution_clock::now();

                queue.submit([&](sycl::handler &h) {
                    h.parallel_for<class duplicates>(sycl::range<1>(numPoints),
                                                     Rasterizer::DuplicateGaussians(pointsBuffer, pointOffsets,
                                                                                    keysBuffer,
                                                                                    valuesBuffer,
                                                                                    numRendered, tileGrid));
                }).wait();

                std::chrono::duration<double, std::milli> duplicateGaussiansDuration =
                        std::chrono::high_resolution_clock::now() - startDuplicateGaussians;
                auto startSorting = std::chrono::high_resolution_clock::now();

                /*
                uint32_t numKeys = 1 << 13;
                std::random_device rd;
                std::mt19937 gen(rd());
                gen.seed(42);
                std::uniform_int_distribution<uint32_t> dis(0, 1 << 5);

                std::vector<uint32_t> keys(numKeys);
                for (auto &key: keys) {
                    key = dis(gen);
                }
                queue.memcpy(keysBuffer, keys.data(), numKeys * sizeof(uint32_t));
                //queue.memcpy(keys.data(), keysBuffer, numRendered * sizeof(uint32_t)).wait();
                */

                // Load keys
                sorter->performOneSweep(keysBuffer, valuesBuffer, numRendered);
                //sorter->verifySort(keysBuffer, numRendered); //, true, keys);
                queue.wait();
                sorter->resetMemory();
                queue.wait();


                std::chrono::duration<double, std::milli> sortingDuration =
                        std::chrono::high_resolution_clock::now() - startSorting;

                auto startIdentifyTileRanges = std::chrono::high_resolution_clock::now();
                queue.memset(rangesBuffer, static_cast<int>(0x00), numTiles * (sizeof(int) * 2)).wait();

                queue.submit([&](sycl::handler &h) {
                    h.parallel_for<class IdentifyTileRanges>(numRendered,
                                                             Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer,
                                                                                            numRendered));
                }).wait();
                std::chrono::duration<double, std::milli> identifyTileRangesDuration =
                        std::chrono::high_resolution_clock::now() - startIdentifyTileRanges;

                auto startRenderGaussians = std::chrono::high_resolution_clock::now();
                // Compute the global work size ensuring it is a multiple of the local work size
                sycl::range<2> localWorkSize(tileHeight, tileWidth);
                size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
                size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
                sycl::range<2> globalWorkSize(globalHeight, globalWidth);
                uint32_t horizontal_blocks = (imageWidth + tileWidth - 1) / tileWidth;


                queue.submit([&](sycl::handler &h) {
                    auto range = sycl::nd_range<2>(globalWorkSize, localWorkSize);
                    h.parallel_for<class RenderGaussians>(range,
                                                          Rasterizer::RasterizeGaussians(rangesBuffer, keysBuffer,
                                                                                         valuesBuffer, pointsBuffer,
                                                                                         imageBuffer, numRendered,
                                                                                         imageWidth, imageHeight,
                                                                                         horizontal_blocks, numTiles));
                }).wait();


                /*
                queue.submit([&](sycl::handler &h) {
                    sycl::range<1> globalWorkSize(imageHeight * imageWidth * 4);
                    sycl::range<1> localWorkSize(32);

                    h.parallel_for<class RenderGaussians>(sycl::range<1>(width * height),
                                                          Rasterizer::RasterizeGaussiansPerPixel(rangesBuffer, keysBuffer,
                                                                                         valuesBuffer, pointsBuffer,
                                                                                         imageBuffer, numRendered,
                                                                                         imageWidth, imageHeight,
                                                                                         horizontal_blocks, numTiles));
                });
                */
                std::chrono::duration<double, std::milli> renderGaussiansDuration =
                        std::chrono::high_resolution_clock::now() - startRenderGaussians;


                auto startCopyImageToHost = std::chrono::high_resolution_clock::now();
                queue.memcpy(m_image, imageBuffer, width * height * 4);
                queue.wait();

                Rasterizer::saveAsPPM(m_image, width, height, "../output.ppm");
                std::chrono::duration<double, std::milli> copyImageDuration =
                        std::chrono::high_resolution_clock::now() - startCopyImageToHost;


                std::chrono::duration<double, std::milli> totalDuration =
                        std::chrono::high_resolution_clock::now() - startAll;

                bool exceedTimeLimit = (totalDuration.count() > 500);
                logTimes(waitForQueueDuration, preprocessDuration, inclusiveSumDuration, duplicateGaussiansDuration,
                         sortingDuration, identifyTileRangesDuration, renderGaussiansDuration, copyImageDuration,
                         totalDuration, exceedTimeLimit);

            } else {
                std::memset(m_image, static_cast<uint8_t>(0x00), width * height * 4);
            }

        } catch (sycl::exception &e) {
            std::cerr << "Caught a SYCL exception: " << e.what() << std::endl;
            return;
        } catch (std::exception &e) {
            std::cerr << "Caught a standard exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr << "Caught an unknown exception." << std::endl;
            return;
        }

        int rowSize = width * 4; // 4 bytes per pixel for RGBA8
        std::vector<uint8_t> tempRow(rowSize);

        for (int y = 0; y < height / 2; ++y) {
            uint8_t* row1 = m_image + y * rowSize;
            uint8_t* row2 = m_image + (height - y - 1) * rowSize;

            // Swap rows
            std::memcpy(tempRow.data(), row1, rowSize);
            std::memcpy(row1, row2, rowSize);
            std::memcpy(row2, tempRow.data(), rowSize);
        }

        m_textureVideo->updateTextureFromBuffer(m_image, width * height * 4);

    }

    void GaussianModelGraphicsPipeline::logTimes(std::chrono::duration<double, std::milli> t1,
                                    std::chrono::duration<double, std::milli> t2,
                                    std::chrono::duration<double, std::milli> t3,
                                    std::chrono::duration<double, std::milli> t4,
                                    std::chrono::duration<double, std::milli> t5,
                                    std::chrono::duration<double, std::milli> t6,
                                    std::chrono::duration<double, std::milli> t7,
                                    std::chrono::duration<double, std::milli> t8,
                                    std::chrono::duration<double, std::milli> t9,
                                    bool error) {
        if (error) {
            Log::Logger::getInstance()->error("3DGS Rendering: Wait for ready queue: {}", t1.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Preprocess: {}", t2.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Inclusive Sum: {}", t3.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Duplicate Gaussians: {}",
                                              t4.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Sorting: {}", t5.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Identify Tile Ranges: {}",
                                              t6.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Render Gaussians: {}", t7.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Copy image to host: {}", t8.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Total function duration: {}", t9.count());
        } else {
            Log::Logger::getInstance()->info("3DGS Rendering: Wait for ready queue: {}", t1.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Preprocess: {}", t2.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Inclusive Sum: {}", t3.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Duplicate Gaussians: {}",
                                              t4.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Sorting: {}", t5.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Identify Tile Ranges: {}",
                                              t6.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Render Gaussians: {}", t7.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Copy image to host: {}", t8.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Total function duration: {}", t9.count());
        }


    }

}