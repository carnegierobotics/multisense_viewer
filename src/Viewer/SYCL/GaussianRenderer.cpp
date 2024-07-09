//
// Created by magnus on 5/17/24.
//

#include "GaussianRenderer.h"


#include "Viewer/Tools/Utils.h"
#include "Rasterizer.h"
#include "Viewer/SYCL/radixsort/RadixSorter.h"

#include <filesystem>
#include <tinyply.h>
#include <glm/gtc/type_ptr.hpp>
#include <random>

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

namespace VkRender {

    uint8_t *GaussianRenderer::getImage() {
        return m_image;
    }

    uint32_t GaussianRenderer::getImageSize() {
        return m_initInfo.imageSize;
    }


    void GaussianRenderer::setup(const VkRender::AbstractRenderer::InitializeInfo &initInfo, bool useCPU) {
        m_initInfo = initInfo;

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

            queue = sycl::queue(useCPU ? cpuSelector : gpuSelector, sycl::property::queue::in_order());
            // Use the queue for your computation
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("GPU device not found");
            Log::Logger::getInstance()->info("Falling back to default device selector");
            // Fallback to default device selector
            queue = sycl::queue(sycl::property::queue::in_order());
        }


        Log::Logger::getInstance()->info("Selected Device {}",
                                         queue.get_device().get_info<sycl::info::device::name>().c_str());

        gs = loadFromFile(Utils::getModelsPath() / "3dgs" / "coordinates.ply", 1);
        //gs = loadFromFile("/home/magnus/crl/multisense_viewer/3dgs_insect.ply", 100);
        setupBuffers(initInfo.camera);
        //simpleRasterizer(camera, false);
    }

    void GaussianRenderer::clearBuffers() {
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

    void GaussianRenderer::setupBuffers(const VkRender::Camera *camera) {
        Log::Logger::getInstance()->info("Loaded {} Gaussians", gs.getSize());
        try {
            uint32_t numPoints = gs.getSize();

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
            sorter = std::make_unique<Sorter>(queue, sortBufferSize);

            width = camera->m_width;
            height = camera->m_height;
            imageBuffer = sycl::malloc_device<uint8_t>(width * height * 4, queue);

            m_image = reinterpret_cast<uint8_t *>(std::malloc(width * height * 4));
            rangesBuffer = sycl::malloc_device<glm::ivec2>((width / 16) * (height / 16), queue);

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


    void GaussianRenderer::render(const AbstractRenderer::RenderInfo &info, const VkRender::RenderUtils *renderUtils) {
        auto *camera = info.camera;
        auto params = Rasterizer::getHtanfovxyFocal(camera->m_Fov, camera->m_height, camera->m_width);
        glm::mat4 viewMatrix = camera->matrices.view;
        glm::mat4 projectionMatrix = camera->matrices.perspective;
        glm::vec3 camPos = camera->pose.pos;
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


            size_t numPoints = gs.getSize();
            Rasterizer::PreprocessInfo scene{};
            scene.projectionMatrix = projectionMatrix;
            scene.viewMatrix = viewMatrix;
            scene.params = params;
            scene.camPos = camPos;
            scene.tileGrid = tileGrid;
            scene.height = height;
            scene.width = width;
            scene.shDim = gs.getShDim();
            scene.shDegree = uint32_t(roundf(sqrtf(gs.getShDim())) - 1);

            auto startPreprocess = std::chrono::high_resolution_clock::now();
            queue.submit([&](sycl::handler &h) {

                h.parallel_for(sycl::range<1>(numPoints), Rasterizer::Preprocess(positionBuffer, scalesBuffer,
                                                                                 quaternionBuffer, opacityBuffer,
                                                                                 sphericalHarmonicsBuffer,
                                                                                 numTilesTouchedBuffer,
                                                                                 pointsBuffer, &scene));
            });
            auto endPreprocess = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> preprocessDuration = endPreprocess - startPreprocess;
            auto startInclusiveSum = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<1>(1),
                               Rasterizer::InclusiveSum(numTilesTouchedBuffer, pointOffsets, numPoints));
            });

            uint32_t numRendered = 0;
            queue.memcpy(&numRendered, pointOffsets + (numPoints - 1), sizeof(uint32_t)).wait();
            Log::Logger::getInstance()->trace("3DGS Rendering: Total Gaussians: {}. {:.3f}M", numRendered,
                                              numRendered / 1e6);
            if (numRendered == 0)
                return;

            std::chrono::duration<double, std::milli> inclusiveSumDuration =
                    std::chrono::high_resolution_clock::now() - startInclusiveSum;

            auto startDuplicateGaussians = std::chrono::high_resolution_clock::now();
            queue.submit([&](sycl::handler &h) {
                h.parallel_for<class duplicates>(sycl::range<1>(numPoints),
                                                 Rasterizer::DuplicateGaussians(pointsBuffer, pointOffsets, keysBuffer,
                                                                                valuesBuffer,
                                                                                numRendered, tileGrid));
            });
            std::chrono::duration<double, std::milli> duplicateGaussiansDuration =
                    std::chrono::high_resolution_clock::now() - startDuplicateGaussians;

            queue.fill(imageBuffer, static_cast<uint8_t>(0x00), width * height * 4);
            auto startSorting = std::chrono::high_resolution_clock::now();
            sorter->performOneSweep(keysBuffer, valuesBuffer, numRendered);
            //sorter->verifySort(keysBuffer, numRendered);
            std::chrono::duration<double, std::milli> sortingDuration =
                    std::chrono::high_resolution_clock::now() - startSorting;

            auto startIdentifyTileRanges = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<1>(numTiles),
                               Rasterizer::IdentifyTileRangesInit(rangesBuffer));
            });

            queue.wait();
            queue.submit([&](sycl::handler &h) {
                h.parallel_for<class IdentifyTileRanges>(numRendered,
                                                         Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer,
                                                                                        numRendered));
            });
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
            });


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
            queue.wait();
            // Copy back to host
            auto* numbersDevice = reinterpret_cast<uint8_t*>(malloc(width * height * 4));
            queue.memcpy(m_image, imageBuffer, width * height * 4);
            queue.wait();

            Rasterizer::saveAsPPM(m_image, width, height, "../output.ppm");
            std::chrono::duration<double, std::milli> copyImageDuration =
                    std::chrono::high_resolution_clock::now() - startCopyImageToHost;

            sorter->resetMemory();

            std::chrono::duration<double, std::milli> totalDuration =
                    std::chrono::high_resolution_clock::now() - startAll;

            bool exceedTimeLimit = (totalDuration.count() > 500);
            logTimes(waitForQueueDuration, preprocessDuration, inclusiveSumDuration, duplicateGaussiansDuration,
                     sortingDuration, identifyTileRangesDuration, renderGaussiansDuration, copyImageDuration,
                     totalDuration, exceedTimeLimit);

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
    }

    void GaussianRenderer::logTimes(std::chrono::duration<double, std::milli> t1,
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
            Log::Logger::getInstance()->trace("3DGS Rendering: Wait for ready queue: {}", t1.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Preprocess: {}", t2.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Inclusive Sum: {}", t3.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Duplicate Gaussians: {}",
                                              t4.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Sorting: {}", t5.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Identify Tile Ranges: {}",
                                              t6.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Render Gaussians: {}", t7.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Copy image to host: {}", t8.count());
            Log::Logger::getInstance()->trace("3DGS Rendering: Total function duration: {}", t9.count());
        }


    }

    GaussianRenderer::GaussianPoints GaussianRenderer::loadFromFile(std::filesystem::path path, int downSampleRate) {
        GaussianPoints data;
        //auto plyFilePath = std::filesystem::path("/home/magnus/phd/SuGaR/output/refined_ply/0000/3dgs.ply");
        auto plyFilePath = std::filesystem::path(path);

        // Open the PLY file
        std::ifstream ss(plyFilePath, std::ios::binary);
        if (!ss.is_open()) {
            throw std::runtime_error("Failed to open PLY file.");
        }

        tinyply::PlyFile file;
        file.parse_header(ss);

        std::shared_ptr<tinyply::PlyData> vertices, scales, quats, opacities, colors, harmonics;

        try { vertices = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { quats = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { opacities = file.request_properties_from_element("vertex", {"opacity"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
        // Request spherical harmonics properties
        std::vector<std::string> harmonics_properties;
        for (int i = 0; i < 45; ++i) {
            harmonics_properties.push_back("f_rest_" + std::to_string(i));
        }
        try { harmonics = file.request_properties_from_element("vertex", harmonics_properties); }
        catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


        file.read(ss);
        const size_t numVertices = vertices->count;

        // Process vertices
        if (vertices) {
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            std::vector<float> vertexBuffer(numVertices * 3);
            std::memcpy(vertexBuffer.data(), vertices->buffer.get(), numVerticesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.positions.emplace_back(vertexBuffer[i * 3], vertexBuffer[i * 3 + 1], vertexBuffer[i * 3 + 2]);
            }
        }

        // Process scales
        if (scales) {
            const size_t numScalesBytes = scales->buffer.size_bytes();
            std::vector<float> scaleBuffer(numVertices * 3);
            std::memcpy(scaleBuffer.data(), scales->buffer.get(), numScalesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                float sx = expf(scaleBuffer[i * 3]) + 0.02f;
                float sy = expf(scaleBuffer[i * 3 + 1]) + 0.02f;
                float sz = expf(scaleBuffer[i * 3 + 2]) + 0.02f;

                data.scales.emplace_back(sx, sy, sz);
            }
        }

        // Process quats
        if (quats) {
            const size_t numQuatsBytes = quats->buffer.size_bytes();
            std::vector<float> quatBuffer(numVertices * 4);
            std::memcpy(quatBuffer.data(), quats->buffer.get(), numQuatsBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.quats.emplace_back(quatBuffer[i * 4], quatBuffer[i * 4 + 1], quatBuffer[i * 4 + 2],
                                        quatBuffer[i * 4 + 3]);
            }
        }

        // Process opacities
        if (opacities) {
            const size_t numOpacitiesBytes = opacities->buffer.size_bytes();
            std::vector<float> opacityBuffer(numVertices);
            std::memcpy(opacityBuffer.data(), opacities->buffer.get(), numOpacitiesBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                float opacity = opacityBuffer[i];
                opacity = 1.0f / (1.0f + expf(-opacity));
                data.opacities.push_back(opacity);
            }
        }


        // Process colors and spherical harmonics
        /*
        if (colors && harmonics) {
            const size_t numColorsBytes = colors->buffer.size_bytes();
            std::vector<float> colorBuffer(numVertices * 3);
            std::memcpy(colorBuffer.data(), colors->buffer.get(), numColorsBytes);

            const size_t numHarmonicsBytes = harmonics->buffer.size_bytes();
            std::vector<float> harmonicsBuffer(numVertices * harmonics_properties.size());
            std::memcpy(harmonicsBuffer.data(), harmonics->buffer.get(), numHarmonicsBytes);

            // Extract DC components
            std::vector<float> features_dc(numVertices * 3);
            for (size_t i = 0; i < numVertices; ++i) {
                features_dc[i * 3 + 0] = colorBuffer[i * 3 + 0];
                features_dc[i * 3 + 1] = colorBuffer[i * 3 + 1];
                features_dc[i * 3 + 2] = colorBuffer[i * 3 + 2];
            }

            // Extract extra features
            std::vector<float> features_extra(numVertices * harmonics_properties.size());
            for (size_t i = 0; i < harmonics_properties.size(); ++i) {
                const size_t offset = i * numVertices;
                for (size_t j = 0; j < numVertices; ++j) {
                    features_extra[j * harmonics_properties.size() + i] = harmonicsBuffer[
                            j * harmonics_properties.size() + i];
                }
            }
            uint32_t max_sh_degree = 3;

            // Reshape and transpose features_extra
            const size_t sh_coeffs = (max_sh_degree + 1) * (max_sh_degree + 1) - 1;
            std::vector<float> reshaped_extra(numVertices * 3 * sh_coeffs);
            for (size_t i = 0; i < numVertices; ++i) {
                for (size_t j = 0; j < sh_coeffs; ++j) {
                    for (size_t k = 0; k < 3; ++k) {
                        reshaped_extra[(i * sh_coeffs + j) * 3 + k] = features_extra[(i * 3 + k) * sh_coeffs + j];
                    }
                }
            }

            // Combine features_dc and reshaped_extra
            data.sphericalHarmonics.resize(numVertices * (3 + sh_coeffs));
            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                for (size_t j = 0; j < 3; ++j) {
                    data.sphericalHarmonics.push_back(features_dc[i * 3 + j]);
                }
                for (size_t j = 0; j < sh_coeffs; ++j) {
                    for (size_t k = 0; k < 3; ++k) {
                        data.sphericalHarmonics.push_back(reshaped_extra[(i * sh_coeffs + j) * 3 + k]);
                    }
                }
            }

            data.shDim = 3 + harmonics_properties.size();
        }
         */

        // Process colors and spherical harmonics
        if (colors && harmonics) {
            const size_t numColorsBytes = colors->buffer.size_bytes();
            std::vector<float> colorBuffer(numVertices * 3);
            std::memcpy(colorBuffer.data(), colors->buffer.get(), numColorsBytes);

            const size_t numHarmonicsBytes = harmonics->buffer.size_bytes();
            std::vector<float> harmonicsBuffer(numVertices * harmonics_properties.size());
            std::memcpy(harmonicsBuffer.data(), harmonics->buffer.get(), numHarmonicsBytes);

            for (size_t i = 0; i < numVertices; i += downSampleRate) {
                data.sphericalHarmonics.push_back(colorBuffer[i * 3]);
                data.sphericalHarmonics.push_back(colorBuffer[i * 3 + 1]);
                data.sphericalHarmonics.push_back(colorBuffer[i * 3 + 2]);

                for (size_t j = 0; j < harmonics_properties.size(); ++j) {
                    data.sphericalHarmonics.push_back(harmonicsBuffer[i * harmonics_properties.size() + j]);
                }
            }

            data.shDim = 3 + harmonics_properties.size();
        }
        return data;
    }

    void GaussianRenderer::singleOneSweep() {

    }


}
