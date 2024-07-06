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


    void GaussianRenderer::setup(const VkRender::AbstractRenderer::InitializeInfo &initInfo) {
        m_initInfo = initInfo;

        try {
            // Create a queue using the CPU device selector
#ifdef GPU_ENABLED
            auto gpuSelector = [](const sycl::device &dev) {
            if (dev.is_gpu()) {
                return 1; // Positive value to prefer GPU devices
            } else {
                return -1; // Negative value to reject non-GPU devices
            }
        };
            queue = sycl::queue(gpuSelector, sycl::property::queue::in_order());
#else
            auto cpuSelector = [](const sycl::device &dev) {
                if (dev.is_cpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };    // Define a callable device selector using a lambda
            queue = sycl::queue(cpuSelector, sycl::property::queue::in_order());
#endif
            // Use the queue for your computation
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("GPU device not found");
            Log::Logger::getInstance()->info("Falling back to default device selector");
            // Fallback to default device selector
            queue = sycl::queue(sycl::property::queue::in_order());
        }


        Log::Logger::getInstance()->info("Selected Device {}",
                                         queue.get_device().get_info<sycl::info::device::name>().c_str());
        sorter = std::make_unique<Sorter>(queue, 1 << 23);

        gs = loadFromFile(Utils::getModelsPath() / "3dgs" / "coordinates.ply", 1);
        //gs = loadFromFile("/home/magnus/crl/multisense_viewer/3dgs_insect.ply", 1);
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


            width = camera->m_width;
            height = camera->m_height;
            imageBuffer = sycl::malloc_device<glm::uint8_t>(width * height * 4, queue);

            m_image = reinterpret_cast<uint8_t *>(std::malloc(width * height * 4));
            rangesBuffer = sycl::malloc_device<glm::ivec2>((width / 16) * (height / 16), queue);


            // Get the device associated with the queue
            sycl::device device = queue.get_device();

            // Query and print the maximum work group size
            size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
            std::cout << "Max work group size: " << max_work_group_size << std::endl;

            // Query and print the maximum number of work items per dimension
            sycl::id<2> max_work_item_sizes = device.get_info<sycl::info::device::max_work_item_sizes<2>>();
            std::cout << "Max work item sizes: "
                      << max_work_item_sizes[0] << " x "
                      << max_work_item_sizes[1] << std::endl;

            // Query and print the maximum number of compute units
            size_t max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
            std::cout << "Max compute units: " << max_compute_units << std::endl;

            auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            std::cout << "Sub group sizes: ";
            for (auto size: sub_group_sizes)
                std::cout << size << " ";
            std::cout << std::endl;

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


    void printDurations(
            const std::chrono::duration<double> &durationRasterization,
            const std::chrono::duration<double> &durationIdentification,
            const std::chrono::duration<double> &durationSorting,
            const std::chrono::duration<double> &durationAccumulation,
            const std::chrono::duration<double> &durationPreprocess,
            const std::chrono::duration<double> &durationInclusiveSum,
            const std::chrono::duration<double> &durationTotal) {
        // Convert durations to milliseconds
        auto durationRasterizationMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationRasterization);
        auto durationIdentificationMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationIdentification);
        auto durationSortingMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationSorting);
        auto durationAccumulationMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationAccumulation);
        auto durationPreprocessUs = std::chrono::duration_cast<std::chrono::microseconds>(durationPreprocess);
        auto durationInclusiveSumMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationInclusiveSum);
        auto durationTotalMs = std::chrono::duration_cast<std::chrono::milliseconds>(durationTotal);

        // Create an output string stream to format the output
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3); // Set fixed-point notation and precision

        oss << "Total Duration: " << durationTotalMs.count() << " ms\n";
        oss << "Rasterization: " << durationRasterizationMs.count() << " ms\n";
        oss << "Identification: " << durationIdentificationMs.count() << " ms\n";
        oss << "Sorting: " << durationSortingMs.count() << " ms\n";
        oss << "Accumulation: " << durationAccumulationMs.count() << " ms\n";
        oss << "Preprocessing: " << durationPreprocessUs.count() << " us\n";
        oss << "Inclusive Scan: " << durationInclusiveSumMs.count() << " ms\n\n";

        // Print the formatted string
        std::cout << oss.str();
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

        auto startAll = std::chrono::high_resolution_clock::now();
        // Start timing
        // Preprocess
        try {
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

            auto startPreprocess = std::chrono::high_resolution_clock::now();
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

            queue.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<1>(1),
                               Rasterizer::InclusiveSum(numTilesTouchedBuffer, pointOffsets, numPoints));
            });
            std::chrono::duration<double, std::milli> inclusiveSumDuration = std::chrono::high_resolution_clock::now() - startInclusiveSum;

            queue.wait();
            uint32_t numRendered = 0;
            queue.memcpy(&numRendered, pointOffsets + (numPoints - 1), sizeof(uint32_t));
            queue.wait();
            if (numRendered == 0)
                return;
            //auto durationInclusiveSum = inclusiveSum(&numRendered);
            keysBuffer = sycl::malloc_device<uint32_t>(numRendered, queue);
            valuesBuffer = sycl::malloc_device<uint32_t>(numRendered, queue);
            auto startDuplicateGaussians = std::chrono::high_resolution_clock::now();
            queue.submit([&](sycl::handler &h) {
                h.parallel_for<class duplicates>(sycl::range<1>(numPoints),
                                                 Rasterizer::DuplicateGaussians(pointsBuffer, pointOffsets, keysBuffer,
                                                                                valuesBuffer,
                                                                                numRendered, tileGrid));
            });
            queue.wait();
            std::chrono::duration<double, std::milli> duplicateGaussiansDuration = std::chrono::high_resolution_clock::now() - startDuplicateGaussians;

            auto startSorting = std::chrono::high_resolution_clock::now();

            // Sort gaussians
            {
                sorter->performOneSweep(keysBuffer, valuesBuffer, numRendered);
                queue.wait();
                std::vector<uint32_t> keys(numRendered);
            }
            std::chrono::duration<double, std::milli> sortingDuration = std::chrono::high_resolution_clock::now() - startSorting;


            auto startIdentifyTileRanges = std::chrono::high_resolution_clock::now();

            queue.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<1>(numTiles),
                               Rasterizer::IdentifyTileRangesInit(rangesBuffer));
            }).wait();

            queue.submit([&](sycl::handler &h) {
                h.parallel_for<class IdentifyTileRanges>(numRendered,
                                                         Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer,
                                                                                        numRendered));
            });
            std::chrono::duration<double, std::milli> identifyTileRangesDuration = std::chrono::high_resolution_clock::now() - startIdentifyTileRanges;

            //std::chrono::duration<double> totalDuration = std::chrono::high_resolution_clock::now() - startAll;
            //printDurations(durationRasterization, durationIdentification, durationSorting, durationAccumulation, durationPreprocess, durationInclusiveSum, totalDuration);
            //auto durationRasterization = rasterizeGaussians();
            auto startRenderGaussians = std::chrono::high_resolution_clock::now();

            // Compute the global work size ensuring it is a multiple of the local work size
            sycl::range<2> localWorkSize(tileHeight, tileWidth);
            size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
            size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
            sycl::range<2> globalWorkSize(globalHeight, globalWidth);
            uint32_t horizontal_blocks = (imageWidth + tileWidth - 1) / tileWidth;
            queue.wait();
            queue.submit([&](sycl::handler &h) {
                h.parallel_for<class RenderGaussians>(sycl::nd_range<2>(globalWorkSize, localWorkSize),
                                                      Rasterizer::RasterizeGaussians(rangesBuffer, keysBuffer,
                                                                                     valuesBuffer, pointsBuffer,
                                                                                     imageBuffer, numRendered,
                                                                                     imageWidth, imageHeight,
                                                                                     horizontal_blocks, numTiles));
            }).wait();
            std::chrono::duration<double, std::milli> renderGaussiansDuration = std::chrono::high_resolution_clock::now() - startRenderGaussians;


        // Copy back to host
        queue.memcpy(m_image, imageBuffer, width * height * 4).wait();
        sycl::free(keysBuffer, queue);
        sycl::free(valuesBuffer, queue);
        std::chrono::duration<double, std::milli> totalDuration = std::chrono::high_resolution_clock::now() - startAll;

        Log::Logger::getInstance()->trace("3DGS Rendering: Preprocess: {}", preprocessDuration.count());
        Log::Logger::getInstance()->trace("3DGS Rendering: Inclusive Sum: {}", inclusiveSumDuration.count());
        Log::Logger::getInstance()->trace("3DGS Rendering: Duplicate Gaussians: {}", duplicateGaussiansDuration.count());
        Log::Logger::getInstance()->trace("3DGS Rendering: Sorting: {}", sortingDuration.count());
        Log::Logger::getInstance()->trace("3DGS Rendering: Identify Tile Ranges: {}", identifyTileRangesDuration.count());
        Log::Logger::getInstance()->trace("3DGS Rendering: Render Gaussians: {}", renderGaussiansDuration.count());
        Log::Logger::getInstance()->trace("3DGS Rendering: Total function duration: {}", totalDuration.count());

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
                data.scales.emplace_back(expf(scaleBuffer[i * 3]), expf(scaleBuffer[i * 3 + 1]),
                                         expf(scaleBuffer[i * 3 + 2]));
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
