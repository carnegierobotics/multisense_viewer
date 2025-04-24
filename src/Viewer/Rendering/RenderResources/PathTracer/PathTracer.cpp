//
// Created by magnus on 11/27/24.
//

#include <utility>

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/LightSourceComponent.h"
#include "Viewer/Tools/SYCLDeviceSelector.h"

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer2DGSKernel.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer2DGSKernelBackward.h"

namespace VkRender::PathTracer {
    PhotonTracer::PhotonTracer(Application *ctx, const PipelineSettings &pipelineSettings,
                               std::shared_ptr<Scene> scene) : m_pipelineSettings(pipelineSettings),
                                                               m_context(ctx) {
        // Load the scene into gpu memory
        // Create image memory
        // Allocate host memory for RGBA image (4 floats per pixel)
        m_imageMemory = new float[pipelineSettings.width * pipelineSettings.height];
        m_renderInformation = std::make_unique<RenderInformation>();
        pipelineSettings.device().wait();
        prepareImageAndInfoBuffers();
        uploadGaussianData(scene);
        uploadQuadricEntities(scene);
        pipelineSettings.device().wait();
        m_gpu.numEntities = m_gpu.numQuadrics + m_gpu.numGaussians;

        m_backwardInfo.photonIDGradient = new glm::mat3[m_pipelineSettings.photonCount];
        m_backwardInfo.gradientPixelCoordinates = new glm::vec2[m_pipelineSettings.photonCount];
        m_backwardInfo.gradientImagePerObject =new float[pipelineSettings.width * pipelineSettings.height];
        m_backwardInfo.gradientImageHoriz =new float[pipelineSettings.width * pipelineSettings.height];
        m_backwardInfo.gradientImageVert =new float[pipelineSettings.width * pipelineSettings.height];
        m_backwardInfo.pixelHitCounter =new float[pipelineSettings.width * pipelineSettings.height];

        Log::Logger::getInstance()->info(
            "PathTracer created, Propterties: PhotonCount: {}, Bounces: {}, Image Size: {}x{}",
            m_pipelineSettings.photonCount, m_pipelineSettings.numBounces, m_pipelineSettings.width,
            m_pipelineSettings.height);
        Log::Logger::getInstance()->info("PathTracer on Device: {}",
                                         m_pipelineSettings.device().get_device().get_info<sycl::info::device::name>().
                                         c_str());
    }

    void PhotonTracer::update(RenderSettings &renderSettings) {
        try {
            auto &queue = m_pipelineSettings.device();
            auto start = std::chrono::steady_clock::now();

            // Update shared GPU/CPU render information
            m_renderInformation->frameID++;
            m_renderInformation->totalPhotons += m_pipelineSettings.photonCount;
            m_renderInformation->gamma = renderSettings.gammaCorrection;
            m_renderInformation->numBounces = m_pipelineSettings.numBounces;
            m_renderInformation->applyBetaWeight = renderSettings.applyBetaContribution;
            Log::Logger::getInstance()->trace("Path Tracer: Uploading Render Information");

            queue.fill(m_gpu.imageMemoryCounter, 0.0f, m_pipelineSettings.width * m_pipelineSettings.height);
            queue.fill(m_gpu.imageMemory, 0.0f, m_pipelineSettings.width * m_pipelineSettings.height);

            queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
            queue.memcpy(m_gpu.pinholeCamera, &renderSettings.camera, sizeof(PinholeCamera));
            queue.memcpy(m_gpu.cameraTransform, &renderSettings.cameraTransform, sizeof(TransformComponent));

            GPUDataOutput output{};
            queue.fill(m_gpuDataOutput, output, m_pipelineSettings.photonCount);

            queue.wait();
            // Kernel Launch
            sycl::range<1> globalRange(m_pipelineSettings.photonCount);
            Log::Logger::getInstance()->trace("Path Tracer: Submitting Kernels");

            if (m_gpu.numGaussians > 0) {
                queue.submit([gpu = m_gpu, gpuOutput = m_gpuDataOutput, random = m_pcg32, photonCount = m_pipelineSettings.photonCount](sycl::handler &cgh) {
                    sycl::range<1> globalRange(photonCount);

                    LightTracerKernel kernel(gpu, gpuOutput, random);
                    cgh.parallel_for(globalRange, kernel);
                });
            }

            queue.wait();
            uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class AverageImageKernel>(
                    sycl::range<1>(imageSize),
                    [gpu = m_gpu, imageSize](sycl::id<1> idx) {
                        size_t pixelIndex = idx[0];
                        if (pixelIndex > imageSize) {
                            return;
                        }
                        // Read the counter value for the pixel.
                        float count = gpu.imageMemoryCounter[pixelIndex];
                        // Only average if the pixel was hit at least once.
                        if (count > 0.0f) {
                            float newContribution = gpu.imageMemory[pixelIndex] / count;
                            //float newContribution = gpu.imageMemory[pixelIndex];
                            gpu.imageMemoryPersistent[pixelIndex] += newContribution;

                            gpu.imageMemoryPersistent[pixelIndex] = std::min(gpu.imageMemoryPersistent[pixelIndex], 1.0f);

                        } else {
                            // Optionally, you can set pixels with no hits to 0.
                            //m_gpu.imageMemory[pixelIndex] = 0.0f;
                        }
                    }
                );
            });


            queue.wait();

            // Retrieve updated information from GPU
            queue.memcpy(m_renderInformation.get(), m_gpu.renderInformation, sizeof(RenderInformation));
            queue.memcpy(m_imageMemory, m_gpu.imageMemoryPersistent, m_pipelineSettings.width * m_pipelineSettings.height * sizeof(float));
            queue.memcpy(m_backwardInfo.pixelHitCounter, m_gpu.imageMemoryCounter, m_pipelineSettings.width * m_pipelineSettings.height * sizeof(float));
            queue.wait();

            auto end = std::chrono::steady_clock::now();
            std::cout << "SYCL kernel time: "
                      << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

            double totalM = static_cast<double>(m_renderInformation->totalPhotons) / 1e6;
            double sensorK = static_cast<double>(m_renderInformation->photonsAccumulated) / 1000.0;
            Log::Logger::getInstance()->trace(
                "Path Tracer:  Simulated {}M photons. About {}k photons hit the sensor",
                totalM, sensorK);
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("Caught exception: {}", e.what());
            std::cerr << "Exception: " << e.what() << std::endl;
            throw std::runtime_error("Caught exception");
        }
    }

    PhotonTracer::BackwardInfo PhotonTracer::backward(RenderSettings &renderSettings) {
        try {
            auto start = std::chrono::steady_clock::now();

            auto &queue = m_pipelineSettings.device();
            uint64_t simulatePhotonCount = m_pipelineSettings.photonCount;
            uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;


            m_renderInformation->totalPhotons += m_pipelineSettings.photonCount;
            m_renderInformation->gamma = renderSettings.gammaCorrection;
            m_renderInformation->numBounces = m_pipelineSettings.numBounces;
            queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
            queue.memcpy(m_gpu.pinholeCamera, &renderSettings.camera, sizeof(PinholeCamera));
            queue.memcpy(m_gpu.cameraTransform, &renderSettings.cameraTransform, sizeof(TransformComponent));
            queue.fill(m_gpu.photonIDGradient, glm::mat3(0.0f),  m_pipelineSettings.photonCount);
            queue.fill(m_gpu.gradientPixelCoordinates, glm::vec2(0.0f),  m_pipelineSettings.photonCount);
            queue.fill(m_gpu.gradientImagePerObject, FLT_MAX, imageSize);
            queue.fill(m_gpu.gradientImageU, 0.0f, imageSize);
            queue.fill(m_gpu.gradientImageV, 0.0f, imageSize);

            queue.wait();
            sycl::range<1> globalRange(simulatePhotonCount);
            queue.submit([&](sycl::handler &cgh) {
                // Capture GPUData, etc. by value or reference as needed
                LightTracerKernelBackward kernel(m_gpu, m_gpuDataOutput, m_pcg32);
                cgh.parallel_for(globalRange, kernel);
            });

            queue.wait();
            queue.memcpy(m_backwardInfo.gradientImagePerObject, m_gpu.gradientImagePerObject, imageSize * sizeof(float));
            queue.memcpy(m_backwardInfo.photonIDGradient, m_gpu.photonIDGradient, sizeof(glm::mat3) *  m_pipelineSettings.photonCount);
            queue.memcpy(m_backwardInfo.gradientPixelCoordinates, m_gpu.gradientPixelCoordinates, sizeof(glm::vec2) *  m_pipelineSettings.photonCount);
            queue.memcpy(m_backwardInfo.gradientImageHoriz, m_gpu.gradientImageU, sizeof(float) * imageSize);
            queue.memcpy(m_backwardInfo.gradientImageVert, m_gpu.gradientImageV, sizeof(float) * imageSize);

            queue.wait();

            auto end = std::chrono::steady_clock::now();
            std::cout << "SYCL Backwards kernel time: "
                      << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
        return m_backwardInfo;
    }

    void PhotonTracer::resetImage() {
        Log::Logger::getInstance()->trace("Resetting Image...");
        auto &queue = m_pipelineSettings.device();
        queue.fill(m_gpu.imageMemory, static_cast<float>(0),
                   m_pipelineSettings.width * m_pipelineSettings.height).wait();
        queue.fill(m_gpu.imageMemoryPersistent, static_cast<float>(0),
                   m_pipelineSettings.width * m_pipelineSettings.height).wait();
        queue.fill(m_gpu.imageMemoryCounter, static_cast<float>(0),
                   m_pipelineSettings.width * m_pipelineSettings.height).wait();
        m_renderInformation->frameID = 0;
        m_renderInformation->photonsAccumulated = 0;
        m_renderInformation->totalPhotons = 0;
        queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation)).wait();
        Log::Logger::getInstance()->trace("Image successfully reset");
    }

    void PhotonTracer::prepareImageAndInfoBuffers() {
        uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;
        auto &queue = m_pipelineSettings.device();
        // Allocate device memory for RGBA image (4 floats per pixel)
        m_gpu.imageMemory = sycl::malloc_device<float>(imageSize, queue);
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        // Allocate device memory for RGBA image (4 floats per pixel)
        m_gpu.imageMemoryPersistent = sycl::malloc_device<float>(imageSize, queue);
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        // Allocate device photon hit counter
        m_gpu.imageMemoryCounter = sycl::malloc_device<float>(imageSize, queue);
        if (!m_gpu.imageMemoryCounter) {
            throw std::runtime_error("Device memory allocation failed.");
        }

        m_gpu.renderInformation = sycl::malloc_device<RenderInformation>(1, queue);
        if (!m_gpu.renderInformation) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
        // Initialize device memory to 0
        // Initialize host memory to 0
        // Initialize RNGs
        // Generate a random seed using std::random_device.
        size_t simulatePhotonCount = m_pipelineSettings.photonCount;
        // Seed a fast PRNG using std::random_device only once
        std::random_device rd;
        std::mt19937_64 engine(rd());
        std::uniform_int_distribution<uint64_t> dist;

        std::vector<PCG32> rng(simulatePhotonCount);
        for (uint64_t i = 0; i < simulatePhotonCount; ++i) {
            uint64_t randomNumber = dist(engine);
            rng[i].init(randomNumber, i);
        }
        // Allocate and copy RNG to GPU
        m_pcg32 = sycl::malloc_device<PCG32>(simulatePhotonCount, queue);
        queue.memcpy(m_pcg32, rng.data(), sizeof(PCG32) * simulatePhotonCount);

        m_gpu.pinholeCamera = sycl::malloc_device<PinholeCamera>(1, queue);
        m_gpu.cameraTransform = sycl::malloc_device<TransformComponent>(1, queue);
        m_gpuDataOutput = sycl::malloc_device<GPUDataOutput>(simulatePhotonCount, queue);
        GPUDataOutput output{};

        uint32_t numGaussiansMemory = 1;
        uint32_t numQuadricsMemory = 3;
        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(numGaussiansMemory, queue);
        m_gpu.quadricInputAssembly = sycl::malloc_device<QuadricInputAssembly>(numQuadricsMemory, queue);
        m_gpu.photonIDGradient = sycl::malloc_device<glm::mat3>(m_pipelineSettings.photonCount, queue);
        m_gpu.gradientPixelCoordinates = sycl::malloc_device<glm::vec2>(m_pipelineSettings.photonCount, queue);
        m_gpu.gradientImageU = sycl::malloc_device<float>(imageSize, queue);
        m_gpu.gradientImageV = sycl::malloc_device<float>(imageSize, queue);
        m_gpu.gradientImagePerObject = sycl::malloc_device<float>(imageSize, queue);

        queue.fill(m_gpu.gradientImagePerObject, 0.0f, imageSize);
        queue.fill(m_gpu.gradientImageV, 0.0f, imageSize);
        queue.fill(m_gpu.gradientImageU, 0.0f, imageSize);
        queue.fill(m_gpu.gradientPixelCoordinates, glm::vec2(0.0f), m_pipelineSettings.photonCount);
        queue.fill(m_gpu.photonIDGradient, glm::mat3(0.0f), m_pipelineSettings.photonCount);
        queue.fill(m_gpuDataOutput, output, simulatePhotonCount);
        queue.fill(m_gpu.imageMemory, 0.0f, imageSize);
        queue.fill(m_gpu.imageMemoryCounter, 0.0f, imageSize);

        queue.fill(m_gpu.imageMemoryPersistent, 0.0f, imageSize);

        queue.wait();
    }


    void PhotonTracer::freeResources() {
        auto &queue = m_pipelineSettings.device();
        Log::Logger::getInstance()->info("Freeing Path tracer GPU/SYCL resources");

        queue.wait();
        if (m_gpu.imageMemory) {
            sycl::free(m_gpu.imageMemory, queue);
            m_gpu.imageMemory = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: imageMemory");
        }
        if (m_gpu.imageMemoryCounter) {
            sycl::free(m_gpu.imageMemoryCounter, queue);
            m_gpu.imageMemoryCounter = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: imageMemory");
        }
        if (m_gpu.imageMemoryPersistent) {
            sycl::free(m_gpu.imageMemoryPersistent, queue);
            m_gpu.imageMemoryPersistent = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: imageMemory");
        }

        if (m_gpu.gaussianInputAssembly) {
            sycl::free(m_gpu.gaussianInputAssembly, queue);
            m_gpu.gaussianInputAssembly = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: gaussianInputAssembly");
        }
        if (m_gpu.quadricInputAssembly) {
            sycl::free(m_gpu.quadricInputAssembly, queue);
            m_gpu.quadricInputAssembly = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: quadricInputAssembly");
        }
        if (m_gpu.bvhNodes) {
            sycl::free(m_gpu.bvhNodes, queue);
            m_gpu.bvhNodes = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: quadricInputAssembly");
        }
        if (m_gpu.gradientPixelCoordinates) {
            sycl::free(m_gpu.gradientPixelCoordinates, queue);
            m_gpu.gradientPixelCoordinates = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: gradientPixelCoordinates");
        }
        if (m_gpu.photonIDGradient) {
            sycl::free(m_gpu.photonIDGradient, queue);
            m_gpu.photonIDGradient = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: photonIDGradient");
        }
        if (m_gpu.gradientImageU) {
            sycl::free(m_gpu.gradientImageU, queue);
            m_gpu.gradientImageU = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: gradientImageU");
        }
        if (m_gpu.gradientImageV) {
            sycl::free(m_gpu.gradientImageV, queue);
            m_gpu.gradientImageV = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: gradientImageV");
        }
        if (m_gpu.gradientImagePerObject) {
            sycl::free(m_gpu.gradientImagePerObject, queue);
            m_gpu.gradientImagePerObject = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: gradientImagePerObject");
        }
        if (m_gpu.pinholeCamera) {
            sycl::free(m_gpu.pinholeCamera, queue);
            m_gpu.pinholeCamera = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: pinholeCamera");
        }
        if (m_gpu.cameraTransform) {
            sycl::free(m_gpu.cameraTransform, queue);
            m_gpu.cameraTransform = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: cameraTransform");
        }
        if (m_gpu.renderInformation) {
            sycl::free(m_gpu.renderInformation, queue);
            m_gpu.renderInformation = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: renderInformation");
        }
        if (m_pcg32) {
            sycl::free(m_pcg32, queue);
            m_pcg32 = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: pcg32");
        }
        if (m_gpuDataOutput) {
            sycl::free(m_gpuDataOutput, queue);
            m_gpuDataOutput = nullptr;
            Log::Logger::getInstance()->trace("Freed GPU Memory: gpuDataOutput");
        }
        queue.wait();

        Log::Logger::getInstance()->info("Freed Path tracer GPU/SYCL resources");
    }


    void PhotonTracer::uploadGaussiansFromTensors(GPUDataTensors &data) {
#ifdef DIFF_RENDERER_ENABLED

        auto &queue = m_pipelineSettings.device();
        uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;
        queue.fill(m_gpu.imageMemoryPersistent, 0.0f, imageSize);

        // 2) Move Tensors to CPU (if they aren't already) so we can extract values
        //    (SYCL can't just copy directly from a PyTorch CUDA device pointer.)
        //    If data is already on CPU, this .cpu() will be basically a no-op.
        auto positionsCpu = data.positions.cpu();
        auto scalesCpu = data.scales.cpu();
        auto normalsCpu = data.normals.cpu();

        auto emissionsCpu = data.emissions.cpu();
        auto colorsCpu = data.colors.cpu();
        auto specularCpu = data.specular.cpu();
        auto diffuseCpu = data.diffuse.cpu();

        // 3) Some basic sanity checks on tensor shapes
        //    Here we assume:
        //      positions: [N, 3]
        //      scales:    [N, 1] or just [N]
        //      normals:   [N, 3]
        TORCH_CHECK(positionsCpu.dim() == 2 && positionsCpu.size(1) == 3,
                    "positions must have shape [N,3]");
        TORCH_CHECK(normalsCpu.dim() == 2 && normalsCpu.size(1) == 3,
                    "normals must have shape [N,3]");

        // For scales, accept either [N] or [N,1]
        TORCH_CHECK((scalesCpu.dim() == 2 && scalesCpu.size(1) == 2),
                    "scales must have shape [N,2]");

        // 4) Get the number of gaussians (N)
        const auto numGaussians = positionsCpu.size(0);

        // 5) Create a host vector of GaussianInputAssembly
        std::vector<GaussianInputAssembly> hostGaussians(numGaussians);

        // 6) Pointers to the underlying float data (on CPU).
        //    We'll read them row-by-row.
        //    Example: positionsCpu.data_ptr<float>() returns a pointer to the 2D array in row-major order.
        const float *posPtr = positionsCpu.data_ptr<float>();
        const float *scalesPtr = scalesCpu.data_ptr<float>();
        const float *normalsPtr = normalsCpu.data_ptr<float>();

        const float *emissionsPtr = emissionsCpu.data_ptr<float>();
        const float *colorsPtr = colorsCpu.data_ptr<float>();
        const float *specularPtr = specularCpu.data_ptr<float>();
        const float *diffusePtr = diffuseCpu.data_ptr<float>();

        // 7) Fill the hostGaussians array
        //    (positions = 3 floats, normals = 3 floats, scale = 1 float, plus defaults)
        for (int i = 0; i < numGaussians; ++i) {
            GaussianInputAssembly point{};

            // positions: [i,0..2]
            point.position.x = posPtr[i * 3 + 0];
            point.position.y = posPtr[i * 3 + 1];
            point.position.z = posPtr[i * 3 + 2];

            // scales: either shape [N,1] or [N].
            // If [N,1], then index is i*1, else it's i
            if (scalesCpu.dim() == 2) {
                point.scale.x = scalesPtr[i * 2 + 0];
                point.scale.y = scalesPtr[i * 2 + 1];
            }

            // normals: [i,0..2]
            point.normal.x = normalsPtr[i * 3 + 0];
            point.normal.y = normalsPtr[i * 3 + 1];
            point.normal.z = normalsPtr[i * 3 + 2];

            // Update colors (each is 4 floats)
            point.color = glm::vec4(
                colorsPtr[i * 4 + 0],
                colorsPtr[i * 4 + 1],
                colorsPtr[i * 4 + 2],
                colorsPtr[i * 4 + 3]
            );

            // Fill default appearance properties
            point.emission = emissionsPtr[i]; // emission = 0
            point.diffuse = diffusePtr[i]; // diffuse = 0.5
            point.specular = specularPtr[i]; // specular = 0.5
            point.phongExponent = 32; // phongExponent = 32

            hostGaussians[i] = point;
        }

        // 8) Allocate device memory and copy
        queue.memcpy(m_gpu.gaussianInputAssembly, hostGaussians.data(), numGaussians * sizeof(GaussianInputAssembly));

        // 9) Set number of gaussians
        m_gpu.numGaussians = numGaussians;

        // Log
        Log::Logger::getInstance()->info("uploadFromTensors: Uploaded {} Gaussians", numGaussians);

        // Upload QUadrics
        float *quadricsPtr = data.quadrics.cpu().data_ptr<float>(); // shape: [numGaussians]
        float *quadricsPosPtr = data.quadricPositions.cpu().data_ptr<float>(); // shape: [numGaussians]
        float *quadricsRotPtr = data.quadricRotations.cpu().data_ptr<float>(); // shape: [numGaussians]

        const auto numQuadrics = data.quadricPositions.size(0);

        std::vector<QuadricInputAssembly> quadricInputAssembly;
        for (int i = 0; i < numQuadrics; ++i) {
            QuadricInputAssembly point{};
            point.a = quadricsPtr[i * 12 + 0];
            point.b = quadricsPtr[i * 12 + 1];
            point.c = quadricsPtr[i * 12 + 2];
            point.t_x = quadricsPtr[i * 12 + 3];
            point.t_y = quadricsPtr[i * 12 + 4];
            point.b_beta = quadricsPtr[i * 12 + 5];
            point.threshold = quadricsPtr[i * 12 + 6];
            point.kernelScale = quadricsPtr[i * 12 + 7];

            point.min.x = quadricsPtr[i * 12 + 8];
            point.max.x = quadricsPtr[i * 12 + 9];
            point.min.y = quadricsPtr[i * 12 + 10];
            point.max.y = quadricsPtr[i * 12 + 11];


            point.emission = 0.0f;
            point.diffuse = 1.0f;
            point.specular = 0.0f;
            point.phongExponent = 32.0f;
            point.color = glm::vec4(0.8f);

            glm::vec3 translation = {quadricsPosPtr[i * 3 + 0], quadricsPosPtr[i * 3 + 1], quadricsPosPtr[i * 3 + 2]};
            point.transform.setPosition(translation);
            glm::quat quat = {
                quadricsRotPtr[i * 4 + 0], quadricsRotPtr[i * 4 + 1], quadricsRotPtr[i * 4 + 2],
                quadricsRotPtr[i * 4 + 3]
            };
            point.transform.setRotationQuaternion(quat);

            quadricInputAssembly.push_back(point);
        }

        queue.memcpy(m_gpu.quadricInputAssembly, quadricInputAssembly.data(),
                     quadricInputAssembly.size() * sizeof(QuadricInputAssembly));
        m_gpu.numQuadrics = quadricInputAssembly.size(); // Number of entities for rendering

        size_t numEntities = quadricInputAssembly.size() + numGaussians;
        m_gpu.numEntities = numEntities;

        Log::Logger::getInstance()->info("Uploaded  {} Quadrics to renderkernel from Tensor", m_gpu.numQuadrics);

        // Build BVH leaves from quadrics.
        auto leaves = buildBVHLeaves(quadricInputAssembly);
        // Build the BVH nodes.
        auto bvhNodes = buildBVH(leaves);

        // Allocate device memory for BVH nodes.
        size_t bvhSize = bvhNodes.size();
        m_gpu.bvhNodes = sycl::malloc_device<BVHNode>(bvhSize, queue);

        // Copy the BVH nodes from host to device.
        queue.memcpy(m_gpu.bvhNodes, bvhNodes.data(), bvhSize * sizeof(BVHNode));

        m_gpu.numBVHNodes = bvhSize;
        Log::Logger::getInstance()->info("Uploaded {} BVH nodes for Quadrics", bvhSize);
        queue.wait();

#endif
    }

    void PhotonTracer::uploadGaussianData(std::shared_ptr<Scene> &scene) {
        auto &queue = m_pipelineSettings.device();
        std::vector<GaussianInputAssembly> gaussianInputAssembly;
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        // Find all entities with GaussianComponent
        auto view = scene->getRegistry().view<LightSourceComponent>();
        for (auto e: view) {
            auto &component = Entity(e, scene.get()).getComponent<LightSourceComponent>();
            for (size_t i = 0; i < component.size(); ++i) {
                GaussianInputAssembly point{};
                point.position = component.positions[i];
                point.scale = component.scales[i];
                point.normal = component.normals[i];

                point.emission = component.emissions[i];
                point.color = component.colors[i];
                point.diffuse = component.diffuse[i];
                point.specular = component.specular[i];
                point.phongExponent = component.phongExponents[i];
                gaussianInputAssembly.push_back(point);
            }
            auto &transform = Entity(e, scene.get()).getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);
        }

        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(gaussianInputAssembly.size(), queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, gaussianInputAssembly.data(),
                     gaussianInputAssembly.size() * sizeof(GaussianInputAssembly));
        m_gpu.numGaussians = gaussianInputAssembly.size(); // Number of entities for rendering

        Log::Logger::getInstance()->info("Uploaded  {} Gaussians to renderkernel", m_gpu.numGaussians);
        queue.wait();
    }

    void PhotonTracer::uploadQuadricEntities(std::shared_ptr<Scene> &scene) {
        auto &queue = m_pipelineSettings.device();
        std::vector<QuadricInputAssembly> quadricInputAssembly;
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        // Find all entities with GaussianComponent
        auto view = scene->getRegistry().view<MeshComponent, MaterialComponent>();
        for (auto e: view) {
            auto &component = Entity(e, scene.get()).getComponent<MeshComponent>();
            auto &material = Entity(e, scene.get()).getComponent<MaterialComponent>();
            if (component.meshDataType() == QUADRIC) {
                auto parameters = std::dynamic_pointer_cast<QuadricMeshParameters>(component.meshParameters);
                if (!parameters) {
                    continue;
                }
                QuadricInputAssembly point{};
                point.a = parameters->a;
                point.b = parameters->b;
                point.c = parameters->c;
                point.t_x = parameters->t_x;
                point.t_y = parameters->t_y;
                point.min = parameters->min;
                point.max = parameters->max;

                point.b_beta = parameters->b_beta;
                point.threshold = parameters->threshold;
                point.kernelScale = parameters->kernelScale;

                point.emission = material.emission;
                point.color = material.albedo;
                point.diffuse = material.diffuse;
                point.specular = material.specular;
                point.phongExponent = material.phongExponent;
                auto &transform = Entity(e, scene.get()).getComponent<TransformComponent>();
                point.transform = transform;

                quadricInputAssembly.push_back(point);
            }
        }

        m_gpu.quadricInputAssembly = sycl::malloc_device<QuadricInputAssembly>(quadricInputAssembly.size(), queue);
        queue.memcpy(m_gpu.quadricInputAssembly, quadricInputAssembly.data(),
                     quadricInputAssembly.size() * sizeof(QuadricInputAssembly));
        m_gpu.numQuadrics = quadricInputAssembly.size(); // Number of entities for rendering

        Log::Logger::getInstance()->info("Uploaded  {} Quadrics to renderkernel", m_gpu.numQuadrics);
        queue.wait();

        // Build BVH leaves from quadrics.
        auto leaves = buildBVHLeaves(quadricInputAssembly);
        // Build the BVH nodes.
        auto bvhNodes = buildBVH(leaves);

        // Allocate device memory for BVH nodes.
        size_t bvhSize = bvhNodes.size();
        m_gpu.bvhNodes = sycl::malloc_device<BVHNode>(bvhSize, queue);

        // Copy the BVH nodes from host to device.
        queue.memcpy(m_gpu.bvhNodes, bvhNodes.data(), bvhSize * sizeof(BVHNode));

        m_gpu.numBVHNodes = bvhSize;

        Log::Logger::getInstance()->info("Uploaded {} BVH nodes for Quadrics", bvhSize);
        queue.wait();
    }


    float PhotonTracer::computeLocalZ(float x, float y, const QuadricInputAssembly &quadric) {
        // Example quadric function.
        // Adjust this to your actual quadric function.
        float alphaX = std::tanh(quadric.t_x);
        float alphaY = std::tanh(quadric.t_y);
        // A simple quadratic form—modify as needed.
        return quadric.c * (alphaX * (x * x) / (quadric.a * quadric.a) +
                            alphaY * (y * y) / (quadric.b * quadric.b));
    }

    PhotonTracer::AABB PhotonTracer::computeLocalAABB(const QuadricInputAssembly &quadric) {
        // Initialize the AABB bounds for the kept vertices.
        glm::vec3 aabbMin(std::numeric_limits<float>::max());
        glm::vec3 aabbMax(std::numeric_limits<float>::lowest());

        float scaleFactor = 1.0f;
        // Helper lambda for Beta kernel:
        auto betaKernel = [&](float r, float bExp) {
            // (1 - r^2)^(4 e^bExp), clipped if r>1
            if (r > 1.0f) r = 1.0f;
            return std::pow(1.0f - r * r, 4.0f * std::exp(bExp));
        };

        auto getVertex = [&](glm::vec3 v) {

        };

        int N = 100; // Grid resoltuion
        float dx = (quadric.max.x - quadric.min.x) / float(N - 1);
        float dy = (quadric.max.y - quadric.min.y) / float(N - 1);

        // Precompute the sign factors alpha_x, alpha_y

        for (int i = 0; i < N; ++i) {
            float x = quadric.min.x + i * dx; // domain from min.x to max.x
            for (int j = 0; j < N; ++j) {
                float y = quadric.min.y + j * dy; // domain from min.y to max.y
                float zSquared = quadric.a * quadric.a *(x*x + y * y);
                float z = sqrtf(zSquared);
                // Build position
                glm::vec3 position(x, y, z);
                // Apply scale factor if needed.
                position *= scaleFactor;
                // Compute radial coordinate for Beta kernel:
                // R_general = sqrt(|alphaX|*x^2/a^2 + |alphaY|*y^2/b^2)
                float geodesic = sqrtf(std::pow(x, 2.0f) + std::pow(y, 2.0f) + std::pow(z, 2.0f));
                // Evaluate kernel
                float bkValue = betaKernel(geodesic, quadric.b_beta);
                // Decide whether to keep the vertex based on the threshold
                bool keepVertex = (bkValue > quadric.threshold);
                if (!keepVertex) {
                    continue;
                }
                // Update the AABB using the current vertex's position.
                aabbMin = glm::min(aabbMin, position);
                aabbMax = glm::max(aabbMax, position);
            }
        }

        float x = 0; // domain from min.x to max.x
        float y = 0; // domain from min.y to max.y
        float zSquared = quadric.a * quadric.a *(x*x + y * y);
        float z = sqrtf(zSquared);
        // Build position
        glm::vec3 position(x, y, z);
        // Apply scale factor if needed.
        position *= scaleFactor;
        // Compute radial coordinate for Beta kernel:
        // R_general = sqrt(|alphaX|*x^2/a^2 + |alphaY|*y^2/b^2)
        float geodesic = sqrtf(std::pow(x, 2.0f) + std::pow(y, 2.0f) + std::pow(z, 2.0f));
        // Evaluate kernel
        float bkValue = betaKernel(geodesic, quadric.b_beta);
        // Decide whether to keep the vertex based on the threshold
        bool keepVertex = (bkValue > quadric.threshold);
        if (keepVertex) {
            aabbMin = glm::min(aabbMin, position);
            aabbMax = glm::max(aabbMax, position);
        }


        // If the quadric is planar (c == 0), the z extents collapse.
        // Artificially expand the z bounds by an epsilon.
        if (quadric.c == 0.0f) {
            const float epsilon = 0.01f; // Adjust this value as needed.
            aabbMin.z -= epsilon;
            aabbMax.z += epsilon;
        }
        // The tightly bound AABB for the kept vertices.
        return {aabbMin, aabbMax};
    }

    PhotonTracer::AABB transformAABB(const PhotonTracer::AABB &localBox, const glm::mat4 &transform) {
        std::array<glm::vec3, 8> localCorners = {
            glm::vec3(localBox.min.x, localBox.min.y, localBox.min.z),
            glm::vec3(localBox.min.x, localBox.min.y, localBox.max.z),
            glm::vec3(localBox.min.x, localBox.max.y, localBox.min.z),
            glm::vec3(localBox.min.x, localBox.max.y, localBox.max.z),
            glm::vec3(localBox.max.x, localBox.min.y, localBox.min.z),
            glm::vec3(localBox.max.x, localBox.min.y, localBox.max.z),
            glm::vec3(localBox.max.x, localBox.max.y, localBox.min.z),
            glm::vec3(localBox.max.x, localBox.max.y, localBox.max.z)
        };

        glm::vec3 worldMin(std::numeric_limits<float>::max());
        glm::vec3 worldMax(std::numeric_limits<float>::lowest());
        for (const auto &corner: localCorners) {
            glm::vec4 cornerWorld4 = transform * glm::vec4(corner, 1.0f);
            glm::vec3 cornerWorld = glm::vec3(cornerWorld4) / cornerWorld4.w;
            worldMin = glm::min(worldMin, cornerWorld);
            worldMax = glm::max(worldMax, cornerWorld);
        }
        return {worldMin, worldMax};
    }


    int buildBVHNode(std::vector<BVHNode> &nodes,
                     std::vector<PhotonTracer::BVHLeaf> &leaves,
                     size_t start, size_t end) {
        BVHNode node;
        node.isLeaf = false;
        node.leftChild = -1;
        node.rightChild = -1;
        node.quadricIndex = -1;

        // Compute the bounding box over leaves[start, end)
        glm::vec3 nodeMin(std::numeric_limits<float>::max());
        glm::vec3 nodeMax(std::numeric_limits<float>::lowest());
        for (size_t i = start; i < end; i++) {
            nodeMin = glm::min(nodeMin, leaves[i].bboxMin);
            nodeMax = glm::max(nodeMax, leaves[i].bboxMax);
        }
        node.bboxMin = nodeMin;
        node.bboxMax = nodeMax;

        size_t count = end - start;
        if (count == 1) {
            // Leaf node: store the single quadric index.
            node.isLeaf = true;
            node.quadricIndex = leaves[start].quadricIndex;
            int nodeIndex = nodes.size();
            nodes.push_back(node);
            return nodeIndex;
        }

        // Choose the axis with the greatest extent.
        glm::vec3 extent = nodeMax - nodeMin;
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z)
            axis = 1;
        else if (extent.z > extent.x && extent.z > extent.y)
            axis = 2;

        // Compute the center along the chosen axis.
        float mid = 0.0f;
        for (size_t i = start; i < end; i++) {
            glm::vec3 center = 0.5f * (leaves[i].bboxMin + leaves[i].bboxMax);
            mid += center[axis];
        }
        mid /= count;

        // Partition the leaves so that those with centers < mid come first.
        size_t pivot = std::partition(leaves.begin() + start, leaves.begin() + end,
                                      [axis, mid](const PhotonTracer::BVHLeaf &leaf) {
                                          glm::vec3 center = 0.5f * (leaf.bboxMin + leaf.bboxMax);
                                          return center[axis] < mid;
                                      }
                       ) - leaves.begin();

        // If the partition fails (all on one side), split in half.
        if (pivot == start || pivot == end) {
            pivot = start + count / 2;
        }

        int leftChild = buildBVHNode(nodes, leaves, start, pivot);
        int rightChild = buildBVHNode(nodes, leaves, pivot, end);
        node.leftChild = leftChild;
        node.rightChild = rightChild;

        int nodeIndex = nodes.size();
        nodes.push_back(node);
        return nodeIndex;
    }

    std::vector<PhotonTracer::BVHLeaf> PhotonTracer::buildBVHLeaves(const std::vector<QuadricInputAssembly> &quadrics) {
        std::vector<BVHLeaf> leaves;
        for (size_t i = 0; i < quadrics.size(); i++) {
            const auto &quad = quadrics[i];
            // Compute local AABB by sampling the quadric's domain.
            AABB localBox = computeLocalAABB(quad);
            // Transform local AABB into world space.
            AABB worldBox = transformAABB(localBox, quad.transform.getTransform());
            BVHLeaf leaf{};
            leaf.bboxMin = worldBox.min;
            leaf.bboxMax = worldBox.max;
            leaf.quadricIndex = i;
            leaves.push_back(leaf);
        }
        return leaves;
    }


    std::vector<BVHNode> PhotonTracer::buildBVH(const std::vector<BVHLeaf> &inputLeaves) {
        std::vector<BVHLeaf> leaves = inputLeaves; // make a copy to allow reordering
        std::vector<BVHNode> nodes;
        if (!leaves.empty())
            buildBVHNode(nodes, leaves, 0, leaves.size());
        return nodes;
    }


    PhotonTracer::~PhotonTracer() {
        if (m_imageMemory) {
            delete[] m_imageMemory;
            Log::Logger::getInstance()->trace("Freed CPU Memory: imageMemory");
        }
        if (m_backwardInfo.photonIDGradient) {
            delete[] m_backwardInfo.photonIDGradient;
            Log::Logger::getInstance()->trace("Freed CPU Memory: photonIDGradient");
        }
        if (m_backwardInfo.gradientPixelCoordinates) {
            delete[] m_backwardInfo.gradientPixelCoordinates;
            Log::Logger::getInstance()->trace("Freed CPU Memory: gradientPixelCoordinates");
        }
        if (m_backwardInfo.gradientImagePerObject) {
            delete[] m_backwardInfo.gradientImagePerObject;
            Log::Logger::getInstance()->trace("Freed CPU Memory: gradientImagePerObject");
        }
        if (m_backwardInfo.gradientImageHoriz) {
            delete[] m_backwardInfo.gradientImageHoriz;
            Log::Logger::getInstance()->trace("Freed CPU Memory: gradientImageHoriz");
        }
        if (m_backwardInfo.gradientImageVert) {
            delete[] m_backwardInfo.gradientImageVert;
            Log::Logger::getInstance()->trace("Freed CPU Memory: gradientImageVert");
        }
        if (m_backwardInfo.pixelHitCounter) {
            delete[] m_backwardInfo.pixelHitCounter;
            Log::Logger::getInstance()->trace("Freed CPU Memory: pixelHitCounter");
        }
        freeResources();
    }


    /* Draw rays
     {
        auto view = m_scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
        for (auto e: view) {
            Entity entity(e, m_scene.get());
            auto &transform = entity.getComponent<TransformComponent>();
            auto camera = std::dynamic_pointer_cast<PinholeCamera>(entity.getComponent<CameraComponent>().camera);
            if (!camera || entity.getComponent<CameraComponent>().isActiveCamera())
                continue;
            float fx = camera->m_fx;
            float fy = camera->m_fy;
            float cx = camera->m_cx;
            float cy = camera->m_cy;
            float width = camera->m_width;
            float height = camera->m_height;


            // Helper lambda to create a ray entity
            auto updateRayEntity = [&](Entity cornerEntity, float x, float y) {
                MeshComponent *mesh;
                if (!cornerEntity.hasComponent<MeshComponent>())
                    mesh = &cornerEntity.addComponent<MeshComponent>(CYLINDER);
                else
                    mesh = &cornerEntity.getComponent<MeshComponent>();

                if (!cornerEntity.hasComponent<TemporaryComponent>())
                    cornerEntity.addComponent<TemporaryComponent>();


                cornerEntity.getComponent<TransformComponent>() = transform;
                auto cylinderParams = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh->meshParameters);
                // The cylinder magnitude is how long the cylinder is.
                // Start the cylinder at the camera origin
                cylinderParams->origin = glm::vec3(0.0f, 0.0f, 0.0f);

                // Choose a plane at Z = -1 for visualization. Objects in front of the camera have negative Z.
                float Z_plane = -1.0f;

                auto mapPixelTo3D = [&](float u, float v) {
                    float X = -(u - cx) * Z_plane / fx;
                    float Y = -(v - cy) * Z_plane / fy; // Notice the minus sign before (v - cy)
                    float Z = Z_plane;
                    return glm::vec3(X, Y, Z);
                };
                glm::vec3 direction = mapPixelTo3D(x, y);


                cylinderParams->direction = glm::normalize(direction);
                cylinderParams->magnitude = glm::length(direction);
                cylinderParams->radius = 0.01f;
                mesh->updateMeshData = true;
            };

            auto groupEntity = m_scene->getOrCreateEntityByName("Rays");
            if (!groupEntity.hasComponent<GroupComponent>())
                groupEntity.addComponent<GroupComponent>();
            if (!groupEntity.hasComponent<TemporaryComponent>())
                groupEntity.addComponent<TemporaryComponent>();
            if (!groupEntity.hasComponent<VisibleComponent>())
                groupEntity.addComponent<VisibleComponent>(); // For visibility toggling

            auto topLeftEntity = m_scene->getOrCreateEntityByName("TopLeft");
            auto topRightEntity = m_scene->getOrCreateEntityByName("TopRight");

            auto bottomLeftEntity = m_scene->getOrCreateEntityByName("BottomLeft");
            auto bottomRightEntity = m_scene->getOrCreateEntityByName("BottomRight");


            updateRayEntity(topLeftEntity, 0.0f, 0.0f);
            updateRayEntity(topRightEntity, width, 0.0f);
            updateRayEntity(bottomLeftEntity, width, height);
            updateRayEntity(bottomRightEntity, 0.0f, height);

            topLeftEntity.setParent(groupEntity);
            topRightEntity.setParent(groupEntity);
            bottomLeftEntity.setParent(groupEntity);
            bottomRightEntity.setParent(groupEntity);

            //auto centerRayEntity = m_scene->getOrCreateEntityByName("CenterRay");
            //updateRayEntity(centerRayEntity, width / 2, height / 2);

            // Generate rays for every 10th pixel
            for (int x = 0; x < width; x += 100) {
                for (int y = 0; y < height; y += 100) {
                    // Create a unique name for the ray entity
                    std::string rayEntityName = "Ray_" + std::to_string(x) + "_" + std::to_string(y);

                    // Get or create the entity for this ray
                    auto rayEntity = m_scene->getOrCreateEntityByName(rayEntityName);
                    rayEntity.setParent(groupEntity);

                    // Update the ray entity's position or other attributes based on the pixel coordinates
                    updateRayEntity(rayEntity, static_cast<float>(x), static_cast<float>(y));
                }
            }
        }
    }
    */


    void PhotonTracer::uploadVertexData(std::shared_ptr<Scene> &scene) {
        /*
        std::vector<InputAssembly> vertexData;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> indexOffsets; // Offset for each entity's indices
        std::vector<uint32_t> vertexOffsets; // Offset for each entity's vertices
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        std::vector<MaterialComponent> materials; // Transformation matrices for entities
        std::vector<TagComponent> tagComponents; // Transformation matrices for entities
        auto view = scene->getRegistry().view<MeshComponent, TransformComponent>();
        uint32_t currentVertexOffset = 0;
        uint32_t currentIndexOffset = 0;

        for (auto e: view) {
            Entity entity(e, scene.get());
            std::string tag = entity.getName();
            // Initialize a flag to determine if we should skip this entity
            bool skipEntity = false;
            // Start with the current entity
            Entity current = entity;
            // Traverse up the parent hierarchy
            while (current.getParent()) {
                // Move to the parent entity
                current = current.getParent();
                // Check if the parent has both GroupComponent and VisibilityComponent
                if (current.hasComponent<GroupComponent>() && current.hasComponent<VisibleComponent>()) {
                    // Retrieve the VisibilityComponent
                    auto &visibility = current.getComponent<VisibleComponent>();
                    // If visibility is set to false, mark to skip this entity
                    if (!visibility.visible) {
                        skipEntity = true;
                        break; // No need to check further ancestors
                    }
                }
            }
            if (entity.hasComponent<CameraComponent>())
                skipEntity = true;

            auto &meshComponent = entity.getComponent<MeshComponent>();
            if (meshComponent.meshDataType() != OBJ_FILE)
                skipEntity = true;

            // If an ancestor with visible == false was found, skip to the next entity
            if (skipEntity) {
                continue;
            }


            auto &transform = entity.getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);

            auto tagComponent = entity.getComponent<TagComponent>();
            tagComponents.emplace_back(tagComponent);

            if (entity.hasComponent<MaterialComponent>()) {
                auto &material = entity.getComponent<MaterialComponent>();
                float diff = material.diffuse;
                float specular = material.specular;
                materials.emplace_back(material);
            }

            MeshManager meshManager;
            std::shared_ptr<MeshData> meshData = meshManager.getMeshData(meshComponent);
            // Store vertex offset
            vertexOffsets.push_back(currentVertexOffset);
            // Add vertex data
            for (auto &vert: meshData->vertices) {
                InputAssembly input{};
                input.position = vert.pos;
                input.color = vert.color;
                input.normal = vert.normal;
                vertexData.emplace_back(input);
            }
            currentVertexOffset += meshData->vertices.size();
            // Store index offset
            indexOffsets.push_back(currentIndexOffset);
            // Add index data
            for (auto idx: meshData->indices) {
                indices.push_back(idx + vertexOffsets.back()); // Adjust indices by vertex offset
            }
            currentIndexOffset += meshData->indices.size();
        }
        // Upload vertex data to GPU // Split each assignment for debug purposes
        auto &queue = m_pipelineSettings.device();
        m_gpu.vertices = sycl::malloc_device<InputAssembly>(vertexData.size(), queue);
        queue.memcpy(m_gpu.vertices, vertexData.data(), vertexData.size() * sizeof(InputAssembly));
        queue.wait();

        // Upload index data to GPU
        m_gpu.indices = sycl::malloc_device<uint32_t>(indices.size(), queue);
        queue.memcpy(m_gpu.indices, indices.data(), indices.size() * sizeof(uint32_t));
        queue.wait();

        // Upload transform matrices to GPU
        m_gpu.transforms = sycl::malloc_device<TransformComponent>(transformMatrices.size(), queue);
        queue.memcpy(m_gpu.transforms, transformMatrices.data(), transformMatrices.size() * sizeof(TransformComponent));
        queue.wait();

        // Upload material data to GPU
        m_gpu.materials = sycl::malloc_device<MaterialComponent>(materials.size(), queue);
        queue.memcpy(m_gpu.materials, materials.data(), materials.size() * sizeof(MaterialComponent));
        queue.wait();

        // Upload tag components to GPU
        m_gpu.tagComponents = sycl::malloc_device<TagComponent>(tagComponents.size(), queue);
        queue.memcpy(m_gpu.tagComponents, tagComponents.data(), tagComponents.size() * sizeof(TagComponent));
        queue.wait();

        // Upload vertex offsets to GPU (if necessary for rendering)
        m_gpu.vertexOffsets = sycl::malloc_device<uint32_t>(vertexOffsets.size(), queue);
        queue.memcpy(m_gpu.vertexOffsets, vertexOffsets.data(), vertexOffsets.size() * sizeof(uint32_t));
        queue.wait();

        // Upload index offsets to GPU
        m_gpu.indexOffsets = sycl::malloc_device<uint32_t>(indexOffsets.size(), queue);
        queue.memcpy(m_gpu.indexOffsets, indexOffsets.data(), indexOffsets.size() * sizeof(uint32_t));
        queue.wait();


        m_gpu.totalVertices = currentVertexOffset; // Number of entities for rendering
        m_gpu.totalIndices = currentIndexOffset; // Number of entities for rendering
        m_gpu.numEntities = static_cast<uint32_t>(transformMatrices.size()); // Number of entities for rendering
        */
    }
}
