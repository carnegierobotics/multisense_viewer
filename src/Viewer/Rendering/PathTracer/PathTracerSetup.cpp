//
// Created by magnus on 5/2/25.
//

#include "PathTracerSetup.h"

#include <stb_image_write.h>
#include <glm/gtc/matrix_inverse.hpp>

#include "PathTracerTypes.h"
#include "Viewer/Rendering/PathTracer/Device/PathTracerAdjointKernel.h"

#include <Viewer/Rendering/MeshManager.h>
#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Rendering/Components/MaterialComponent.h>
#include <Viewer/Rendering/Components/QuadricCollectionComponent.h>
#include <Viewer/Scenes/Entity.h>

#include "Viewer/Rendering/PathTracer/BVH.h"
#include "Viewer/Rendering/PathTracer/Device/KernelHelpers.h"

#include <yaml-cpp/yaml.h>
#include <OpenImageDenoise/oidn.hpp>

namespace YAML {
    template<>
    struct convert<float3> {
        static Node encode(const float3 &v) {
            Node node;
            node.push_back(v.x());
            node.push_back(v.y());
            node.push_back(v.z());
            return node;
        }

        static bool decode(const Node &node, float3 &v) {
            if (!node.IsSequence() || node.size() != 3) return false;
            v.x() = node[0].as<float>();
            v.y() = node[1].as<float>();
            v.z() = node[2].as<float>();
            return true;
        }
    };

    template<>
    struct convert<float4x4> {
        static Node encode(const float4x4 &m) {
            Node node;
            // each row is a sequence of 4 floats
            for (size_t r = 0; r < 4; ++r) {
                Node row;
                for (size_t c = 0; c < 4; ++c)
                    row.push_back(m.row[r][c]);
                node.push_back(row);
            }
            return node;
        }

        static bool decode(const Node &node, float4x4 &m) {
            if (!node.IsSequence() || node.size() != 4) return false;
            for (size_t r = 0; r < 4; ++r) {
                const Node &row = node[r];
                if (!row.IsSequence() || row.size() != 4) return false;
                for (size_t c = 0; c < 4; ++c)
                    m.row[r][c] = row[c].as<float>();
            }
            return true;
        }
    };
} // namespace YAML

// ------------------------------------------------------------------
// Emitter overloads: control how they appear when you do `out << myFloat3;`
// ------------------------------------------------------------------
YAML::Emitter &operator<<(YAML::Emitter &out, const float3 &v) {
    out << YAML::Flow
            << YAML::BeginSeq
            << v.x() << v.y() << v.z()
            << YAML::EndSeq;
    return out;
}

YAML::Emitter &operator<<(YAML::Emitter &out, const float4x4 &m) {
    // emit as sequence of 4 flow‐style row sequences
    out << YAML::BeginSeq;
    for (size_t r = 0; r < 4; ++r) {
        out << YAML::Flow
                << YAML::BeginSeq
                << m.row[r][0]
                << m.row[r][1]
                << m.row[r][2]
                << m.row[r][3]
                << YAML::EndSeq;
    }
    out << YAML::EndSeq;
    return out;
}

namespace VkRender::PathTracer {
    PathTracerSetup::~PathTracerSetup() {
        freeDeviceMemory();
    }


    void PathTracerSetup::setupFrameBuffers() {
        Utils::ScopedTimer timer("PathTracer: Setup Framebuffers");
        // Host framebuffer
        if (m_frameBuffers.memory) {
            free(m_frameBuffers.memory);
        }
        if (m_frameBuffers.residuals) {
            free(m_frameBuffers.residuals);
        }
        if (m_frameBuffers.photonHits) {
            free(m_frameBuffers.photonHits);
        }
        if (d_frameBuffers.memory) {
            sycl::free(d_frameBuffers.memory, m_queue);
            d_frameBuffers.memory = nullptr;
        }
        if (d_frameBuffers.residuals) {
            sycl::free(d_frameBuffers.residuals, m_queue);
            d_frameBuffers.residuals = nullptr;
        }
        if (d_frameBuffers.photonHits) {
            sycl::free(d_frameBuffers.photonHits, m_queue);
            d_frameBuffers.photonHits = nullptr;
        }
        auto &ci = m_createInfo;
        uint32_t blockSize = ci.framebufferSize;

        Log::Logger::getInstance()->info("Creating framebuffers on host with size: {:.2f}Mb", blockSize / 1e6);
        m_frameBuffers.memory = static_cast<float4 *>(malloc(blockSize));
        memset(m_frameBuffers.memory, 0, blockSize);
        m_frameBuffers.frameBufferSize = blockSize;


        Log::Logger::getInstance()->info("Creating framebuffers on device with size: {:.2f}Mb", blockSize / 1e6);

        auto *deviceMemory = deviceAlloc<float>(blockSize);
        d_frameBuffers.memory = reinterpret_cast<float4 *>(deviceMemory);
        d_frameBuffers.frameBufferSize = blockSize;

        Log::Logger::getInstance()->info("Done Creating Framebuffers");


        uint32_t residualBlockSize = 600 * 600 * 4;
        Log::Logger::getInstance()->info("Creating residuals buffer on host with size: {:.2f}Mb",
                                         residualBlockSize / 1e6);
        m_frameBuffers.residuals = static_cast<float *>(malloc(residualBlockSize));
        memset(m_frameBuffers.residuals, 0, residualBlockSize);
        m_frameBuffers.residualBufferSize = residualBlockSize;

        Log::Logger::getInstance()->info("Creating residuals buffer on host with size: {:.2f}Mb",
                                         residualBlockSize / 1e6);
        auto *deviceMemoryResiduals = deviceAlloc<float>(residualBlockSize);
        d_frameBuffers.residuals = deviceMemoryResiduals;
        d_frameBuffers.residualBufferSize = residualBlockSize;
        Log::Logger::getInstance()->info("Done Creating Residuals buffer");


        // Photon Map
        uint32_t photonMapSize = 1e8;
        Log::Logger::getInstance()->info("Creating photon hit buffer on host with size: {:.2f}Mb", photonMapSize / 1e6);
        m_frameBuffers.photonHits = static_cast<Photon *>(malloc(photonMapSize));
        memset(m_frameBuffers.photonHits, 0, photonMapSize);
        m_frameBuffers.photonHitBufferSize = photonMapSize;

        Log::Logger::getInstance()->info("Creating photon hit buffer on host with size: {:.2f}Mb", photonMapSize / 1e6);
        auto *photonHitBuffer = deviceAlloc<Photon>(photonMapSize);
        d_frameBuffers.photonHits = photonHitBuffer;
        d_frameBuffers.photonHitBufferSize = photonMapSize;
        Log::Logger::getInstance()->info("Done Creating photon hit buffer buffer");

        Photon photon;
        photon.position = {0.0f};
        photon.power = -1.0f;
        m_queue.fill(d_frameBuffers.photonHits, photon, photonMapSize).wait(); // Only copy first camera instance
    }

    void PathTracerSetup::uploadScene(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
        // free existing GPU memory
        freeDeviceMemory();
        collectCameras(scene, editorCamera);
        //// collect host data
        collectGeometry(scene);
        collectInstances(scene);
        collectLights(scene);

        buildBLASForAllMeshes();
        buildBLASForPointCloud();
        buildTopLevelBVH();

        // build and upload scene descriptor
        Utils::ScopedTimer timer("PathTracer: Build Scene Description");
        buildSceneDesc();
        d_sceneDesc = deviceAlloc<SceneDesc>(1);
        m_queue.memcpy(d_sceneDesc, &m_sceneDescDevice, sizeof(SceneDesc)).wait();
    }


    void PathTracerSetup::collectCameras(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
        m_cameras.clear();

        uint32_t pixelOffset = 0;
        // EDITOR CAMERA
        if (editorCamera.camera) {
            PinholeParameters pinholeParameters;
            SharedCameraSettings cameraSettings;
            pinholeParameters.width = editorCamera.editorWidth;
            pinholeParameters.height = editorCamera.editorHeight;
            pinholeParameters.cx = pinholeParameters.width / 2.0f;
            pinholeParameters.cy = pinholeParameters.height / 2.0f;
            pinholeParameters.fx = 600.0f;
            pinholeParameters.fy = 600.0f;
            // Construct the pinhole
            PinholeCamera defaultCam(cameraSettings, pinholeParameters);
            Camera cam{};
            cam.width = editorCamera.editorWidth;
            cam.height = editorCamera.editorHeight;
            cam.pos = glm2sycl(editorCamera.camera->matrices.position);
            cam.proj = glm2sycl(editorCamera.camera->matrices.projection);
            cam.view = glm2sycl(editorCamera.camera->matrices.view);
            cam.invProj = glm2sycl(glm::inverse(editorCamera.camera->matrices.projection));
            cam.invView = glm2sycl(glm::inverse(editorCamera.camera->matrices.view));
            cam.firstPixel = pixelOffset;

            glm::vec3 cameraNormal(0.0f, 0.0f, -1.0f);
            glm::mat4 modelMatrix = editorCamera.camera->matrices.transform;
            glm::mat3 normalMat = glm::inverseTranspose(glm::mat3(modelMatrix));
            glm::vec3 rotatedNormal = glm::normalize(normalMat * cameraNormal);
            cam.forward = glm2sycl(rotatedNormal);

            m_cameras.push_back(cam);
            pixelOffset += cam.width * cam.height * 4;
        }
        // SCENE CAMERAS
        auto view = scene->getRegistry().view<CameraComponent, TransformComponent>();
        for (auto id: view) {
            Entity e(id, scene.get());
            auto &cameraComponent = e.getComponent<CameraComponent>();
            if (cameraComponent.cameraType != CameraComponent::PINHOLE)
                continue;
            auto &transformComponent = e.getComponent<TransformComponent>();
            auto sceneCameraParameters = cameraComponent.getPinholeCamera()->parameters();
            Camera cam;
            cam.width = static_cast<uint32_t>(sceneCameraParameters.width);
            cam.height = static_cast<uint32_t>(sceneCameraParameters.height);
            cam.pos = glm2sycl(transformComponent.getPosition());
            cam.proj = glm2sycl(cameraComponent.camera->matrices.projection);
            cam.view = glm2sycl(cameraComponent.camera->matrices.view);
            cam.invProj = glm2sycl(glm::inverse(cameraComponent.camera->matrices.projection));
            cam.invView = glm2sycl(glm::inverse(cameraComponent.camera->matrices.view));
            cam.firstPixel = pixelOffset;

            glm::vec3 cameraNormal(0.0f, 0.0f, -1.0f);
            glm::mat4 modelMatrix = cameraComponent.camera->matrices.transform;
            glm::mat3 normalMat = glm::inverseTranspose(glm::mat3(modelMatrix));
            glm::vec3 rotatedNormal = glm::normalize(normalMat * cameraNormal);
            cam.forward = glm2sycl(rotatedNormal);
            cam.entity = e;
            pixelOffset += cam.width * cam.height * 4;
            m_cameras.push_back(cam);
        }


        if (pixelOffset >= m_createInfo.framebufferSize) {
            Log::Logger::getInstance()->error("More cameras than framebuffers");
            throw std::runtime_error("PathTracerSetup::PathTracerSetup(): More cameras than framebuffers");
        }
    }

    void PathTracerSetup::printMaterialIDs() {
        auto logger = Log::Logger::getInstance();
        logger->info("=== Material slots (entity name → materialIndex) ===");
        for (size_t i = 0; i < m_materials.size(); ++i) {
            logger->info("  [{}] '{}' → materialIndex = {}", i, m_materialNames[i], i);
        }
        logger->info("======================================================");
    }

    void PathTracerSetup::updateDynamic(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
        Utils::ScopedTimer timer("PathTracer: Update Dynamic Data");

        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
        }

        collectInstances(scene);

        Log::Logger::getInstance()->trace("Updating transforms");
        m_queue.memcpy(d_transforms, m_transforms.data(), m_transforms.size() * sizeof(Transform));

        // view both mesh & material & transform
        auto view = scene->getRegistry().view<LightSourceComponent>();

        for (int i = 0; auto id: view) {
            Entity e(id, scene.get());
            auto &transformComponent = e.getComponent<TransformComponent>();
            m_lights[i].transform.objectToWorld = glm2sycl(transformComponent.getTransform());
            m_lights[i].transform.worldToObject = glm2sycl(glm::inverse(transformComponent.getTransform()));
            ++i;
        }
        Log::Logger::getInstance()->trace("Updating Lights");
        m_queue.memcpy(d_lights, m_lights.data(), m_lights.size() * sizeof(MeshLight));


        auto &camera = m_cameras.front();
        camera.pos = glm2sycl(editorCamera.camera->matrices.position);
        camera.proj = glm2sycl(editorCamera.camera->matrices.projection);
        camera.view = glm2sycl(editorCamera.camera->matrices.view);
        glm::vec3 cameraNormal(0.0f, 0.0f, -1.0f);
        glm::mat4 modelMatrix = editorCamera.camera->matrices.transform;
        glm::mat3 normalMat = glm::inverseTranspose(glm::mat3(modelMatrix));
        glm::vec3 rotatedNormal = glm::normalize(normalMat * cameraNormal);
        camera.forward = glm2sycl(rotatedNormal);

        Log::Logger::getInstance()->trace("Updating Cameras");
        m_queue.memcpy(d_cameras, &camera, sizeof(Camera)); // Only copy first camera instance
        m_sceneDescDevice.cameras = d_cameras;
        m_sceneDescDevice.lights = d_lights;
        m_sceneDescHost.cameras = d_cameras;
        m_sceneDescHost.lights = d_lights;

        // IF camera was moving then clear the image data for that camera
        if (editorCamera.movedSinceLastFrame) {
            resetImageMemory();
        }

        m_queue.wait();
    }

    void PathTracerSetup::resetImageMemory() {
        Log::Logger::getInstance()->info("Resetting camera framebuffers");

        // Editor camera
        const auto &camera = m_cameras.front();
        const uint32_t width = camera.width;
        const uint32_t height = camera.height;
        const size_t pixelCount = static_cast<size_t>(width) * height;
        const size_t floatComponents = pixelCount * 4; // 4 floats per pixel (RGBA32F)
        m_queue.fill(d_frameBuffers.memory, 0.0f, floatComponents); // Only copy first camera instance

        // Scene cameras
        for (int i = 1; i < m_cameras.size(); ++i) {
            auto &camera = m_cameras[i];
            size_t pixelCount = camera.width * camera.height;
            uint32_t imageSize = pixelCount * 4;
            m_queue.fill(d_frameBuffers.memory + camera.firstPixel, 0.0f, imageSize);
        }
        m_totalPhotons = 0;
        m_frameID = 0;

        // Residuals buffer:
        uint32_t residualBlockSize = 600 * 600 * 4;
        m_queue.fill(d_frameBuffers.residuals, 0.0f, residualBlockSize); // Only copy first camera instance

        // Photon map
        uint32_t photonMapSize = 1e8;
        Photon photon;
        photon.position = {0.0f};
        photon.power = -1.0f;
        m_queue.fill(d_frameBuffers.photonHits, photon, photonMapSize).wait(); // Only copy first camera instance

        m_queue.fill(m_sceneDescDevice.photonMapHitCount, static_cast<uint32_t>(0), sizeof(uint32_t)).wait();
        // Only copy first camera instance
    }

    RenderInfoOutput PathTracerSetup::renderFrame(const RenderSettings &settings) {
        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
        }
        Utils::ScopedTimer timer("PathTracer: RenderFrame");

        auto conf = settings;
        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<uint32_t> dist;
        // 4) Draw one seed value:
        conf.randomSeed = dist(rng);
        auto event = m_queue.submit([scene=d_sceneDesc, fb = d_frameBuffers, config=conf](sycl::handler &cgh) {
            PathTracerMeshKernel kernel(scene, fb, config);
            cgh.parallel_for(sycl::range<1>(config.photonCount), kernel);
        });

        event.wait();

        m_photonCount = settings.photonCount;
        m_totalPhotons += settings.photonCount;
        m_frameID++;

        return {m_frameID, m_totalPhotons};
    }

    RenderInfoOutput PathTracerSetup::radiativeBackprop(std::shared_ptr<EditorPathTracerLayerUI> imageUI,
                                                        const RenderSettings &settings) {
        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
        }
        Utils::ScopedTimer timer("PathTracer: RenderFrame");

        // calculate residual
        // - Load reference image
        std::vector<float> residual;

        auto &camera = m_cameras[1];

        std::string name = camera.entity.getName();
        std::string gtPath = imageUI->gtFolderPath.string() + "/" + name + ".pfm";

        int gtW, gtH, gtC; // will be populated by loadPFM
        std::vector<float> gtData; // flattened, size = gtW*gtH*gtC
        if (!Utils::loadPFM(gtPath, gtData, gtW, gtH, gtC)) {
            throw std::runtime_error("Failed to load GT PFM: " + gtPath);
        }
        if (gtW != camera.width || gtH != camera.height || gtC != 1) {
            throw std::runtime_error("Size mismatch between GT and render for " + name);
        }

        // 1) read back your device float RGBA buffer
        size_t pixelCount = camera.width * camera.height;
        uint32_t imageSize = pixelCount * 4;
        float *hostMemory = new float[imageSize];
        float *floatImageAfterTonemap = new float[pixelCount];

        m_queue.memcpy(hostMemory,
                       d_frameBuffers.memory + camera.firstPixel,
                       imageSize * sizeof(float)).wait();

        // 2) compute residual in linear domain
        //    Here we take only the “R” channel of GT (gtC>=1) vs. your L=hostMemory[r]
        //    If you want full-rgba residual, set channels accordingly.
        std::vector<uint8_t> rgb8(pixelCount * 3);

        residual.resize(pixelCount);
        float gammaInv = 1.0f / imageUI->gamma; // sRGB ≈ 1/2.2
        for (size_t i = 0; i < pixelCount; ++i) {
            float L = hostMemory[i * 4];

            float Lm = 1.f - std::exp(-L * imageUI->exposure);
            float val = std::pow(std::clamp(Lm, 0.f, 1.f), gammaInv);

            floatImageAfterTonemap[i] = val;
            float v = val * 255.f + 0.5f;
            rgb8[i * 3 + 0] = static_cast<uint8_t>(v);
            rgb8[i * 3 + 1] = static_cast<uint8_t>(v);
            rgb8[i * 3 + 2] = static_cast<uint8_t>(v);

            float L_gt = gtData[i * gtC + 0];
            residual[i] = val - L_gt;
        }

        delete[] hostMemory;

        uint32_t residualBlockSize = 600 * 600 * 4;
        m_queue.fill(d_frameBuffers.residuals, 0.0f, residualBlockSize).wait(); // Only copy first camera instance


        m_queue.memcpy(d_frameBuffers.residuals,
                       residual.data(),
                       pixelCount * sizeof(float)).wait();

        // 3) write out the residual as a single-channel PFM
        std::filesystem::path outPath = imageUI->gtFolderPath / "optimization" / std::filesystem::path(
                                            name + "_residual.pfm");
        Utils::savePFM(outPath, residual.data(), camera.width, camera.height, 1);


        // --- 4) Write pfm
        std::filesystem::path pfmImagePath = outPath.replace_filename(name + "_measured").replace_extension(".pfm");
        Utils::savePFM(pfmImagePath, floatImageAfterTonemap, camera.width, camera.height, 1);
        delete[] floatImageAfterTonemap;

        // --- 4) Write PNG
        std::filesystem::path pngImagePath = outPath.replace_filename(name + "_measured").replace_extension(".png");

        if (!stbi_write_png(pngImagePath.string().c_str(),
                            camera.width, camera.height,
                            3, rgb8.data(), camera.width * 3)) {
            throw std::runtime_error("Failed to write PNG file: " + pngImagePath.string());
        }

        Log::Logger::getInstance()->info("Wrote {} to {}. Parameter slots: {}", name, pngImagePath.string(),
                                         m_totalParamSlots);

        m_queue.fill(d_sceneDesc->gradAll, 0.0f, m_totalParamSlots).wait(); // Only copy first camera instance

        sycl::range<2> threadsPerBlock(16, 16);
        sycl::range<2> numBlocks(
            (camera.height + threadsPerBlock[0] - 1) / threadsPerBlock[0],
            (camera.width + threadsPerBlock[1] - 1) / threadsPerBlock[1]);

        m_sceneDescDevice.gradDebugMaterialID = imageUI->gradMaterialID;
        d_sceneDesc->gradDebugMaterialID = imageUI->gradMaterialID;

        Log::Logger::getInstance()->info("Reset gradient image. Launching backprop kernel...");


        auto conf = settings;
        conf.iteration = m_backpropIterations;
        auto event = m_queue.submit(
            [scene=d_sceneDesc, fb = d_frameBuffers, config=conf, threadsPerBlock, numBlocks](sycl::handler &cgh) {
                PathTracerAdjointKernel kernel(scene, fb, config);
                cgh.parallel_for(sycl::nd_range<2>(numBlocks * threadsPerBlock, threadsPerBlock), kernel);
            });

        event.wait();
        Log::Logger::getInstance()->info("Ran Radiative Backprop");

        // 1) read back your device float RGBA buffer
        float *allGradientsHost = new float[m_totalParamSlots];
        m_queue.memcpy(allGradientsHost,
                       d_sceneDesc->gradAll,
                       m_totalParamSlots * sizeof(float)).wait();
        // 3) write out the residual as a single-channel PFM

        std::filesystem::path csvPath = imageUI->gtFolderPath
                                        / "optimization"
                                        / std::filesystem::path("all_gradients.csv");

        std::ofstream out(csvPath);
        if (!out) {
            throw std::runtime_error("Failed to open CSV for gradients: " + csvPath.string());
        }

        // Optional header
        out << "paramIndex,gradient\n";

        for (uint32_t i = 0; i < m_totalParamSlots; ++i) {
            out << i << ',' << allGradientsHost[i] << '\n';
        }

        out.close();
        Log::Logger::getInstance()->info("Wrote all_gradients.csv with {} entries", m_totalParamSlots);
        delete[] allGradientsHost;

        const Camera &camDbg = m_cameras[1]; // ← same cam you rendered
        size_t nPix = camDbg.width * camDbg.height;

        std::vector<float> gradImgHost(nPix);
        m_queue.memcpy(gradImgHost.data(),
                       d_sceneDesc->gradKdImage,
                       nPix * sizeof(float)).wait();

        /* ------------------------------------------------------------------ */
        /* 3)  write the gradient image to disk (PFM, 1 channel)              */
        /* ------------------------------------------------------------------ */
        std::filesystem::path gradPath = imageUI->gtFolderPath /
                                         "optimization" /
                                         std::filesystem::path("dI_dkd1.pfm");

        Utils::savePFM(gradPath,
                       gradImgHost.data(),
                       camDbg.width, camDbg.height, 1);

        m_backpropIterations++;

        return {m_frameID, m_totalPhotons};
    }


    void PathTracerSetup::generateImages(std::shared_ptr<EditorPathTracerLayerUI> imageUI) {
        float gammaInv = 1.0f / imageUI->gamma; // sRGB ≈ 1/2.2

        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
        }

        for (int camIdx = 1; camIdx < m_cameras.size(); ++camIdx) {
            auto &camera = m_cameras[camIdx];

            // --- 1) Read back float RGBA buffer
            size_t pixelCount = camera.width * camera.height;
            uint32_t imageSize = pixelCount * 4;
            float *hostMemory = new float[imageSize];
            float *floatImageAfterTonemap = new float[pixelCount];
            m_queue.memcpy(hostMemory, d_frameBuffers.memory + camera.firstPixel,
                           imageSize * sizeof(float)).wait();

            // Optional denoising (linear HDR, before exposure/gamma)
            std::unique_ptr<float[]> denoisedRGB; // Float3 buffer if we denoise
            if (imageUI->denoiseImage) {
                try {
                    // Prepare an RGB buffer for OIDN (Float3)
                    std::unique_ptr<float[]> colorRGB{new float[pixelCount * 3]};
                    // If you already have RGB in hostMemory use those; otherwise replicate R->RGB.
                    // Assuming R,G,B are valid in hostMemory:
                    for (size_t i = 0; i < pixelCount; ++i) {
                        const float r = hostMemory[i * 4 + 0];
                        const float g = hostMemory[i * 4 + 1];
                        const float b = hostMemory[i * 4 + 2];
                        colorRGB[i * 3 + 0] = r;
                        colorRGB[i * 3 + 1] = g;
                        colorRGB[i * 3 + 2] = b;
                    }

                    denoisedRGB.reset(new float[pixelCount * 3]);

                    // Create OIDN device (CPU by default; you can switch to SYCL device later)
                    oidn::DeviceRef device = oidn::newDevice(); // default: CPU
                    device.commit();

                    // Classic ray tracing denoiser
                    oidn::FilterRef filter = device.newFilter("RT");
                    filter.set("hdr", true); // input is HDR linear
                    filter.set("cleanAux", true); // noisy/noisy aux acceptable; harmless here

                    // Set images with row/byte strides (contiguous here)
                    filter.setImage("color",
                                    colorRGB.get(), oidn::Format::Float3,
                                    camera.width, camera.height,
                                    /*byteOffset*/ 0,
                                    /*bytePixelStride*/ sizeof(float) * 3,
                                    /*byteRowStride*/ sizeof(float) * 3 * camera.width);
                    filter.setImage("output",
                                    denoisedRGB.get(), oidn::Format::Float3,
                                    camera.width, camera.height,
                                    0, sizeof(float) * 3, sizeof(float) * 3 * camera.width);

                    // (Optional) If you have auxiliary buffers:
                    // filter.setImage("albedo",  albedoPtr,  oidn::Format::Float3, w, h, ...);
                    // filter.setImage("normal",  normalPtr,  oidn::Format::Float3, w, h, ...);

                    filter.commit();
                    filter.execute();

                    // Check for errors
                    const char *errMsg = nullptr;
                    if (device.getError(errMsg) != oidn::Error::None) {
                        Log::Logger::getInstance()->warning(
                            "{}", std::string("OIDN warning: ") + (errMsg ? errMsg : ""));
                        // Fall back to non-denoised path by clearing pointer
                        denoisedRGB.reset();
                    }
                } catch (const std::exception &e) {
                    Log::Logger::getInstance()->error("{}", std::string("OIDN exception: ") + e.what());
                    denoisedRGB.reset();
                }
            }

            // --- 2) Tonemap & gamma‐correct to 8-bit grayscale
            std::vector<uint8_t> rgb8(pixelCount * 3);

            // Use denoised RGB (if present) to compute luminance; else use raw
            for (size_t i = 0; i < pixelCount; ++i) {
                float L_linear;
                if (denoisedRGB) {
                    // Luminance from denoised RGB
                    const float r = denoisedRGB[i * 3 + 0];
                    const float g = denoisedRGB[i * 3 + 1];
                    const float b = denoisedRGB[i * 3 + 2];
                    // Rec.709 luminance
                    L_linear = 0.2126f * r + 0.7152f * g + 0.0722f * b;
                } else {
                    // Fallback: use R as your original code did
                    L_linear = hostMemory[i * 4 + 0];
                }

                // Simple photographic tonemap with exposure
                const float Lm = 1.f - std::exp(-L_linear * imageUI->exposure);
                const float val = std::pow(std::clamp(Lm, 0.f, 1.f), gammaInv);

                floatImageAfterTonemap[i] = val;
                const float v = val * 255.f + 0.5f;
                const uint8_t u8 = static_cast<uint8_t>(std::clamp(v, 0.f, 255.f));
                rgb8[i * 3 + 0] = u8;
                rgb8[i * 3 + 1] = u8;
                rgb8[i * 3 + 2] = u8;
            }
            // --- 3) Build paths
            std::filesystem::path imgPath = camera.entity
                                                ? std::filesystem::path(camera.entity.getName()).replace_extension(
                                                    ".png")
                                                : std::filesystem::path("default.png");
            imgPath = imageUI->saveImagePath / imgPath;

            std::filesystem::path yamlPath = imgPath;
            yamlPath.replace_extension(".yaml");

            std::filesystem::path pfmPath = imgPath;
            pfmPath.replace_extension(".pfm");

            Utils::savePFM(pfmPath, floatImageAfterTonemap, camera.width, camera.height, 1);

            delete[] hostMemory;
            delete[] floatImageAfterTonemap;

            // --- 4) Write PNG
            if (!stbi_write_png(imgPath.string().c_str(),
                                camera.width, camera.height,
                                3, rgb8.data(), camera.width * 3)) {
                throw std::runtime_error("Failed to write PNG file: " + imgPath.string());
            }

            // --- 5) Emit YAML metadata
            YAML::Emitter out;
            out << YAML::BeginMap;

            // top‐level render settings
            out << YAML::Key << "FrameID" << YAML::Value << m_frameID;
            out << YAML::Key << "PhotonCount" << YAML::Value << m_photonCount;
            out << YAML::Key << "TotalPhotons" << YAML::Value << m_totalPhotons;
            out << YAML::Key << "Gamma" << YAML::Value << imageUI->gamma;
            out << YAML::Key << "Exposure" << YAML::Value << imageUI->exposure;

            // camera block
            out << YAML::Key << "Camera" << YAML::Value << YAML::BeginMap;
            out << YAML::Key << "Name" << YAML::Value << (camera.entity ? camera.entity.getName() : "default");
            out << YAML::Key << "Width" << YAML::Value << camera.width;
            out << YAML::Key << "Height" << YAML::Value << camera.height;
            out << YAML::Key << "Position" << YAML::Value << camera.pos; // uses your vec3 << overload
            out << YAML::Key << "Forward" << YAML::Value << camera.forward; // vec3
            out << YAML::Key << "ViewMatrix" << YAML::Value << camera.view; // float4x4
            out << YAML::Key << "ProjectionMatrix" << YAML::Value << camera.proj; // float4x4
            out << YAML::EndMap; // end camera map

            out << YAML::EndMap; // end top‐level map

            // write to disk
            std::ofstream fout(yamlPath);
            if (!fout) {
                Log::Logger::getInstance()->error("Cannot open YAML file: {}", yamlPath.string());
            } else {
                fout << out.c_str();
                fout.close();
            }

            // --- 6) Advance frame
            ++m_frameID;
        }

        // Write out the photon map
        // --- 1) Read back float RGBA buffer
        uint32_t photonMapSize = 1e8;
        std::vector<Photon> photonHits(photonMapSize);
        m_queue.memcpy(photonHits.data(), d_frameBuffers.photonHits, photonMapSize * sizeof(Photon)).wait();

        std::filesystem::path plyPath = imageUI->saveImagePath / "photons.ply";

        writePhotonPLY(photonHits, plyPath);

        Photon photon;
        photon.position = {0.0f};
        photon.power = -1.0f;
        m_queue.fill(d_frameBuffers.photonHits, photon, photonMapSize).wait(); // Only copy first camera instance
    }


    // ── writePhotonPLY : dumps N photons to binary-little-endian PLY ──────
    void PathTracerSetup::writePhotonPLY(const std::vector<Photon> &photons,
                                         const std::string &filename) {
        const std::size_t N = photons.size();
        if (N == 0) {
            std::cerr << "[writePhotonPLY] photon array is empty.\n";
            return;
        }

        /* 1. flatten into contiguous float arrays ------------------------- */
        std::vector<float> positions;
        std::vector<uint8_t> powers;
        positions.reserve(N * 3);
        powers.reserve(N * 3);

        float scale = 255.0f;

        uint32_t numPhotons = 0;
        for (const Photon &ph: photons) {
            if (ph.power <= 0.0f)
                continue;

            positions.push_back(ph.position.x());
            positions.push_back(ph.position.y());
            positions.push_back(ph.position.z());
            powers.push_back(static_cast<uint8_t>(ph.power * scale));
            powers.push_back(static_cast<uint8_t>(ph.power * scale));
            powers.push_back(static_cast<uint8_t>(ph.power * scale));
            numPhotons++;
        }

        /* 2. assemble the PLY file --------------------------------------- */
        tinyply::PlyFile ply;

        ply.add_properties_to_element(
            /*element*/ "vertex",
                        /*prop names*/ {"x", "y", "z"},
                        tinyply::Type::FLOAT32,
                        numPhotons,
                        reinterpret_cast<uint8_t *>(positions.data()),
                        tinyply::Type::INVALID, 0);

        ply.add_properties_to_element("vertex",
                                      {"red", "green", "blue"},
                                      tinyply::Type::UINT8, numPhotons,
                                      reinterpret_cast<uint8_t *>(powers.data()),
                                      tinyply::Type::INVALID, 0);

        /* 3. write to disk -------------------------------------------------- */
        std::ofstream ofs(filename, std::ios::out | std::ios::binary);
        if (!ofs)
            throw std::runtime_error("failed to open " + filename);

        ply.write(ofs, /*isBinary*/ true); // binary-little-endian
        ofs.close();

        std::cout << "[writePhotonPLY] wrote "
                << N << " photons to " << filename << '\n';
    }

    void PathTracerSetup::generateEditorImage(const std::shared_ptr<VulkanTexture2D> &viewportTexture,
                                              std::shared_ptr<EditorPathTracerLayerUI> imageUI) {
        Utils::ScopedTimer timer("PathTracer: Generate Editor Image");

        const float gammaInv = 1.0f / imageUI->gamma;

        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
            return;
        }

        const auto &camera = m_cameras.front();
        const uint32_t width = camera.width;
        const uint32_t height = camera.height;
        const size_t pixelCount = static_cast<size_t>(width) * height;
        const size_t floatComponents = pixelCount * 4; // RGBA32F
        const size_t floatByteSize = floatComponents * sizeof(float);

        Log::Logger::getInstance()->trace("PathTracerSetup::generateEditorImage(): {}/{} MB",
                                          static_cast<float>(floatByteSize) / 1.0e6f,
                                          static_cast<float>(m_createInfo.framebufferSize) / 1.0e6f);

        // 1) Copy device -> host
        std::vector<float> hostRGBA(floatComponents);
        m_queue.memcpy(hostRGBA.data(), d_frameBuffers.memory, floatByteSize).wait();

        // 1b) Optional denoising (linear HDR)
        std::unique_ptr<float[]> denoisedRGB; // Float3 buffer if we denoise
        if (imageUI->denoiseImage) {
            try {
                std::unique_ptr<float[]> colorRGB{new float[pixelCount * 3]};
                for (size_t i = 0; i < pixelCount; ++i) {
                    colorRGB[i * 3 + 0] = hostRGBA[i * 4 + 0];
                    colorRGB[i * 3 + 1] = hostRGBA[i * 4 + 1];
                    colorRGB[i * 3 + 2] = hostRGBA[i * 4 + 2];
                }

                denoisedRGB.reset(new float[pixelCount * 3]);

                // 1) Create device (SYCL/GPU)
                oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
                device.commit();

                // 2) Create a device-accessible buffer and upload your RGB floats
                const size_t bytes = sizeof(float) * 3 * width * height;
                oidn::BufferRef colorBuf = device.newBuffer(bytes);
                colorBuf.write(0, bytes, colorRGB.get());   // upload host -> device buffer

                oidn::BufferRef outBuf   = device.newBuffer(bytes);

                // 3) Set images using buffers (no raw pointers)
                oidn::FilterRef filter = device.newFilter("RT");
                filter.set("hdr", false);

                filter.setImage("color",  colorBuf, oidn::Format::Float3,
                                width, height, /*byteOffset*/0,
                                /*bytePixelStride*/ sizeof(float)*3,
                                /*byteRowStride*/  sizeof(float)*3*width);

                filter.setImage("output", outBuf,   oidn::Format::Float3,
                                width, height, 0,
                                sizeof(float)*3,
                                sizeof(float)*3*width);

                filter.commit();
                filter.execute();

                // 4) Download result back to host memory you own
                outBuf.read(0, bytes, denoisedRGB.get());

                const char *errMsg = nullptr;
                if (device.getError(errMsg) != oidn::Error::None) {
                    Log::Logger::getInstance()->warning("OIDN warning: {}", errMsg ? errMsg : "");
                    denoisedRGB.reset();
                }
            } catch (const std::exception &e) {
                Log::Logger::getInstance()->error("OIDN exception: {}", e.what());
                denoisedRGB.reset();
            }
        }

        // 2) Tonemap & gamma-correct to 8-bit RGBA (currently grayscale)
        std::vector<uint8_t> rgba8(pixelCount * 4);
        for (size_t i = 0; i < pixelCount; ++i) {
            float L_linear;
            if (denoisedRGB) {
                const float r = denoisedRGB[i * 3 + 0];
                const float g = denoisedRGB[i * 3 + 1];
                const float b = denoisedRGB[i * 3 + 2];
                L_linear = 0.2126f * r + 0.7152f * g + 0.0722f * b; // Rec.709
            } else {
                // fallback: luminance approx from raw RGB
                const float r = hostRGBA[i * 4 + 0];
                const float g = hostRGBA[i * 4 + 1];
                const float b = hostRGBA[i * 4 + 2];
                L_linear = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            }

            const float Lm = 1.f - std::exp(-L_linear * imageUI->exposure);
            const float val = std::pow(std::clamp(Lm, 0.f, 1.f), gammaInv);
            const uint8_t u8 = static_cast<uint8_t>(std::clamp(val * 255.f + 0.5f, 0.f, 255.f));

            rgba8[i * 4 + 0] = u8;
            rgba8[i * 4 + 1] = u8;
            rgba8[i * 4 + 2] = u8;
            rgba8[i * 4 + 3] = 255;
        }

        // 3) Upload to Vulkan texture
        viewportTexture->loadImage(rgba8.data());
    }


    void PathTracerSetup::collectGeometry(const std::shared_ptr<Scene> &scene) {
        Utils::ScopedTimer timer("PathTracer: Collect Geometry");

        m_points.clear();
        m_pointRanges.clear();

        m_vertices.clear();
        m_tris.clear();
        m_meshRanges.clear();
        m_meshIndexMap.clear();
        m_meshNames.clear();

        // collect unique meshes
        auto view = scene->getRegistry().view<MeshComponent>();
        for (auto id: view) {
            Entity e(id, scene.get());
            if (!e.isVisible()) {
                continue;
            }

            auto &mc = e.getComponent<MeshComponent>();
            auto mesh = MeshManager::instance().getMeshData(mc);
            if (!mesh)
                continue;

            std::string meshID = mc.getCacheIdentifier();
            if (m_meshIndexMap.count(meshID)) continue;

            if (mc.meshDataType() == QUADRIC) {
                auto params = std::dynamic_pointer_cast<QuadricMeshParameters>(mc.meshParameters);
                OrientedPoint point;
                point.beta = params->b_beta;
                point.c = params->c;
                point.threshold = params->threshold;
                point.type = QuadricPoint;
                m_points.emplace_back(point);
                // record mesh range
                PointCloudRange range{};
                range.pointCount = 1;
                range.firstPoint = m_pointRanges.size();
                m_pointRanges.push_back(range);
                m_meshIndexMap[meshID] = static_cast<uint32_t>(m_pointRanges.size() - 1);
            } else if (mc.meshDataType() == GAUSSIAN_2D) {
                auto params = std::dynamic_pointer_cast<Gaussian2DMeshParameters>(mc.meshParameters);
                OrientedPoint point;
                point.opacity = params->opacity;
                point.color = params->color;
                point.covX = params->covX;
                point.covY = params->covY;
                point.threshold = params->threshold;
                point.type = Gaussian2DPoint;
                m_points.emplace_back(point);
                // record mesh range
                PointCloudRange range{};
                range.pointCount = 1;
                range.firstPoint = m_pointRanges.size();
                m_pointRanges.push_back(range);
                m_meshIndexMap[meshID] = static_cast<uint32_t>(m_pointRanges.size() - 1);
            } else {
                // base offsets
                uint32_t vertBase = static_cast<uint32_t>(m_vertices.size());
                uint32_t triBase = static_cast<uint32_t>(m_tris.size());

                // append vertices
                for (auto &v: mesh->m_vertices) {
                    Vertex vertex;
                    vertex.pos = float3{v.pos.x, v.pos.y, v.pos.z};
                    vertex.norm = float3{v.normal.x, v.normal.y, v.normal.z};
                    m_vertices.push_back(vertex);
                }
                // append triangles
                const float inv3 = 1.0f / 3.0f;
                for (size_t i = 0; i < mesh->m_indices.size(); i += 3) {
                    uint32_t i0 = mesh->m_indices[i + 0] + vertBase;
                    uint32_t i1 = mesh->m_indices[i + 1] + vertBase;
                    uint32_t i2 = mesh->m_indices[i + 2] + vertBase;
                    Triangle t{};
                    t.v0 = i0;
                    t.v1 = i1;
                    t.v2 = i2;
                    // compute centroid from the three vertex positions
                    float3 p0 = m_vertices[i0].pos;
                    float3 p1 = m_vertices[i1].pos;
                    float3 p2 = m_vertices[i2].pos;
                    t.centroid = (p0 + p1 + p2) * inv3;
                    m_tris.push_back(t);
                }
                // record mesh range
                MeshRange range{};
                range.firstVert = vertBase;
                range.vertCount = static_cast<uint32_t>(mesh->m_vertices.size());
                range.firstTri = triBase;
                range.triCount = static_cast<uint32_t>(mesh->m_indices.size() / 3);
                m_meshRanges.push_back(range);
                m_meshNames.push_back(meshID); // <— keep name in sync
                m_meshIndexMap[meshID] = static_cast<uint32_t>(m_meshRanges.size() - 1);
            }
        }
    }

    void PathTracerSetup::collectInstances(const std::shared_ptr<Scene> &scene) {
        Utils::ScopedTimer timer("PathTracer: Collect Instances");
        m_instances.clear();
        m_transforms.clear();
        m_materials.clear(); // one material slot per instance

        auto view = scene->getRegistry().view<MeshComponent, MaterialComponent, TransformComponent>(
            entt::exclude<RasterizerRenderingComponent, LightSourceComponent, CameraComponent,
                QuadricCollectionComponent>);
        for (auto entID: view) {
            Entity e(entID, scene.get());
            if (!e.isVisible()) {
                continue;
            }

            std::string name = e.getName();
            // --- 1) look up the mesh index we built in collectGeometry() ---
            auto &mc = e.getComponent<MeshComponent>();
            if (!MeshManager::instance().getMeshData(mc).get())
                continue;
            bool isMeshPointType = mc.meshDataType() == QUADRIC || mc.meshDataType() == GAUSSIAN_2D;


            const std::string mid = mc.getCacheIdentifier();
            if (!m_meshIndexMap.contains(mid)) {
                continue;
            }
            uint32_t geomIdx = m_meshIndexMap.at(mid);

            // --- 2) append this entity's material parameters ---
            auto &matComp = e.getComponent<MaterialComponent>();
            Material gpuMat{};
            gpuMat.baseColor = {
                matComp.albedo.x
            };
            gpuMat.specular = {
                matComp.specular
            };
            gpuMat.diffuse = {
                matComp.diffuse
            };
            gpuMat.phongExp = matComp.phongExponent;

            uint32_t matIdx = static_cast<uint32_t>(m_materials.size());
            m_materials.push_back(gpuMat);
            m_materialNames.push_back(name);

            // --- 3) record the instance record pointing at mesh+material+transform ---
            uint32_t xfIdx = static_cast<uint32_t>(m_transforms.size());
            m_instances.push_back({
                /* geomType       */
                isMeshPointType ? (GeometryType::PointCloud) : (GeometryType::Mesh),
                /* geomIndex      */ geomIdx,
                /* materialIndex  */ matIdx,
                /* transformIndex */ xfIdx,
            });

            // --- 4) store the transform for this instance ---
            auto &tc = e.getComponent<TransformComponent>();
            Transform xf{};
            xf.objectToWorld = glm2sycl(tc.getTransform());
            xf.worldToObject = glm2sycl(glm::inverse(tc.getTransform()));

            m_transforms.push_back(xf);
        }

        //----------------------------------------------------------
        // build parameter → slot mapping
        //----------------------------------------------------------
        m_kdOffset.resize(m_materials.size());
        m_vtxOffset.resize(m_vertices.size());

        uint32_t slot = 0;

        // diffuse kd  (1 float each)
        for (uint32_t i = 0; i < m_materials.size(); ++i) {
            m_kdOffset[i] = {slot, 1};
            slot += 1;
        }

        /*
        // optional: vertex positions  (3 floats each)
        for (uint32_t v=0; v<m_vertices.size(); ++v){
            m_vtxOffset[v]={slot,3}; slot+=3;
        }
        */

        m_totalParamSlots = slot;
    }

    void PathTracerSetup::collectLights(const std::shared_ptr<Scene> &scene) {
        m_lights.clear();

        // View both mesh, material, and transform components
        auto view = scene->getRegistry()
                .view<MeshComponent, TransformComponent, LightSourceComponent>();

        for (auto id: view) {
            Entity e(id, scene.get());
            auto &meshComponent = e.getComponent<MeshComponent>();
            auto &transformComponent = e.getComponent<TransformComponent>();
            auto &lightSourceComponent = e.getComponent<LightSourceComponent>();

            // Skip if not emissive
            if (lightSourceComponent.flux <= 0.0f) continue;

            const auto &mesh = MeshManager::instance().getMeshData(meshComponent);
            const auto world = transformComponent.getTransform();

            MeshLight meshLight;
            meshLight.flux = lightSourceComponent.flux;

            // 1) Loop through triangles
            for (size_t t = 0; t < mesh->m_indices.size(); t += 3) {
                auto i0 = mesh->m_indices[t + 0];
                auto i1 = mesh->m_indices[t + 1];
                auto i2 = mesh->m_indices[t + 2];

                glm::vec3 p1 = mesh->m_vertices[i0].pos;
                glm::vec3 p2 = mesh->m_vertices[i1].pos;
                glm::vec3 p3 = mesh->m_vertices[i2].pos;

                // Transform to world space
                glm::vec4 P0 = world * glm::vec4(p1, 1.0f);
                glm::vec4 P1 = world * glm::vec4(p2, 1.0f);
                glm::vec4 P2 = world * glm::vec4(p3, 1.0f);

                float3 v0 = {P0.x, P0.y, P0.z};
                float3 e1 = {P1.x - P0.x, P1.y - P0.y, P1.z - P0.z};
                float3 e2 = {P2.x - P0.x, P2.y - P0.y, P2.z - P0.z};
                float3 n = normalize(cross(e1, e2));
                float area = 0.5f * length(cross(e1, e2));

                meshLight.addTriangle(v0, e1, e2, n, area);
            }

            // 2) Finalize the CDF and radiance
            if (meshLight.triangleCount > 0) {
                meshLight.finalize();
                meshLight.transform.objectToWorld = glm2sycl(transformComponent.getTransform());
                meshLight.transform.worldToObject = glm2sycl(glm::inverse(transformComponent.getTransform()));
                m_lights.push_back(meshLight);
            }
        }
    }


    void PathTracerSetup::buildSceneDesc() {
        const size_t vertCount = m_vertices.size();

        //—— allocate & copy SoA (positions + normals) ——
        d_vertices = deviceAlloc<Vertex>(vertCount);
        m_queue.memcpy(d_vertices, m_vertices.data(), vertCount * sizeof(Vertex));

        //—— allocate & copy indexed arrays ——
        const size_t triCount = m_tris.size();
        d_tris = deviceAlloc<Triangle>(triCount);
        m_queue.memcpy(d_tris, m_tris.data(), triCount * sizeof(Triangle));

        const size_t meshCount = m_meshRanges.size();
        d_meshRanges = deviceAlloc<MeshRange>(meshCount);
        m_queue.memcpy(d_meshRanges, m_meshRanges.data(), meshCount * sizeof(MeshRange));

        const size_t instCount = m_instances.size();
        d_instances = deviceAlloc<Instance>(instCount);
        m_queue.memcpy(d_instances, m_instances.data(), instCount * sizeof(Instance));

        const size_t xfCount = m_transforms.size();
        d_transforms = deviceAlloc<Transform>(xfCount);
        m_queue.memcpy(d_transforms, m_transforms.data(), xfCount * sizeof(Transform));

        const size_t matCount = m_materials.size();
        d_materials = deviceAlloc<Material>(matCount);
        m_queue.memcpy(d_materials, m_materials.data(), matCount * sizeof(Material));

        const size_t lightCount = m_lights.size();
        d_lights = deviceAlloc<MeshLight>(lightCount);
        m_queue.memcpy(d_lights, m_lights.data(), lightCount * sizeof(MeshLight));

        const size_t camCount = m_cameras.size();
        d_cameras = deviceAlloc<Camera>(camCount);
        m_queue.memcpy(d_cameras, m_cameras.data(), camCount * sizeof(Camera));

        const size_t pointCount = m_points.size();
        d_points = deviceAlloc<OrientedPoint>(pointCount);
        m_queue.memcpy(d_points, m_points.data(), pointCount * sizeof(OrientedPoint));


        // —— upload BLAS pool ————————————————————————————
        d_blasNodes = deviceAlloc<BVHNode>(m_blasNodes.size());
        m_queue.memcpy(d_blasNodes, m_blasNodes.data(),
                       m_blasNodes.size() * sizeof(BVHNode));

        d_blasRanges = deviceAlloc<BLASRange>(m_blasRanges.size());
        m_queue.memcpy(d_blasRanges, m_blasRanges.data(),
                       m_blasRanges.size() * sizeof(BLASRange));

        // —— upload BLAS pool ————————————————————————————
        d_betaNodes = deviceAlloc<BVHNode>(m_betaNodes.size());
        m_queue.memcpy(d_betaNodes, m_betaNodes.data(),
                       m_betaNodes.size() * sizeof(BVHNode));

        d_betaRanges = deviceAlloc<BLASRange>(m_betaRanges.size());
        m_queue.memcpy(d_betaRanges, m_betaRanges.data(),
                       m_betaRanges.size() * sizeof(BLASRange));

        // —— upload TLAS ————————————————————————————————
        d_tlasNodes = deviceAlloc<TLASNode>(m_tlasNodes.size());
        m_queue.memcpy(d_tlasNodes, m_tlasNodes.data(),
                       m_tlasNodes.size() * sizeof(TLASNode));

        // —— upload TLAS ————————————————————————————————
        d_trianglePerm = deviceAlloc<uint32_t>(m_trianglePerm.size());
        m_queue.memcpy(d_trianglePerm, m_trianglePerm.data(),
                       m_trianglePerm.size() * sizeof(uint32_t));

        d_photonHitCounter = deviceAlloc<uint32_t>(1);
        m_queue.fill(d_photonHitCounter, static_cast<uint32_t>(0), sizeof(uint32_t)).wait();
        m_sceneDescDevice.photonMapHitCount = d_photonHitCounter;

        //—— fill SceneDesc ——
        m_sceneDescDevice.points = d_points;
        m_sceneDescDevice.vertices = d_vertices;
        m_sceneDescDevice.triangles = d_tris;
        m_sceneDescDevice.meshes = d_meshRanges;
        m_sceneDescDevice.instances = d_instances;
        m_sceneDescDevice.transforms = d_transforms;
        m_sceneDescDevice.materials = d_materials;
        m_sceneDescDevice.lights = d_lights;
        m_sceneDescDevice.cameras = d_cameras;

        m_sceneDescDevice.tlasNodes = d_tlasNodes;
        m_sceneDescDevice.blasRanges = d_blasRanges;
        m_sceneDescDevice.blasNodes = d_blasNodes;
        m_sceneDescDevice.betaNodes = d_betaNodes;
        m_sceneDescDevice.betaRanges = d_betaRanges;
        m_sceneDescDevice.triPerm = d_trianglePerm;


        m_sceneDescDevice.pointCount = static_cast<uint32_t>(pointCount);
        m_sceneDescDevice.triCount = static_cast<uint32_t>(triCount);
        m_sceneDescDevice.vertexCount = static_cast<uint32_t>(vertCount);
        m_sceneDescDevice.meshCount = static_cast<uint32_t>(meshCount);
        m_sceneDescDevice.instanceCount = static_cast<uint32_t>(instCount);
        m_sceneDescDevice.transformCount = static_cast<uint32_t>(xfCount);
        m_sceneDescDevice.materialCount = static_cast<uint32_t>(matCount);
        m_sceneDescDevice.lightCount = static_cast<uint32_t>(lightCount);
        m_sceneDescDevice.cameraCount = static_cast<uint32_t>(camCount);

        /* ---------- HOST  descriptor (For Debug Purposes, not used for rendering) ---------- */
        m_sceneDescHost.points = m_points.data();
        m_sceneDescHost.vertices = m_vertices.data();
        m_sceneDescHost.triangles = m_tris.data();
        m_sceneDescHost.meshes = m_meshRanges.data();
        m_sceneDescHost.instances = m_instances.data();
        m_sceneDescHost.transforms = m_transforms.data();
        m_sceneDescHost.materials = m_materials.data();
        m_sceneDescHost.lights = m_lights.data();
        m_sceneDescHost.cameras = m_cameras.data();

        m_sceneDescHost.blasNodes = m_blasNodes.data();
        m_sceneDescHost.blasRanges = m_blasRanges.data();
        m_sceneDescHost.betaNodes = m_betaNodes.data();
        m_sceneDescHost.betaRanges = m_betaRanges.data();
        m_sceneDescHost.tlasNodes = m_tlasNodes.data();
        m_sceneDescHost.triPerm = m_trianglePerm.data();


        /* counts are identical */
        m_sceneDescHost.pointCount = m_sceneDescDevice.pointCount;
        m_sceneDescHost.triCount = m_sceneDescDevice.triCount;
        m_sceneDescHost.vertexCount = m_sceneDescDevice.vertexCount;
        m_sceneDescHost.meshCount = m_sceneDescDevice.meshCount;
        m_sceneDescHost.instanceCount = m_sceneDescDevice.instanceCount;
        m_sceneDescHost.transformCount = m_sceneDescDevice.transformCount;
        m_sceneDescHost.materialCount = m_sceneDescDevice.materialCount;
        m_sceneDescHost.lightCount = m_sceneDescDevice.lightCount;
        m_sceneDescHost.cameraCount = m_sceneDescDevice.cameraCount;


        // Radiative Backpropagation parameters:
        /* ---- upload parameter offset tables -------------------------------- */
        d_kdOffset = deviceAlloc<ParamOffset>(m_kdOffset.size());
        m_queue.memcpy(d_kdOffset, m_kdOffset.data(),
                       m_kdOffset.size() * sizeof(ParamOffset));

        d_vtxOffset = deviceAlloc<ParamOffset>(m_vtxOffset.size());
        m_queue.memcpy(d_vtxOffset, m_vtxOffset.data(),
                       m_vtxOffset.size() * sizeof(ParamOffset));

        /* ---- allocate gradient vector (zero-initialised) ------------------- */
        d_gradAll = deviceAlloc<float>(m_totalParamSlots);
        m_queue.fill(d_gradAll, 0.f, m_totalParamSlots); // set to 0 once at init

        /* ---- allocate gradient debug image (zero-initialised) ------------------- */
        size_t nPix = 600 * 600;
        d_gradKdImage = deviceAlloc<float>(nPix);
        m_queue.fill(d_gradKdImage, 0.f, nPix); // set to 0 once at init

        /* ---- expose to SceneDesc so kernels can reach them ----------------- */
        m_sceneDescDevice.kdOffset = d_kdOffset;
        m_sceneDescDevice.vtxOffset = d_vtxOffset;
        m_sceneDescDevice.gradAll = d_gradAll;
        m_sceneDescDevice.gradKdImage = d_gradKdImage;
    }


    static void gatherLeavesDFS(std::vector<BVHNode> &nodes,
                                uint32_t root,
                                std::vector<LeafRange> &out) {
        struct StackItem {
            uint32_t n;
        };
        SmallStack<256> st;
        st.push({root});
        while (!st.empty()) {
            uint32_t idx = st.pop();
            BVHNode &N = nodes[idx];
            if (N.isLeaf()) {
                out.push_back({idx, N.leftFirst, N.triCount});
            } else {
                // push right first so left is visited next (classic DFS)
                st.push({N.leftFirst + 1});
                st.push({N.leftFirst});
            }
        }
    }

    void PathTracerSetup::buildBLASForAllMeshes() {
        Utils::ScopedTimer timer("PathTracer: Build BLAS");

        m_blasNodes.clear(); // flat pool that will hold *every* mesh BVH
        m_blasRanges.clear(); // one record per mesh
        m_blasNames.clear(); // purely for debug drawing

        /* --------------------------------------------------------------------- */
        /* Loop over *unique* meshes (one per entry in m_meshRanges)              */
        /* --------------------------------------------------------------------- */
        for (uint32_t m = 0; m < m_meshRanges.size(); ++m) {
            const MeshRange &meshRange = m_meshRanges[m];
            const std::string name = m_meshNames[m]; // for the UI

            // ──────────────────────────────────────────────────────────────
            // 1.  Gather **local vertices** (just copy the structs)
            // ──────────────────────────────────────────────────────────────
            std::vector<Vertex> localVerts;
            localVerts.reserve(meshRange.vertCount);

            for (uint32_t v = 0; v < meshRange.vertCount; ++v)
                localVerts.push_back(m_vertices[meshRange.firstVert + v]);

            // ──────────────────────────────────────────────────────────────
            // 2.  Gather & re‑index triangles so they refer to localVerts[]
            // ──────────────────────────────────────────────────────────────
            std::vector<Triangle> localTris;
            localTris.reserve(meshRange.triCount);

            for (uint32_t t = 0; t < meshRange.triCount; ++t) {
                Triangle T = m_tris[meshRange.firstTri + t];
                T.v0 -= meshRange.firstVert; // now between 0 … mr.vertCount‑1
                T.v1 -= meshRange.firstVert;
                T.v2 -= meshRange.firstVert;
                localTris.push_back(T);
            }

            // ──────────────────────────────────────────────────────────────
            // 3.  Build the mesh‑local BVH
            // ──────────────────────────────────────────────────────────────
            std::vector<BVHNode> localNodes;
            std::vector<uint32_t> triIdx; // permutation (ignored later)

            BasicBVH::build(localTris,
                            localVerts, // ← vertex array is required
                            localNodes,
                            triIdx,
                            /*maxLeaf*/ 4);

            // ---------- A. reorder the global triangle array ---------------------------
            // global index where this mesh's triangles start
            uint32_t globalTriStart = meshRange.firstTri;

            // 1.  Temporary copy that will hold triangles in BVH order
            std::vector<Triangle> reordered;
            reordered.reserve(localTris.size());

            for (unsigned int i: triIdx) {
                Triangle T = localTris[i];

                // convert vertex indices back to GLOBAL space
                T.v0 += meshRange.firstVert;
                T.v1 += meshRange.firstVert;
                T.v2 += meshRange.firstVert;

                reordered.push_back(T);
            }

            // 2.  Overwrite the slice in m_tris with the reordered triangles
            std::copy(reordered.begin(), reordered.end(),
                      m_tris.begin() + globalTriStart);

            // 3.  Now patch the BVH nodes ----------------------------------------------
            //     (children still contiguous, only need global offset)

            uint32_t firstNode = uint32_t(m_blasNodes.size());

            // 4C) Patch all *leaf* nodes so their triangle slices shift by globalTriStart
            for (BVHNode &N: localNodes) {
                if (N.isLeaf())
                    N.leftFirst += globalTriStart;
            }

            //---------------- 5.  Append to the big BLAS pool & record range -----
            m_blasNodes.insert(m_blasNodes.end(),
                               localNodes.begin(), localNodes.end());

            m_blasRanges.push_back({
                firstNode,
                uint32_t(localNodes.size())
            });

            m_blasNames.push_back(name); // purely for your debug renderer
        }
    }


    void PathTracerSetup::buildBLASForPointCloud() {
        Utils::ScopedTimer t("PathTracer: Build BLAS – point clouds");

        m_betaNodes.clear();
        m_betaRanges.clear();

        for (uint32_t pc = 0; pc < m_pointRanges.size(); ++pc) {
            const PointCloudRange &pr = m_pointRanges[pc];

            /* 1) copy this cloud’s patches into a local buffer            */
            std::vector<OrientedPoint> localPts(pr.pointCount);
            for (uint32_t i = 0; i < pr.pointCount; ++i)
                localPts[i] = m_points[pr.firstPoint + i];

            /* 2) BVH build                                                */
            std::vector<BVHNode> localNodes;
            std::vector<uint32_t> perm;
            QuadricBVH2D::build(localPts, localNodes, perm, 2);

            /* 3) optional: reorder global storage in BVH order            */
            {
                std::vector<OrientedPoint> tmp(pr.pointCount);
                for (uint32_t i = 0; i < pr.pointCount; ++i)
                    tmp[i] = localPts[perm[i]];
                std::copy(tmp.begin(), tmp.end(),
                          m_points.begin() + pr.firstPoint);
            }

            /* 4) patch leaves → global indices                            */
            for (BVHNode &N: localNodes)
                if (N.isLeaf())
                    N.leftFirst = pr.firstPoint + N.leftFirst;

            /* 5) append to big pool                                       */
            uint32_t first = uint32_t(m_betaNodes.size());
            m_betaNodes.insert(m_betaNodes.end(),
                               localNodes.begin(), localNodes.end());
            m_betaRanges.push_back({first, uint32_t(localNodes.size())});
        }
    }


    //──────────────────────────────────────────────────────────────────────────
    // Build TLAS over *instances*  (one leaf = one Instance struct)
    //──────────────────────────────────────────────────────────────────────────
    void PathTracerSetup::buildTopLevelBVH() {
        Utils::ScopedTimer timer("PathTracer: Build TLAS");

        using Box = struct {
            float3 bmin, bmax;
            uint32_t inst;
        };

        /* 1) gather instance‑space AABBs */
        std::vector<Box> boxes;
        boxes.reserve(m_instances.size());

        for (uint32_t i = 0; i < m_instances.size(); ++i) {
            const Instance &inst = m_instances[i];
            const Transform &xf = m_transforms[inst.transformIndex];

            /* root node of this mesh’s BLAS */
            BLASRange br{};
            BVHNode root;
            switch (inst.geomType) {
                case GeometryType::PointCloud:
                    br = m_betaRanges[inst.geomIndex];
                    root = m_betaNodes[br.firstNode];
                    break;
                case GeometryType::Mesh:
                    br = m_blasRanges[inst.geomIndex];
                    root = m_blasNodes[br.firstNode];
                    break;
            }

            /* object‑space corners → world space, track min/max */
            float3 wmin{FLT_MAX}, wmax{-FLT_MAX};

            for (int c = 0; c < 8; ++c) {
                bool bx = c & 4, by = c & 2, bz = c & 1;
                float3 pObj = {
                    bx ? root.aabbMax.x() : root.aabbMin.x(),
                    by ? root.aabbMax.y() : root.aabbMin.y(),
                    bz ? root.aabbMax.z() : root.aabbMin.z()
                };
                float3 pW = toWorldPoint(pObj, xf);
                wmin = min(wmin, pW);
                wmax = max(wmax, pW);
            }
            boxes.push_back({wmin, wmax, i});
        }

        /* 2) recursive median‑split builder (identical to BLAS style) */
        m_tlasNodes.clear();
        m_tlasNodes.reserve(boxes.size() * 2);

        std::function<int(int, int)> build = [&](int start, int end) -> int {
            int n = int(m_tlasNodes.size());
            m_tlasNodes.emplace_back();
            TLASNode &N = m_tlasNodes.back();

            /* compute bounds of current set */
            float3 bmin{FLT_MAX}, bmax{-FLT_MAX};
            for (int i = start; i < end; ++i) {
                bmin = min(bmin, boxes[i].bmin);
                bmax = max(bmax, boxes[i].bmax);
            }
            N.aabbMin = bmin;
            N.aabbMax = bmax;

            int count = end - start;
            if (count == 1) {
                N.count = 1; // leaf
                N.leftChild = boxes[start].inst; // points to Instance index
                N.rightChild = 0;
            } else {
                N.count = 0; // internal

                /* centroid bounds → pick longest axis */
                float3 cmin{FLT_MAX}, cmax{-FLT_MAX};
                for (int i = start; i < end; ++i) {
                    float3 cent = (boxes[i].bmin + boxes[i].bmax) * 0.5f;
                    cmin = min(cmin, cent);
                    cmax = max(cmax, cent);
                }
                float3 ext = cmax - cmin;
                int axis = (ext.x() > ext.y() && ext.x() > ext.z()) ? 0 : (ext.y() > ext.z()) ? 1 : 2;
                float pivot = (cmin[axis] + cmax[axis]) * 0.5f;

                auto midIter = std::partition(boxes.begin() + start, boxes.begin() + end,
                                              [&](const Box &b) {
                                                  float3 cc = (b.bmin + b.bmax) * 0.5f;
                                                  return cc[axis] < pivot;
                                              });
                int mid = int(midIter - boxes.begin());
                if (mid == start || mid == end) mid = start + count / 2;

                N.leftChild = build(start, mid);
                N.rightChild = build(mid, end);
            }
            return n;
        };

        if (!boxes.empty()) build(0, int(boxes.size()));
    }


    void PathTracerSetup::freeDeviceMemory() {
        // free descriptor
        if (d_sceneDesc) {
            sycl::free(d_sceneDesc, m_queue);
            d_sceneDesc = nullptr;
        }
        // free buffers
        auto freeIf = [&](void *p) {
            if (p) {
                sycl::free(p, m_queue);
                p = nullptr;
            }
        };
        freeIf(d_vertices);
        freeIf(d_tris);
        freeIf(d_meshRanges);
        freeIf(d_instances);
        freeIf(d_transforms);
        freeIf(d_materials);
        freeIf(d_lights);
        freeIf(d_cameras);

        // BVH
        freeIf(d_blasRanges);
        freeIf(d_blasNodes);
        freeIf(d_tlasNodes);
    }
}
