//
// Created by magnus on 5/2/25.
//

#include "PathTracerSetup.h"

#include <glm/gtc/matrix_inverse.hpp>

#include "PathTracerTypes.h"

#include <Viewer/Rendering/MeshManager.h>
#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Rendering/Components/MaterialComponent.h>
#include <Viewer/Rendering/Components/QuadricCollectionComponent.h>
#include <Viewer/Scenes/Entity.h>

#include "Viewer/Rendering/PathTracer/BVH.h"
#include "Viewer/Rendering/PathTracer/Device/KernelHelpers.h"


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


        if (d_frameBuffers.memory) {
            sycl::free(d_frameBuffers.memory, m_queue);
            d_frameBuffers.memory = nullptr;
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
            cam.firstPixel = pixelOffset;

            cam.forward = float3(0.0f, 0.0f, -1.0f);

            pixelOffset += cam.width * cam.height * 4;
            m_cameras.push_back(cam);
        }


        if (pixelOffset >= m_createInfo.framebufferSize) {
            Log::Logger::getInstance()->error("More cameras than framebuffers");
            throw std::runtime_error("PathTracerSetup::PathTracerSetup(): More cameras than framebuffers");
        }
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

        Log::Logger::getInstance()->trace("Updating Cameras");
        m_queue.memcpy(d_cameras, &camera, sizeof(Camera)); // Only copy first camera instance
        m_sceneDescDevice.cameras = d_cameras;
        m_sceneDescDevice.lights = d_lights;
        m_sceneDescHost.cameras = d_cameras;
        m_sceneDescHost.lights = d_lights;

        // IF camera was moving then clear the image data for that camera
        if (editorCamera.movedSinceLastFrame) {
            const auto &camera = m_cameras.front();
            const uint32_t width = camera.width;
            const uint32_t height = camera.height;
            const size_t pixelCount = static_cast<size_t>(width) * height;
            const size_t floatComponents = pixelCount * 4; // 4 floats per pixel (RGBA32F)
            Log::Logger::getInstance()->info("Clearing Camera framebuffers");
            m_queue.fill(d_frameBuffers.memory, 0.0f, floatComponents); // Only copy first camera instance
        }

        m_queue.wait();
    }

    void PathTracerSetup::renderFrame(const RenderSettings &settings) {
        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
        }
        Utils::ScopedTimer timer("PathTracer: RenderFrame");


        auto event = m_queue.submit([scene=d_sceneDesc, fb = d_frameBuffers, config=settings](sycl::handler &cgh) {
            PathTracerMeshKernel kernel(scene, fb, config);
            cgh.parallel_for(sycl::range<1>(config.photonCount), kernel);
        });

        event.wait();
    }

    void PathTracerSetup::generateImages(std::span<std::byte> outRGBA32f) {
        auto &ci = m_createInfo;
        for (auto &camera: m_cameras) {
            uint32_t imageSize = static_cast<uint32_t>(camera.width) * static_cast<uint32_t>(camera.height);

            break;
        }
        size_t bytes = outRGBA32f.size();
        //m_queue.memcpy(outRGBA32f.data(), d_frameBuffer, bytes).wait();
    }

    void PathTracerSetup::generateEditorImage(const std::shared_ptr<VulkanTexture2D> &viewportTexture, float gamma,
                                             float exposure) {
        Utils::ScopedTimer timer("PathTracer: Generate Editor Image");

        float gammaInv = 1.0f / gamma; // sRGB ≈ 1/2.2

        if (!d_sceneDesc) {
            Log::Logger::getInstance()->error("Path Tracer has not been initialized");
        }

        const auto &camera = m_cameras.front();
        const uint32_t width = camera.width;
        const uint32_t height = camera.height;
        const size_t pixelCount = static_cast<size_t>(width) * height;
        const size_t floatComponents = pixelCount * 4; // 4 floats per pixel (RGBA32F)s
        const size_t floatByteSize = floatComponents * sizeof(float);

        // 1) Copy float32 RGBA image from device to host scratch buffer
        Log::Logger::getInstance()->trace("PathTracerSetup::generateEditorImage(): {}/{}",
                                          static_cast<float>(floatByteSize) / 1000000.0f,
                                          static_cast<float>(m_createInfo.framebufferSize) / 1000000.0f);


        m_queue.memcpy(m_frameBuffers.memory, d_frameBuffers.memory, floatByteSize).wait();
        // 2) Convert to 8-bit RGBA
        std::vector<uint8_t> rgba8;
        rgba8.resize(pixelCount * 4);
        float *src = reinterpret_cast<float *>(m_frameBuffers.memory);
        for (size_t i = 0; i < pixelCount; ++i) {
            // clamp to [0,1], then map to [0,255]
            float v = src[i * 4];

            auto tonemap = [&](float L) {
                float Lm = 1.f - std::exp(-L * exposure); // filmic-ish
                return Lm;
            };
            v = tonemap(v);

            v = std::pow(std::clamp(v, 0.f, 1.f), gammaInv);

            rgba8[i * 4 + 0] = static_cast<uint8_t>(v * 255.f + 0.5f);
            rgba8[i * 4 + 1] = static_cast<uint8_t>(v * 255.f + 0.5f);
            rgba8[i * 4 + 2] = static_cast<uint8_t>(v * 255.f + 0.5f);
            rgba8[i * 4 + 3] = 255;
        }
        // 3) Upload RGBA8 image to the Vulkan texture
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

            auto &mc = e.getComponent<MeshComponent>();
            auto mesh = MeshManager::instance().getMeshData(mc);
            if (!mesh)
                continue;

            std::string meshID = mc.getCacheIdentifier();
            if (m_meshIndexMap.count(meshID)) continue;

            if ( mc.meshDataType() == QUADRIC) {
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
            }
            else if (mc.meshDataType() == GAUSSIAN_2D) {
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
            }
            else {
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
            std::string name = e.getName();
            // --- 1) look up the mesh index we built in collectGeometry() ---
            auto &mc = e.getComponent<MeshComponent>();
            if (!MeshManager::instance().getMeshData(mc).get())
                continue;
            bool isMeshPointType = mc.meshDataType() == QUADRIC || mc.meshDataType() == GAUSSIAN_2D;


            const std::string mid = mc.getCacheIdentifier();
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
                float3 n = sycl::normalize(sycl::cross(e1, e2));
                float area = 0.5f * sycl::length(sycl::cross(e1, e2));

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
                            /*maxLeaf*/ 2);

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

            for (BVHNode &N: localNodes) {
                if (N.isLeaf()) {
                    N.leftFirst += globalTriStart; // only leaves need patching
                }
            }

            #ifndef NDEBUG
            for(const BVHNode& n : localNodes)
                if(n.isLeaf())
                    for(uint32_t k=0;k<n.triCount;++k){
                        uint32_t g = n.leftFirst+k;
                        assert(g < m_tris.size());                // 1. triangle in range
                        const Triangle& T = m_tris[g];
                        assert(T.v0 < m_vertices.size() &&        // 2. vertices in range
                               T.v1 < m_vertices.size() &&
                               T.v2 < m_vertices.size());
                    }
            #endif

            //---------------- 5.  Append to the big BLAS pool & record range -----
            uint32_t firstNode = uint32_t(m_blasNodes.size());
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
                wmin = sycl::min(wmin, pW);
                wmax = sycl::max(wmax, pW);
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
                bmin = sycl::min(bmin, boxes[i].bmin);
                bmax = sycl::max(bmax, boxes[i].bmax);
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
                    cmin = sycl::min(cmin, cent);
                    cmax = sycl::max(cmax, cent);
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
