//
// Created by magnus on 5/2/25.
//

#include "PathTracerSYCL.h"
#include "PathTracerTypes.h"

#include <Viewer/Rendering/MeshManager.h>
#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Scenes/Entity.h>

#include "BVH.h"
#include "KernelHelpers.h"


namespace VkRender::PathTracer {
    PathTracerSYCL::~PathTracerSYCL() {
        freeDeviceMemory();
    }


    void PathTracerSYCL::setupFrameBuffers() {
        // Host framebuffer
        if (m_frameBuffers.memory) {
            free(m_frameBuffers.memory);
        }
        auto &ci = m_createInfo;
        uint32_t blockSize = ci.framebufferSize;
        m_frameBuffers.memory = static_cast<float4 *>(malloc(blockSize));
        memset(m_frameBuffers.memory, 0, blockSize);

        d_memory = deviceAlloc<float4>(blockSize);

        m_frameBuffers.memory = d_memory;
        m_frameBuffers.frameBufferSize = blockSize;
        d_frameBuffers = deviceAlloc<FrameBuffer>(1);
        m_queue.memcpy(d_frameBuffers, &m_frameBuffers, sizeof(FrameBuffer)).wait();
    }


    void PathTracerSYCL::uploadScene(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
        // free existing GPU memory
        freeDeviceMemory();
        collectCameras(scene, editorCamera);
        // collect host data
        collectGeometry(scene);
        collectInstances(scene);
        collectLights(scene);

        std::vector<uint32_t> triangleIndices;
        std::vector<Vertex> worldVerts;


        //BasicBVH::build(m_tris, m_vertices, m_bvhNodes, triangleIndices);


        buildBLASForAllMeshes();

        buildTopLevelBVH();

        /*
        // Reorder triangles in BVH order not in mesh order.
        {
            std::vector<Triangle> ordered(m_tris.size());
            for (size_t i = 0; i < triangleIndices.size(); ++i)
                ordered[i] = m_tris[triangleIndices[i]];
            m_tris.swap(ordered);
        }
        */

        // build and upload scene descriptor
        buildSceneDesc();
        d_sceneDesc = deviceAlloc<SceneDesc>(1);
        m_queue.memcpy(d_sceneDesc, &m_sceneDesc, sizeof(SceneDesc)).wait();
    }

    void PathTracerSYCL::traverseBVH() {
    }

    void PathTracerSYCL::intersectBVH(uint32_t nodeIndex) {
    }


    void PathTracerSYCL::collectCameras(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
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

            pixelOffset += cam.width * cam.height * 4;
            m_cameras.push_back(cam);
        }


        if (pixelOffset >= m_createInfo.framebufferSize) {
            Log::Logger::getInstance()->error("More cameras than framebuffers");
            throw std::runtime_error("PathTracerSYCL::PathTracerSYCL(): More cameras than framebuffers");
        }
    }


    void PathTracerSYCL::updateDynamic(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
        m_queue.memcpy(d_transforms, m_transforms.data(), m_transforms.size() * sizeof(Transform));

        // view both mesh & material & transform
        auto view = scene->getRegistry()
                .view<MeshComponent, TransformComponent, LightSourceComponent>();

        for (int i = 0; auto id: view) {
            Entity e(id, scene.get());
            auto &transformComponent = e.getComponent<TransformComponent>();
            m_lights[i].transform.objectToWorld = glm2sycl(transformComponent.getTransform());
            m_lights[i].transform.worldToObject = glm2sycl(glm::inverse(transformComponent.getTransform()));
            ++i;
        }


        m_queue.memcpy(d_lights, m_lights.data(), m_lights.size() * sizeof(MeshLight));

        auto &camera = m_cameras.front();
        camera.pos = glm2sycl(editorCamera.camera->matrices.position);
        camera.proj = glm2sycl(editorCamera.camera->matrices.projection);
        camera.view = glm2sycl(editorCamera.camera->matrices.view);

        m_queue.memcpy(d_cameras, &camera, sizeof(Camera)); // Only copy first camera instance
        m_sceneDesc.cameras = d_cameras;
        m_sceneDesc.lights = d_lights;

        // IF camera was moving then clear the image data for that camera
        if (editorCamera.movedSinceLastFrame) {
            const auto &camera = m_cameras.front();
            const uint32_t width = camera.width;
            const uint32_t height = camera.height;
            const size_t pixelCount = static_cast<size_t>(width) * height;
            const size_t floatComponents = pixelCount * 4; // 4 floats per pixel (RGBA32F)

            m_queue.fill(d_frameBuffers->memory, 0.0f, floatComponents); // Only copy first camera instance
        }

        m_queue.wait();
    }

    void PathTracerSYCL::renderFrame(int photonCount) {
        if (!d_sceneDesc) {
            return;
        }

        m_queue.submit([scene=d_sceneDesc, fb = d_frameBuffers, photonCount](sycl::handler &cgh) {
            PathTracerMeshKernel kernel(scene, fb);
            cgh.parallel_for(sycl::range<1>(photonCount), kernel);
        }).wait();
    }

    void PathTracerSYCL::generateImages(std::span<std::byte> outRGBA32f) {
        auto &ci = m_createInfo;
        for (auto &camera: m_cameras) {
            uint32_t imageSize = static_cast<uint32_t>(camera.width) * static_cast<uint32_t>(camera.height);

            break;
        }
        size_t bytes = outRGBA32f.size();
        //m_queue.memcpy(outRGBA32f.data(), d_frameBuffer, bytes).wait();
    }

    void PathTracerSYCL::generateEditorImage(const std::shared_ptr<VulkanTexture2D> &viewportTexture) {
        const auto &camera = m_cameras.front();
        const uint32_t width = camera.width;
        const uint32_t height = camera.height;
        const size_t pixelCount = static_cast<size_t>(width) * height;
        const size_t floatComponents = pixelCount * 4; // 4 floats per pixel (RGBA32F)
        const size_t floatByteSize = floatComponents * sizeof(float);

        // 1) Copy float32 RGBA image from device to host scratch buffer
        m_queue.memcpy(m_frameBuffers.memory, d_frameBuffers->memory, floatByteSize).wait();

        // 2) Convert to 8-bit RGBA
        std::vector<uint8_t> rgba8;
        rgba8.resize(pixelCount * 4);

        float *src = reinterpret_cast<float *>(m_frameBuffers.memory);
        for (size_t i = 0; i < floatComponents; ++i) {
            // clamp to [0,1], then map to [0,255]
            float v = std::clamp(src[i], 0.0f, 1.0f);
            rgba8[i] = static_cast<uint8_t>(v * 255.0f);
        }

        // 3) Upload RGBA8 image to the Vulkan texture
        viewportTexture->loadImage(rgba8.data());
    }

    void PathTracerSYCL::collectGeometry(const std::shared_ptr<Scene> &scene) {
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
            std::string meshID = mc.getCacheIdentifier();
            if (m_meshIndexMap.count(meshID)) continue;
            auto mesh = MeshManager::instance().getMeshData(mc);
            // base offsets
            uint32_t vertBase = static_cast<uint32_t>(m_vertices.size());
            uint32_t triBase = static_cast<uint32_t>(m_tris.size());

            // append vertices
            for (auto &v: mesh->m_vertices) {
                Vertex vertex;
                vertex.pos = float3{v.pos.x, v.pos.y, v.pos.z};
                vertex.norm = float3{v.normal.x, v.normal.y, v.normal.z};
                m_vertices.push_back(vertex);

                /*
                Vertex vertexWorld;
                glm::vec3 vWorld = glm::vec3(transformComponent.getTransform() * glm::vec4(v.pos, 1.0f));
                vertexWorld.pos = float3{vWorld.x, vWorld.y, vWorld.z};
                vertexWorld.norm = float3{v.normal.x, v.normal.y, v.normal.z};
                m_verticesWorld.push_back(vertexWorld);
                */
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

    void PathTracerSYCL::collectInstances(const std::shared_ptr<Scene> &scene) {
        m_instances.clear();
        m_transforms.clear();
        m_materials.clear(); // one material slot per instance

        auto view = scene->getRegistry().view<MeshComponent, MaterialComponent, TransformComponent>(
            );

        for (auto entID: view) {
            Entity e(entID, scene.get());
            std::string name = e.getName();
            // --- 1) look up the mesh index we built in collectGeometry() ---
            auto &mc = e.getComponent<MeshComponent>();
            const std::string mid = mc.getCacheIdentifier();
            uint32_t geomIdx = m_meshIndexMap.at(mid);

            // --- 2) append this entity's material parameters ---
            auto &matComp = e.getComponent<MaterialComponent>();
            Material gpuMat{};
            gpuMat.baseColor = {
                matComp.albedo.x,
                matComp.albedo.y,
                matComp.albedo.z
            };
            gpuMat.specular = {
                matComp.specular,
                matComp.specular,
                matComp.specular
            };
            gpuMat.phongExp = matComp.phongExponent;

            uint32_t matIdx = static_cast<uint32_t>(m_materials.size());
            m_materials.push_back(gpuMat);

            // --- 3) record the instance record pointing at mesh+material+transform ---
            uint32_t xfIdx = static_cast<uint32_t>(m_transforms.size());
            m_instances.push_back({
                /* geomType       */ uint32_t(GeometryType::Mesh),
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

    void PathTracerSYCL::collectLights(const std::shared_ptr<Scene> &scene) {
        m_lights.clear();

        // view both mesh & material & transform
        auto view = scene->getRegistry()
                .view<MeshComponent, TransformComponent, LightSourceComponent>();

        for (auto id: view) {
            Entity e(id, scene.get());
            auto &meshComponent = e.getComponent<MeshComponent>();
            auto &transformComponent = e.getComponent<TransformComponent>();
            auto &lightSourceComponent = e.getComponent<LightSourceComponent>();

            // skip if not emissive
            if (lightSourceComponent.flux <= 0.0f) continue;

            const auto &mesh = MeshManager::instance().getMeshData(meshComponent); // supplies v & index arrays
            const auto world = transformComponent.getTransform();

            MeshLight ML;
            ML.flux = lightSourceComponent.flux; // Φ for this mesh
            ML.totalArea = 0.0f;

            // 1) Loop triangles
            for (size_t t = 0; t < mesh->m_indices.size(); t += 3) {
                auto i0 = mesh->m_indices[t + 0],
                        i1 = mesh->m_indices[t + 1],
                        i2 = mesh->m_indices[t + 2];

                glm::vec3 p1 = mesh->m_vertices[i0].pos;
                glm::vec3 p2 = mesh->m_vertices[i1].pos;
                glm::vec3 p3 = mesh->m_vertices[i2].pos;
                // fetch positions
                glm::vec4 P0 = world * glm::vec4(p1, 1.0f);
                glm::vec4 P1 = world * glm::vec4(p2, 1.0f);
                glm::vec4 P2 = world * glm::vec4(p3, 1.0f);

                glm::vec3 v0 = {P0.x, P0.y, P0.z};
                glm::vec3 e1 = glm::vec3{P1.x, P1.y, P1.z} - v0;
                glm::vec3 e2 = glm::vec3{P2.x, P2.y, P2.z} - v0;
                glm::vec3 n = glm::normalize(glm::cross(e1, e2));
                float area = 0.5f * glm::length(glm::cross(e1, e2));

                ML.v0.emplace_back(v0.x, v0.y, v0.z);
                ML.edge1.emplace_back(e1.x, e1.y, e1.z);
                ML.edge2.emplace_back(e2.x, e2.y, e2.z);
                ML.normal.emplace_back(n.x, n.y, n.z);

                ML.totalArea += area;
                ML.cdf.push_back(ML.totalArea);
            }

            // 2) finalize CDF and radiance
            for (auto &c: ML.cdf) c /= ML.totalArea;
            ML.radiance = ML.flux / (M_PI * ML.totalArea);

            ML.transform.objectToWorld = glm2sycl(transformComponent.getTransform());
            ML.transform.worldToObject = glm2sycl(glm::inverse(transformComponent.getTransform()));

            m_lights.push_back(std::move(ML));
        }
    }


    void PathTracerSYCL::buildSceneDesc() {
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


        // —— upload BLAS pool ————————————————————————————
        d_blasNodes  = deviceAlloc<BVHNode>( m_blasNodes.size() );
        m_queue.memcpy(d_blasNodes, m_blasNodes.data(),
                       m_blasNodes.size()*sizeof(BVHNode));

        d_blasRanges = deviceAlloc<BLASRange>( m_blasRanges.size() );
        m_queue.memcpy(d_blasRanges, m_blasRanges.data(),
                       m_blasRanges.size()*sizeof(BLASRange));

        // —— upload TLAS ————————————————————————————————
       d_tlasNodes = deviceAlloc<TLASNode>( m_tlasNodes.size() );
       m_queue.memcpy(d_tlasNodes, m_tlasNodes.data(),
                      m_tlasNodes.size()*sizeof(TLASNode));


        //—— fill SceneDesc ——
        m_sceneDesc.vertices = d_vertices;
        m_sceneDesc.triangles = d_tris;
        m_sceneDesc.meshes = d_meshRanges;
        m_sceneDesc.instances = d_instances;
        m_sceneDesc.transforms = d_transforms;
        m_sceneDesc.materials = d_materials;
        m_sceneDesc.lights = d_lights;
        m_sceneDesc.cameras = d_cameras;

        m_sceneDesc.tlasNodes = d_tlasNodes;
        m_sceneDesc.blasRanges = d_blasRanges;
        m_sceneDesc.blasNodes = d_blasNodes;
        m_sceneDesc.blasNodeCount = static_cast<uint32_t>(m_blasNodes.size());
        m_sceneDesc.tlasNodeCount = static_cast<uint32_t>(m_tlasNodes.size());

        m_sceneDesc.triCount = static_cast<uint32_t>(triCount);
        m_sceneDesc.meshCount = static_cast<uint32_t>(meshCount);
        m_sceneDesc.instanceCount = static_cast<uint32_t>(instCount);
        m_sceneDesc.transformCount = static_cast<uint32_t>(xfCount);
        m_sceneDesc.materialCount = static_cast<uint32_t>(matCount);
        m_sceneDesc.lightCount = static_cast<uint32_t>(lightCount);
        m_sceneDesc.cameraCount = static_cast<uint32_t>(camCount);
    }

    void PathTracerSYCL::buildBLASForAllMeshes() {
        m_blasNodes.clear(); // flat pool that will hold *every* mesh BVH
        m_blasRanges.clear(); // one record per mesh
        m_blasNames.clear(); // purely for debug drawing

        /* --------------------------------------------------------------------- */
        /* Loop over *unique* meshes (one per entry in m_meshRanges)              */
        /* --------------------------------------------------------------------- */
        for (uint32_t m = 0; m < m_meshRanges.size(); ++m) {
            const MeshRange &mr = m_meshRanges[m];
            const std::string name = m_meshNames[m]; // for the UI

            // ──────────────────────────────────────────────────────────────
            // 1.  Gather **local vertices** (just copy the structs)
            // ──────────────────────────────────────────────────────────────
            std::vector<Vertex> localVerts;
            localVerts.reserve(mr.vertCount);

            for (uint32_t v = 0; v < mr.vertCount; ++v)
                localVerts.push_back(m_vertices[mr.firstVert + v]);

            // ──────────────────────────────────────────────────────────────
            // 2.  Gather & re‑index triangles so they refer to localVerts[]
            // ──────────────────────────────────────────────────────────────
            std::vector<Triangle> localTris;
            localTris.reserve(mr.triCount);

            for (uint32_t t = 0; t < mr.triCount; ++t) {
                Triangle T = m_tris[mr.firstTri + t];
                T.v0 -= mr.firstVert; // now between 0 … mr.vertCount‑1
                T.v1 -= mr.firstVert;
                T.v2 -= mr.firstVert;
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
                            /*maxLeaf*/ 8);

            //---------------- 4.  Patch child indices so they point into          */
            //----------------     the *global* triangle array again -------------
            for (BVHNode &N: localNodes) {
                if (N.isLeaf()) {
                    // leaf: convert local triangle index → global index
                    N.leftFirst += mr.firstTri;
                } else {
                    // internal: children become global node indices
                    N.leftFirst += m_blasNodes.size();
                }
            }

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

    //──────────────────────────────────────────────────────────────────────────
    // Build TLAS over *instances*  (one leaf = one Instance struct)
    //──────────────────────────────────────────────────────────────────────────
    void PathTracerSYCL::buildTopLevelBVH() {
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
            const BLASRange &br = m_blasRanges[inst.geomIndex];
            const BVHNode &root = m_blasNodes[br.firstNode];

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


    void PathTracerSYCL::freeDeviceMemory() {
        // free descriptor
        if (d_sceneDesc) free(d_sceneDesc, m_queue);
        // free buffers
        auto freeIf = [&](void *p) {
            if (p) {
                free(p, m_queue);
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
