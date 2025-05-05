//
// Created by magnus on 5/2/25.
//

#include "PathTracerSYCL.h"

#include <Viewer/Rendering/MeshManager.h>
#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Scenes/Entity.h>

#include "KernelHelpers.h"


namespace VkRender {
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
        d_frameBuffers = deviceAlloc<PathTracer::FrameBuffer>(1);
        m_queue.memcpy(d_frameBuffers, &m_frameBuffers,sizeof(PathTracer::FrameBuffer)).wait();
    }


    void PathTracerSYCL::uploadScene(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera) {
        // free existing GPU memory
        freeDeviceMemory();
        collectCameras(scene, editorCamera);
        // collect host data
        collectGeometry(scene);
        collectInstances(scene);
        collectLights(scene);

        buildBLASForAllMeshes();
        buildTopLevelBVH();
        // build and upload scene descriptor
        buildSceneDesc();
        d_sceneDesc = deviceAlloc<PathTracer::SceneDesc>(1);
        m_queue.memcpy(d_sceneDesc, &m_sceneDesc,sizeof(PathTracer::SceneDesc)).wait();
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
            PathTracer::Camera cam{};
            cam.width = editorCamera.editorWidth;
            cam.height = editorCamera.editorHeight;
            cam.pos = PathTracer::glm2sycl(editorCamera.camera->matrices.position);
            cam.proj = PathTracer::glm2sycl(editorCamera.camera->matrices.projection);
            cam.view = PathTracer::glm2sycl(editorCamera.camera->matrices.view);
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
            PathTracer::Camera cam;
            cam.width = static_cast<uint32_t>(sceneCameraParameters.width);
            cam.height = static_cast<uint32_t>(sceneCameraParameters.height);
            cam.pos = PathTracer::glm2sycl(transformComponent.getPosition());
            cam.proj = PathTracer::glm2sycl(cameraComponent.camera->matrices.projection);
            cam.view = PathTracer::glm2sycl(cameraComponent.camera->matrices.view);
            cam.firstPixel = pixelOffset;

            pixelOffset += cam.width * cam.height * 4;
            m_cameras.push_back(cam);
        }


        if (pixelOffset >= m_createInfo.framebufferSize) {
            Log::Logger::getInstance()->error("More cameras than framebuffers");
            throw std::runtime_error("PathTracerSYCL::PathTracerSYCL(): More cameras than framebuffers");
        }
    }


    void PathTracerSYCL::updateDynamic(const std::shared_ptr<Scene> &scene) {
        m_queue.memcpy(d_transforms, m_transforms.data(),
                       m_transforms.size() * sizeof(PathTracer::Transform));
        m_queue.memcpy(d_lights, m_lights.data(),
                       m_lights.size() * sizeof(PathTracer::MeshLight));
    }

    void PathTracerSYCL::renderFrame() {
        if (!d_sceneDesc) {
            return;
        }
        uint32_t photonCount = 1;

        m_queue.submit([scene=d_sceneDesc, fb = d_frameBuffers, photonCount](sycl::handler &cgh) {
            PathTracer::PathTracerMeshKernel kernel(scene, fb);
            cgh.parallel_for(sycl::range<1>(photonCount), kernel);
        }).wait();
    }

    void PathTracerSYCL::generateImages(std::span<std::byte> outRGBA32f) {
        size_t bytes = outRGBA32f.size();
        //m_queue.memcpy(outRGBA32f.data(), d_frameBuffer, bytes).wait();
    }

    void PathTracerSYCL::collectGeometry(const std::shared_ptr<Scene> &scene) {
        m_px.clear();
        m_py.clear();
        m_pz.clear();
        m_nx.clear();
        m_ny.clear();
        m_nz.clear();
        m_tris.clear();
        m_meshRanges.clear();

        // collect unique meshes
        auto view = scene->getRegistry().view<MeshComponent>();
        for (auto id: view) {
            Entity e(id, scene.get());
            std::string name = e.getName();
            auto &mc = e.getComponent<MeshComponent>();
            std::string mid = mc.getCacheIdentifier();
            if (m_meshIndexMap.count(mid)) continue;
            auto mesh = MeshManager::instance().getMeshData(mc);
            // base offsets
            uint32_t vertBase = static_cast<uint32_t>(m_px.size());
            uint32_t triBase = static_cast<uint32_t>(m_tris.size());

            // append vertices
            for (auto &v: mesh->m_vertices) {
                m_px.push_back(v.pos.x);
                m_py.push_back(v.pos.y);
                m_pz.push_back(v.pos.z);
                m_nx.push_back(v.normal.x);
                m_ny.push_back(v.normal.y);
                m_nz.push_back(v.normal.z);
            }

            // append triangles
            for (size_t i = 0; i < mesh->m_indices.size(); i += 3) {
                PathTracer::Triangle t{};
                t.v0 = mesh->m_indices[i + 0] + vertBase;
                t.v1 = mesh->m_indices[i + 1] + vertBase;
                t.v2 = mesh->m_indices[i + 2] + vertBase;
                m_tris.push_back(t);
            }

            // record mesh range
            PathTracer::MeshRange range{};
            range.firstVert = vertBase;
            range.vertCount = static_cast<uint32_t>(mesh->m_vertices.size());
            range.firstTri = triBase;
            range.triCount = static_cast<uint32_t>(mesh->m_indices.size() / 3);
            m_meshRanges.push_back(range);

            m_meshIndexMap[mid] = static_cast<uint32_t>(m_meshRanges.size() - 1);
        }
    }

    void PathTracerSYCL::collectInstances(const std::shared_ptr<Scene> &scene) {
        m_instances.clear();
        m_transforms.clear();
        m_materials.clear(); // one material slot per instance

        auto view = scene->getRegistry().view<MeshComponent, MaterialComponent, TransformComponent>(
            entt::exclude<LightSourceComponent>);

        for (auto entID: view) {
            Entity e(entID, scene.get());
            std::string name = e.getName();
            // --- 1) look up the mesh index we built in collectGeometry() ---
            auto &mc = e.getComponent<MeshComponent>();
            const std::string mid = mc.getCacheIdentifier();
            uint32_t geomIdx = m_meshIndexMap.at(mid);

            // --- 2) append this entity's material parameters ---
            auto &matComp = e.getComponent<MaterialComponent>();
            PathTracer::Material gpuMat{};
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
                /* geomIndex      */ geomIdx,
                /* materialIndex  */ matIdx,
                /* transformIndex */ xfIdx,
                /* geomType       */ uint32_t(PathTracer::GeometryType::Mesh)
            });

            // --- 4) store the transform for this instance ---
            auto &tc = e.getComponent<TransformComponent>();
            PathTracer::Transform xf{};
            xf.objectToWorld = PathTracer::glm2sycl(tc.getTransform());
            xf.worldToObject = PathTracer::glm2sycl(glm::inverse(tc.getTransform()));

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

            const auto &mesh = MeshManager::instance().getMeshData(meshComponent); // supplies vertex & index arrays
            const auto world = transformComponent.getTransform();

            PathTracer::MeshLight ML;
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

            ML.transform.objectToWorld = PathTracer::glm2sycl(transformComponent.getTransform());
            ML.transform.worldToObject = PathTracer::glm2sycl(glm::inverse(transformComponent.getTransform()));

            m_lights.push_back(std::move(ML));
        }
    }


    void PathTracerSYCL::buildSceneDesc() {
        const size_t vertCount = m_px.size();

        //—— allocate & copy SoA (positions + normals) ——
        d_px = deviceAlloc<float>(vertCount);
        m_queue.memcpy(d_px, m_px.data(), vertCount * sizeof(float));

        d_py = deviceAlloc<float>(vertCount);
        m_queue.memcpy(d_py, m_py.data(), vertCount * sizeof(float));

        d_pz = deviceAlloc<float>(vertCount);
        m_queue.memcpy(d_pz, m_pz.data(), vertCount * sizeof(float));

        d_nx = deviceAlloc<float>(vertCount);
        m_queue.memcpy(d_nx, m_nx.data(), vertCount * sizeof(float));

        d_ny = deviceAlloc<float>(vertCount);
        m_queue.memcpy(d_ny, m_ny.data(), vertCount * sizeof(float));

        d_nz = deviceAlloc<float>(vertCount);
        m_queue.memcpy(d_nz, m_nz.data(), vertCount * sizeof(float));

        //—— allocate & copy indexed arrays ——
        const size_t triCount = m_tris.size();
        d_tris = deviceAlloc<PathTracer::Triangle>(triCount);
        m_queue.memcpy(d_tris, m_tris.data(), triCount * sizeof(PathTracer::Triangle));

        const size_t meshCount = m_meshRanges.size();
        d_meshRanges = deviceAlloc<PathTracer::MeshRange>(meshCount);
        m_queue.memcpy(d_meshRanges, m_meshRanges.data(), meshCount * sizeof(PathTracer::MeshRange));

        const size_t instCount = m_instances.size();
        d_instances = deviceAlloc<PathTracer::Instance>(instCount);
        m_queue.memcpy(d_instances, m_instances.data(), instCount * sizeof(PathTracer::Instance));

        const size_t xfCount = m_transforms.size();
        d_transforms = deviceAlloc<PathTracer::Transform>(xfCount);
        m_queue.memcpy(d_transforms, m_transforms.data(), xfCount * sizeof(PathTracer::Transform));

        const size_t matCount = m_materials.size();
        d_materials = deviceAlloc<PathTracer::Material>(matCount);
        m_queue.memcpy(d_materials, m_materials.data(), matCount * sizeof(PathTracer::Material));

        const size_t lightCount = m_lights.size();
        d_lights = deviceAlloc<PathTracer::MeshLight>(lightCount);
        m_queue.memcpy(d_lights, m_lights.data(), lightCount * sizeof(PathTracer::MeshLight));

        const size_t camCount = m_cameras.size();
        d_cameras = deviceAlloc<PathTracer::Camera>(camCount);
        m_queue.memcpy(d_cameras, m_cameras.data(), camCount * sizeof(PathTracer::Camera));


        size_t blasCount = m_blasNodes.size();
        d_blasNodes = deviceAlloc<PathTracer::BVHNode>(blasCount);
        m_queue.memcpy(d_blasNodes, m_blasNodes.data(), blasCount * sizeof(PathTracer::BVHNode));
        m_sceneDesc.blasNodes = d_blasNodes;
        m_sceneDesc.blasNodeCount = static_cast<uint32_t>(blasCount);
        // copy BLAS ranges
        size_t blasRangesSize = m_blasRanges.size();
        d_blasRanges = deviceAlloc<PathTracer::BLASRange>(blasRangesSize);
        m_queue.memcpy(d_blasRanges, m_blasRanges.data(), blasRangesSize * sizeof(PathTracer::BLASRange));
        m_sceneDesc.blasRanges = d_blasRanges;

        // copy TLAS
        size_t tlasCount = m_tlasNodes.size();
        d_tlasNodes = deviceAlloc<PathTracer::BVHNode>(tlasCount);
        m_queue.memcpy(d_tlasNodes, m_tlasNodes.data(), tlasCount * sizeof(PathTracer::BVHNode));
        m_sceneDesc.tlas = d_tlasNodes;
        m_sceneDesc.tlasNodeCount = static_cast<uint32_t>(tlasCount);

        //—— fill SceneDesc ——
        m_sceneDesc.vertices = PathTracer::VertexSOA{d_px, d_py, d_pz, d_nx, d_ny, d_nz};
        m_sceneDesc.triangles = d_tris;
        m_sceneDesc.meshes = d_meshRanges;
        m_sceneDesc.instances = d_instances;
        m_sceneDesc.transforms = d_transforms;
        m_sceneDesc.materials = d_materials;
        m_sceneDesc.lights = d_lights;
        m_sceneDesc.cameras = d_cameras;

        m_sceneDesc.triCount = static_cast<uint32_t>(triCount);
        m_sceneDesc.meshCount = static_cast<uint32_t>(meshCount);
        m_sceneDesc.instanceCount = static_cast<uint32_t>(instCount);
        m_sceneDesc.transformCount = static_cast<uint32_t>(xfCount);
        m_sceneDesc.materialCount = static_cast<uint32_t>(matCount);
        m_sceneDesc.lightCount = static_cast<uint32_t>(lightCount);
        m_sceneDesc.cameraCount = static_cast<uint32_t>(camCount);
    }

    // CPU-side BVH building method integrated into PathTracerSYCL
    void PathTracerSYCL::buildBLASForAllMeshes() {
        m_blasNodes.clear();
        m_blasRanges.clear();

        for (uint32_t m = 0; m < m_meshRanges.size(); ++m) {
            auto &mr = m_meshRanges[m];

            // 1) Gather local vertices
            std::vector<float> pxBLAS, pyBLAS, pzBLAS;
            pxBLAS.reserve(mr.vertCount);
            pyBLAS.reserve(mr.vertCount);
            pzBLAS.reserve(mr.vertCount);
            for (uint32_t v = 0; v < mr.vertCount; ++v) {
                pxBLAS.push_back(m_px[mr.firstVert + v]);
                pyBLAS.push_back(m_py[mr.firstVert + v]);
                pzBLAS.push_back(m_pz[mr.firstVert + v]);
            }

            // 2) Gather & remap triangles
            std::vector<PathTracer::Triangle> triBLAS;
            triBLAS.reserve(mr.triCount);
            for (uint32_t t = 0; t < mr.triCount; ++t) {
                auto tri = m_tris[mr.firstTri + t];
                tri.v0 -= mr.firstVert;
                tri.v1 -= mr.firstVert;
                tri.v2 -= mr.firstVert;
                triBLAS.push_back(tri);
            }

            // 3) Build into a local node array
            std::vector<PathTracer::BVHNode> localNodes;
            buildBVHNodes(triBLAS, pxBLAS, pyBLAS, pzBLAS, localNodes);

            // 4) Patch leaf offsets to global
            //    firstNode = where these nodes will live in m_blasNodes
            uint32_t firstNode = uint32_t(m_blasNodes.size());
            for (auto &n : localNodes) {
                if (n.count == 0) {
                    // internal node: child indices are relative to localNodes
                    n.leftChild  += firstNode;
                    n.rightChild += firstNode;
                } else {
                    // leaf: tri‐list offset
                    n.leftChild += mr.firstTri;
                }
            }

            // 5) Append into the big BLAS tree
            m_blasNodes.insert(m_blasNodes.end(),
                               localNodes.begin(),
                               localNodes.end());
            m_blasRanges.push_back({ firstNode,
                                     uint32_t(localNodes.size()) });
        }
    }

    // CPU-side BVH building method integrated into PathTracerSYCL
    void PathTracerSYCL::buildBVHNodes(
        const std::vector<PathTracer::Triangle> &triangles,
        const std::vector<float> &px,
        const std::vector<float> &py,
        const std::vector<float> &pz,
        std::vector<PathTracer::BVHNode> &outNodes) {
        if (triangles.empty()) return;

        // working copy for partitioning
        std::vector<PathTracer::Triangle> triBuf = triangles;

        // clear & reserve
        outNodes.clear();
        outNodes.reserve(triBuf.size() * 2);

        // recursive builder returns index of the node it just created
        std::function<int(int, int)> build = [&](int start, int end) -> int {
            int nodeIdx = static_cast<int>(outNodes.size());
            outNodes.emplace_back(); // append a new node
            auto &node = outNodes.back();

            // 1) compute bounding box for [start,end)
            sycl::float3 bmin{FLT_MAX}, bmax{-FLT_MAX};
            for (int i = start; i < end; ++i) {
                const auto &t = triBuf[i];
                for (int v = 0; v < 3; ++v) {
                    uint32_t vi = (v == 0 ? t.v0 : (v == 1 ? t.v1 : t.v2));
                    sycl::float3 p{px[vi], py[vi], pz[vi]};
                    bmin = sycl::min(bmin, p);
                    bmax = sycl::max(bmax, p);
                }
            }
            node.bboxMin = bmin;
            node.bboxMax = bmax;

            int nPrims = end - start;
            if (nPrims <= 2) {
                // --- leaf ------------------------
                node.count     = nPrims;
                node.leftChild = start;    // index into *local* triBuf
                node.rightChild= 0;
            } else {
                // --- internal --------------------
                node.count = 0;

                // split by centroid
                sycl::float3 cmin{FLT_MAX}, cmax{-FLT_MAX};
                for (int i = start; i < end; ++i) {
                    const auto &t = triBuf[i];
                    sycl::float3 v0{px[t.v0], py[t.v0], pz[t.v0]};
                    sycl::float3 v1{px[t.v1], py[t.v1], pz[t.v1]};
                    sycl::float3 v2{px[t.v2], py[t.v2], pz[t.v2]};
                    sycl::float3 cent = (v0 + v1 + v2) / 3.f;
                    cmin = sycl::min(cmin, cent);
                    cmax = sycl::max(cmax, cent);
                }
                sycl::float3 ext = cmax - cmin;
                int axis = (ext.x() > ext.y() && ext.x() > ext.z()
                                ? 0
                                : ext.y() > ext.z()
                                      ? 1
                                      : 2);
                float mid = (cmin[axis] + cmax[axis]) * 0.5f;

                // partition
                auto it = std::partition(triBuf.begin() + start, triBuf.begin() + end,
                                         [&](auto &t) {
                                             sycl::float3 v0{px[t.v0], py[t.v0], pz[t.v0]};
                                             sycl::float3 v1{px[t.v1], py[t.v1], pz[t.v1]};
                                             sycl::float3 v2{px[t.v2], py[t.v2], pz[t.v2]};
                                             return ((v0 + v1 + v2) / 3.f)[axis] < mid;
                                         });
                int midIdx = static_cast<int>(it - triBuf.begin());
                if (midIdx == start || midIdx == end)
                    midIdx = start + nPrims / 2;

                // recursively build children
                node.leftChild = build(start, midIdx);
                node.rightChild = build(midIdx, end);
            }

            return nodeIdx;
        };

        // kick off recursion
        build(0, static_cast<int>(triBuf.size()));
    }

    //------------------------------------------------------------------------------
    // Build TLAS over instance AABBs (world-space, handles non-uniform scales)
    //------------------------------------------------------------------------------
    void PathTracerSYCL::buildTopLevelBVH() {
        // 1) Gather each instance’s world-space AABB
        struct AABB {
            sycl::float3 min, max;
            uint32_t instIdx;
        };
        std::vector<AABB> boxes;
        boxes.reserve(m_instances.size());

        for (uint32_t i = 0; i < m_instances.size(); ++i) {
            const auto &inst = m_instances[i];
            const auto &xf = m_transforms[inst.transformIndex];

            // fetch the BLAS root node for this mesh
            const auto &root = m_blasNodes[m_blasRanges[inst.geomIndex].firstNode];

            // build the 8 object-space corners
            sycl::float3 corners[8] = {
                {root.bboxMin.x(), root.bboxMin.y(), root.bboxMin.z()},
                {root.bboxMin.x(), root.bboxMin.y(), root.bboxMax.z()},
                {root.bboxMin.x(), root.bboxMax.y(), root.bboxMin.z()},
                {root.bboxMin.x(), root.bboxMax.y(), root.bboxMax.z()},
                {root.bboxMax.x(), root.bboxMin.y(), root.bboxMin.z()},
                {root.bboxMax.x(), root.bboxMin.y(), root.bboxMax.z()},
                {root.bboxMax.x(), root.bboxMax.y(), root.bboxMin.z()},
                {root.bboxMax.x(), root.bboxMax.y(), root.bboxMax.z()}
            };

            // transform to world and find min/max
            sycl::float3 wmin{FLT_MAX}, wmax{-FLT_MAX};
            for (int c = 0; c < 8; ++c) {
                sycl::float4 wc = xf.objectToWorld * sycl::float4{corners[c], 1.0f};
                sycl::float3 p{wc.x(), wc.y(), wc.z()};
                wmin = sycl::min(wmin, p);
                wmax = sycl::max(wmax, p);
            }

            boxes.push_back({wmin, wmax, i});
        }

        // 2) Recursively build the TLAS into m_tlasNodes
        m_tlasNodes.clear();
        m_tlasNodes.reserve(boxes.size() * 2);

        std::function<int(int, int)> buildTL = [&](int start, int end) -> int {
            int nodeIdx = static_cast<int>(m_tlasNodes.size());
            m_tlasNodes.emplace_back();
            auto &node = m_tlasNodes.back();

            // compute this node’s bounds
            sycl::float3 bmin{FLT_MAX}, bmax{-FLT_MAX};
            for (int i = start; i < end; ++i) {
                bmin = sycl::min(bmin, boxes[i].min);
                bmax = sycl::max(bmax, boxes[i].max);
            }
            node.bboxMin = bmin;
            node.bboxMax = bmax;

            int count = end - start;
            if (count == 1) {
                // Leaf: one instance
                node.count = 1;
                node.leftChild = boxes[start].instIdx; // instIdx baked in
                node.rightChild = 0; // unused
            } else {
                // Internal: split by centroid
                //  compute centroid bounds
                sycl::float3 cmin{FLT_MAX}, cmax{-FLT_MAX};
                for (int i = start; i < end; ++i) {
                    sycl::float3 cent = (boxes[i].min + boxes[i].max) * 0.5f;
                    cmin = sycl::min(cmin, cent);
                    cmax = sycl::max(cmax, cent);
                }
                sycl::float3 ext = cmax - cmin;
                int axis = (ext.x() > ext.y() && ext.x() > ext.z()
                                ? 0
                                : ext.y() > ext.z()
                                      ? 1
                                      : 2);
                float mid = (cmin[axis] + cmax[axis]) * 0.5f;

                // partition the range
                auto it = std::partition(
                    boxes.begin() + start, boxes.begin() + end,
                    [&](auto &b) {
                        sycl::float3 cent = (b.min + b.max) * 0.5f;
                        return cent[axis] < mid;
                    }
                );
                int midIdx = static_cast<int>(it - boxes.begin());
                // fallback to half-split if unbalanced
                if (midIdx == start || midIdx == end)
                    midIdx = start + count / 2;

                // recurse
                node.count = 0; // internal
                node.leftChild = buildTL(start, midIdx);
                node.rightChild = buildTL(midIdx, end);
            }

            return nodeIdx;
        };

        if (!boxes.empty())
            buildTL(0, static_cast<int>(boxes.size()));
    }


    void PathTracerSYCL::freeDeviceMemory() {
        // free descriptor
        if (d_sceneDesc) free(d_sceneDesc, m_queue);
        // free buffers
        auto freeIf = [&](void *p) {
            if (p) free(p, m_queue);
            p = nullptr;
        };
        freeIf(d_px);
        freeIf(d_py);
        freeIf(d_pz);
        freeIf(d_nx);
        freeIf(d_ny);
        freeIf(d_nz);
        freeIf(d_tris);
        freeIf(d_meshRanges);
        freeIf(d_instances);
        freeIf(d_transforms);
        freeIf(d_materials);
        freeIf(d_lights);
        freeIf(d_cameras);
        freeIf(d_blasNodes);
        freeIf(d_tlasNodes);
        freeIf(d_blasRanges);
    }
}
