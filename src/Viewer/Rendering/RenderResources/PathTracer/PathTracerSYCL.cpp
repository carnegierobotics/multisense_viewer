//
// Created by magnus on 5/2/25.
//

#include "PathTracerSYCL.h"

#include <Viewer/Rendering/MeshManager.h>
#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Scenes/Entity.h>


namespace VkRender {


PathTracerSYCL::~PathTracerSYCL() {
    freeDeviceMemory();
}

void PathTracerSYCL::uploadScene(const std::shared_ptr<Scene>& scene) {
    // free existing GPU memory
    freeDeviceMemory();

    // collect host data
    collectGeometry(scene);
    collectInstances(scene);
    collectLights(scene);
    collectCameras(scene);

    // build and upload scene descriptor
    buildSceneDesc();
    d_sceneDesc = static_cast<PathTracer::SceneDesc*>(
        sycl::malloc_device(sizeof(PathTracer::SceneDesc), m_queue));
    m_queue.memcpy(d_sceneDesc, &m_SceneDesc,
                   sizeof(PathTracer::SceneDesc)).wait();

    // setup output
    setupFrameBuffers();
}

void PathTracerSYCL::setupFrameBuffers() {
    /*
    // free old
    if (d_frameBuffer)
        sycl::free(d_frameBuffer, m_queue);

    // total pixels
    size_t total = 0;
    for (auto &cam : m_cameras) {
        total += size_t(cam.width) * cam.height;
    }

    d_frameBuffer = static_cast<float4*>(
        sycl::malloc_device(total * sizeof(float4), m_queue));
    m_queue.memset(d_frameBuffer, 0, total * sizeof(float4)).wait();
    */
}

void PathTracerSYCL::updateDynamic(const std::shared_ptr<Scene>& scene) {
    m_queue.memcpy(d_transforms, m_transforms.data(),
                   m_transforms.size()*sizeof(PathTracer::Transform));
    m_queue.memcpy(d_lights, m_lights.data(),
                   m_lights.size()*sizeof(PathTracer::AreaLight));
}

void PathTracerSYCL::renderFrame() {
    uint32_t photonCount = 0;
    for (auto &cam : m_cameras)
        photonCount += cam.width * cam.height;

    m_queue.submit([scene=d_sceneDesc, photonCount](sycl::handler &cgh) {
        PathTracer::PathTracerMeshKernel kernel(scene);
        cgh.parallel_for(sycl::range<1>(photonCount), kernel);
    }).wait();
}

void PathTracerSYCL::generateImages(std::span<std::byte> outRGBA32f) {
    size_t bytes = outRGBA32f.size();
    //m_queue.memcpy(outRGBA32f.data(), d_frameBuffer, bytes).wait();
}

void PathTracerSYCL::collectGeometry(const std::shared_ptr<Scene>& scene) {
    m_px.clear(); m_py.clear(); m_pz.clear();
    m_nx.clear(); m_ny.clear(); m_nz.clear();
    m_tris.clear(); m_meshRanges.clear(); m_bvh.clear();

    // collect unique meshes
    auto view = scene->getRegistry().view<MeshComponent>();
    for (auto id : view) {
        Entity e(id, scene.get());
        auto &mc = e.getComponent<MeshComponent>();
        std::string mid = mc.getCacheIdentifier();
        if (m_meshIndexMap.count(mid)) continue;
        auto mesh = MeshManager::instance().getMeshData(mc);
        // base offsets
        uint32_t vertBase = static_cast<uint32_t>(m_px.size());
        uint32_t triBase  = static_cast<uint32_t>(m_tris.size());

        // append vertices
        for (auto &v : mesh->m_vertices) {
            m_px.push_back(v.pos.x);
            m_py.push_back(v.pos.y);
            m_pz.push_back(v.pos.z);
            m_nx.push_back(v.normal.x);
            m_ny.push_back(v.normal.y);
            m_nz.push_back(v.normal.z);

        }

        // append triangles
        for (size_t i=0; i<mesh->m_indices.size(); i+=3) {
            PathTracer::Triangle t{};
            t.v0 = mesh->m_indices[i+0] + vertBase;
            t.v1 = mesh->m_indices[i+1] + vertBase;
            t.v2 = mesh->m_indices[i+2] + vertBase;
            m_tris.push_back(t);
        }

        // record mesh range
        PathTracer::MeshRange range{};
        range.firstVert = vertBase;
        range.vertCount = static_cast<uint32_t>(mesh->m_vertices.size());
        range.firstTri  = triBase;
        range.triCount  = static_cast<uint32_t>(mesh->m_indices.size()/3);
        m_meshRanges.push_back(range);

        m_meshIndexMap[mid] = static_cast<uint32_t>(m_meshRanges.size()-1);
    }
    // build BVH
        PathTracerSYCL::buildBVHNodes(m_tris, m_px, m_py, m_pz, m_bvh);
}

void PathTracerSYCL::collectInstances(const std::shared_ptr<Scene>& scene) {
    m_instances.clear();
    m_transforms.clear();
    m_materials.clear();        // one material slot per instance

    auto view = scene->getRegistry()
                    .view<MeshComponent, MaterialComponent, TransformComponent>();

    for (auto entID : view) {
        Entity e(entID, scene.get());

        // --- 1) look up the mesh index we built in collectGeometry() ---
        auto &mc = e.getComponent<MeshComponent>();
        const std::string mid = mc.getCacheIdentifier();
        uint32_t geomIdx = m_meshIndexMap.at(mid);

        // --- 2) append this entity's material parameters ---
        auto &matComp = e.getComponent<MaterialComponent>();
        PathTracer::Material gpuMat{};
        gpuMat.baseColor  = {
            matComp.albedo.x,
            matComp.albedo.y,
            matComp.albedo.z
        };
        gpuMat.specular   = {
            matComp.specular,
            matComp.specular,
            matComp.specular
        };
        gpuMat.phongExp   = matComp.phongExponent;

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
        // TODO convert glm to sycl matrix
        m_transforms.push_back(xf);
    }
}

void PathTracerSYCL::collectLights(const std::shared_ptr<Scene>& scene) {
    m_lights.clear();
    auto view = scene->getRegistry().view<LightSourceComponent, TransformComponent>();
    for (auto id : view) {
        Entity e(id, scene.get());
        auto &lightSourceComponent = e.getComponent<LightSourceComponent>();
        auto &transformComponent = e.getComponent<TransformComponent>();
        PathTracer::AreaLight L;
        // TODO IMPLEMENT L.origin   = tc.worldMatrix() * sycl::float4(0,0,0,1);
        // TODO IMPLEMENT L.edgeU    = tc.worldMatrix() * sycl::float4(1,0,0,0);
        // TODO IMPLEMENT L.edgeV    = tc.worldMatrix() * sycl::float4(0,1,0,0);
        // TODO IMPLEMENT L.area     = sycl::length(sycl::cross(L.edgeU, L.edgeV));
        // TODO IMPLEMENT L.radiance = lc.emission;
        m_lights.push_back(L);
    }
}

void PathTracerSYCL::collectCameras(const std::shared_ptr<Scene>& scene) {
    m_cameras.clear();
    uint32_t pixelOffset = 0;
    auto view = scene->getRegistry().view<CameraComponent, TransformComponent>();
    for (auto id : view) {
        Entity e(id, scene.get());
        auto &cc = e.getComponent<CameraComponent>();
        auto &tc = e.getComponent<TransformComponent>();
        PathTracer::Camera cam;
        // TODO IMPLEMENT  cam.view       = cc.viewMatrix();
        // TODO IMPLEMENT  cam.proj       = cc.projMatrix();
        // TODO IMPLEMENT  cam.pos        = tc.worldMatrix() * sycl::float4(0,0,0,1);
        // TODO IMPLEMENT  cam.width      = cc.resolution.x;
        // TODO IMPLEMENT  cam.height     = cc.resolution.y;
         cam.firstPixel = pixelOffset;
        pixelOffset   += cam.width*cam.height;
        m_cameras.push_back(cam);
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
    const size_t triCount  = m_tris.size();
    d_tris      = deviceAlloc<PathTracer::Triangle>(triCount);
    m_queue.memcpy(d_tris, m_tris.data(), triCount * sizeof(PathTracer::Triangle));

    const size_t meshCount = m_meshRanges.size();
    d_meshRanges = deviceAlloc<PathTracer::MeshRange>(meshCount);
    m_queue.memcpy(d_meshRanges, m_meshRanges.data(), meshCount * sizeof(PathTracer::MeshRange));

    const size_t bvhCount  = m_bvh.size();
    d_bvh        = deviceAlloc<PathTracer::BVHNode>(bvhCount);
    m_queue.memcpy(d_bvh, m_bvh.data(), bvhCount * sizeof(PathTracer::BVHNode));

    const size_t instCount = m_instances.size();
    d_instances  = deviceAlloc<PathTracer::Instance>(instCount);
    m_queue.memcpy(d_instances, m_instances.data(), instCount * sizeof(PathTracer::Instance));

    const size_t xfCount   = m_transforms.size();
    d_transforms = deviceAlloc<PathTracer::Transform>(xfCount);
    m_queue.memcpy(d_transforms, m_transforms.data(), xfCount * sizeof(PathTracer::Transform));

    const size_t matCount  = m_materials.size();
    d_materials  = deviceAlloc<PathTracer::Material>(matCount);
    m_queue.memcpy(d_materials, m_materials.data(), matCount * sizeof(PathTracer::Material));

    const size_t lightCount = m_lights.size();
    d_lights     = deviceAlloc<PathTracer::AreaLight>(lightCount);
    m_queue.memcpy(d_lights, m_lights.data(), lightCount * sizeof(PathTracer::AreaLight));

    const size_t camCount   = m_cameras.size();
    d_cameras    = deviceAlloc<PathTracer::Camera>(camCount);
    m_queue.memcpy(d_cameras, m_cameras.data(), camCount * sizeof(PathTracer::Camera));

    //—— fill SceneDesc ——
    m_SceneDesc.vertices       = PathTracer::VertexSOA{ d_px, d_py, d_pz, d_nx, d_ny, d_nz };
    m_SceneDesc.triangles      = d_tris;
    m_SceneDesc.meshes         = d_meshRanges;
    m_SceneDesc.bvh            = d_bvh;
    m_SceneDesc.instances      = d_instances;
    m_SceneDesc.transforms     = d_transforms;
    m_SceneDesc.materials      = d_materials;
    m_SceneDesc.lights         = d_lights;
    m_SceneDesc.cameras        = d_cameras;

    m_SceneDesc.triCount       = static_cast<uint32_t>(triCount);
    m_SceneDesc.meshCount      = static_cast<uint32_t>(meshCount);
    m_SceneDesc.instanceCount  = static_cast<uint32_t>(instCount);
    m_SceneDesc.transformCount = static_cast<uint32_t>(xfCount);
    m_SceneDesc.materialCount  = static_cast<uint32_t>(matCount);
    m_SceneDesc.lightCount     = static_cast<uint32_t>(lightCount);
    m_SceneDesc.cameraCount    = static_cast<uint32_t>(camCount);
}

// CPU-side BVH building method integrated into PathTracerSYCL
void PathTracerSYCL::buildBVHNodes(
    const std::vector<PathTracer::Triangle> &triangles,
    const std::vector<float>    &px,
    const std::vector<float>    &py,
    const std::vector<float>    &pz,
    std::vector<PathTracer::BVHNode>        &outNodes)
{
    outNodes.clear();
    if (triangles.empty()) return;

    // working copy for in-place partitioning
    std::vector<PathTracer::Triangle> triBuf = triangles;
    outNodes.reserve(triangles.size() * 2);

    // recursive lambda for node construction
    std::function<int(int,int)> build = [&](int start, int end) {
        int idx = static_cast<int>(outNodes.size());
        outNodes.emplace_back();
        PathTracer::BVHNode &node = outNodes.back();

        // compute bounds
        sycl::float3 bmin{FLT_MAX,FLT_MAX,FLT_MAX}, bmax{-FLT_MAX,-FLT_MAX,-FLT_MAX};
        for (int i = start; i < end; ++i) {
            const auto &t = triBuf[i];
            for (int v = 0; v < 3; ++v) {
                uint32_t vi = (v==0?t.v0:(v==1?t.v1:t.v2));
                sycl::float3 p{px[vi], py[vi], pz[vi]};
                bmin = sycl::min(bmin, p);
                bmax = sycl::max(bmax, p);
            }
        }
        node.bboxMin = bmin;
        node.bboxMax = bmax;

        int count = end - start;
        if (count <= 2) {
            node.leftFirst = start;
            node.count     = count;
        } else {
            // centroid bounds
            sycl::float3 cmin{FLT_MAX,FLT_MAX,FLT_MAX}, cmax{-FLT_MAX,-FLT_MAX,-FLT_MAX};
            for (int i = start; i < end; ++i) {
                const auto &t = triBuf[i];
                sycl::float3 v0{px[t.v0],py[t.v0],pz[t.v0]};
                sycl::float3 v1{px[t.v1],py[t.v1],pz[t.v1]};
                sycl::float3 v2{px[t.v2],py[t.v2],pz[t.v2]};
                sycl::float3 c = (v0+v1+v2)/3.f;
                cmin = sycl::min(cmin, c);
                cmax = sycl::max(cmax, c);
            }
            sycl::float3 ext = cmax - cmin;
            int axis = (ext.x() > ext.y() && ext.x() > ext.z() ? 0
                    : ext.y() > ext.z() ? 1 : 2);
            float mid = (cmin[axis] + cmax[axis]) * 0.5f;

            auto it = std::partition(
                triBuf.begin()+start, triBuf.begin()+end,
                [&](auto &t) {
                    sycl::float3 v0{px[t.v0],py[t.v0],pz[t.v0]};
                    sycl::float3 v1{px[t.v1],py[t.v1],pz[t.v1]};
                    sycl::float3 v2{px[t.v2],py[t.v2],pz[t.v2]};
                    return ((v0+v1+v2)/3.f)[axis] < mid;
                }
            );
            int midIdx = static_cast<int>(it - triBuf.begin());
            if (midIdx==start || midIdx==end)
                midIdx = start + count/2;

            node.leftFirst = static_cast<uint32_t>(outNodes.size());
            node.count     = 0;
            build(start,  midIdx);
            build(midIdx, end);
        }
        return idx;
    };

    // kick off build
    build(0, static_cast<int>(triBuf.size()));
}


void PathTracerSYCL::freeDeviceMemory() {
    // free descriptor
    if (d_sceneDesc)           sycl::free(d_sceneDesc,    m_queue);
    // free buffers
    auto freeIf = [&](void* p){ if(p) sycl::free(p, m_queue); };
    freeIf(d_px);    freeIf(d_py);    freeIf(d_pz);
    freeIf(d_nx);    freeIf(d_ny);    freeIf(d_nz);
    freeIf(d_tris);  freeIf(d_meshRanges);
    freeIf(d_bvh);
    freeIf(d_instances);
    freeIf(d_transforms);
    freeIf(d_materials);
    freeIf(d_lights);
    freeIf(d_cameras);
}
}
