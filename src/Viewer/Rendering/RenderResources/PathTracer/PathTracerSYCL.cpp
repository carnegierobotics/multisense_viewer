//
// Created by magnus on 5/2/25.
//

#include "PathTracerSYCL.h"

#include <Viewer/Rendering/MeshManager.h>
#include <Viewer/Scenes/Entity.h>


namespace VkRender {

// -------------------------
// PathTracerSYCL impl
// -------------------------
PathTracerSYCL::PathTracerSYCL(
    sycl::queue &m_queue,
    uint32_t m_width, uint32_t m_height
) : m_queue(m_queue), m_width(m_width), m_height(m_height) {
    d_image = sycl::malloc_device<uint8_t>(m_width * m_height * 4, m_queue);
}

PathTracerSYCL::~PathTracerSYCL() {
    sycl::free(d_vertices, m_queue);
    sycl::free(d_tris,     m_queue);
    sycl::free(d_mats,     m_queue);
    sycl::free(d_nodes,    m_queue);
    sycl::free(d_image,    m_queue);
}

void PathTracerSYCL::uploadScene(VkRender::Scene &scene) {
    std::vector<PathTracer::Vertex>   verts;
    std::vector<PathTracer::Triangle> tris;
    std::vector<PathTracer::Material> mats;

    auto view = scene.getRegistry().view<
        MeshComponent,
        MaterialComponent,
        TransformComponent
    >();
    for (auto id : view) {
        Entity e(id, &scene);
        auto &mc  = e.getComponent<MeshComponent>();
        auto &mat = e.getComponent<MaterialComponent>();
        auto &xf  = e.getComponent<TransformComponent>();
        auto mesh = MeshManager::instance().getMeshData(mc);

        size_t baseV = verts.size();
        for (auto &v : mesh->m_vertices) {
            PathTracer::Vertex vv;
            // TODO: apply xf transform to v.position & v.normal
            vv.x = v.pos.x;
            vv.y = v.pos.y;
            vv.z = v.pos.z;
            verts.push_back(vv);
        }
        PathTracer::Material M;
        // TODO: fill from mat.albedo/emission
        M.diffuse[0] = 1.0f; M.diffuse[1] = 1.0f; M.diffuse[2] = 1.0f;
        M.emission[0] = 0.0f; M.emission[1] = 0.0f; M.emission[2] = 0.0f;
        mats.push_back(M);

        for (size_t i = 0; i + 2 < mesh->m_indices.size(); i += 3) {
            tris.push_back({
                uint32_t(baseV + mesh->m_indices[i + 0]),
                uint32_t(baseV + mesh->m_indices[i + 1]),
                uint32_t(baseV + mesh->m_indices[i + 2]),
                uint32_t(mats.size() - 1)
            });
        }
    }

    m_vertexCount = verts.size();
    m_triCount    = tris.size();
    m_matCount    = mats.size();

    d_vertices = sycl::malloc_device<PathTracer::Vertex>(m_vertexCount, m_queue);
    d_tris     = sycl::malloc_device<PathTracer::Triangle>(m_triCount, m_queue);
    d_mats     = sycl::malloc_device<PathTracer::Material>(m_matCount, m_queue);

    m_queue.memcpy(d_vertices, verts.data(), m_vertexCount * sizeof(PathTracer::Vertex));
    m_queue.memcpy(d_tris,     tris.data(),    m_triCount    * sizeof(PathTracer::Triangle));
    m_queue.memcpy(d_mats,     mats.data(),    m_matCount    * sizeof(PathTracer::Material));
    m_queue.wait();

    std::vector<PathTracer::BVHNode> nodes;
    buildBVH_CPU(verts, tris, nodes);
    m_nodeCount = nodes.size();
    d_nodes     = sycl::malloc_device<PathTracer::BVHNode>(m_nodeCount, m_queue);
    m_queue.memcpy(d_nodes, nodes.data(), m_nodeCount * sizeof(PathTracer::BVHNode));
    m_queue.wait();
}

std::vector<uint8_t> PathTracerSYCL::render() {
    PathTracer::PathTracerKernels kernels(
        m_queue,
        d_vertices, m_vertexCount,
        d_tris,     m_triCount,
        d_mats,
        d_nodes,    m_nodeCount,
        d_image,
        m_width, m_height
    );
    kernels.renderFrame();

    std::vector<uint8_t> out(m_width * m_height * 4);
    m_queue.memcpy(out.data(), d_image, out.size()).wait();
    return out;
}

void PathTracerSYCL::buildBVH_CPU(
    const std::vector<PathTracer::Vertex> &verts,
    const std::vector<PathTracer::Triangle> &tris,
    std::vector<PathTracer::BVHNode> &nodes
) {
    // TODO: implement SAH or median-split BVH builder
    }
}
