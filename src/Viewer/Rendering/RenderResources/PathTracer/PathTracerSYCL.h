//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERSYCL_H
#define PATHTRACERSYCL_H

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracerKernels.h"

namespace VkRender {
    // -------------------------
// Kernel launcher class
// -------------------------
class PathTracerKernels {
public:
    PathTracerKernels(
        sycl::queue &m_queue,
        Vertex *d_vertices, size_t m_vertexCount,
        Triangle *d_tris,   size_t m_triCount,
        Material *d_mats,
        BVHNode *d_nodes,   size_t m_nodeCount,
        uint8_t *d_image,
        uint32_t m_width, uint32_t m_height
    );

    // Launches SYCL kernel to fill image buffer
    void renderFrame();

private:
    sycl::queue &m_queue;
    Vertex     *d_vertices;
    Triangle   *d_tris;
    Material   *d_mats;
    BVHNode    *d_nodes;
    uint8_t    *d_image;
    uint32_t    m_width;
    uint32_t    m_height;

    bool intersectAABB(
        /* ray params */,
        const float minB[3], const float maxB[3]
    ) const;

    bool intersectTriangle(
        /* ray params */,
        const Vertex &v0,
        const Vertex &v1,
        const Vertex &v2,
        float &t
    ) const;
};

} // namespace VkRender::PathTracer

// -------------------------
// Main tracer class
// -------------------------
namespace VkRender {

class PathTracerSYCL {
public:
    PathTracerSYCL(sycl::queue &m_queue, uint32_t m_width, uint32_t m_height);
    ~PathTracerSYCL();

    // Uploads all meshes, materials, transforms from the scene
    void uploadScene(Scene &scene);

    // Renders one frame and returns CPU-side RGBA8 image
    std::vector<uint8_t> render();

private:
    sycl::queue &m_queue;
    uint32_t     m_width;
    uint32_t     m_height;

    PathTracer::Vertex *d_vertices = nullptr;
    size_t              m_vertexCount = 0;
    PathTracer::Triangle *d_tris     = nullptr;
    size_t              m_triCount    = 0;
    PathTracer::Material *d_mats     = nullptr;
    size_t              m_matCount    = 0;
    PathTracer::BVHNode *d_nodes     = nullptr;
    size_t              m_nodeCount   = 0;
    uint8_t            *d_image      = nullptr;

    // Builds CPU BVH and fills 'nodes'
    void buildBVH_CPU(
        const std::vector<PathTracer::Vertex> &verts,
        const std::vector<PathTracer::Triangle> &tris,
        std::vector<PathTracer::BVHNode> &nodes
    );
};

} // namespace VkRender


} // VkRender

#endif //PATHTRACERSYCL_H
