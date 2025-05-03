//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERSYCL_H
#define PATHTRACERSYCL_H

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracerKernels.h"

// -------------------------
// Main tracer class
// -------------------------
namespace VkRender {
class PathTracerSYCL {
public:
    explicit PathTracerSYCL(sycl::queue q) : m_queue(std::move(q)) { }

    /** (re)allocates all GPU buffers that depend on scene topology */
    void uploadScene(const Scene& scene);

    /** reallocates output image if #cameras / resolution changed */
    void setupFrameBuffers();

    /** per‑frame fast update of transforms, animated emissive, … */
    void updateDynamic(const Scene& scene);

    /** launches photon + contribution kernels */
    void renderFrame();

    /** copies the device framebuffer back to host */
    void generateImages(std::span<std::byte> outRGBA32f);

private:
    /*--- helpers called only from uploadScene() ---*/
    void collectGeometry(const Scene&);
    void collectInstances(const Scene&);
    void collectLights(const Scene&);
    void collectCameras(const Scene&);
    void buildSceneDesc();

    /*--- device clean‑up ---*/
    void freeDeviceMemory();

    /*----------------------------------------------*/
    sycl::queue m_queue;

    /* device‑side master descriptor */
    PathTracer::SceneDesc        m_SceneDesc{};   // host copy (fillable with std::vector)
    PathTracer::SceneDesc*       d_SceneDesc = nullptr;

    /* device arrays (raw USM pointers) --------------*/
    // geometry
    float*           d_px = nullptr;  // part of VertexSOA
    float*           d_py = nullptr;
    float*           d_pz = nullptr;
    float*           d_nx = nullptr;
    float*           d_ny = nullptr;
    float*           d_nz = nullptr;

    PathTracer::Triangle*        d_tris = nullptr;
    PathTracer::MeshRange*       d_meshRanges = nullptr;
    PathTracer::OrientedPoint*   d_points = nullptr;
    PathTracer::PointCloudRange* d_pcRanges = nullptr;
    PathTracer::BVHNode*         d_bvh = nullptr;

    // scene graph
    PathTracer::Instance*        d_instances = nullptr;
    PathTracer::Transform*       d_transforms = nullptr;

    // appearance
    PathTracer::Material*        d_materials = nullptr;
    PathTracer::AreaLight*       d_lights = nullptr;

    // view
    PathTracer::Camera*          d_cameras = nullptr;
    glm::vec4*                   d_framebuffer = nullptr;

    /* host‑side staging vectors for dynamic updates */
    std::vector<PathTracer::Transform>  m_transforms;
    std::vector<PathTracer::AreaLight>  m_lights;

    /* counts / state */
    uint32_t triCount{}, meshCount{}, pointCount{}, pcCount{};
    uint32_t instanceCount{}, transformCount{};
    uint32_t materialCount{}, lightCount{}, cameraCount{};
    size_t   framebufferPixels{0};
};


} // VkRender

#endif //PATHTRACERSYCL_H
