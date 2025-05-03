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

    void PathTracerSYCL::uploadScene(Scene& scene) {

        {
            auto view = scene.getRegistry().view<
               MeshComponent, MaterialComponent, TransformComponent
           >();
            for (auto id : view) {

            }
        }

        {
            auto view = scene.getRegistry().view<CameraComponent>();
            std::vector<PathTracer::Camera> cameras(view.size());

            for (size_t i = 0; auto id : view) {
                Entity e(id, &scene);
                auto& component = e.getComponent<CameraComponent>();
            }
        }

    }


    void PathTracerSYCL::uploadScene(const Scene& scene)
    {
        freeDeviceMemory();                 // if this is a reload

        collectGeometry(scene);                 // fills host vectors of verts, tris…
        collectInstances(scene);
        collectLights(scene);
        collectCameras(scene);

        /* 1. allocate USM ↓ */
        d_px = static_cast<float*>(sycl::malloc_device(sizeof(float)*px.size(), m_queue));
        /* …repeat for every array … */

        /* 2. copy immutable host → device ↓ */
        m_queue.memcpy(d_px, px.data(), px.size()*sizeof(float));
        /* …repeat… */

        /* 3. build the master descriptor (host side) */
        buildSceneDesc();                   // sets hSceneDesc.* pointers & counts

        /* 4. copy that single struct to device constant memory */
        d_SceneDesc = static_cast<PathTracer::SceneDesc*>(
            sycl::malloc_device(sizeof(PathTracer::SceneDesc), m_queue));
        m_queue.memcpy(d_SceneDesc, &m_SceneDesc, sizeof(PathTracer::SceneDesc)).wait();

        /* 5. allocate / resize framebuffers */
        setupFrameBuffers();
    }

    void PathTracerSYCL::setupFrameBuffers() {
    }

    void PathTracerSYCL::updateDynamic(const Scene& scene) {

        m_queue.memcpy(d_transforms, h_transforms.data(),
           h_transforms.size()*sizeof(PathTracer::Transform));
        m_queue.memcpy(d_lights, h_lights.data(),
                   h_lights.size()*sizeof(PathTracer::AreaLight));

    }

    void PathTracerSYCL::renderFrame() {
    }

    void PathTracerSYCL::generateImages(std::span<std::byte> outRGBA32f) {
    }

    void PathTracerSYCL::collectGeometry(const Scene&) {
    }

    void PathTracerSYCL::collectInstances(const Scene&) {
    }

    void PathTracerSYCL::collectLights(const Scene&) {
    }

    void PathTracerSYCL::collectCameras(const Scene&) {
    }

    void PathTracerSYCL::buildSceneDesc() {
    }

    void PathTracerSYCL::freeDeviceMemory() {
    }
}
