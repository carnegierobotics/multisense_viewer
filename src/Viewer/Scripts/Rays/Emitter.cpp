//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "Emitter.h"

#include <Viewer/Application/Application.h>

#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Rendering/PathTracer/Device/KernelHelpers.h>
#include <Viewer/Rendering/PathTracer/Device/PathTracerKernels.h>


namespace VkRender {
    void Emitter::onUpdate(Timestep ts) {
        if (!hasComponent<RasterizerRenderingComponent>())
            addComponent<RasterizerRenderingComponent>();

        if (!hasComponent<MeshComponent>())
            return;
        auto &meshComponent = getComponent<MeshComponent>();
        auto rayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(meshComponent.meshParameters);
        if (!rayParams)
            return;

        auto scene = m_entity.getScene();
        glm::vec3 rayOrigin(0.0f);
        glm::vec3 rayDir = rayParams->direction;

        auto &rayTransform = getComponent<TransformComponent>();

        auto view = scene->getRegistry().view<LightSourceComponent>();

        for (auto e: view) {
            Entity entity(e, scene);
            auto lightTransform = entity.getComponent<TransformComponent>();
            rayOrigin = lightTransform.getPosition();
        }

        rayParams->setOrigin(rayOrigin);

        auto blasNodes = m_pathTracerSYCL->getBLASNodes();
        auto tlasNodes = m_pathTracerSYCL->getTLASNodes();


        PathTracer::Ray worldRay = PathTracer::makeRay(PathTracer::glm2sycl(rayOrigin), PathTracer::glm2sycl(rayDir));

        auto sceneDescription = m_pathTracerSYCL->getSceneDescription();
        PathTracer::Hit hit;
        bool intersected = PathTracer::PathTracerMeshKernel::intersectScene(worldRay, &hit, sceneDescription);


        if (intersected) {
            // Calculate direction of hit point:
            glm::vec3 hitPoint = PathTracer::sycl2glm(hit.hitPoint);

            float magnitude = glm::length(hitPoint - rayOrigin);
            rayParams->setMagnitude(magnitude);

            // Contribution ray

            auto entity = scene->getOrCreateEntityByName("ContributionRay");
            if (!entity.hasComponent<RasterizerRenderingComponent>()) {
                entity.addComponent<RasterizerRenderingComponent>();
                auto &mesh = entity.addComponent<MeshComponent>(CYLINDER);
                auto &material = entity.addComponent<MaterialComponent>();
            } else {
                auto &mesh = entity.getComponent<MeshComponent>();
                auto contributionRayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
                auto &material = entity.getComponent<MaterialComponent>();

                contributionRayParams->setOrigin(hitPoint);

                // calculate direction to camera:
                auto cameraEntity = scene->getActiveCameraEntity();
                if (cameraEntity) {
                    auto cameraPos = cameraEntity.getComponent<TransformComponent>();
                    glm::vec3 dir = glm::normalize(cameraPos.getPosition() - hitPoint);

                    PathTracer::Ray contributionRay = PathTracer::makeRay(
                        PathTracer::glm2sycl(hitPoint), PathTracer::glm2sycl(dir));
                    if (PathTracer::PathTracerMeshKernel::intersectScene(contributionRay, &hit, sceneDescription)) {
                        contributionRayParams->setMagnitude(0.0f);
                    } else {
                        contributionRayParams->setDirection(dir);
                        contributionRayParams->setMagnitude(glm::length(cameraPos.getPosition() - hitPoint));
                    }
                }
            }
        } else {
            rayParams->setMagnitude(99.0f);
        }
    }

    void Emitter::onDestroy() {
    }

    void Emitter::onCreate() {
        auto dev = m_context->getSyclDeviceSelector().getDevice(SYCLDeviceType::Default);
        PathTracer::PathTracerSYCLCreateInfo pipelineSettings(dev);
        pipelineSettings.framebufferSize = 1920 * 1080 * 4 * 10 * sizeof(float4); // ~82 MB of framebuffers
        pipelineSettings.queue = dev->getQueue();
        pipelineSettings.device = dev;
        // Re-create your path-tracer with the updated settings:
        m_pathTracerSYCL = std::make_unique<PathTracer::PathTracerSYCL>(pipelineSettings);
        m_pathTracerSYCL->uploadScene(m_context->activeScene());
    }
}
