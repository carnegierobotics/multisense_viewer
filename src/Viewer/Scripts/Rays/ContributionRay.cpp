//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "ContributionRay.h"

#include <Viewer/Rendering/Components/ScriptableComponent.h>
#include <Viewer/Rendering/Core/KeyInput.h>

#include "Helpers.h"

namespace VkRender {
    struct ScriptableComponent;

    void ContributionRay::onUpdate(Timestep ts) {
        /*
        auto &mesh = getComponent<MeshComponent>();
        auto cylinder = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
        if (!cylinder)
            return;

        auto scene = m_entity.getScene();

        auto scriptView = scene->getRegistry().view<ScriptableComponent>();
        for (auto e: scriptView) {
            auto entity = Entity(e, scene);
            auto &script = entity.getComponent<ScriptableComponent>();
            if (script.scriptName == "VkRender::Emitter")
            // TODO replace with a slots in properties view for which entity to attach to.
            {
                emitter = reinterpret_cast<Emitter *>(script.instance);
            }
        }
        if (!emitter)
            return;


        auto cameraEntity = scene->getEntityByName("Camera1"); // TODO replace with connectable slot in Properties view


        if (cameraEntity) {
            auto &camera = cameraEntity.getComponent<CameraComponent>();
            glm::vec3 &cameraPosition = cameraEntity.getComponent<TransformComponent>().getPosition();

            glm::vec3 hitPosition = emitter->hitPosition;
            glm::vec3 hitNormal = emitter->hitNormal;

            glm::vec3 newRayOrigin = hitPosition + hitNormal * 1e-3f;


            glm::vec3 delta = cameraPosition - hitPosition;
            glm::vec3 direction = glm::normalize(delta);
            float magnitude = glm::length(delta);


            bool occluded = false;

            auto quadricCollection = scene->getEntityByName("QuadricCollection");
            if (quadricCollection && quadricCollection.hasChildren()) {
                auto children = quadricCollection.getChildren();

                for (auto quadricEntity: children) {
                    auto &mesh = quadricEntity.getComponent<MeshComponent>();
                    auto &transform = quadricEntity.getComponent<TransformComponent>();
                    auto quadric = std::dynamic_pointer_cast<QuadricMeshParameters>(mesh.meshParameters);
                    glm::vec3 occludedHitPosition, occludedhitPosition;

                    PathTracer::QuadricInputAssembly quad;
                    quad.a = quadric->a;
                    quad.b = quadric->b;
                    quad.c = quadric->c;
                    quad.t_x = quadric->t_x;
                    quad.t_y = quadric->t_y;
                    quad.transform = transform;
                    float beta = 0.0f;
                    if (RayHelpers::checkContributionCollision(newRayOrigin, direction, quad, occludedhitPosition,
                                                               occludedHitPosition, beta)) {
                        occluded = true;
                    }
                }
            }

            if (occluded) {
                cylinder->setOrigin({-99, 0, 0});
            } else {


                auto cameraTransform = cameraEntity.getComponent<TransformComponent>().getTransform();
                glm::vec3 camHitWorld(0.0f);
                float incidentAngle = 0.0f;
                float camera_t = FLT_MAX;
                RayHelpers::checkCameraPlaneIntersection(hitPosition, direction, camHitWorld, camera_t, incidentAngle,
                                                         cameraTransform,
                                                         cameraEntity.getComponent<CameraComponent>().
                                                         pinholeParameters);

                auto parameters = cameraEntity.getComponent<CameraComponent>().getPinholeCamera();

                glm::mat4 worldToCamera = glm::inverse(cameraTransform);
                glm::vec4 hitPointCam4 = worldToCamera * glm::vec4(camHitWorld, 1.0f);
                glm::vec3 camHitLocal = hitPointCam4 / hitPointCam4.w;

                // use pinhole projection
                float fx = parameters->m_parameters.fx;
                float fy = parameters->m_parameters.fy;
                float cx = parameters->m_parameters.cx;
                float cy = parameters->m_parameters.cy;
                float X = camHitLocal.x;
                float Y = camHitLocal.y;
                float Z = camHitLocal.z;
                float xPixel = (fx * X / Z) + cx;
                float yPixel = (fy * Y / Z) + cy;

                uv0 = glm::vec2(xPixel, yPixel);

                cylinder->setOrigin(hitPosition);
                cylinder->setDirection(direction);
                cylinder->setMagnitude(magnitude + 1.0f);
            }
        }
        */
    }


    void ContributionRay::onDestroy() {
    }

    void ContributionRay::onCreate() {
    }
}
