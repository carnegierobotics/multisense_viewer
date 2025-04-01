//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "IntensityGradientRay.h"

#include <Viewer/Rendering/Components/ScriptableComponent.h>
#include <Viewer/Rendering/Core/UbuntuKeyInput.h>
#include <Viewer/Rendering/RenderResources/PathTracer/Definitions.h>

#include "ContributionRay.h"
#include "Helpers.h"


namespace VkRender {
    void IntensityGradientRay::onUpdate(Timestep ts) {
        auto scene = m_entity.getScene();

        auto scriptView = scene->getRegistry().view<ScriptableComponent>();
        for (auto e: scriptView) {
            auto entity = Entity(e, scene);
            auto &script = entity.getComponent<ScriptableComponent>();
            if (script.scriptName == "VkRender::ContributionRay")
                // TODO replace with a slots in properties view for which entity to attach to.
            {
                contributionRay = reinterpret_cast<ContributionRay *>(script.instance);
            }
        }
        if (!contributionRay)
            return;


        auto &transformComponent = getComponent<TransformComponent>();
        auto &mesh = getComponent<MeshComponent>();
        auto cylinder = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);

        if (cylinder) {

            auto cameraEntity = scene->getEntityByName("Camera1");
            if (cameraEntity) {
                auto cameraTransform = cameraEntity.getComponent<TransformComponent>();
                auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
                auto param = cameraComponent.getPinholeCamera()->m_parameters;



                bool updateDirection = false;

                if (UbuntuKeyInput::isKeyPressed(GLFW_KEY_SPACE)) {
                    uv0 = contributionRay->uv0;
                    updateDirection = true;
                }

                if (UbuntuKeyInput::isKeyClicked(GLFW_KEY_UP)) {
                    uv0.y -= 1.0f;
                    updateDirection = true;
                }
                if (UbuntuKeyInput::isKeyClicked(GLFW_KEY_RIGHT)) {
                    uv0.x += 1.0f;
                    updateDirection = true;
                }
                if (UbuntuKeyInput::isKeyClicked(GLFW_KEY_DOWN)) {
                    uv0.y += 1.0f;
                    updateDirection = true;
                }
                if (UbuntuKeyInput::isKeyClicked(GLFW_KEY_LEFT)) {
                    uv0.x -= 1.0f;
                    updateDirection = true;
                }


                directionCamera.x = -(uv0.x - param.cx) / param.fx;
                directionCamera.y = -(uv0.y - param.cy) / param.fy;
                directionCamera.z = -1.0;
                directionCamera = glm::normalize(directionCamera);

                glm::vec3 directionWorld = glm::normalize(cameraTransform.getRotationQuaternion() * directionCamera);

                if (updateDirection) {
                    cylinder->setDirection(directionWorld);
                }
                cylinder->setOrigin(cameraTransform.getPosition());


                auto quadricEntity = scene->getEntityByName("Quadric");
                if (quadricEntity) {
                    auto &mesh = quadricEntity.getComponent<MeshComponent>();
                    auto &transform = quadricEntity.getComponent<TransformComponent>();
                    auto quadric = std::dynamic_pointer_cast<QuadricMeshParameters>(mesh.meshParameters);
                    PathTracer::QuadricInputAssembly quad;
                    quad.a = quadric->a;
                    quad.b = quadric->b;
                    quad.c = quadric->c;
                    quad.t_x = quadric->t_x;
                    quad.t_y = quadric->t_y;
                    quad.transform = transform;
                    float beta = 0.0f;
                    glm::vec3 hitPosition, hitNormal;
                    if (RayHelpers::computeWorldHitPoint(cameraTransform.getPosition(), directionWorld, quad, hitPosition, hitNormal, beta)) {
                        cylinder->setMagnitude(glm::length(hitPosition - cameraTransform.getPosition()));
                    } else {
                        hitPosition = {-99.0f, 0.0f, 0.0f};
                        cylinder->setOrigin(hitPosition);
                    }
                }
            }
        }
    }

    void IntensityGradientRay::onDestroy() {
    }

    void IntensityGradientRay::onCreate() {
    }
}
