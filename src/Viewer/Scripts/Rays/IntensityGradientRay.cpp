//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "IntensityGradientRay.h"


namespace VkRender {
    void IntensityGradientRay::onUpdate(Timestep ts) {

      auto& transformComponent = getComponent<TransformComponent>();


        auto &mesh = getComponent<MeshComponent>();
        auto cylinder = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
        if (cylinder) {

            glm::vec3 directionCamera(0.0f);

             auto cameraEntity = m_entity.getScene()->getEntityByName("Camera1");
            if (cameraEntity) {
                auto& cameraComponent = cameraEntity.getComponent<CameraComponent>();
                auto param = cameraComponent.getPinholeCamera()->m_parameters;
                directionCamera.x = ((param.width / 2) - param.cx)/param.fx;
                directionCamera.y = ((param.height / 2) - param.cy)/param.fy;
                directionCamera.z = -1.0;

                auto cameraTransform = cameraEntity.getComponent<TransformComponent>();
                glm::vec3 directionWorld = glm::normalize(cameraTransform.getRotationQuaternion() * directionCamera);
                cylinder->setDirection(directionWorld);

                cylinder->setOrigin(cameraTransform.getPosition());

            }





        }

    }

    void IntensityGradientRay::onDestroy() {
    }

    void IntensityGradientRay::onCreate() {

    }
}