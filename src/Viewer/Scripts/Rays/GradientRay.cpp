//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "GradientRay.h"


namespace VkRender {
    void GradientRay::onUpdate(Timestep ts) {
        auto &mesh = getComponent<MeshComponent>();
        auto cylinder = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
        if (!cylinder)
            return;


        glm::vec3 normalizedDirection = glm::normalize(ray);
        cylinder->setOrigin(origin);
        cylinder->setDirection(normalizedDirection);
        cylinder->setMagnitude(1.0f);

    }

    void GradientRay::onDestroy() {
    }

    void GradientRay::onCreate() {

    }
}