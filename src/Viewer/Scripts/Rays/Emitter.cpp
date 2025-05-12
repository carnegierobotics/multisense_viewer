//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "Emitter.h"
#include "Helpers.h"

#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Rendering/Core/KeyInput.h>

#include "imgui.h"


namespace VkRender {
    void Emitter::onUpdate(Timestep ts) {
        if (!hasComponent<RasterizerRenderingComponent>())
            addComponent<RasterizerRenderingComponent>();

        if (!hasComponent<MeshComponent>())
            return;
        auto& meshComponent = getComponent<MeshComponent>();
        auto rayParams = std::dynamic_pointer_cast<CylinderMeshParameters>(meshComponent.meshParameters);
        if (!rayParams)
            return;

        auto scene = m_entity.getScene();
        glm::vec3 rayOrigin(0.0f);
        glm::vec3 direction(0.0f);
        float magnitude = 1.0f;

        auto& rayTransform = getComponent<TransformComponent>();

        auto view = scene->getRegistry().view<LightSourceComponent>();

        for (auto e : view) {
            Entity entity(e, scene);
            auto lightTransform = entity.getComponent<TransformComponent>();
            rayOrigin = lightTransform.getPosition();
        }



        rayParams->setOrigin(rayOrigin);
    }

    void Emitter::onDestroy() {
    }

    void Emitter::onCreate() {
        Scene scenePtr = m_entity.getScene();

        m_pathTracerSYCL->uploadScene(scenePtr);
    }
}


