//
// Created by magnus on 3/11/25.
//
#define GLM_ENABLE_EXPERIMENTAL

#include "ContributionRay.h"

#include <Viewer/Rendering/Components/ScriptableComponent.h>
#include <Viewer/Rendering/Core/KeyInput.h>

#include "Emitter.h"

namespace VkRender {
    struct ScriptableComponent;

    void ContributionRay::onUpdate(Timestep ts) {
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

        glm::vec3 position = emitter->hitPosition;
        auto cameraEntity = scene->getEntityByName("Camera1"); // TODO replace with connectable slot in Properties view


        if (cameraEntity) {
            auto &camera = cameraEntity.getComponent<CameraComponent>();
            glm::vec3 &cameraPosition = cameraEntity.getComponent<TransformComponent>().getPosition();
            cylinder->setOrigin(position);
            glm::vec3 delta = cameraPosition - position;
            cylinder->setDirection(glm::normalize(delta));
            cylinder->setMagnitude(glm::length(delta));
        }
    }

    void ContributionRay::onDestroy() {
    }

    void ContributionRay::onCreate() {
    }
}
