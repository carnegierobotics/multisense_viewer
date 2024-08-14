//
// Created by magnus on 8/13/24.
//

#include "Entity.h"

#include "Viewer/VkRender/Scene.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Components/CameraModelComponent.h"

namespace VkRender {
    Entity Scene::createEntityWithUUID(UUID uuid, const std::string &name) {

        Entity entity = {m_registry.create(), this};
        entity.addComponent<IDComponent>(uuid);
        entity.addComponent<TransformComponent>();
        auto &tag = entity.addComponent<TagComponent>();
        tag.Tag = name.empty() ? "Entity" : name;
        Log::Logger::getInstance()->info("Created Entity with UUID: {} and Tag: {}",
                                         entity.getUUID().operator std::string(), entity.getName());

        return entity;

    }

    Entity Scene::findEntityByName(std::string_view name) {
        {auto view = m_registry.view<TagComponent>();
            for (auto entity: view) {
                const TagComponent &tc = view.get<TagComponent>(entity);
                if (tc.Tag == name)
                    return Entity{entity, this};
            }
            return {};

        }

    }

    Entity Scene::createEntity(const std::string &name) {
        return createEntityWithUUID(UUID(), name);

    }

    void Scene::destroyEntity(Entity entity)
    {
        if (!entity) {
            Log::Logger::getInstance()->warning("Attempted to delete an entity that doesn't exist");
            return;
        }
        // Checking if the entity is still valid before attempting to delete
        if (m_registry.valid(entity)) {
            Log::Logger::getInstance()->info("Deleting Entity with UUID: {} and Tag: {}",
                                             entity.getUUID().operator std::string(), entity.getName());

            // Perform the deletion
            m_registry.destroy(entity);
        } else {
            Log::Logger::getInstance()->warning(
                    "Attempted to delete an invalid or already deleted entity with UUID: {}",
                    entity.getUUID().operator std::string());
        }
    }

    void Scene::createNewCamera(const std::string &name, uint32_t width, uint32_t height) {
            auto e = createEntity(name);
            auto &c = e.addComponent<CameraComponent>(Camera(width, height));
            auto &transform = e.getComponent<TransformComponent>();
            //transform.setPosition(glm::vec3(1.0f, 0.0f, 2.0f));
            //c.camera.pose.pos = transform.getPosition();
            auto &gizmo = e.addComponent<CameraModelComponent>();

    }

    void Scene::onMouseEvent(const MouseButtons &mouse) {

    }

    void Scene::onMouseScroll(float change) {

    }

    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
    template<typename T>
    void Scene::onComponentAdded(Entity entity, T &component) {
        static_assert(sizeof(T) == 0);
    }

    template<>
    void Scene::onComponentAdded<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<Rigidbody2DComponent>(Entity entity, Rigidbody2DComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<OBJModelComponent>(Entity entity, OBJModelComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<CameraModelComponent>(Entity entity, CameraModelComponent &component) {
    }

    DISABLE_WARNING_POP

}