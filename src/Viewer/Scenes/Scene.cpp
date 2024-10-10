//
// Created by magnus on 8/13/24.
//

#include "Viewer/VkRender/Core/Entity.h"

#include "Viewer/Scenes/Scene.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"
#include "Viewer/VkRender/Components/ImageComponent.h"
#include "Viewer/Application/Application.h"

namespace VkRender {
    Scene::Scene(const std::string &name, VkRender::Application *context) {
        m_sceneName = name;
        m_context = context;
    }

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
        {
            auto view = m_registry.view<TagComponent>();
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

    void Scene::destroyEntity(Entity entity) {
        if (!entity) {
            Log::Logger::getInstance()->warning("Attempted to delete an entity that doesn't exist");
            return;
        }
        // Checking if the entity is still valid before attempting to delete
        if (m_registry.valid(entity)) {
            Log::Logger::getInstance()->info("Deleting Entity with UUID: {} and Tag: {}",
                                             entity.getUUID().operator std::string(), entity.getName());
            notifyComponentRemoval(entity);

            // Perform the deletion
            m_registry.destroy(entity);
        } else {
            Log::Logger::getInstance()->warning(
                "Attempted to delete an invalid or already deleted entity with UUID: {}",
                entity.getUUID().operator std::string());
        }
    }

    void Scene::notifyComponentRemoval(Entity entity) {
        // Check for each component type, and remove if the entity has the component
        if (entity.hasComponent<MeshComponent>()) {
            entity.removeComponent<MeshComponent>();
        }

        if (entity.hasComponent<MaterialComponent>()) {
            entity.removeComponent<MaterialComponent>();
        }

        // Repeat for other components, adding more checks for each type of component
        // if (entity.hasComponent<OtherComponent>()) {
        //     entity.removeComponent<OtherComponent>();
        // }
    }

    Entity Scene::createNewCamera(const std::string &name, uint32_t width, uint32_t height) {
        auto e = createEntity(name);
        auto &c = e.addComponent<CameraComponent>(Camera(width, height));
        c.camera.setType(Camera::flycam);
        auto &transform = e.getComponent<TransformComponent>();
        c.camera.pose.pos = transform.getPosition();
        return e;
    }

    void Scene::onMouseEvent(const MouseButtons &mouse) {
    }

    void Scene::onMouseScroll(float change) {
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }
    void Scene::notifyEditorsComponentRemoved(Entity entity, MeshComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }

    void Scene::notifyEditorsComponentAdded(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentAdded(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentAdded(entity, component);
        }
    }

    void Scene::notifyEditorsComponentUpdated(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentUpdated(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentUpdated(entity, component);
        }
    }
    void Scene::notifyEditorsComponentRemoved(Entity entity, MaterialComponent &component) {
        for (auto &editor: m_context->m_sceneRenderers) {
            editor.second->onComponentRemoved(entity, component);
        }
        for (auto &editor: m_context->m_editors) {
            editor->onComponentRemoved(entity, component);
        }
    }


    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER


    /** COMPONENT ADDED **/

    template<>
    void Scene::onComponentAdded<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentAdded(entity, component);
    }
    template<>
    void Scene::onComponentAdded<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentAdded(entity, component);
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
    void Scene::onComponentAdded<GaussianModelComponent>(Entity entity, GaussianModelComponent &component) {
    }

    template<>
    void Scene::onComponentAdded<ImageComponent>(Entity entity, ImageComponent &component) {
    }
    /** COMPONENT REMOVE **/

    template<>
    void Scene::onComponentRemoved<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }
    template<>
    void Scene::onComponentRemoved<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentRemoved(entity, component);
    }
    template<>
    void Scene::onComponentRemoved<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<Rigidbody2DComponent>(Entity entity, Rigidbody2DComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<GaussianModelComponent>(Entity entity, GaussianModelComponent &component) {
    }

    template<>
    void Scene::onComponentRemoved<ImageComponent>(Entity entity, ImageComponent &component) {
    }


    /** COMPONENT UPDATE **/
    template<>
    void Scene::onComponentUpdated<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<MeshComponent>(Entity entity, MeshComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }

    template<>
    void Scene::onComponentUpdated<MaterialComponent>(Entity entity, MaterialComponent &component) {
        notifyEditorsComponentUpdated(entity, component);
    }
    template<>
    void Scene::onComponentUpdated<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<Rigidbody2DComponent>(Entity entity, Rigidbody2DComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<GaussianModelComponent>(Entity entity, GaussianModelComponent &component) {
    }

    template<>
    void Scene::onComponentUpdated<ImageComponent>(Entity entity, ImageComponent &component) {
    }


    DISABLE_WARNING_POP
}
