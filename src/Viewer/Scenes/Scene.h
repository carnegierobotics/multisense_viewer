//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_SCENE_H
#define MULTISENSE_VIEWER_SCENE_H

#include <entt/entt.hpp>
#include <Viewer/VkRender/Components/MaterialComponent.h>

#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Core/UUID.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Components/PointCloudComponent.h"

namespace VkRender {
    class Application;
    class Entity;

    class Scene {
        using DestroyCallback = std::function<void(entt::entity)>;

    public:
        Scene(const std::string &name, Application *context);

        void deleteAllEntities();

        ~Scene() {
            deleteAllEntities();
        }

        Entity createEntity(const std::string &name);

        Entity createEntityWithUUID(UUID uuid, const std::string &name);

        Entity findEntityByName(std::string_view name);

        void destroyEntity(Entity entity);

        void notifyComponentRemoval(Entity entity);

        Entity createNewCamera(const std::string &name, uint32_t width, uint32_t height);

        entt::registry &getRegistry() { return m_registry; };

        const entt::registry &getRegistry() const {
            return m_registry;
        }
        const std::string &getSceneName() { return m_sceneName; }

        template<class T>
        void onComponentUpdated(Entity entity, T &component);
    protected:
        entt::registry m_registry;
        std::unordered_map<void *, std::deque<DestroyCallback>> m_destroyCallbacks;

        friend class Entity;
        friend class SceneSerializer;

        template<typename T>
        void onComponentAdded(Entity entity, T &component);

        template<class T>
        void onComponentRemoved(Entity entity, T &component);


    private:
        void onEntityDestroyed(entt::registry &registry, entt::entity entity) {
            for (auto &[owner, callbacks]: m_destroyCallbacks) {
                for (auto &callback: callbacks) {
                    callback(entity);
                }
            }
        }

        void notifyEditorsComponentAdded(Entity entity, MeshComponent &component);
        void notifyEditorsComponentUpdated(Entity entity, MeshComponent &component);
        void notifyEditorsComponentRemoved(Entity entity, MeshComponent &component);
        void notifyEditorsComponentAdded(Entity entity, MaterialComponent &component);

        void notifyEditorsComponentAdded(Entity entity, PointCloudComponent &component);

        void notifyEditorsComponentUpdated(Entity entity, PointCloudComponent &component);

        void notifyEditorsComponentRemoved(Entity entity, PointCloudComponent &component);

        void notifyEditorsComponentUpdated(Entity entity, MaterialComponent &component);
        void notifyEditorsComponentRemoved(Entity entity, MaterialComponent &component);

        template<class T>
        void notifyEditorsComponentAdded(Entity entity, T &component);

        std::string m_sceneName = "Unnamed Scene";
        std::filesystem::path filePath;
        Application *m_context;

    };


}


#endif //MULTISENSE_VIEWER_SCENE_H
