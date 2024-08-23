//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_SCENE_H
#define MULTISENSE_VIEWER_SCENE_H

#include <entt/entt.hpp>

#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Core/UUID.h"
#include "Viewer/VkRender/Components/Components.h"

namespace VkRender {
    class Scene {
        using DestroyCallback = std::function<void(entt::entity)>;

    public:
        Scene() = default;

        virtual void update(uint32_t i) {};

        virtual ~Scene() {
            // Manually destroy all entities to trigger the on_destroy events
            auto view = m_registry.view<entt::entity>();
            for (auto entity: view) {
                m_registry.destroy(entity);  // This will trigger any registered destroy callbacks
            }

            // Now clear the registry to clean up any remaining components
            m_registry.clear<>();
        }

        Entity createEntity(const std::string &name);

        Entity createEntityWithUUID(UUID uuid, const std::string &name);

        Entity findEntityByName(std::string_view name);

        void destroyEntity(Entity entity);

        Entity createNewCamera(const std::string &name, uint32_t width, uint32_t height);

        void onMouseEvent(const MouseButtons &mouseButtons);

        void onMouseScroll(float change);

        /*
        Camera &Renderer::getCamera() {
            if (!m_selectedCameraTag.empty()) {
                auto it = m_cameras.find(m_selectedCameraTag);
                if (it != m_cameras.end()) {
                    return m_cameras[m_selectedCameraTag];
                }
            } // TODO create a new camera with tag if it doesn't exist
        }

        Camera &Renderer::getCamera(std::string tag) {
            if (!m_selectedCameraTag.empty()) {
                auto it = m_cameras.find(tag);
                if (it != m_cameras.end()) {
                    return m_cameras[tag];
                }
            }
            // TODO create a new camera with tag if it doesn't exist
        }
        */

        entt::registry &getRegistry() { return m_registry; };

        const entt::registry& getRegistry() const {
            return m_registry;
        }

        //entt::view get(){return m_registry.view<>();}

        const std::string &getSceneName() { return m_sceneName; }

        void addDestroyFunction(void* owner, DestroyCallback callback) {
            m_destroyCallbacks[owner].push_back(std::move(callback));
            m_registry.on_destroy<entt::entity>().connect<&Scene::onEntityDestroyed>(*this);
        }

        void removeDestroyFunction(void* owner) {
            m_destroyCallbacks.erase(owner);
        }

    protected:
        entt::registry m_registry;
        std::string m_sceneName = "Unnamed Scene";
        std::unordered_map<void*, std::deque<DestroyCallback>> m_destroyCallbacks;

        friend class Entity;

        template<typename T>
        void onComponentAdded(Entity entity, T &component);

    private:
        void onEntityDestroyed(entt::registry &registry, entt::entity entity) {
            for (auto& [owner, callbacks] : m_destroyCallbacks) {
                for (auto& callback : callbacks) {
                    callback(entity);
                }
            }
        }

    };


}


#endif //MULTISENSE_VIEWER_SCENE_H
