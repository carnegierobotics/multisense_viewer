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

    public:
        Scene() = default;

        virtual void render(CommandBuffer &drawCmdBuffers) {};

        virtual void update(uint32_t i) {};

        ~Scene(){
            m_registry.clear<>();
        }

        Entity createEntity(const std::string &name);

        Entity createEntityWithUUID(UUID uuid, const std::string &name);

        Entity findEntityByName(std::string_view name);

        void destroyEntity(Entity entity);

        void createNewCamera(const std::string &name, uint32_t width, uint32_t height);

        void onMouseEvent(MouseButtons& mouseButtons);
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

        entt::registry& getRegistry() {return m_registry;};
        const std::string& getSceneName() {return m_sceneName;}

    protected:
        entt::registry m_registry;
        std::string m_sceneName = "Unnamed Scene";
        friend class Entity;

        template<typename T>
        void onComponentAdded(Entity entity, T &component);
    };


}


#endif //MULTISENSE_VIEWER_SCENE_H
