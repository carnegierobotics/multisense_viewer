//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_ENTITY_H
#define MULTISENSE_VIEWER_ENTITY_H

#include <cassert>
#include <multisense_viewer/external/entt/include/entt/entt.hpp>

#include "Viewer/Tools/Macros.h"
#include "Viewer/Scenes/Scene.h"

namespace VkRender {

    class Entity
    {
    public:
        Entity() = default;
        Entity(entt::entity handle, Scene *scene);
        Entity(const Entity& other) = default;

        template<typename T, typename... Args>
        T& addComponent(Args&&... args)
        {
            VK_ASSERT(!hasComponent<T>(), "Entity already has component!");
            T& component = m_scene->m_registry.emplace<T>(m_entityHandle, std::forward<Args>(args)...);
            m_scene->onComponentAdded<T>(*this, component);
            return component;
        }

        template<typename T, typename... Args>
        T& addOrReplaceComponent(Args&&... args)
        {
            T& component = m_scene->m_registry.emplace_or_replace<T>(m_entityHandle, std::forward<Args>(args)...);
            m_scene->onComponentAdded<T>(*this, component);
            return component;
        }

        template<typename T>
        T& getComponent()
        {
            VK_ASSERT(hasComponent<T>(), "Entity does not have component!");
            return m_scene->m_registry.get<T>(m_entityHandle);
        }

        template<typename T>
        bool hasComponent()
        {
            return  m_scene->m_registry.any_of<T>(m_entityHandle);
        }
        template<typename T>
        void removeComponent()
        {
            VK_ASSERT(hasComponent<T>(), "Entity does not have component!");
            m_scene->m_registry.remove<T>(m_entityHandle);
        }


        operator bool() const { return m_entityHandle != entt::null; }
        operator entt::entity() const { return m_entityHandle; }
        operator uint32_t() const { return static_cast<uint32_t>(m_entityHandle); }


        const std::string& getName() { return getComponent<TagComponent>().Tag; }
        UUID getUUID() { return getComponent<IDComponent>().ID; }


        bool operator==(const Entity& other) const
        {
            return m_entityHandle == other.m_entityHandle && m_scene == other.m_scene;
        }

        bool operator!=(const Entity& other) const
        {
            return !(*this == other);
        }


    private:
        entt::entity m_entityHandle{ entt::null };
        Scene* m_scene = nullptr;

    };

}

#endif //MULTISENSE_VIEWER_ENTITY_H
