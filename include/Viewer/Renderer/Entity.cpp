//
// Created by magnus on 4/11/24.
//

#include "Viewer/Renderer/Entity.h"


namespace VkRender {

    Entity::Entity(entt::entity handle, Renderer *scene)
            : m_entityHandle(handle), m_renderer(scene) {
    }

}