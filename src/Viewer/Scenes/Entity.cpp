//
// Created by magnus on 4/11/24.
//

#include "Entity.h"


namespace VkRender {

    Entity::Entity(entt::entity handle, Scene *scene)
            : m_entityHandle(handle), m_scene(scene) {
    }

}