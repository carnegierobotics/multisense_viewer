//
// Created by magnus on 3/10/25.
//

#ifndef SCRIPTABLEENTITY_H
#define SCRIPTABLEENTITY_H
#include "Entity.h"
#include "Viewer/Rendering/Core/Timestep.h"

namespace VkRender {

    class ScriptableEntity {
    public:
        virtual ~ScriptableEntity() = default;

        template<typename T>
        T &getComponent() {
            return m_entity.getComponent<T>();
        }
        template<typename T>
        T &addComponent() {
            return m_entity.addComponent<T>();
        }
        template<typename T>
        bool hasComponent() {
            return m_entity.hasComponent<T>();
        }

        virtual void onCreate() = 0;

        virtual void onUpdate(Timestep ts) = 0;

        virtual void onDestroy() = 0;

    protected:
        Entity m_entity;

    private:
        friend class Scene;
    };
}
#endif //SCRIPTABLEENTITY_H
