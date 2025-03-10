//
// Created by magnus on 3/10/25.
//

#ifndef SCRIPTABLEENTITY_H
#define SCRIPTABLEENTITY_H
#include "Entity.h"

namespace VkRender {

    class ScriptableEntity {
    public:
        template<typename T>
        T& getComponent() {
            return m_entity.getComponent<T>();
        }

        virtual void onCreate() = 0;
        virtual void onUpdate() = 0;
        virtual void onDestroy() = 0;

    private:
        Entity m_entity;
        friend class Scene;

    };
}
#endif //SCRIPTABLEENTITY_H
