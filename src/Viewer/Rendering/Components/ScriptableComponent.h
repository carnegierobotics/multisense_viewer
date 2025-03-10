//
// Created by magnus on 3/10/25.
//

#ifndef SCRIPTABLECOMPONENT_H
#define SCRIPTABLECOMPONENT_H

#include "Viewer/Rendering/Core/UUID.h"
#include "Viewer/Scenes/ScriptableEntity.h"


namespace VkRender {
    struct ScriptableComponent {
        ScriptableEntity *instance = nullptr;

        std::function<ScriptableEntity *()> instantiateScript;
        std::function<void()> destroyScript;

        template<typename T>
        void bind() {
            instantiateScript = [this]() {return new T();};
            destroyScript = [this]() {delete (instance); instance = nullptr;};
        }
    };


}


#endif //SCRIPTABLECOMPONENT_H
