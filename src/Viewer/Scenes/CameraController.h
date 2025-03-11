//
// Created by magnus on 3/10/25.
//

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "Viewer/Rendering/Core/KeyInput.h"
#include "Viewer/Scenes/ScriptableEntity.h"


namespace VkRender {
    class DefaultController : public ScriptableEntity {
        TransformComponent*transform = nullptr;
        float posX = 0.0f;
    public:
        void onUpdate(Timestep ts) override {
            posX += ts.getSeconds();

            if (Input::isKeyPressed(GLFW_KEY_W))
            {
                transform->getPosition().z += (0.5f * ts);

            }
            if (Input::isKeyPressed(GLFW_KEY_S))
            {
                transform->getPosition().z -= (0.5f * ts);

            }

        }

        void onDestroy() override {

        }

        void onCreate() override {
            transform = &getComponent<TransformComponent>();

        }
    };

}
#endif //CONTROLLER_H
