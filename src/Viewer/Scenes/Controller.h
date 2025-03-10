//
// Created by magnus on 3/10/25.
//

#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "ScriptableEntity.h"


namespace VkRender {
    class Controller : public ScriptableEntity {
        TransformComponent*transform = nullptr;
    public:
        void onUpdate() override {
            Log::Logger::getInstance()->info("OnUpdate");
            transform->setPosition(glm::vec3(3.0f, 0.0f, 0.0f));
        }

        void onDestroy() override {
            Log::Logger::getInstance()->info("OnDestroy");

        }

        void onCreate() override {
            transform = &getComponent<TransformComponent>();
            Log::Logger::getInstance()->info("OnCreate");


        }
    };
}
#endif //CONTROLLER_H
