//
// Created by magnus on 3/10/25.
//

#ifndef VECTORSCRIPTS_H
#define VECTORSCRIPTS_H

#include "Viewer/Scenes/ScriptableEntity.h"


namespace VkRender {

class VectorScripts : ScriptableEntity {
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

#endif //VECTORSCRIPTS_H
