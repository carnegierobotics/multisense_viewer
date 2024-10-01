//
// Created by mgjer on 01/10/2024.
//

#ifndef SCENESERIALIZER_H
#define SCENESERIALIZER_H


#include "Viewer/Scenes/Scene.h"
#include "Viewer/Application/pch.h"


namespace VkRender {
    class SceneSerializer {

    public:
        explicit SceneSerializer(const std::shared_ptr<Scene>& scene);

        void serialize(const std::filesystem::path& filePath);
        void serializeRuntime(const std::filesystem::path& filePath);

        bool deserialize(const std::filesystem::path& filePath);
        bool deserializeRuntime(const std::filesystem::path& filePath);


    private:
        std::shared_ptr<Scene> m_scene;
    };
}



#endif //SCENESERIALIZER_H
