//
// Created by mgjer on 01/10/2024.
//

#ifndef SCENESERIALIZER_H
#define SCENESERIALIZER_H

#include <yaml-cpp/yaml.h>
#include <pugixml.hpp>

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


        bool deserializeYAML(const YAML::Node& data);
        bool deserializeXML(const std::filesystem::path& filePath);

    private:
        std::shared_ptr<Scene> m_scene;

    private:
        bool loadShape(const pugi::xml_node& shape);
        bool loadSensorToCamera(const pugi::xml_node& sensor);
    };
}



#endif //SCENESERIALIZER_H
