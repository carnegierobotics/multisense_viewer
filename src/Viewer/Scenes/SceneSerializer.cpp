//
// Created by mgjer on 01/10/2024.
//

#include <yaml-cpp/yaml.h>

#include "Viewer/Scenes/SceneSerializer.h"

#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/Components.h"

namespace YAML {
    template<>
    struct convert<glm::vec3> {
        static Node encode(const glm::vec3 &rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node &node, glm::vec3 &rhs) {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }
            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            rhs.z = node[2].as<float>();
            return true;
        }
    };

    template<>
    struct convert<glm::quat> {
        static Node encode(const glm::quat &rhs) {
            Node node;
            node.push_back(rhs.w);
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node &node, glm::quat &rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }
            rhs.w = node[0].as<float>();
            rhs.x = node[1].as<float>();
            rhs.y = node[2].as<float>();
            rhs.z = node[3].as<float>();
            return true;
        }
    };
}

namespace VkRender {
    YAML::Emitter &operator <<(YAML::Emitter &out, const glm::vec3 &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter &operator <<(YAML::Emitter &out, const glm::vec4 &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter &operator <<(YAML::Emitter &out, const glm::quat &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    SceneSerializer::SceneSerializer(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
    }

    static void SerializeEntity(YAML::Emitter &out, Entity entity) {
        out << YAML::BeginMap;
        out << YAML::Key << "Entity";
        out << YAML::Value << "123123123";

        if (entity.hasComponent<TagComponent>()) {
            out << YAML::Key << "TagComponent";
            out << YAML::BeginMap;
            auto &tag = entity.getComponent<TagComponent>().Tag;
            out << YAML::Key << "Tag";
            out << YAML::Value << tag;
            out << YAML::EndMap;
        }

        if (entity.hasComponent<TransformComponent>()) {
            out << YAML::Key << "TransformComponent";
            out << YAML::BeginMap;
            auto& transform = entity.getComponent<TransformComponent>();
            out << YAML::Key << "Position";
            out << YAML::Value << transform.getPosition();
            out << YAML::Key << "Rotation";
            out << YAML::Value << transform.getQuaternion();
            out << YAML::Key << "Scale";
            out << YAML::Value << transform.getScale();
            out << YAML::EndMap;
        }

        if (entity.hasComponent<MeshComponent>()) {
            out << YAML::Key << "MeshComponent";
            out << YAML::BeginMap;
            auto& mesh = entity.getComponent<MeshComponent>();
            out << YAML::Key << "ModelPath";
            out << YAML::Value << mesh.m_modelPath.string();
            out << YAML::EndMap;
        }
        if (entity.hasComponent<CameraComponent>()) {
            out << YAML::Key << "CameraComponent";
            out << YAML::BeginMap;
            auto& camera = entity.getComponent<CameraComponent>();
            out << YAML::Key << "render";
            out << YAML::Value << camera.render;
            auto& cameraProps = camera.camera;
            out << YAML::Key << "Width"; out << YAML::Value << cameraProps.m_width;
            out << YAML::Key << "Height"; out << YAML::Value << cameraProps.m_height;
            out << YAML::Key << "ZNear"; out << YAML::Value << cameraProps.m_Znear;
            out << YAML::Key << "ZFar"; out << YAML::Value << cameraProps.m_Zfar;
            out << YAML::Key << "FOV"; out << YAML::Value << cameraProps.m_Fov;


            out << YAML::EndMap;
        }

        out << YAML::EndMap;
    }

    void SceneSerializer::serialize(const std::filesystem::path &filePath) {
        // Ensure the directory exists
        if (filePath.has_parent_path()) {
            std::filesystem::create_directories(filePath.parent_path());
        }

        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "Scene";
        out << YAML::Value << "Scene name";
        out << YAML::Key << "Entities";
        out << YAML::Value << YAML::BeginSeq;
        for (auto entity: m_scene->m_registry.view<entt::entity>()) {
            Entity e(entity, m_scene.get());
            if (!e)
                return;
            SerializeEntity(out, e);
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;

        std::ofstream fout(filePath);
        fout << out.c_str();
    }

    void SceneSerializer::serializeRuntime(const std::filesystem::path &filePath) {
        throw std::runtime_error("Not implemented");
    }


    bool SceneSerializer::deserialize(const std::filesystem::path &filePath) {
        std::ifstream stream(filePath);
        std::stringstream stringStream;
        stringStream << stream.rdbuf();

        YAML::Node data = YAML::Load(stringStream.str());
        if (!data["Scene"])
            return false;

        std::string sceneName = data["Scene"].as<std::string>();
        Log::Logger::getInstance()->info("Deserializing scene {}", sceneName);
        auto entities = data["Entities"];
        if (entities) {
            for (auto entity: entities) {
                uint64_t entityId = entity["Entity"].as<uint64_t>(); // todo uuid
                std::string name;
                auto tagComponent = entity["TagComponent"];
                if (tagComponent)
                    name = tagComponent["Tag"].as<std::string>();

                Entity deserializedEntity = m_scene->createEntity(name); // TOdo uuid

                auto transformComponent = entity["TransformComponent"];
                if (transformComponent) {
                    auto &tc = deserializedEntity.getComponent<TransformComponent>();
                    tc.setPosition(transformComponent["Position"].as<glm::vec3>());
                    tc.setQuaternion(transformComponent["Rotation"].as<glm::quat>());
                }
                auto cameraComponent = entity["CameraComponent"];
                if (cameraComponent) {
                    auto &camera = deserializedEntity.addComponent<CameraComponent>();
                }

                auto meshComponent = entity["MeshComponent"];
                if (meshComponent) {
                    std::filesystem::path path(meshComponent["ModelPath"].as<std::string>());
                    if (std::filesystem::exists(path)) {
                        auto &mesh = deserializedEntity.addComponent<MeshComponent>(path);
                    } else {
                        Log::Logger::getInstance()->error("Failed to load mesh at {}", path.string());
                    }
                }
            }
        }

        return true;
    }

    bool SceneSerializer::deserializeRuntime(const std::filesystem::path &filePath) {
        // Not implement
        throw std::runtime_error("Not implemented");
        return false;
    }
}
