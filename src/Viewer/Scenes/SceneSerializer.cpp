//
// Created by mgjer on 01/10/2024.
//

#include <yaml-cpp/yaml.h>

#include "Viewer/Scenes/SceneSerializer.h"

#include <Viewer/VkRender/Components/MaterialComponent.h>

#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/Components.h"

namespace VkRender::Serialize {
    static std::string PolygonModeToString(VkPolygonMode mode) {
        switch (mode) {
            case VK_POLYGON_MODE_FILL:
                return "Fill";
            case VK_POLYGON_MODE_LINE:
                return "Line";
            case VK_POLYGON_MODE_POINT:
                return "Point";
            default:
                return "Unknown";
        }
    }

    static VkPolygonMode StringToPolygonMode(const std::string &modeStr) {
        if (modeStr == "Fill")
            return VK_POLYGON_MODE_FILL;
        if (modeStr == "Line")
            return VK_POLYGON_MODE_LINE;
        if (modeStr == "Point")
            return VK_POLYGON_MODE_POINT;

        // Default case, or handle unknown input
        return VK_POLYGON_MODE_FILL;
    }
}

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
            auto &transform = entity.getComponent<TransformComponent>();
            out << YAML::Key << "Position";
            out << YAML::Value << transform.getPosition();
            out << YAML::Key << "Rotation";
            out << YAML::Value << transform.getRotationQuaternion();
            out << YAML::Key << "Scale";
            out << YAML::Value << transform.getScale();
            out << YAML::EndMap;
        }

        if (entity.hasComponent<MeshComponent>()) {
            out << YAML::Key << "MeshComponent";
            out << YAML::BeginMap;
            auto &mesh = entity.getComponent<MeshComponent>();
            out << YAML::Key << "ModelPath";
            out << YAML::Value << mesh.meshPath.string();
            out << YAML::Key << "PolygonMode";
            out << YAML::Value << Serialize::PolygonModeToString(mesh.polygonMode); // Serialize PolygonMode as a string
            out << YAML::EndMap;
        }
        if (entity.hasComponent<CameraComponent>()) {
            out << YAML::Key << "CameraComponent";
            out << YAML::BeginMap;
            auto &camera = entity.getComponent<CameraComponent>();
            out << YAML::Key << "render";
            out << YAML::Value << camera.render;
            auto &cameraProps = camera.camera;
            out << YAML::Key << "Width";
            out << YAML::Value << cameraProps.m_width;
            out << YAML::Key << "Height";
            out << YAML::Value << cameraProps.m_height;
            out << YAML::Key << "ZNear";
            out << YAML::Value << cameraProps.m_Znear;
            out << YAML::Key << "ZFar";
            out << YAML::Value << cameraProps.m_Zfar;
            out << YAML::Key << "FOV";
            out << YAML::Value << cameraProps.m_Fov;
            out << YAML::EndMap;
        }
        if (entity.hasComponent<MaterialComponent>()) {
            out << YAML::Key << "MaterialComponent";
            out << YAML::BeginMap;

            auto &material = entity.getComponent<MaterialComponent>();

            // Serialize baseColor (glm::vec4)
            out << YAML::Key << "BaseColor";
            out << YAML::Value << YAML::Flow << std::vector<float>{
                material.baseColor.r, material.baseColor.g, material.baseColor.b, material.baseColor.a
            };

            // Serialize metallic factor (float)
            out << YAML::Key << "Metallic";
            out << YAML::Value << material.metallic;

            // Serialize roughness factor (float)
            out << YAML::Key << "Roughness";
            out << YAML::Value << material.roughness;

            // Serialize usesTexture flag (bool)
            out << YAML::Key << "UsesTexture";
            out << YAML::Value << material.usesTexture;

            // Serialize emissiveFactor (glm::vec4)
            out << YAML::Key << "EmissiveFactor";
            out << YAML::Value << YAML::Flow << std::vector<float>{
                material.emissiveFactor.r, material.emissiveFactor.g, material.emissiveFactor.b,
                material.emissiveFactor.a
            };

            // Serialize vertex shader name (std::filesystem::path)
            out << YAML::Key << "VertexShader";
            out << YAML::Value << material.vertexShaderName.string(); // Convert path to string

            // Serialize fragment shader name (std::filesystem::path)
            out << YAML::Key << "FragmentShader";
            out << YAML::Value << material.fragmentShaderName.string(); // Convert path to string

            // Serialize albedo texture path (std::filesystem::path), only if usesTexture is true
            if (material.usesTexture) {
                out << YAML::Key << "AlbedoTexture";
                out << YAML::Value << material.albedoTexturePath.string(); // Convert path to string
            }

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
                    tc.setRotationQuaternion(transformComponent["Rotation"].as<glm::quat>());
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
                        mesh.polygonMode = Serialize::StringToPolygonMode(meshComponent["PolygonMode"].as<std::string>());
                    } else {
                        Log::Logger::getInstance()->error("Failed to load mesh at {}", path.string());
                    }
                }

                auto materialComponent = entity["MaterialComponent"];
                if (materialComponent) {
                    auto &material = deserializedEntity.addComponent<MaterialComponent>();
                    // Deserialize base color
                    auto baseColor = materialComponent["BaseColor"].as<std::vector<float> >();
                    if (baseColor.size() == 4) {
                        material.baseColor = glm::vec4(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                    }
                    // Deserialize metallic factor
                    material.metallic = materialComponent["Metallic"].as<float>();
                    // Deserialize roughness factor
                    material.roughness = materialComponent["Roughness"].as<float>();
                    // Deserialize uses texture flag
                    material.usesTexture = materialComponent["UsesTexture"].as<bool>();
                    // Deserialize emissive factor
                    auto emissiveFactor = materialComponent["EmissiveFactor"].as<std::vector<float> >();
                    if (emissiveFactor.size() == 4) {
                        material.emissiveFactor = glm::vec4(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2],
                                                            emissiveFactor[3]);
                    }
                    // Deserialize vertex shader name
                    if (materialComponent["VertexShader"]) {
                        material.vertexShaderName = std::filesystem::path(
                            materialComponent["VertexShader"].as<std::string>());
                    }
                    // Deserialize fragment shader name
                    if (materialComponent["FragmentShader"]) {
                        material.fragmentShaderName = std::filesystem::path(
                            materialComponent["FragmentShader"].as<std::string>());
                    }
                    // Deserialize albedo texture path (only if usesTexture is true)
                    if (material.usesTexture && materialComponent["AlbedoTexture"]) {
                        material.albedoTexturePath = std::filesystem::path(
                            materialComponent["AlbedoTexture"].as<std::string>());
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
