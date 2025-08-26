//
// Created by mgjer on 01/10/2024.
//

#include <yaml-cpp/yaml.h>

#include "Viewer/Scenes/SceneSerializer.h"

#include <Viewer/Application/ApplicationConfig.h>
#include <Viewer/Rendering/Components/MaterialComponent.h>
#include <Viewer/Rendering/Components/QuadricCollectionComponent.h>
#include <Viewer/Rendering/Components/ScriptableComponent.h>
#include <Viewer/Scripts/VectorScripts.h>
#include <Viewer/Scripts/Rays/ContributionRay.h>

#include "Viewer/Scenes/CameraController.h"

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/LightSourceComponent.h"

namespace VkRender::Serialize {
    static std::string polygonModeToString(VkPolygonMode mode) {
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

    static VkPolygonMode stringToPolygonMode(const std::string &modeStr) {
        if (modeStr == "Fill")
            return VK_POLYGON_MODE_FILL;
        if (modeStr == "Line")
            return VK_POLYGON_MODE_LINE;
        if (modeStr == "Point")
            return VK_POLYGON_MODE_POINT;

        // Default case, or handle unknown input
        return VK_POLYGON_MODE_FILL;
    }

    /*
    // Convert CameraType to string
    std::string cameraTypeToString(Camera::CameraType type) {
        switch (type) {
            case Camera::arcball:
                return "arcball";
            case Camera::flycam:
                return "flycam";
            case Camera::pinhole:
                return "pinhole";
            default:
                throw std::invalid_argument("Unknown CameraType");
        }
    }

    // Convert string to CameraType
    Camera::CameraType stringToCameraType(const std::string &str) {
        if (str == "arcball") return Camera::arcball;
        if (str == "flycam") return Camera::flycam;
        if (str == "pinhole") return Camera::pinhole;
        throw std::invalid_argument("Unknown CameraType: " + str);
    }
    */
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

    template<>
    struct convert<glm::vec4> {
        static Node encode(const glm::vec4 &rhs) {
            Node node;
            node.push_back(rhs.w);
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node &node, glm::vec4 &rhs) {
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
    YAML::Emitter &operator<<(YAML::Emitter &out, const glm::vec3 &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter &operator<<(YAML::Emitter &out, const glm::vec4 &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    YAML::Emitter &operator<<(YAML::Emitter &out, const glm::quat &v) {
        out << YAML::Flow;
        out << YAML::BeginSeq << v.w << v.x << v.y << v.z << YAML::EndSeq;
        return out;
    }

    SceneSerializer::SceneSerializer(const std::shared_ptr<Scene> &scene) : m_scene(scene) {
    }

    static void SerializeEntity(YAML::Emitter &out, Entity entity) {
        out << YAML::BeginMap;
        out << YAML::Key << "Entity";
        out << YAML::Value << entity.getUUID().operator std::string();
        // Serialize Parent UUID if the entity has a ParentComponent
        if (entity.hasComponent<ParentComponent>()) {
            auto parentEntity = entity.getParent();
            out << YAML::Key << "Parent";
            out << YAML::Value << parentEntity.getUUID().operator std::string();
        }
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
        // Serialize VisibleComponent
        if (entity.hasComponent<VisibleComponent>()) {
            out << YAML::Key << "VisibleComponent";
            out << YAML::BeginMap;
            auto &visible = entity.getComponent<VisibleComponent>().visible;
            out << YAML::Key << "Visible";
            out << YAML::Value << visible;
            out << YAML::EndMap;
        }
        // Serialize GroupComponent
        if (entity.hasComponent<GroupComponent>()) {
            out << YAML::Key << "GroupComponent";
            out << YAML::BeginMap;
            // Add any group-specific serialization if needed
            out << YAML::EndMap;
        }

        if (entity.hasComponent<ScriptableComponent>()) {
            auto& scriptComp = entity.getComponent<ScriptableComponent>();
            out << YAML::Key << "ScriptableComponent";
            out << YAML::BeginMap;
            out << YAML::Key << "ScriptName";
            out << YAML::Value << scriptComp.scriptName;
            out << YAML::EndMap;
        }

        if (entity.hasComponent<MeshComponent>()) {
            out << YAML::Key << "MeshComponent";
            out << YAML::BeginMap;
            auto &mesh = entity.getComponent<MeshComponent>();
            switch (mesh.meshDataType()) {
                case OBJ_FILE: {
                    auto params = std::dynamic_pointer_cast<OBJFileMeshParameters>(mesh.meshParameters);
                    out << YAML::Key << "ModelPath";
                    out << YAML::Value << params->path.string();
                }
                break;
                case PLY_FILE: {
                    auto params = std::dynamic_pointer_cast<PLYFileMeshParameters>(mesh.meshParameters);
                    out << YAML::Key << "ModelPath";
                    out << YAML::Value << params->path.string();
                }
                break;
                case CYLINDER: {
                    auto params = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
                    out << YAML::Key << "Origin";
                    out << YAML::Value << YAML::Flow << std::vector<float>{
                        params->origin.x, params->origin.y, params->origin.z
                    };

                    out << YAML::Key << "Direction";
                    out << YAML::Value << YAML::Flow << std::vector<float>{
                        params->direction.x, params->direction.y, params->direction.z
                    };

                    out << YAML::Key << "Magnitude";
                    out << YAML::Value << params->magnitude;

                    out << YAML::Key << "Radius";
                    out << YAML::Value << params->radius;
                }
                break;
                case QUADRIC: {
                    // Cast to our quadric type.
                    auto params = std::dynamic_pointer_cast<QuadricMeshParameters>(mesh.meshParameters);

                    out << YAML::Key << "a" << YAML::Value << params->a;
                    out << YAML::Key << "b" << YAML::Value << params->b;
                    out << YAML::Key << "c" << YAML::Value << params->c;
                    out << YAML::Key << "t_x" << YAML::Value << params->t_x;
                    out << YAML::Key << "t_y" << YAML::Value << params->t_y;

                    out << YAML::Key << "gridResolution" << YAML::Value << params->gridResolution;

                    out << YAML::Key << "min";
                    out << YAML::Value << YAML::Flow << std::vector<float>{
                        params->min.x, params->min.y
                    };

                    out << YAML::Key << "max";
                    out << YAML::Value << YAML::Flow << std::vector<float>{
                        params->max.x, params->max.y
                    };

                    out << YAML::Key << "BBeta" << YAML::Value << params->b_beta;
                    out << YAML::Key << "Threshold" << YAML::Value << params->threshold;
                    out << YAML::Key << "KernelScale" << YAML::Value << params->kernelScale;
                }
                break;
                default:
                    break;
            }
            out << YAML::Key << "MeshDataType";
            out << YAML::Value << meshDataTypeToString(mesh.meshDataType());
            out << YAML::Key << "PolygonMode";
            out << YAML::Value << Serialize::polygonModeToString(mesh.polygonMode());
            // Serialize PolygonMode as a string
            out << YAML::EndMap;
        }

        if (entity.hasComponent<CameraComponent>()) {
            out << YAML::Key << "CameraComponent";
            out << YAML::BeginMap;
            auto &camera = entity.getComponent<CameraComponent>();
            auto type = camera.cameraType;
            // Serialize CameraType
            out << YAML::Key << "CameraType";
            out << YAML::Value << CameraComponent::cameraTypeToString(camera.cameraType);
            out << YAML::Key << "RenderFromViewpoint";
            out << YAML::Value << camera.isActiveCamera();
            out << YAML::Key << "FlipX";
            out << YAML::Value << camera.cameraSettings.flipX;
            out << YAML::Key << "FlipY";
            out << YAML::Value << camera.cameraSettings.flipY;

            // Serialize based on CameraType
            switch (camera.cameraType) {
                case CameraComponent::ARCBALL:
                    // ARCBALL-specific serialization (if any) can be added here
                    break;
                case CameraComponent::PERSPECTIVE: {
                    auto &params = camera.baseCameraParameters;
                    out << YAML::Key << "ProjectionParameters";
                    out << YAML::BeginMap;
                    out << YAML::Key << "Near" << YAML::Value << params.nearPlane;
                    out << YAML::Key << "Far" << YAML::Value << params.farPlane;
                    out << YAML::Key << "Aspect" << YAML::Value << params.aspect;
                    out << YAML::Key << "FOV" << YAML::Value << params.fov;
                    out << YAML::EndMap;
                    break;
                }
                case CameraComponent::PINHOLE: {
                    auto &params = camera.pinholeParameters;
                    out << YAML::Key << "PinHoleParameters";
                    out << YAML::BeginMap;
                    out << YAML::Key << "Height" << YAML::Value << params.height;
                    out << YAML::Key << "Width" << YAML::Value << params.width;
                    out << YAML::Key << "Fx" << YAML::Value << params.fx;
                    out << YAML::Key << "Fy" << YAML::Value << params.fy;
                    out << YAML::Key << "Cx" << YAML::Value << params.cx;
                    out << YAML::Key << "Cy" << YAML::Value << params.cy;
                    out << YAML::Key << "Focal Length" << YAML::Value << params.focalLength;
                    out << YAML::Key << "Aperture" << YAML::Value << params.fNumber;
                    out << YAML::EndMap;
                    break;
                }
                default:
                    Log::Logger::getInstance()->warning("Fallback: Cannot serialize camera type");
            }
            out << YAML::EndMap;
        }

        if (entity.hasComponent<MaterialComponent>()) {
            out << YAML::Key << "MaterialComponent";
            out << YAML::BeginMap;
            auto &material = entity.getComponent<MaterialComponent>();
            // Serialize baseColor (glm::vec4)
            out << YAML::Key << "BaseColor";
            out << YAML::Value << YAML::Flow << std::vector<float>{
                material.albedo.r, material.albedo.g, material.albedo.b, material.albedo.a
            };
            // Serialize metallic factor (float)
            out << YAML::Key << "Emission";
            out << YAML::Value << material.emission;
            out << YAML::Key << "AlphaMode";
            out << YAML::Value << toString(material.alphaMode);
            out << YAML::Key << "Diffuse";
            out << YAML::Value << material.diffuse;
            out << YAML::Key << "Specular";
            out << YAML::Value << material.specular;
            out << YAML::Key << "PhongExponent";
            out << YAML::Value << material.phongExponent;
            out << YAML::Key << "UseVertexColor";
            out << YAML::Value << material.useTexture;
            // Serialize vertex shader name (std::filesystem::path)
            out << YAML::Key << "VertexShader";
            out << YAML::Value << material.vertexShaderName.string(); // Convert path to string
            // Serialize fragment shader name (std::filesystem::path)
            out << YAML::Key << "FragmentShader";
            out << YAML::Value << material.fragmentShaderName.string(); // Convert path to string
            out << YAML::Key << "AlbedoTexturePath";
            out << YAML::Value << material.albedoTexturePath.string(); // Convert path to string
            out << YAML::EndMap;
        }


        if (entity.hasComponent<LightSourceComponent>()) {
            out << YAML::Key << "LightSourceComponent";
            auto &component = entity.getComponent<LightSourceComponent>();
            out << YAML::BeginMap;
            // Serialize positions
            out << YAML::Key << "Flux";
            out << YAML::Value <<  component.flux;

            out << YAML::EndMap;
        }

        if (entity.hasComponent<RasterizerRenderingComponent>()) {
            out << YAML::Key << "RasterizerRenderingComponent";
            auto &component = entity.getComponent<RasterizerRenderingComponent>();
            out << YAML::BeginMap;

            out << YAML::EndMap;
        }

        if (entity.hasComponent<VkRender::QuadricCollectionComponent>()) {
            out << YAML::Key << "QuadricCollectionComponent";
            auto &component = entity.getComponent<VkRender::QuadricCollectionComponent>();
            out << YAML::BeginMap;

            // Serialize positions
            out << YAML::Key << "Positions";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto &pos: component.positions) {
                out << YAML::Flow << YAML::BeginSeq << pos.x << pos.y << pos.z << YAML::EndSeq;
            }
            out << YAML::EndSeq;

            // Serialize rotations (quaternions as: w, x, y, z)
            out << YAML::Key << "Rotations";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto &rot: component.rotations) {
                out << YAML::Flow << YAML::BeginSeq << rot.w << rot.x << rot.y << rot.z << YAML::EndSeq;
            }
            out << YAML::EndSeq;


            // Serialize positions
            out << YAML::Key << "Min";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto &pos: component.min) {
                out << YAML::Flow << YAML::BeginSeq << pos.x << pos.y <<YAML::EndSeq;
            }
            out << YAML::EndSeq;

            // Serialize positions
            out << YAML::Key << "Max";
            out << YAML::Value << YAML::BeginSeq;
            for (const auto &pos: component.max) {
                out << YAML::Flow << YAML::BeginSeq << pos.x << pos.y << YAML::EndSeq;
            }
            out << YAML::EndSeq;

            // Lambda to serialize float arrays
            auto serializeFloatArray = [&](const std::vector<float> &values, const std::string &key) {
                out << YAML::Key << key;
                out << YAML::Value << YAML::BeginSeq;
                for (const auto &value: values) {
                    out << value;
                }
                out << YAML::EndSeq;
            };

            // Serialize shape parameters
            serializeFloatArray(component.a, "A");
            serializeFloatArray(component.b, "B");
            serializeFloatArray(component.c, "C");
            serializeFloatArray(component.t_x, "T_x");
            serializeFloatArray(component.t_y, "T_y");

            // Serialize additional constants
            serializeFloatArray(component.kernelScale, "KernelScale");
            serializeFloatArray(component.threshold, "Threshold");
            serializeFloatArray(component.beta, "Beta");

            out << YAML::EndMap;
        }


        if (entity.hasComponent<GroupComponent>()) {
            out << YAML::Key << "GroupComponent";
            out << YAML::BeginMap;
            auto &groupComponent = entity.getComponent<GroupComponent>();
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
        out << YAML::Value << filePath.filename().string();

        out << YAML::Key << "Base Path";
        out << YAML::Value << filePath.parent_path().string();

        out << YAML::Key << "Entities";
        out << YAML::Value << YAML::BeginSeq;
        for (auto entity: m_scene->m_registry.view<entt::entity>()) {
            Entity e(entity, m_scene.get());
            if (!e || e.hasComponent<TemporaryComponent>())
                continue;
            SerializeEntity(out, e);
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;

        std::ofstream fout(filePath);
        fout << out.c_str();
        Log::Logger::getInstance()->info("Saved scene: {} to {}", filePath.filename().string(), filePath.string());
    }

    void SceneSerializer::serializeRuntime(const std::filesystem::path &filePath) {
        throw std::runtime_error("Not implemented");
    }

    bool SceneSerializer::deserialize(const std::filesystem::path &filePath) {
        if (filePath.extension() == ".multisense") {
            std::ifstream stream(filePath);
            std::stringstream stringStream;
            stringStream << stream.rdbuf();
            YAML::Node data = YAML::Load(stringStream.str());
            return deserializeYAML(data);
        }

        if (filePath.extension() == ".xml") {
            return deserializeXML(filePath);
        }
        //Log::Logger::getInstance()->info("Loaded scene: {} from {}", filePath.filename().string(), filePath.string());

        return false;
    }

    bool SceneSerializer::deserializeXML(const std::filesystem::path &filePath) {


        return true;
    }

    bool SceneSerializer::deserializeYAML(const YAML::Node& data) {
        // TODO sanitize input

        if (!data["Scene"])
            return false;
        std::string sceneName = data["Scene"].as<std::string>();

        auto entities = data["Entities"];
        if (entities) {
            std::unordered_map<uint64_t, Entity> entityMap;

            for (auto entity: entities) {
                auto entityId = UUID(entity["Entity"].as<uint64_t>()); // todo uuid
                std::string name = "Unnamed";
                auto tagComponent = entity["TagComponent"];
                if (tagComponent)
                    name = tagComponent["Tag"].as<std::string>();

                Entity deserializedEntity = m_scene->createEntityWithUUID(entityId, name); // TOdo uuid

                auto transformComponent = entity["TransformComponent"];
                if (transformComponent) {
                    auto &tc = deserializedEntity.getComponent<TransformComponent>();
                    tc.setPosition(transformComponent["Position"].as<glm::vec3>());
                    tc.setRotationQuaternion(transformComponent["Rotation"].as<glm::quat>());
                    tc.setScale(transformComponent["Scale"].as<glm::vec3>());
                }

                // Deserialize VisibleComponent
                auto visibleComponentNode = entity["VisibleComponent"];
                if (visibleComponentNode) {
                    auto &visibleComponent = deserializedEntity.addComponent<VisibleComponent>();
                    visibleComponent.visible = visibleComponentNode["Visible"].as<bool>();
                }

                auto cameraComponent = entity["CameraComponent"];
                if (cameraComponent) {
                    auto &camera = deserializedEntity.addComponent<CameraComponent>();

                    // Deserialize CameraType
                    if (cameraComponent["CameraType"]) {
                        std::string cameraTypeStr = cameraComponent["CameraType"].as<std::string>();
                        camera.cameraType = CameraComponent::stringToCameraType(cameraTypeStr);
                    }
                    // Deserialize CameraType
                    if (cameraComponent["RenderFromViewpoint"]) {
                        camera.isActiveCamera() = cameraComponent["RenderFromViewpoint"].as<bool>();
                    }
                    // Deserialize CameraType
                    if (cameraComponent["FlipX"]) {
                        camera.cameraSettings.flipX = cameraComponent["FlipX"].as<bool>();
                    }
                    // Deserialize CameraType
                    if (cameraComponent["FlipY"]) {
                        camera.cameraSettings.flipY = cameraComponent["FlipY"].as<bool>();
                    }

                    // Deserialize based on CameraType
                    switch (camera.cameraType) {
                        case CameraComponent::ARCBALL:
                            // ARCBALL-specific deserialization (if any) can be added here
                            break;

                        case CameraComponent::PERSPECTIVE: {
                            auto projectionParams = cameraComponent["ProjectionParameters"];
                            if (projectionParams) {
                                camera.baseCameraParameters.nearPlane = projectionParams["Near"].as<float>(0.1f);
                                camera.baseCameraParameters.farPlane = projectionParams["Far"].as<float>(100.0f);
                                camera.baseCameraParameters.aspect = projectionParams["Aspect"].as<float>(1.6f);
                                camera.baseCameraParameters.fov = projectionParams["FOV"].as<float>(60.0f);
                                camera.updateParametersChanged();
                            }
                            break;
                        }

                        case CameraComponent::PINHOLE: {
                            auto pinholeParams = cameraComponent["PinHoleParameters"];
                            if (pinholeParams) {
                                camera.pinholeParameters.height = pinholeParams["Height"].as<int>(720);
                                camera.pinholeParameters.width = pinholeParams["Width"].as<int>(1280);
                                camera.pinholeParameters.fx = pinholeParams["Fx"].as<float>(1280.0f);
                                camera.pinholeParameters.fy = pinholeParams["Fy"].as<float>(720.0f);
                                camera.pinholeParameters.cx = pinholeParams["Cx"].as<float>(640.0f);
                                camera.pinholeParameters.cy = pinholeParams["Cy"].as<float>(360.0f);
                                if (pinholeParams["Focal Length"])
                                    camera.pinholeParameters.focalLength = pinholeParams["Focal Length"].as<float>(
                                        100.0f);
                                if (pinholeParams["Aperture"])
                                    camera.pinholeParameters.fNumber = pinholeParams["Aperture"].as<float>(4.0f);
                            }
                            break;
                        }

                        default:
                            Log::Logger::getInstance()->warning("Fallback: Cannot deserialize camera type");
                    }
                    camera.updateParametersChanged();
                }

                auto meshComponentNode = entity["MeshComponent"];
                if (meshComponentNode) {
                    std::filesystem::path path;
                    if (meshComponentNode["ModelPath"]) {
                        path = meshComponentNode["ModelPath"].as<std::string>();
                    }

                    auto meshDataTypeStr = meshComponentNode["MeshDataType"].as<std::string>();
                    MeshDataType meshDataType = stringToMeshDataType(meshDataTypeStr);

                    // Add MeshComponent to the entity
                    auto &mesh = deserializedEntity.addComponent<MeshComponent>(meshDataType, path);
                    // Deserialize PolygonMode
                    if (meshComponentNode["PolygonMode"] && meshComponentNode["PolygonMode"].IsScalar()) {
                        std::string polygonModeStr = meshComponentNode["PolygonMode"].as<std::string>();
                        mesh.polygonMode() = Serialize::stringToPolygonMode(polygonModeStr);
                    } else {
                        // Handle missing PolygonMode (optional: set default or throw error)
                        mesh.polygonMode() = VK_POLYGON_MODE_FILL; // Default value
                    }

                    switch (meshDataType) {
                        case CYLINDER: {
                            auto params = std::make_shared<CylinderMeshParameters>();
                            auto originNode = meshComponentNode["Origin"];
                            if (originNode && originNode.IsSequence() && originNode.size() == 3) {
                                params->origin = glm::vec3(originNode[0].as<float>(), originNode[1].as<float>(),
                                                           originNode[2].as<float>());
                            }

                            auto directionNode = meshComponentNode["Direction"];
                            if (directionNode && directionNode.IsSequence() && directionNode.size() == 3) {
                                params->direction = glm::vec3(directionNode[0].as<float>(),
                                                              directionNode[1].as<float>(),
                                                              directionNode[2].as<float>());
                            }

                            if (meshComponentNode["Magnitude"]) {
                                params->magnitude = meshComponentNode["Magnitude"].as<float>();
                            }

                            if (meshComponentNode["Radius"]) {
                                params->radius = meshComponentNode["Radius"].as<float>();
                            }

                            mesh.meshParameters = params;
                        }
                        break;

                        case QUADRIC: {
                            auto params = std::make_shared<QuadricMeshParameters>();

                            if (meshComponentNode["a"])
                                params->a = meshComponentNode["a"].as<float>();
                            if (meshComponentNode["b"])
                                params->b = meshComponentNode["b"].as<float>();
                            if (meshComponentNode["c"])
                                params->c = meshComponentNode["c"].as<float>();
                            if (meshComponentNode["BBeta"])
                                params->b_beta = meshComponentNode["BBeta"].as<float>();
                            if (meshComponentNode["Threshold"])
                                params->threshold = meshComponentNode["Threshold"].as<float>();
                            if (meshComponentNode["KernelScale"])
                                params->kernelScale = meshComponentNode["KernelScale"].as<float>();
                            if (meshComponentNode["t_x"])
                                params->t_x = meshComponentNode["t_x"].as<float>();
                            if (meshComponentNode["t_y"])
                                params->t_y = meshComponentNode["t_y"].as<float>();

                            if (meshComponentNode["gridResolution"])
                                params->gridResolution = meshComponentNode["gridResolution"].as<int>();

                            if (meshComponentNode["min"] &&
                                meshComponentNode["min"].IsSequence() &&
                                meshComponentNode["min"].size() == 2) {
                                params->min = glm::vec2(
                                    meshComponentNode["min"][0].as<float>(),
                                    meshComponentNode["min"][1].as<float>()
                                );
                            }

                            if (meshComponentNode["max"] &&
                                meshComponentNode["max"].IsSequence() &&
                                meshComponentNode["max"].size() == 2) {
                                params->max = glm::vec2(
                                    meshComponentNode["max"][0].as<float>(),
                                    meshComponentNode["max"][1].as<float>()
                                );
                            }

                            mesh.meshParameters = params;
                        }

                        default: ;
                    }
                }

                auto materialComponent = entity["MaterialComponent"];
                if (materialComponent) {
                    auto &material = deserializedEntity.addComponent<MaterialComponent>();
                    // Deserialize base color
                    auto baseColor = materialComponent["BaseColor"].as<std::vector<float> >();
                    if (baseColor.size() == 4) {
                        material.albedo = glm::vec4(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                    }
                    if (materialComponent["AlphaMode"]) {
                        material.alphaMode = fromString(materialComponent["AlphaMode"].as<std::string>());
                    } else {
                        material.alphaMode = AlphaMode::Opaque; // Default value or handle as needed
                    }
                    if (materialComponent["Emission"]) {
                        material.emission = materialComponent["Emission"].as<float>();
                    } else {
                        material.emission = 0.0f; // Default value or handle as needed
                    }
                    if (materialComponent["Diffuse"]) {
                        material.diffuse = materialComponent["Diffuse"].as<float>();
                    } else {
                        material.diffuse = 0.0f; // Default value or handle as needed
                    }
                    if (materialComponent["Specular"]) {
                        material.specular = materialComponent["Specular"].as<float>();
                    } else {
                        material.specular = 0.0f; // Default value or handle as needed
                    }
                    if (materialComponent["PhongExponent"]) {
                        material.phongExponent = materialComponent["PhongExponent"].as<float>();
                    } else {
                        material.phongExponent = 32.0f; // Default value or handle as needed
                    }
                    if (materialComponent["UseVertexColor"]) {
                        material.useTexture = materialComponent["UseVertexColor"].as<bool>();
                    } else {
                        material.useTexture = false; // Default value or handle as needed
                    }
                    // Deserialize uses texture flag
                    if (materialComponent["VertexShader"]) {
                        material.vertexShaderName = std::filesystem::path(
                            materialComponent["VertexShader"].as<std::string>());
                    }
                    // Deserialize fragment shader name
                    if (materialComponent["FragmentShader"]) {
                        material.fragmentShaderName = std::filesystem::path(
                            materialComponent["FragmentShader"].as<std::string>());
                    }
                }
                auto groupComponent = entity["GroupComponent"];
                if (groupComponent) {
                    auto &component = deserializedEntity.addComponent<GroupComponent>();
                }

                auto lightSourceNode = entity["LightSourceComponent"];
                if (lightSourceNode) {
                    auto &component = deserializedEntity.addComponent<LightSourceComponent>();
                    auto &node = lightSourceNode;
                    // Deserialize fragment shader name
                    if (node["Flux"]) {
                        component.flux = node["Flux"].as<float>(100.0f);
                    }
                }
                auto rasterizerRenderingNode = entity["RasterizerRenderingComponent"];
                if (rasterizerRenderingNode) {
                    auto &component = deserializedEntity.addComponent<RasterizerRenderingComponent>();
                    auto &node = rasterizerRenderingNode;
                }

                auto quadricNode = entity["QuadricCollectionComponent"];
                if (quadricNode) {
                    auto &component = deserializedEntity.addComponent<VkRender::QuadricCollectionComponent>();
                    auto &node = quadricNode;

                    // Deserialize positions
                    if (node["Positions"]) {
                        for (const auto &positionNode: node["Positions"]) {
                            glm::vec3 position(
                                positionNode[0].as<float>(),
                                positionNode[1].as<float>(),
                                positionNode[2].as<float>()
                            );
                            component.positions.push_back(position);
                        }
                    }

                    // Deserialize rotations (assuming order: w, x, y, z)
                    if (node["Rotations"]) {
                        for (const auto &rotationNode: node["Rotations"]) {
                            glm::quat rotation(
                                rotationNode[0].as<float>(), // w
                                rotationNode[1].as<float>(), // x
                                rotationNode[2].as<float>(), // y
                                rotationNode[3].as<float>() // z
                            );
                            component.rotations.push_back(rotation);
                        }
                    }
                    // Deserialize rotations (assuming order: w, x, y, z)
                    if (node["Rotations"]) {
                        for (const auto &rotationNode: node["Rotations"]) {
                            glm::quat rotation(
                                rotationNode[0].as<float>(), // w
                                rotationNode[1].as<float>(), // x
                                rotationNode[2].as<float>(), // y
                                rotationNode[3].as<float>() // z
                            );
                            component.rotations.push_back(rotation);
                        }
                    }

                    // Deserialize rotations (assuming order: w, x, y, z)
                    if (node["Min"]) {
                        for (const auto &minNode: node["Min"]) {
                            glm::vec2 minVal(
                                minNode[0].as<float>(), // w
                                minNode[1].as<float>() // x
                            );
                            component.min.push_back(minVal);
                        }
                    }

                    // Deserialize rotations (assuming order: w, x, y, z)
                    if (node["Max"]) {
                        for (const auto &maxNode: node["Max"]) {
                            glm::vec2 maxVal(
                                maxNode[0].as<float>(), // w
                                maxNode[1].as<float>() // x
                            );
                            component.max.push_back(maxVal);
                        }
                    }

                    // Helper lambda to deserialize float arrays.
                    auto deserializeFloatArray = [&](std::vector<float> &values, const std::string &key,
                                                     size_t defaultSize = 0, float defaultValue = 0.0f) {
                        if (node[key]) {
                            for (const auto &valueNode: node[key]) {
                                values.push_back(valueNode.as<float>());
                            }
                        } else {
                            values.resize(defaultSize, defaultValue);
                        }
                    };

                    // Expected size from the positions array.
                    size_t expectedSize = component.positions.size();

                    deserializeFloatArray(component.a, "A", expectedSize, 1.0f);
                    deserializeFloatArray(component.b, "B", expectedSize, 1.0f);
                    deserializeFloatArray(component.c, "C", expectedSize, 1.0f);
                    deserializeFloatArray(component.t_x, "T_x", expectedSize, 2.0f);
                    deserializeFloatArray(component.t_y, "T_y", expectedSize, 2.0f);
                    deserializeFloatArray(component.kernelScale, "KernelScale", expectedSize, 1.0f);
                    deserializeFloatArray(component.threshold, "Threshold", expectedSize, 0.01f);
                    deserializeFloatArray(component.beta, "Beta", expectedSize, 0.0f);

                }


                auto scriptNode = entity["ScriptableComponent"];
                if (scriptNode) {
                    auto& scriptComp = deserializedEntity.addComponent<ScriptableComponent>();
                    std::string storedScriptName = scriptNode["ScriptName"].as<std::string>();

                    if (storedScriptName == std::string(getTypeName<DefaultController>())) {
                        scriptComp.bind<DefaultController>();
                    } else if (storedScriptName == std::string(getTypeName<VectorScripts>())) {
                        scriptComp.bind<VectorScripts>();
                    }else if (storedScriptName == std::string(getTypeName<ContributionRay>())) {
                        scriptComp.bind<ContributionRay>();
                    }
                }

                // Store the entity in the map
                entityMap[entityId] = deserializedEntity;

            }

            for (auto entityNode: entities) {
                uint64_t uuid = entityNode["Entity"].as<uint64_t>();
                Entity deserializedEntity = entityMap[uuid];

                // Check if the entity has a parent
                auto parentUUIDNode = entityNode["Parent"];
                if (parentUUIDNode) {
                    uint64_t parentUUID = parentUUIDNode.as<uint64_t>();
                    if (entityMap.find(parentUUID) != entityMap.end()) {
                        Entity parentEntity = entityMap[parentUUID];
                        deserializedEntity.setParent(parentEntity);
                    } else {
                        Log::Logger::getInstance()->warning("Parent entity with UUID {} not found.", parentUUID);
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
