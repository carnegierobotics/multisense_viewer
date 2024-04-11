//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_COMPONENTS_H
#define MULTISENSE_VIEWER_COMPONENTS_H

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/quaternion.hpp>

#include "Viewer/Core/Camera.h"
#include "Viewer/Core/UUID.h"

namespace VkRender {

    struct IDComponent {
        UUID ID{};

        IDComponent() = default;

        IDComponent(const IDComponent &) = default;

        explicit IDComponent(const UUID &uuid) : ID(uuid) {}

        IDComponent &operator=(const IDComponent &other) = default;

    };

    struct TagComponent {
        std::string Tag;

        TagComponent() = default;

        TagComponent(const TagComponent &other) = default;

        TagComponent &operator=(const TagComponent &other) = default;

    };

    struct TransformComponent {
        glm::vec3 Translation = {0.0f, 0.0f, 0.0f};
        glm::vec3 Rotation = {0.0f, 0.0f, 0.0f};
        glm::vec3 Scale = {1.0f, 1.0f, 1.0f};

        TransformComponent() = default;

        TransformComponent(const TransformComponent &other) = default;

        TransformComponent &operator=(const TransformComponent &other) = default;


        [[nodiscard]] glm::mat4 GetTransform() const {
            glm::mat4 rotation = glm::toMat4(glm::quat(Rotation));

            return glm::translate(glm::mat4(1.0f), Translation)
                   * rotation
                   * glm::scale(glm::mat4(1.0f), Scale);
        }
    };


    struct CameraComponent {
        Camera camera;
        bool primary = true; // TODO: think about moving to Scene
        bool fixedAspectRatio = false;

        CameraComponent() = default;

        CameraComponent(const CameraComponent &) = default;

        explicit CameraComponent(Camera *cam) : camera(*cam) {}
    };

    struct ScriptComponent {
        std::string ClassName;
        std::shared_ptr<Base2> script;
        ScriptComponent() = default;

        ScriptComponent(const ScriptComponent &) = default;

        ScriptComponent(std::string scriptName, Renderer* m_context){
            script = VkRender::ComponentMethodFactory::Create(scriptName);
            script->m_context = m_context;
            if (script.get() == nullptr) {
                Log::Logger::getInstance()->error("Failed to register script {}.", scriptName);
                return;
            }
            Log::Logger::getInstance()->info("Registered script: {} in factory", scriptName.c_str());
        }
    };


    struct Rigidbody2DComponent {
        enum class BodyType {
            Static = 0, Dynamic, Kinematic
        };
        BodyType Type = BodyType::Static;
        bool FixedRotation = false;

        // Storage for runtime
        void *RuntimeBody = nullptr;

        Rigidbody2DComponent() = default;

        Rigidbody2DComponent(const Rigidbody2DComponent &) = default;
    };


    struct TextComponent {
        std::string TextString;
        glm::vec4 Color{1.0f};
        float Kerning = 0.0f;
        float LineSpacing = 0.0f;
    };


    template<typename... Component>
    struct ComponentGroup {
    };

    using AllComponents =
            ComponentGroup<TransformComponent,
                    CameraComponent, ScriptComponent,
                    Rigidbody2DComponent, TextComponent>;

}

#endif //MULTISENSE_VIEWER_COMPONENTS_H
