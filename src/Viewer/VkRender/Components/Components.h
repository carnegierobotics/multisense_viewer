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
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/VkRender/Core/UUID.h"
#include "Viewer/Scenes/ScriptSupport/Base.h"
#include "Viewer/Scenes/ScriptSupport/ScriptBuilder.h"

namespace VkRender {
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

    struct IDComponent {
        UUID ID{};

        IDComponent() = default;

        IDComponent(const IDComponent &) = default;

        explicit IDComponent(const UUID &uuid) : ID(uuid) {
        }

        // Explicitly define the copy assignment operator
        IDComponent &operator=(const IDComponent &other) {
            return *this;
        }
    };

    struct TagComponent {
        std::string Tag;

        TagComponent() = default;

        TagComponent(const TagComponent &other) = default;

        TagComponent &operator=(const TagComponent &other) = default;
    };

    struct TransformComponent {
        enum class RotationType {
            Euler,
            Quaternion
        };

        union {
            struct {
                float pitch;
                float yaw;
                float roll;
            } euler;

            glm::quat quaternion;
        };

        RotationType type;

        glm::vec3 translation = {0.0f, 0.0f, 0.0f};
        glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
        glm::vec3 scale = {1.0f, 1.0f, 1.0f};

        bool m_flipUpAxis = false;

        TransformComponent() = default;

        TransformComponent(const TransformComponent &other) = default;

        TransformComponent &operator=(const TransformComponent &other) = default;


        [[nodiscard]] glm::mat4 GetTransform() const {
            glm::mat4 rot = getRotMat();

            return glm::translate(glm::mat4(1.0f), translation)
                   * rot
                   * glm::scale(glm::mat4(1.0f), scale);
        }

        void setEuler(float pitch, float yaw, float roll) {
            euler.pitch = pitch;
            euler.yaw = yaw;
            euler.roll = roll;
            type = RotationType::Euler;
        }

        void setQuaternion(const glm::quat &q) {
            quaternion = q;
            type = RotationType::Quaternion;
        }

        void setPosition(const glm::vec3 &v) {
            translation = v;
        }

        void setFlipUpOption(const bool flipUp) {
            m_flipUpAxis = flipUp;
        }

        bool &getFlipUpOption() {
            return m_flipUpAxis;
        }

        glm::mat4 getRotMat() const {
            if (type == RotationType::Euler) {
                glm::vec3 radiansEuler = glm::radians(glm::vec3(euler.pitch, euler.yaw, euler.roll));
                glm::quat q = glm::quat(radiansEuler);
                glm::mat4 rot = glm::toMat4(q);
                if (m_flipUpAxis) {
                    glm::mat4 m_flipUpRotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                                                                   glm::vec3(1.0f, 0.0f, 0.0f));
                    rot = m_flipUpRotationMatrix * rot;
                }
                return rot;
            }
            glm::mat4 rot = glm::toMat4(quaternion);
            if (m_flipUpAxis) {
                glm::mat4 m_flipUpRotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                                                               glm::vec3(1.0f, 0.0f, 0.0f));
                rot = m_flipUpRotationMatrix * rot;
            }
            return rot;
        }
    };


    struct CameraComponent {
        Camera camera;
        bool drawGizmo = true;

        explicit CameraComponent(const Camera &cam) : camera(cam) {
        }

        // Explicitly define the copy assignment operator
        CameraComponent &operator=(const CameraComponent &other) {
            return *this;
        }

        // Overload the function call operator
        Camera &operator()() {
            return camera;
        }
    };

    struct ScriptComponent {
        std::string ClassName;
        std::shared_ptr<Base> script;

        ScriptComponent() = default;

        ScriptComponent(const ScriptComponent &) = default;

        ScriptComponent(std::string scriptName, Renderer* m_context) {
            script = ComponentMethodFactory::Create(scriptName);
            script->m_context = m_context;
            if (script == nullptr) {
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

    DISABLE_WARNING_POP
}

#endif //MULTISENSE_VIEWER_COMPONENTS_H
