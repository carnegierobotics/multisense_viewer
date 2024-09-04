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
#include "MeshComponent.h"

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

        struct Rotation {

            glm::vec3 rotation;

            glm::quat quaternion;
        } rot;

        bool m_flipUpAxis = false;

        TransformComponent() = default;

        TransformComponent(const TransformComponent &other) = default;

        TransformComponent &operator=(const TransformComponent &other) = default;


        [[nodiscard]] glm::mat4 GetTransform() {
            glm::mat4 rot = getRotMat();

            return glm::translate(glm::mat4(1.0f), translation)
                   * rot
                   * glm::scale(glm::mat4(1.0f), scale);
        }



        void setQuaternion(const glm::quat &q) {
            rot.quaternion = q;
            type = RotationType::Quaternion;
        }
        glm::quat &getQuaternion() {
            return rot.quaternion;
        }
        glm::vec3 &getRotation() {
            return rot.rotation;
        }

        void setPosition(const glm::vec3 &v) {
            translation = v;
        }

        void setScale(const glm::vec3& s){
            scale = s;
        }

        glm::vec3 &getPosition() {
            return translation;
        }
        glm::vec3 &getScale() {
            return scale;
        }

        void setFlipUpOption(const bool flipUp) {
            m_flipUpAxis = flipUp;
        }

        bool &getFlipUpOption() {
            return m_flipUpAxis;
        }

        glm::mat4 getRotMat() {
            if (type == RotationType::Euler) {
                glm::vec3 radiansEuler = glm::radians(rot.rotation);
                rot.quaternion = glm::quat(radiansEuler);
                glm::mat4 rotMat = glm::toMat4(rot.quaternion);
                if (m_flipUpAxis) {
                    glm::mat4 m_flipUpRotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                                                                   glm::vec3(1.0f, 0.0f, 0.0f));
                    rotMat = m_flipUpRotationMatrix * rotMat;
                }
                return rotMat;
            }
            glm::mat4 rotMat = glm::toMat4(rot.quaternion);
            rot.rotation = glm::eulerAngles(rot.quaternion);

            if (m_flipUpAxis) {
                glm::mat4 m_flipUpRotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                                                               glm::vec3(1.0f, 0.0f, 0.0f));
                rotMat = m_flipUpRotationMatrix * rotMat;
            }
            return rotMat;
        }
    private:

        RotationType type;

        glm::vec3 translation = {0.0f, 0.0f, 0.0f};
        glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
        glm::vec3 scale = {1.0f, 1.0f, 1.0f};

    };


    struct CameraComponent {
        Camera camera;
        bool drawGizmo = true;

        CameraComponent() = default;

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

        ScriptComponent(std::string scriptName, Application *m_context) {
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
