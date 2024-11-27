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
#include <multisense_viewer/external/entt/include/entt/entt.hpp>
#include "Viewer/Rendering/Editors/Camera.h"
#include "Viewer/Rendering/Core/UUID.h"
#include "MeshComponent.h"

namespace VkRender {
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

    struct IDComponent {
        UUID ID{};

        IDComponent() = default;

        explicit IDComponent(const UUID &uuid) : ID(uuid) {
        }

    };

    struct TagComponent {
        std::string Tag;

        std::string &getTag() { return Tag; }

        void setTag(const std::string &tag) { Tag = tag; }

    };

    struct TransformComponent {
        // Flip the up axis option
        bool m_flipUpAxis = false;

        // Transformation components
        glm::vec3 translation = {0.0f, 0.0f, 0.0f};
        glm::vec3 rotationEuler = {0.0f, 0.0f, 0.0f}; // Euler angles in degrees
        glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
        glm::vec3 scale = {1.0f, 1.0f, 1.0f};

        // Constructors
        TransformComponent() = default;

        // Get the transformation matrix
        [[nodiscard]] glm::mat4 getTransform() const {
            glm::mat4 rotMat = glm::toMat4(rotation);

            if (m_flipUpAxis) {
                glm::mat4 flipUpRotationMatrix = glm::rotate(
                        glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                rotMat = flipUpRotationMatrix * rotMat;
            }

            return glm::translate(glm::mat4(1.0f), translation) * rotMat *
                   glm::scale(glm::mat4(1.0f), scale);
        }

        // Set rotation using Euler angles (degrees)
        void setRotationEuler(const glm::vec3 &eulerAnglesDegrees) {
            rotationEuler = eulerAnglesDegrees;
            glm::vec3 radians = glm::radians(rotationEuler);
            rotation = glm::quat(radians);
        }

        // Get rotation as Euler angles (degrees)
        glm::vec3 &getRotationEuler() {
            return rotationEuler;
        }

        void updateFromEulerRotation() {
            setRotationEuler(rotationEuler);
        }

        // Set rotation using quaternion
        void setRotationQuaternion(const glm::quat &q) {
            rotation = q;
            rotationEuler = glm::degrees(glm::eulerAngles(rotation));
        }

        // Get rotation as quaternion
        glm::quat &getRotationQuaternion() {
            return rotation;
        }

        // Position setters and getters
        void setPosition(const glm::vec3 &v) {
            translation = v;
        }

        glm::vec3 &getPosition() {
            return translation;
        }

        // Scale setters and getters
        void setScale(const glm::vec3 &s) {
            scale = s;
        }

        glm::vec3 &getScale() {
            return scale;
        }

        // Flip up axis option setters and getters
        void setFlipUpOption(const bool flipUp) {
            m_flipUpAxis = flipUp;
        }

        bool getFlipUpOption() const {
            return m_flipUpAxis;
        }
    };

    struct CameraComponent {
        // we use a shared pointer as storage since most often we need to share this data with the rendering loop.
        std::shared_ptr<Camera> camera;
        bool render = true;

        CameraComponent() {
            camera = std::make_shared<Camera>();
        }

        explicit CameraComponent(const Camera &cam) : camera(std::make_shared<Camera>(cam)) {
        }

        bool &renderFromViewpoint() { return render; }
    };

    struct ScriptComponent {
        std::string className;
    };


    struct ParentComponent {
        entt::entity parent = entt::null;
    };

    /** @brief Temporary components are not saved to scene file */
    struct TemporaryComponent {
        entt::entity entity;
    };

    struct ChildrenComponent {
        std::vector<entt::entity> children{};
    };
    struct VisibleComponent {
        bool visible = true;
    };
    struct GroupComponent {
        std::string placeHolder;

        std::filesystem::path colmapPath;
    };

    struct TextComponent {
        std::string TextString;
        glm::vec4 Color{1.0f};
        float Kerning = 0.0f;
        float LineSpacing = 0.0f;
    };

    struct VectorComponent {
        glm::vec3 origin;
        glm::vec3 direction;
        float magnitude;

    };
    DISABLE_WARNING_POP
}

#endif //MULTISENSE_VIEWER_COMPONENTS_H
