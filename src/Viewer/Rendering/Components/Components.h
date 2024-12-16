//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_COMPONENTS_H
#define MULTISENSE_VIEWER_COMPONENTS_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <filesystem>
#include <entt/entt.hpp>
#include <Viewer/Rendering/Editors/BaseCamera.h>
#include <Viewer/Tools/Macros.h>

#include "Viewer/Rendering/Core/UUID.h"
#include "Viewer/Rendering/Editors/PinholeCamera.h"

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


    struct CameraComponent {
        enum CameraType : uint32_t {
            PERSPECTIVE, // Classical Computer Graphics camera
            ARCBALL, // Classical Computer Graphics camera for orbiting camera around a point
            PINHOLE, // Computer Vision

        };

        // Utility function to convert CameraType to string
        static std::string cameraTypeToString(CameraType cameraType) {
            switch (cameraType) {
                case PERSPECTIVE: return "PERSPECTIVE";
                case PINHOLE: return "PINHOLE";
                case ARCBALL: return "ARCBALL";
                default: throw std::invalid_argument("Invalid CameraType");
            }
        }

// Utility function to convert string to CameraType
        static CameraType stringToCameraType(const std::string& cameraTypeStr) {
            static const std::unordered_map<std::string, CameraType> stringToEnum = {
                    {"PERSPECTIVE", PERSPECTIVE},
                    {"PINHOLE", PINHOLE},
                    {"ARCBALL", ARCBALL}
            };

            auto it = stringToEnum.find(cameraTypeStr);
            if (it != stringToEnum.end()) {
                return it->second;
            } else {
                throw std::invalid_argument("Invalid CameraType string: " + cameraTypeStr);
            }
        }

// Utility function to get all CameraType values as an array
        static std::array<CameraType, 3> getAllCameraTypes() {
            return {PERSPECTIVE, PINHOLE, ARCBALL};
        }
        // we use a shared pointer as storage since most often we need to share this data with the rendering loop.
        std::shared_ptr<BaseCamera> camera = std::make_shared<BaseCamera>(); // Possibly not required to be a pointer type, but we're passing it quite often so might be beneficial at the risk of safety
        bool render = false;
        bool flipY = true;
        CameraType cameraType = PERSPECTIVE;
        struct PinHoleParameters {
            int height = 720;         // Default image height
            int width = 1280;         // Default image width
            float fx = 1280.0f;          // Default horizontal focal length (pixels)
            float fy = 720.0f;          // Default vertical focal length (pixels)
            float cx = 640.0f;          // Default principal point x-coordinate (pixels)
            float cy = 360.0f;          // Default principal point y-coordinate (pixels)

            // Overload equality operator
            bool operator==(const PinHoleParameters& other) const {
                return height == other.height &&
                       width == other.width &&
                       fx == other.fx &&
                       fy == other.fy &&
                       cx == other.cx &&
                       cy == other.cy;
            }

            // Optional: Overload inequality operator for convenience
            bool operator!=(const PinHoleParameters& other) const {
                return !(*this == other);
            }
        } pinHoleParameters;

        struct ProjectionParameters {
            float near = 0.1f;
            float far = 100.0f;
            float aspect = 1.6f;
            float fov = 60.0f;

            // Overload equality operator
            bool operator==(const ProjectionParameters& other) const {
                return near == other.near &&
                       far == other.far &&
                       aspect == other.aspect &&
                       fov == other.fov;
            }
            // Overload inequality operator for convenience
            bool operator!=(const ProjectionParameters& other) const {
                return !(*this == other);
            }
        }projectionParameters;

        void updateParametersChanged(){
            switch (cameraType) {
                case PERSPECTIVE:
                    camera = std::make_shared<BaseCamera>(projectionParameters.aspect, projectionParameters.fov, projectionParameters.near, projectionParameters.far);
                    camera->m_flipYProjection = flipY;
                    break;
                case PINHOLE:
                    camera = std::make_shared<PinholeCamera>(pinHoleParameters.width, pinHoleParameters.height, pinHoleParameters.fx, pinHoleParameters.fy, pinHoleParameters.cx, pinHoleParameters.cy);
                    camera->m_flipYProjection = flipY;
                    break;
                default:
                    Log::Logger::getInstance()->warning("Camera type not implemented in scene. Reverting to Perspective");
                    camera = std::make_shared<BaseCamera>();
                    break;
            }
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

        std::filesystem::path colmapPath; // TODO remove
    };

    struct TextComponent {
        std::string TextString;
        glm::vec4 Color{1.0f};
        float Kerning = 0.0f;
        float LineSpacing = 0.0f;
    };

    DISABLE_WARNING_POP
}

#endif //MULTISENSE_VIEWER_COMPONENTS_H
