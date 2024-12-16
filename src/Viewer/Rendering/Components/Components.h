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
