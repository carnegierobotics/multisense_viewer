//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_OBJMODELCOMPONENT
#define MULTISENSE_VIEWER_OBJMODELCOMPONENT


#include <vulkan/vulkan_core.h>

#include <utility>


#include "Viewer/Rendering/MeshData.h"

#include "Viewer/Tools/Utils.h"

namespace VkRender {

    struct MeshComponent {
        MeshComponent() = default;

        MeshComponent(MeshDataType type, std::filesystem::path path) : m_meshType(type), m_modelPath(std::move(path)) {}

        explicit MeshComponent(MeshDataType type) : m_meshType(type) {}

        std::filesystem::path &modelPath() { return m_modelPath; }

        MeshDataType &meshDataType() { return m_meshType; }

        VkPolygonMode &polygonMode() { return m_polygonMode; }

        std::string getCacheIdentifier() const {
            // Combine mesh type and model path to create a unique identifier
            std::string identifier = std::to_string(static_cast<int>(m_meshType)) + "_";

            if (!m_modelPath.empty()) {
                identifier += m_modelPath.string();
            } else {
                identifier += "NoPath";
            }

            return identifier;
        }

    private:
        VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;
        std::filesystem::path m_modelPath;
        MeshDataType m_meshType = MeshDataType::CUSTOM;

    };


};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT
