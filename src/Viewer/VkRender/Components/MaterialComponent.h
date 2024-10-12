//
// Created by mgjer on 04/10/2024.
//

#ifndef MATERIALCOMPONENT_H
#define MATERIALCOMPONENT_H

#include <glm/glm.hpp>
#include <filesystem>

#include "Viewer/VkRender/Editors/PipelineKey.h"
#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/VulkanTexture.h"

namespace VkRender {
    enum class AlphaMode {
        Opaque,
        Mask,
        Blend
    };
    struct MaterialComponent {
        glm::vec4 baseColor = glm::vec4(1.0f);    // Base color (could be an albedo color)
        float metallic = 1.0f;                    // Metallic factor
        float roughness = 1.0f;                   // Roughness factor
        // Emissive properties
        glm::vec4 emissiveFactor = glm::vec4(0.0f); // Default to no emission

        std::filesystem::path vertexShaderName = "defaultBasic.vert";
        std::filesystem::path fragmentShaderName = "defaultBasic.frag";
        std::filesystem::path albedoTexturePath = "default.png";

        bool usesVideoSource = false;
        std::filesystem::path videoFolderSource = "path/to/images";
        size_t videoFileNameIndex;
        std::vector<std::filesystem::path> videoFileNames;
    };

    struct MaterialInstance {
        RenderMode renderMode = RenderMode::Opaque;
        // Textures (may be null if not present)
        std::shared_ptr<VulkanTexture2D> vulkanTexture;

        std::shared_ptr<Texture> baseColorTexture; // TODO replace with VulkanTexture
        std::shared_ptr<Texture> metallicRoughnessTexture; // TODO replace with VulkanTexture
        std::shared_ptr<Texture> normalTexture; // TODO replace with VulkanTexture
        std::shared_ptr<Texture> occlusionTexture; // TODO replace with VulkanTexture
        std::shared_ptr<Texture> emissiveTexture; // TODO replace with VulkanTexture
        // Rendering properties
        AlphaMode alphaMode = AlphaMode::Opaque;
        float alphaCutoff = 0.5f;  // Used if alphaMode is Mask
        bool doubleSided = false;
    };

}

#endif //MATERIALCOMPONENT_H
