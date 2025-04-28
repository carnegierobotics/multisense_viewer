//
// Created by mgjer on 04/10/2024.
//

#ifndef MATERIALCOMPONENT_H
#define MATERIALCOMPONENT_H

#include <glm/glm.hpp>
#include <filesystem>

#include <Viewer/Rendering/Core/RenderDefinitions.h>
#include <Viewer/Rendering/Core/VulkanShaderModule.h>
#include "Viewer/Rendering/Core/VulkanTexture.h"

namespace VkRender {
    enum class AlphaMode {
        Opaque,
        Mask,
        Blend
    };
    struct MaterialComponent {
        glm::vec4 albedo = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);    // Base color (could be an albedo color)
        float emission = 0.0f;         // Emissive power
        float diffuse = 1.0f;           // Diffuse coefficient
        float specular = 0.0f;          // Specular coefficient
        float phongExponent = 32.0f;     // Shininess exponent


        bool reloadShader = false;
        bool useTexture = false;

        std::filesystem::path vertexShaderName = "BlinnPhongShader.vert";
        std::filesystem::path fragmentShaderName = "BlinnPhongShader.frag";
        std::filesystem::path albedoTexturePath = "default.png";
    };

    struct MaterialInstance {
        std::shared_ptr<VulkanTexture2D> baseColorTexture;
        // Rendering properties
        AlphaMode alphaMode = AlphaMode::Opaque;
        float alphaCutoff = 0.5f;  // Used if alphaMode is Mask
        bool doubleSided = false;


        std::vector<std::shared_ptr<VulkanShaderModule>> shaders;
        void addShader(VulkanShaderModuleCreateInfo info) {
            shaders.push_back(std::make_shared<VulkanShaderModule>(info));
        }
        void addShader(std::shared_ptr<VulkanShaderModule> shader) {
            shaders.push_back(shader);
        }
    };

    static MaterialBufferObject makeMaterialBuffer(const MaterialComponent& c)
    {
        MaterialBufferObject mbo{};
        mbo.baseColor      = c.albedo;
        mbo.specular       = c.specular;
        mbo.diffuse        = c.diffuse;
        mbo.emissiveFactor = glm::vec4(glm::vec3(c.emission), 1.0f);
        mbo.useVertexColor = c.useTexture ? 1.0f : 0.0f;
        // note: we no longer fill light data here, that’s in the global UBO
        return mbo;
    }
}

#endif //MATERIALCOMPONENT_H
