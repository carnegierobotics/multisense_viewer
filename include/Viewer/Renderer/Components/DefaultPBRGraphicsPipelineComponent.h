//
// Created by magnus on 4/12/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H
#define MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H

#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/GLTFDefines.h"
#include "Viewer/Tools/Utils.h"

namespace RenderResource {

    struct DefaultPBRGraphicsPipelineComponent {
        DefaultPBRGraphicsPipelineComponent() = default;

        DefaultPBRGraphicsPipelineComponent(const DefaultPBRGraphicsPipelineComponent &) = default;

        DefaultPBRGraphicsPipelineComponent &
        operator=(const DefaultPBRGraphicsPipelineComponent &other) { return *this; }

        DefaultPBRGraphicsPipelineComponent(VkRender::RenderUtils *utils,
                                            const VkRender::GLTFModelComponent &modelComponent,
                                            const RenderResource::SkyboxGraphicsPipelineComponent &skyboxComponent) {
            renderUtils = utils;
            vulkanDevice = utils->device;
            emptyTexture.fromKtxFile(Utils::getTexturePath() / "empty.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice,
                                     vulkanDevice->m_TransferQueue);

            createMaterialBuffer(modelComponent);
            setupUniformBuffers();
            setupDescriptors(modelComponent, skyboxComponent);
            setupPipelines();
            shaderValuesParams = skyboxComponent.shaderValuesParams;

        }

        ~DefaultPBRGraphicsPipelineComponent() {
            for (auto &pipeline: pipelines)
                vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline.second, nullptr);

            for (auto &pipelineLayout: pipelineLayouts)
                vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout.second, nullptr);

            vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayouts.scene, nullptr);
            vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayouts.material, nullptr);
            vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayouts.node, nullptr);
            vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayouts.materialBuffer, nullptr);
            vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
        };

        VkRender::RenderUtils *renderUtils;
        VulkanDevice *vulkanDevice;

        std::unordered_map<std::string, VkPipeline> pipelines;
        std::unordered_map<std::string, VkPipelineLayout> pipelineLayouts;
        VkPipeline boundPipeline = VK_NULL_HANDLE;
        struct DescriptorSetLayouts {
            VkDescriptorSetLayout scene;
            VkDescriptorSetLayout material;
            VkDescriptorSetLayout node;
            VkDescriptorSetLayout materialBuffer;
        } descriptorSetLayouts;

        std::vector<VkDescriptorSet> descriptorSets;
        VkDescriptorPool descriptorPool;
        std::vector<Buffer> bufferParams;
        std::vector<Buffer> bufferScene;

        VkRender::UBOMatrix uboMatrix;
        VkRender::ShaderValuesParams shaderValuesParams;

        Texture2D emptyTexture;

        // We use a material buffer to pass material data ind image indices to the shader
        struct alignas(16) ShaderMaterial {
            glm::vec4 baseColorFactor;
            glm::vec4 emissiveFactor;
            glm::vec4 diffuseFactor;
            glm::vec4 specularFactor;
            float workflow;
            int colorTextureSet;
            int PhysicalDescriptorTextureSet;
            int normalTextureSet;
            int occlusionTextureSet;
            int emissiveTextureSet;
            float metallicFactor;
            float roughnessFactor;
            float alphaMask;
            float alphaMaskCutoff;
            float emissiveStrength;
        };
        Buffer shaderMaterialBuffer;
        VkDescriptorSet descriptorSetMaterials;

        enum PBRWorkflows {
            PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1
        };

        void setupUniformBuffers();

        void setupPipelines();

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex, const VkRender::GLTFModelComponent &component);

        void createMaterialBuffer(const VkRender::GLTFModelComponent &component);

        void setupNodeDescriptorSet(VkRender::Node *pNode);

        void addPipelineSet(std::string prefix, std::string vertexShader, std::string fragmentShader);

        void
        setupDescriptors(const VkRender::GLTFModelComponent &component,
                         const SkyboxGraphicsPipelineComponent &skyboxComponent);

        void renderNode(CommandBuffer *commandBuffer, uint32_t cbIndex, VkRender::Node *node,
                        VkRender::Material::AlphaMode alphaMode);

        void update();
    };

}
#endif //MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H
