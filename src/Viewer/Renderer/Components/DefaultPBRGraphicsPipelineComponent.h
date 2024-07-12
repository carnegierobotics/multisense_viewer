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
#include "Viewer/Renderer/Components.h"

namespace VkRender {

    struct DefaultPBRGraphicsPipelineComponent {
        DefaultPBRGraphicsPipelineComponent() = default;

        DefaultPBRGraphicsPipelineComponent(const DefaultPBRGraphicsPipelineComponent &) = delete;

        DefaultPBRGraphicsPipelineComponent &
        operator=(const DefaultPBRGraphicsPipelineComponent &other) { return *this; }

        DefaultPBRGraphicsPipelineComponent(RenderUtils *utils,
                                            const GLTFModelComponent &modelComponent,
                                            const SkyboxGraphicsPipelineComponent &skyboxComponent) {
            renderUtils = utils;
            vulkanDevice = utils->device;
            resources.resize(renderUtils->swapchainImages);
            emptyTexture.fromKtxFile((Utils::getTexturePath() / "empty.ktx").string(), VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, vulkanDevice->m_TransferQueue);
            for(size_t i = 0; i < resources.size(); ++i){
            createMaterialBuffer(resources[i], modelComponent);
            setupUniformBuffers(resources[i]);
            setupDescriptors(resources[i], modelComponent, skyboxComponent);
            resources[i].shaderValuesParams = skyboxComponent.shaderValuesParams;
                setupPipelines(resources[i]);
            }
        }

        ~DefaultPBRGraphicsPipelineComponent() {

            for(auto& res : resources) {

                for (auto &pipeline: res.pipelines)
                    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline.second, nullptr);

                for (auto &pipelineLayout: res.pipelineLayouts)
                    vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout.second, nullptr);

                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayouts.scene, nullptr);
                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayouts.material, nullptr);
                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayouts.node, nullptr);
                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayouts.materialBuffer,
                                             nullptr);
                vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, res.descriptorPool, nullptr);
            }
        };

        RenderUtils *renderUtils;
        VulkanDevice *vulkanDevice;

        struct DescriptorSetLayouts {
            VkDescriptorSetLayout scene;
            VkDescriptorSetLayout material;
            VkDescriptorSetLayout node;
            VkDescriptorSetLayout materialBuffer;
        };

        struct Resource {

            Buffer bufferParams;
            Buffer bufferScene;
            Buffer shaderMaterialBuffer;

            VkDescriptorPool descriptorPool;
            std::vector<VkDescriptorSet> descriptorSets;
            DescriptorSetLayouts descriptorSetLayouts;

            VkPipeline boundPipeline = VK_NULL_HANDLE;
            std::unordered_map<std::string, VkPipeline> pipelines;
            std::unordered_map<std::string, VkPipelineLayout> pipelineLayouts;

            VkDescriptorSet descriptorSetMaterials;

            UBOMatrix uboMatrix;
            ShaderValuesParams shaderValuesParams;

            bool busy = false;
        };
        bool markedForDeletion = false;
        Texture2D emptyTexture; // TODO Possibly make more empty textures to match our triple buffering?
        std::vector<Resource> resources;
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


        enum PBRWorkflows {
            PBR_WORKFLOW_METALLIC_ROUGHNESS = 0, PBR_WORKFLOW_SPECULAR_GLOSINESS = 1
        };

        void setupUniformBuffers(Resource& res);

        void setupPipelines(Resource& res);

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex, const GLTFModelComponent &component);

        void createMaterialBuffer(Resource& res, const GLTFModelComponent &component);

        void setupNodeDescriptorSet(Node *pNode, VkDescriptorPool pool, VkDescriptorSetLayout* layout);

        void addPipelineSet(Resource& resource, std::string prefix, std::string vertexShader, std::string fragmentShader);

        void
        setupDescriptors(Resource& res, const GLTFModelComponent &component,
                         const SkyboxGraphicsPipelineComponent &skyboxComponent);

        void renderNode(CommandBuffer *commandBuffer, uint32_t cbIndex, Node *node,
                        Material::AlphaMode alphaMode);

        void update(uint32_t frameID);

        void updateView(const Camera &camera);

        void updateTransform(const TransformComponent &transform);

    };

}
#endif //MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H
