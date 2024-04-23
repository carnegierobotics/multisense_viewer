//
// Created by magnus on 4/20/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT_H
#define MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT_H


#include "SkyboxGraphicsPipelineComponent.h"
#include "OBJModelComponent.h"

namespace VkRender {

    struct DefaultGraphicsPipelineComponent {
        DefaultGraphicsPipelineComponent() = default;

        DefaultGraphicsPipelineComponent(const DefaultGraphicsPipelineComponent &) = default;

        DefaultGraphicsPipelineComponent &
        operator=(const DefaultGraphicsPipelineComponent &other) { return *this; }

        DefaultGraphicsPipelineComponent(VkRender::RenderUtils *utils,
                                         const OBJModelComponent &modelComponent) {
            renderUtils = utils;
            vulkanDevice = utils->device;
            resources.resize(renderUtils->UBCount);
            emptyTexture.fromKtxFile(Utils::getTexturePath() / "empty.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice,
                                     vulkanDevice->m_TransferQueue);

            for (auto &resource: resources) {
                setupUniformBuffers(resource);
                setupDescriptors(resource, modelComponent);
                setupPipeline(resource);
            }
        }

        ~DefaultGraphicsPipelineComponent() {


            for(auto& res : resources) {

                vkDestroyPipeline(vulkanDevice->m_LogicalDevice, res.pipeline, nullptr);

                vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, res.pipelineLayout, nullptr);

                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayout, nullptr);

                vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, res.descriptorPool, nullptr);
            }

        };

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex);
        void update();
        void updateGraphicsPipeline();

        bool markedForDeletion = false;
        struct Resource {

            Buffer bufferParams;
            Buffer bufferScene;
            Buffer shaderMaterialBuffer;

            VkDescriptorPool descriptorPool;
            std::vector<VkDescriptorSet> descriptorSets;
            VkDescriptorSetLayout descriptorSetLayout;

            VkPipeline pipeline;
            VkPipelineLayout pipelineLayout;

            VkRender::UBOMatrix uboMatrix;
            VkRender::ShaderValuesParams shaderValuesParams;

            bool busy = false;
            bool requestIdle = false;
        };

        std::vector<Resource> resources;
    private:


        void setupUniformBuffers(Resource &res);

        void setupPipeline(Resource &res);

        void setupDescriptors(Resource &resource, const OBJModelComponent &modelComponent);


        Texture2D emptyTexture; // TODO Possibly make more empty textures to match our triple buffering?

        VkRender::RenderUtils *renderUtils;
        VulkanDevice *vulkanDevice;

    };


};
#endif //MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT_H
