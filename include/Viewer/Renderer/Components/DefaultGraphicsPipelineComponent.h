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
                                         const OBJModelComponent &modelComponent,
                                         bool secondaryAvailable = false,
                                         const std::string& vertexShader = "default.vert.spv",
                                         const std::string& fragmentShader = "default.frag.spv") {
            renderUtils = utils;
            vulkanDevice = utils->device;
            emptyTexture.fromKtxFile(Utils::getTexturePath() / "empty.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice,
                                     vulkanDevice->m_TransferQueue);

            uint32_t numSwapChainImages = renderUtils->UBCount;
            uint32_t numResourcesToAllocate = 1;
            std::vector<VkRenderPass> passes = {{*renderUtils->renderPass}};
            if (secondaryAvailable) {
                numResourcesToAllocate++;
                passes = {{*renderUtils->renderPass, renderUtils->secondaryRenderPasses->front().renderPass}};
            }

            resources.resize(numResourcesToAllocate);
            renderData.resize(numSwapChainImages);
            setupUniformBuffers();
            setupDescriptors(modelComponent);

            for (size_t i = 0; i < resources.size(); ++i) {
                resources[i].res.resize(numSwapChainImages);
                resources[i].renderPass = passes[i];
                resources[i].vertexShader = vertexShader;
                resources[i].fragmentShader = fragmentShader;
                for (size_t j = 0; j < resources[i].res.size(); ++j) {
                    setupPipeline(resources[i].res[j], vertexShader, fragmentShader,
                                  renderData[j].descriptorSetLayout, resources[i].renderPass);
                }
            }
        }


        ~DefaultGraphicsPipelineComponent() {


            for (auto &resource: resources) {
                for (auto &res: resource.res) {
                    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, res.pipeline, nullptr);
                    vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, res.pipelineLayout, nullptr);
                    //vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayout, nullptr);
                    //vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, res.descriptorPool, nullptr);
                }
            }

            for (auto &res: renderData) {
                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayout, nullptr);
                vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, res.descriptorPool, nullptr);
            }

        };

        bool draw(CommandBuffer *commandBuffer, uint32_t cbIndex, uint32_t renderPassIndex = 0);

        void update();

        void updateGraphicsPipeline();

        bool markedForDeletion = false;
        struct Resource {

            VkPipeline pipeline{};
            VkPipelineLayout pipelineLayout{};

            bool busy = false;
            bool requestIdle = false;
        };

        struct RenderResource {
            std::vector<Resource> res{};
            VkRenderPass renderPass = VK_NULL_HANDLE;
            std::string vertexShader;
            std::string fragmentShader;
        };

        struct RenderData {
            Buffer bufferParams;
            Buffer bufferScene;
            Buffer shaderMaterialBuffer;
            VkDescriptorPool descriptorPool{};
            std::vector<VkDescriptorSet> descriptorSets;
            VkDescriptorSetLayout descriptorSetLayout{};
            VkRender::UBOMatrix uboMatrix;
            VkRender::ShaderValuesParams shaderValuesParams;
        };

        std::vector<RenderResource> resources;
        std::vector<RenderData> renderData;

    private:
        void setupUniformBuffers();

        void setupDescriptors(const OBJModelComponent &modelComponent);

        Texture2D emptyTexture; // TODO Possibly make more empty textures to match our triple buffering?
        VkRender::RenderUtils *renderUtils;
        VulkanDevice *vulkanDevice;

        void setupPipeline(Resource &resource, const std::string &vertexShader, const std::string &fragmentShader,
                           VkDescriptorSetLayout &pT, VkRenderPass pPassT);
    };


};
#endif //MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT_H
