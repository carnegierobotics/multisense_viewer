//
// Created by magnus on 4/20/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
#define MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H


#include <stb_image.h>
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Renderer/Components/RenderComponents/RenderBase.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {

    class DefaultGraphicsPipelineComponent2 : RenderBase {
    public:

        DefaultGraphicsPipelineComponent2() = default;

        /** @brief
        // Delete copy constructors, we dont want to perform shallow copied of vulkan resources leading to double deletion.
        // If copy is necessary define custom copy constructor and use move semantics or references
        */
        DefaultGraphicsPipelineComponent2(const DefaultGraphicsPipelineComponent2 &) = delete;

        DefaultGraphicsPipelineComponent2 &operator=(const DefaultGraphicsPipelineComponent2 &) = delete;

        explicit DefaultGraphicsPipelineComponent2(VkRender::RenderUtils *utils,
                                                   const std::string &vertexShader = "default.vert.spv",
                                                   const std::string &fragmentShader = "default.frag.spv") {

            m_numSwapChainImages = utils->swapchainImages;
            m_vulkanDevice = utils->device;
            m_utils = utils;
            // Number of resources per render pass
            m_renderData.resize(m_numSwapChainImages);
            m_emptyTexture.fromKtxFile((Utils::getTexturePath() / "empty.ktx").string(), VK_FORMAT_R8G8B8A8_UNORM, m_vulkanDevice, m_vulkanDevice->m_TransferQueue);

            // Assume we get a modelComponent that has vertex and index buffers in gpu memory. We need to create graphics resources which are:
            // Descriptor sets: pool, layout, sets
            // Uniform Buffers:
            setupUniformBuffers();
            setupDescriptors();
            // First create normal render pass resources
            // Graphics pipelines
            for (auto &data: m_renderData) {

                setupPipeline(data, RENDER_PASS_COLOR, vertexShader, fragmentShader, utils->msaaSamples,
                              *utils->renderPass);

                setupPipeline(data, RENDER_PASS_SECOND, vertexShader, fragmentShader, utils->msaaSamples,
                              *utils->renderPass);

                setupPipeline(data, RENDER_PASS_DEPTH_ONLY, vertexShader, fragmentShader,
                              VK_SAMPLE_COUNT_1_BIT,
                              utils->depthRenderPass->renderPass);
            }

            m_vertexShader = vertexShader;
            m_fragmentShader = fragmentShader;
        }

        ~DefaultGraphicsPipelineComponent2() override {
            if (!resourcesDeleted)
                cleanUp(0, true);
        };


        void draw(CommandBuffer *cmdBuffers) override;

        bool cleanUp(uint32_t currentFrame, bool force = false) override;

        void update(uint32_t currentFrame) override;

        void reloadShaders();

        template <typename T>
        void bind(T &modelComponent);

        void setTexture(const VkDescriptorImageInfo* info);

    public:
        UBOMatrix mvp;
        FragShaderParams fragShaderParams;

    private:

        struct Vertices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t vertexCount = 0;
        };
        struct Indices {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            uint32_t indexCount = 0;
        };


        void setupUniformBuffers();

        void setupDescriptors();

        void setupPipeline(DefaultRenderData &data, RenderPassType type, const std::string &vertexShader,
                           const std::string &fragmentShader,
                           VkSampleCountFlagBits sampleCountFlagBits, VkRenderPass renderPass);

    private:
        VulkanDevice *m_vulkanDevice = nullptr;
        uint32_t m_numSwapChainImages = 0;
        Texture2D m_emptyTexture;
        Texture2D m_objTexture;
        std::vector<DefaultRenderData> m_renderData;
        bool boundToModel = false;
        bool resourcesDeleted = false;
        Indices indices{};
        Vertices vertices{};

        bool requestRebuildPipelines = false;
        std::string m_vertexShader;
        std::string m_fragmentShader;
        VkRender::RenderUtils *m_utils;
    };


};
#endif //MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
