//
// Created by magnus on 4/20/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
#define MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H


#include <Viewer/VkRender/Editors/PipelineKey.h>

#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/RenderPipelines/RenderBase.h"

#include "Viewer/VkRender/Components/Components.h"


namespace VkRender {
    class Application;

    class DefaultGraphicsPipeline {
    public:
        DefaultGraphicsPipeline() = delete;

        /** @brief
        // Delete copy constructors, we dont want to perform shallow copied of vulkan resources leading to double deletion.
        // If copy is necessary define custom copy constructor and use move semantics or references
        */
        DefaultGraphicsPipeline(const DefaultGraphicsPipeline &) = delete;

        DefaultGraphicsPipeline &operator=(const DefaultGraphicsPipeline &) = delete;

        ~DefaultGraphicsPipeline();

        explicit DefaultGraphicsPipeline(Application &m_context, const RenderPassInfo &renderPassInfo,
                                         const PipelineKey &key);


        void cleanUp();

        void bind(CommandBuffer &commandBuffer) const;

        std::shared_ptr<VulkanGraphicsPipeline> pipeline() { return m_graphicsPipeline; }

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

        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        Texture2D m_emptyTexture;
        Texture2D m_objTexture;

        Indices indices{};
        Vertices vertices{};

        std::filesystem::path m_vertexShader;
        std::filesystem::path m_fragmentShader;

        UBOMatrix m_vertexParams; // Non GPU-accessible data, shared across frames
        FragShaderParams m_fragParams; // Non GPU-accessible data, shared across frames
        std::vector<DefaultRenderData> m_renderData;
        SharedRenderData m_sharedRenderData;

        std::shared_ptr<VulkanGraphicsPipeline> m_graphicsPipeline;

        void setupUniformBuffers();

        void setupDescriptors();

        void setupPipeline();

        void setTexture(const VkDescriptorImageInfo *info);
    };
};
#endif //MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
