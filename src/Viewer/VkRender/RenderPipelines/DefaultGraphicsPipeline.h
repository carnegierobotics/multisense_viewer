//
// Created by magnus on 4/20/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
#define MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H


#include <stb_image.h>
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/RenderPipelines/RenderBase.h"
#include "Viewer/VkRender/RenderPipelines/GraphicsPipeline.h"

#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {

    class DefaultGraphicsPipeline : public GraphicsPipeline {
    public:

        DefaultGraphicsPipeline() = delete;

        /** @brief
        // Delete copy constructors, we dont want to perform shallow copied of vulkan resources leading to double deletion.
        // If copy is necessary define custom copy constructor and use move semantics or references
        */
        DefaultGraphicsPipeline(const DefaultGraphicsPipeline &) = delete;

        DefaultGraphicsPipeline &operator=(const DefaultGraphicsPipeline &) = delete;

        ~DefaultGraphicsPipeline();

        explicit DefaultGraphicsPipeline(Renderer &m_context, const RenderPassInfo &renderPassInfo,
                                                  const std::string &vertexShader = "default.vert.spv",
                                                  const std::string &fragmentShader = "default.frag.spv");


        void cleanUp();

        void update(uint32_t currentFrame) override;

        void updateTransform(TransformComponent &transform) override;

        void draw(CommandBuffer &cmdBuffers) override;

        void updateView(const Camera &camera) override;

        void bind(MeshComponent &meshComponent) override;

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

        std::string m_vertexShader;
        std::string m_fragmentShader;

        UBOMatrix m_vertexParams; // Non GPU-accessible data, shared across frames
        FragShaderParams m_fragParams; // Non GPU-accessible data, shared across frames
        std::vector<DefaultRenderData> m_renderData;
        SharedRenderData m_sharedRenderData;

        void setupUniformBuffers();

        void setupDescriptors();

        void setupPipeline();

        void setTexture(const VkDescriptorImageInfo *info);
    };


};
#endif //MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
