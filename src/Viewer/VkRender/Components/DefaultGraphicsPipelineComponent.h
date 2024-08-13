//
// Created by magnus on 4/20/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
#define MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H


#include <stb_image.h>
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Components/RenderBase.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/VkRender/Components/Components.h"

namespace VkRender {

    class DefaultGraphicsPipelineComponent : RenderBase {
    public:
        struct RenderPassInfo {
            VkSampleCountFlagBits sampleCount;
            VkRenderPass renderPass;
        };


        DefaultGraphicsPipelineComponent() = default;

        /** @brief
        // Delete copy constructors, we dont want to perform shallow copied of vulkan resources leading to double deletion.
        // If copy is necessary define custom copy constructor and use move semantics or references
        */
        DefaultGraphicsPipelineComponent(const DefaultGraphicsPipelineComponent &) = delete;

        DefaultGraphicsPipelineComponent &operator=(const DefaultGraphicsPipelineComponent &) = delete;

        ~DefaultGraphicsPipelineComponent() override;

        explicit DefaultGraphicsPipelineComponent(Renderer &m_context, const RenderPassInfo &renderPassInfo,
                                                  const std::string &vertexShader = "default.vert.spv",
                                                  const std::string &fragmentShader = "default.frag.spv");


        bool cleanUp();

        void update(uint32_t currentFrame) override;

        void updateTransform(const TransformComponent &transform);

        void draw(CommandBuffer &cmdBuffers);

        void updateView(const Camera &camera);

        template<typename T>
        void bind(T &modelComponent);

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
