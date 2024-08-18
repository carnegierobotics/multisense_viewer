//
// Created by mgjer on 14/08/2024.
//

#ifndef MULTISENSE_VIEWER_UBOGRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_UBOGRAPHICSPIPELINE_H

#include "Viewer/VkRender/RenderPipelines//RenderBase.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Components/Components.h"
#include "GraphicsPipeline.h"

namespace VkRender {
    class Renderer;

    class UboGraphicsPipeline : public GraphicsPipeline {
    public:


        UboGraphicsPipeline(Renderer &m_context, const RenderPassInfo &renderPassInfo);
        ~UboGraphicsPipeline();

        void update(uint32_t currentFrame) override;

        void updateTransform(TransformComponent &transform) override;

        void draw(CommandBuffer &cmdBuffers) override;

        void updateView(const Camera &camera) override;

        void bind(MeshComponent &meshComponent) override;

    private:
        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        uint32_t m_vertexCount;
        std::string m_vertexShader;
        std::string m_fragmentShader;
        UBOMatrix m_vertexParams; // Non GPU-accessible data, shared across frames
        std::vector<DefaultRenderData> m_renderData;
        SharedRenderData m_sharedRenderData;
        void setupUniformBuffers();
        void setupDescriptors();
        void setupPipeline();
    };
}
#endif //MULTISENSE_VIEWER_UBOGRAPHICSPIPELINE_H
