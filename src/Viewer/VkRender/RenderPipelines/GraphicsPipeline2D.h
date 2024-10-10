//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H
#define MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H

#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/RenderPipelines/RenderBase.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/GraphicsPipeline.h"
#include "Viewer/VkRender/Core/VulkanTexture.h"

namespace VkRender {
    class Application;

    // TODO instead of inherting from standard 3D Graphics pipeline, we should inherit from a separate type just for 2D to avoid messy overrides
    class GraphicsPipeline2D {
    public:
        GraphicsPipeline2D(Application &context, const RenderPassInfo &renderPassInfo);

        ~GraphicsPipeline2D();

        void updateTransform(TransformComponent &transform);

        void updateView(const Camera &camera);

        void update(uint32_t currentFrameIndex);
        void updateTexture(void* data, size_t size);

        void bindTexture(std::shared_ptr<VulkanTexture> texture);

        void draw(CommandBuffer &commandBuffer);

        void setTexture(const VkDescriptorImageInfo *info);

    private:
        void setupPipeline();

        void setupDescriptors(const std::shared_ptr<VulkanTexture>& ptr);

        void setupUniformBuffers();


        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        std::unique_ptr<TextureVideo> m_renderTexture;

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

        Vertices m_vertices;
        Indices m_indices;

        std::string m_vertexShader;
        std::string m_fragmentShader;

        UBOMatrix m_vertexParams; // Non GPU-accessible data, shared across frames
        FragShaderParams m_fragParams; // Non GPU-accessible data, shared across frames
        std::vector<DefaultRenderData> m_renderData;
        SharedRenderData m_sharedRenderData;

    };
};


#endif //MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H
