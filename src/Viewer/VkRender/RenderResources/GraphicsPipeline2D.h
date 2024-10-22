//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H
#define MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H

#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Core/VulkanTexture.h"

namespace VkRender {
    class Application;

    // TODO instead of inherting from standard 3D Graphics pipeline, we should inherit from a separate type just for 2D to avoid messy overrides
    class GraphicsPipeline2D {
    public:
        GraphicsPipeline2D(Application &context, const RenderPassInfo &renderPassInfo);

        ~GraphicsPipeline2D();

        void bindTexture(const std::shared_ptr<VulkanTexture>& texture);

        void draw(CommandBuffer &commandBuffer);

        void setTexture(const VkDescriptorImageInfo *info);

    private:
        void setupPipeline();

        void setupDescriptors(const std::shared_ptr<VulkanTexture>& ptr);


        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        std::shared_ptr<VulkanTexture> m_defaultTexture;
        std::shared_ptr<VulkanImage> m_defaultImage;

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
        struct DefaultRenderData {
            std::unique_ptr<Buffer> fragShaderParamsBuffer; // GPU Accessible, triple-buffered
            std::unique_ptr<Buffer> mvpBuffer; // GPU Accessible, triple-buffered
            VkDescriptorSet descriptorSet; // Triple-buffered
        };

        std::vector<DefaultRenderData> m_renderData;

        VkDescriptorPool m_descriptorPool{};
        VkDescriptorSetLayout m_descriptorSetLayout{};
        std::unique_ptr<VulkanGraphicsPipeline> m_graphicsPipeline = nullptr;

    };
};


#endif //MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H
