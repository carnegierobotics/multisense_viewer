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

namespace VkRender {
    class Renderer;

    class GraphicsPipeline2D {
    public:
        GraphicsPipeline2D(Renderer &context, const RenderPassInfo &renderPassInfo);

        ~GraphicsPipeline2D();

        void draw(CommandBuffer &cmdBuffers);

        void setTexture(const VkDescriptorImageInfo *info);

        TextureVideo& getVideoTexture(){return m_textureVideo;}


        void update(uint32_t currentFrame);

        void updateTransform(const TransformComponent &transform);

        void updateView(const Camera &camera);

        template<typename T>
        void bind(T &modelComponent);

    private:
        void setupPipeline();

        void setupDescriptors();

        void setupBuffers();

        void cleanUp();

        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        Texture2D m_emptyTexture;
        Texture2D m_objTexture;
        TextureVideo m_textureVideo;

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

        Indices indices{};
        Vertices vertices{};

        std::string m_vertexShader;
        std::string m_fragmentShader;

        UBOMatrix m_vertexParams; // Non GPU-accessible data, shared across frames
        FragShaderParams m_fragParams; // Non GPU-accessible data, shared across frames
        std::vector<DefaultRenderData> m_renderData;
        SharedRenderData m_sharedRenderData;

        void setupUniformBuffers();
    };
};


#endif //MULTISENSE_VIEWER_GRAPHICSPIPELINE2D_H
