//
// Created by mgjer on 14/08/2024.
//

#ifndef MULTISENSE_VIEWER_CAMERAMODELGRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_CAMERAMODELGRAPHICSPIPELINE_H

#include "Viewer/VkRender/Components/RenderBase.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Components.h"

namespace VkRender {
    class Renderer;

    class CameraModelGraphicsPipeline {
    public:


        CameraModelGraphicsPipeline(Renderer &m_context, const RenderPassInfo &renderPassInfo);
        ~CameraModelGraphicsPipeline();

        template<typename T>
        void bind(T &modelComponent);

        void draw(CommandBuffer &cmdBuffers);

        void update(uint32_t currentFrame);

        void updateTransform(const VkRender::TransformComponent &transform);

        void updateView(const Camera &camera);
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

        Indices indices{};
        Vertices vertices{};

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
#endif //MULTISENSE_VIEWER_CAMERAMODELGRAPHICSPIPELINE_H
