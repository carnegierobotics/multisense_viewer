//
// Created by magnus on 4/20/24.
//

#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Core/VulkanResourceManager.h"

namespace VkRender {
    DefaultGraphicsPipeline::DefaultGraphicsPipeline(Application& m_context, const RenderPassInfo& renderPassInfo,
                                                     const PipelineKey& key) : m_vulkanDevice(m_context.vkDevice()),
                                                                               m_renderPassInfo(
                                                                                   std::move(renderPassInfo)) {
        m_numSwapChainImages = m_context.swapChainBuffers().size();
        m_vulkanDevice = m_context.vkDevice();
        m_vertexShader = key.vertexShaderName;
        m_fragmentShader = key.fragmentShaderName;
        // Vertex bindings an attributes

        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        vertexInputStateCI.pVertexBindingDescriptions = key.vertexInputBindingDescriptions.data();
        vertexInputStateCI.pVertexAttributeDescriptions = key.vertexInputAttributes.data();
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(key.vertexInputAttributes.size());
        vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(key.vertexInputBindingDescriptions.size());


        std::vector<VkPipelineShaderStageCreateInfo> shaderStages(2);
        VkShaderModule vertModule{};
        VkShaderModule fragModule{};
        shaderStages[0] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_vertexShader.string(),
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_fragmentShader.string(),
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        VulkanGraphicsPipelineCreateInfo createInfo(m_renderPassInfo.renderPass, m_vulkanDevice);
        createInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(
            key.polygonMode, VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE);
        createInfo.msaaSamples = m_renderPassInfo.sampleCount;
        createInfo.shaders = shaderStages;
        createInfo.descriptorSetLayouts = key.setLayouts;
        createInfo.vertexInputState = vertexInputStateCI;

        m_graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(createInfo);

        for (auto shaderStage : shaderStages) {
            vkDestroyShaderModule(m_vulkanDevice.m_LogicalDevice, shaderStage.module, nullptr);
        }
    }


    void DefaultGraphicsPipeline::bind(CommandBuffer& commandBuffer) const {
        vkCmdBindPipeline(
            commandBuffer.getActiveBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline->getPipeline());
    }
};
