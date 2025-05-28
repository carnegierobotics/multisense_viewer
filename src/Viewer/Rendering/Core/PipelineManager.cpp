//
// Created by mgjer on 04/10/2024.
//
#include "PipelineKey.h"

#include "PipelineManager.h"

#include <Viewer/Application/Application.h>


namespace VkRender {
    std::shared_ptr<VulkanGraphicsPipeline> PipelineManager::getOrCreatePipeline(const PipelineKey &key, const PipelineInfo& pipelineInfo, const RenderPassInfo& renderPassInfo, VkPipelineLayout globalPipelineLayout, Application* context) {
        auto it = m_pipelineCache.find(key);
        if (it != m_pipelineCache.end()) {
            return it->second;
        }

        // Vertex bindings an attributes
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        vertexInputStateCI.pVertexBindingDescriptions = pipelineInfo.bindings.data();
        vertexInputStateCI.pVertexAttributeDescriptions = pipelineInfo.attrs.data();
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(pipelineInfo.attrs.size());
        vertexInputStateCI.vertexBindingDescriptionCount = static_cast<uint32_t>(pipelineInfo.bindings.size());

        VulkanGraphicsPipelineCreateInfo createInfo(renderPassInfo.renderPass, context->vkDevice());
        createInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(
            key.polygonMode, VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE);
        createInfo.msaaSamples = renderPassInfo.sampleCount;

        std::vector<VkPipelineShaderStageCreateInfo> shadersStageInfo{};
        if (pipelineInfo.materialInstance) {
            auto shaders = pipelineInfo.materialInstance->shaders;
            for (const auto& shader : shaders) {
                shadersStageInfo.emplace_back(shader->stageInfo());
            }
        }

        createInfo.shaders = shadersStageInfo;

        for (auto& setLayout : pipelineInfo.setLayouts) {
            createInfo.descriptorSetLayouts.emplace_back(setLayout);
        }
        createInfo.vertexInputState = vertexInputStateCI;
        createInfo.debugInfo = renderPassInfo.debugName;
        createInfo.globalPipelineLayout = globalPipelineLayout;
        createInfo.materialInstance = pipelineInfo.materialInstance;

        auto pipeline = std::make_shared<VulkanGraphicsPipeline>(createInfo);
        // Create the graphics pipeline using the pipeline layout
        m_pipelineCache[key] = pipeline;
        return pipeline;
    }

    void PipelineManager::removePipeline(const PipelineKey &key)  {
        auto it = m_pipelineCache.find(key);
        if (it != m_pipelineCache.end()) {
            m_pipelineCache.erase(it);
        }
    }
}
