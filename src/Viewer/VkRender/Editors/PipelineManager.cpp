//
// Created by mgjer on 04/10/2024.
//
#include "Viewer/VkRender/Editors/PipelineKey.h"

#include "Viewer/VkRender/Editors/PipelineManager.h"


namespace VkRender {
    std::shared_ptr<DefaultGraphicsPipeline> PipelineManager::getOrCreatePipeline(const PipelineKey &key, const VkRenderPass& renderPass) {
        auto it = m_pipelineCache.find(key);
        if (it != m_pipelineCache.end()) {
            return it->second;
        }

        // Create a new pipeline
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &key.descriptorSetLayout;
        // Add push constant ranges if necessary

        VkPipelineLayout pipelineLayout;
        if (vkCreatePipelineLayout(nullptr, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        renderPassInfo.renderPass = renderPass;

        // Create the graphics pipeline using the pipeline layout
        auto pipeline = std::make_shared<DefaultGraphicsPipeline>(*m_context, renderPassInfo);
        m_pipelineCache[key] = pipeline;
        return pipeline;
    }
}
