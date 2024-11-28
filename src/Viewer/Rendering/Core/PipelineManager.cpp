//
// Created by mgjer on 04/10/2024.
//
#include "PipelineKey.h"

#include "PipelineManager.h"


namespace VkRender {
    std::shared_ptr<DefaultGraphicsPipeline> PipelineManager::getOrCreatePipeline(const PipelineKey &key, const RenderPassInfo& renderPassInfo, Application* context) {
        auto it = m_pipelineCache.find(key);
        if (it != m_pipelineCache.end()) {
            return it->second;
        }
        // Create the graphics pipeline using the pipeline layout
        auto pipeline = std::make_shared<DefaultGraphicsPipeline>(*context, renderPassInfo, key);
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