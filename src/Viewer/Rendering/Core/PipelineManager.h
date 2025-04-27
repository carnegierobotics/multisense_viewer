//
// Created by mgjer on 04/10/2024.
//

#ifndef PIPELINEMANAGER_H
#define PIPELINEMANAGER_H

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Core/PipelineKey.h"
#include "Viewer/Rendering/Core/VulkanGraphicsPipeline.h"

namespace VkRender {


    class PipelineManager {
    public:
        PipelineManager() = default;
        std::shared_ptr<VulkanGraphicsPipeline> getOrCreatePipeline(const PipelineKey &key, const PipelineInfo& pipelineInfo, const RenderPassInfo &renderPassInfo, Application *context);

        // Function to remove a pipeline by key
        void removePipeline(const PipelineKey &key);

    private:
        std::unordered_map<PipelineKey, std::shared_ptr<VulkanGraphicsPipeline>, PipelineKeyHash> m_pipelineCache;
    };
}

#endif //PIPELINEMANAGER_H
