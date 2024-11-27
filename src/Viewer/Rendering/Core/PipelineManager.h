//
// Created by mgjer on 04/10/2024.
//

#ifndef PIPELINEMANAGER_H
#define PIPELINEMANAGER_H

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Core/PipelineKey.h"
#include "Viewer/Rendering/MeshInstance.h"
#include "Viewer/Rendering/RenderResources/DefaultGraphicsPipeline.h"

namespace VkRender {


    class PipelineManager {
    public:
        PipelineManager() = default;
        std::shared_ptr<DefaultGraphicsPipeline> getOrCreatePipeline(const PipelineKey &key, const RenderPassInfo &renderPassInfo, Application *context);

        // Function to remove a pipeline by key
        void removePipeline(const PipelineKey &key);

    private:
        std::unordered_map<PipelineKey, std::shared_ptr<DefaultGraphicsPipeline>> m_pipelineCache;
    };
}

#endif //PIPELINEMANAGER_H
