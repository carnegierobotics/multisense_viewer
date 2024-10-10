//
// Created by mgjer on 04/10/2024.
//

#ifndef PIPELINEMANAGER_H
#define PIPELINEMANAGER_H

#include <Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h>

#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/RenderPipelines/GraphicsPipeline.h"
#include "Viewer/VkRender/Components/MaterialComponent.h"
#include "Viewer/VkRender/Editors/PipelineKey.h"

namespace VkRender {

    struct RenderCommand {
        Entity entity;
        std::shared_ptr<DefaultGraphicsPipeline> pipeline;
        MeshInstance* mesh;             // GPU-specific mesh data
        MaterialInstance* materialInstance;  // GPU-specific material data
        TransformComponent* transform; // ECS component with world transform
    };

    class PipelineManager {
    public:
        PipelineManager() = default;
        std::shared_ptr<DefaultGraphicsPipeline> getOrCreatePipeline(const PipelineKey &key, const RenderPassInfo &renderPassInfo, Application *context);

    private:
        std::unordered_map<PipelineKey, std::shared_ptr<DefaultGraphicsPipeline>> m_pipelineCache;
    };
}

#endif //PIPELINEMANAGER_H
