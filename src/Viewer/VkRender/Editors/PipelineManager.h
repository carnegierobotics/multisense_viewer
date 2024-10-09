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
        MeshComponent* mesh;
        MaterialComponent* material;
        TransformComponent* transform;
        // Other necessary data
    };

    class PipelineManager {
    public:

        PipelineManager() = default;

        std::shared_ptr<DefaultGraphicsPipeline> getOrCreatePipeline(const PipelineKey &key, const VkRenderPass &renderPass);

    private:
        std::unordered_map<PipelineKey, std::shared_ptr<DefaultGraphicsPipeline>> m_pipelineCache;
        Application* m_context{};
    };
}

#endif //PIPELINEMANAGER_H
