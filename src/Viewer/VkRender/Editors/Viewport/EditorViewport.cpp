//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Viewport/EditorViewport.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"

namespace VkRender {

    void EditorViewport::onRender(CommandBuffer& drawCmdBuffers) {

        m_context->scene()->render(drawCmdBuffers);

    }

    void EditorViewport::onUpdate() {
        m_context->scene()->update();
    }

    EditorViewport::EditorViewport(VulkanRenderPassCreateInfo &createInfo) : Editor(createInfo) {

        addUI("EditorUILayer");
        addUI("DebugWindow");

        // Grid and objects

    }

    void EditorViewport::onSceneLoad() {

        DefaultGraphicsPipelineComponent::RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        auto entity = m_context->findEntityByName("FirstEntity");
        if (entity) {
            auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);
            res.bind(entity.getComponent<VkRender::OBJModelComponent>());
        }
    }
}
