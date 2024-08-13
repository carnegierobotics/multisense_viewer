//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/Viewport/EditorViewport.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"

namespace VkRender {

    void EditorViewport::onRender(CommandBuffer& drawCmdBuffers) {
        if (m_activeScene)
            m_activeScene->render(drawCmdBuffers);

    }

    void EditorViewport::onUpdate() {
        if (m_activeScene)
            m_activeScene->update(m_context->currentFrameIndex());

    }

    EditorViewport::EditorViewport(EditorCreateInfo &createInfo) : Editor(createInfo) {

        addUI("EditorUILayer");
        addUI("DebugWindow");

        // Grid and objects

    }

    void EditorViewport::onSceneLoad() {

        DefaultGraphicsPipelineComponent::RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        m_activeScene = m_context->activeScene();

        auto entity = m_activeScene->findEntityByName("FirstEntity");
        if (entity && !entity.hasComponent<DefaultGraphicsPipelineComponent>()) {
            auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);
            res.bind(entity.getComponent<VkRender::OBJModelComponent>());
        }

    }
}
