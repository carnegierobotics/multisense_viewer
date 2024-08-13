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

        auto renderEntity = m_activeScene->findEntityByName("RenderEntity"+ std::to_string(m_createInfo.editorIndex)); // Replace with renderables in the scene
        if (renderEntity) {
            if (renderEntity.hasComponent<DefaultGraphicsPipelineComponent>()){
                auto &resources = renderEntity.getComponent<DefaultGraphicsPipelineComponent>();
                resources.draw(drawCmdBuffers);
            }
        }

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
        auto renderEntity =m_activeScene->createEntity("RenderEntity"+ std::to_string(m_createInfo.editorIndex));
        auto& pipelineEntity = renderEntity.addComponent<DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);

        auto entity = m_activeScene->findEntityByName("FirstEntity"); // Replace with renderables in the scene
        // Then for each renderables create a graphics pipeline that fits
        if (entity &&  entity.hasComponent<VkRender::OBJModelComponent>()) {
            // Assuming the pipeline is already set up for the render pass
            //auto& model = ;
            pipelineEntity.bind(entity.getComponent<VkRender::OBJModelComponent>());
        }
    }
}
