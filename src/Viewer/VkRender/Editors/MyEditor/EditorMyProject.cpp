//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/MyEditor/EditorMyProject.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/Components.h"

namespace VkRender {

    void EditorMyProject::onRender(CommandBuffer& drawCmdBuffers) {


    }

    void EditorMyProject::onUpdate() {
    }

    EditorMyProject::EditorMyProject(VulkanRenderPassCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");

    }

    void EditorMyProject::onSceneLoad() {

        /*
        DefaultGraphicsPipelineComponent::RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        auto entity = m_context->findEntityByName("FirstEntity");
        if (entity) {
            auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);
            res.bind(entity.getComponent<VkRender::OBJModelComponent>());
        }
        */
    }


}