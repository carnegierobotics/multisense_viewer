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

    void EditorViewport::onRender(CommandBuffer &drawCmdBuffers) {
        if (!m_activeScene)
            return;

        if (m_activeScene)
            m_activeScene->render(drawCmdBuffers);


        for (auto& pipeline : renderPipelines){
            pipeline->draw(drawCmdBuffers);
        }

    }

    void EditorViewport::onUpdate() {
        if (!m_activeScene)
            return;

        m_activeScene->update(m_context->currentFrameIndex());

        auto view = m_activeScene->getRegistry().view<VkRender::TransformComponent, OBJModelComponent>();
        // Iterate over the entities in the view
        for (size_t i = 0; auto entity : view) {
            auto& object = view.get<VkRender::TransformComponent>(entity);
            renderPipelines[i]->updateTransform(object);
            i++;
        }

        // active camera
        auto cameraView = m_activeScene->getRegistry().view<VkRender::CameraComponent>();
        // Iterate over the entities in the view
        Camera* camera;
        for (auto entity : cameraView) {
            auto& c = cameraView.get<VkRender::CameraComponent>(entity);
            camera = &c();
        }
        for (auto& pipeline : renderPipelines){
            pipeline->updateView(*camera);
            pipeline->update(m_context->currentFrameIndex());
        }
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

        auto view = m_activeScene->getRegistry().view<VkRender::OBJModelComponent>();
        // Iterate over the entities in the view
        renderPipelines.resize(view.size_hint());
        for (size_t i = 0; auto entity : view) {
            renderPipelines[i] = std::make_unique<DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);
            auto& model = view.get<VkRender::OBJModelComponent>(entity);
            renderPipelines[i]->bind(model);
            i++;
        }

    }
}
