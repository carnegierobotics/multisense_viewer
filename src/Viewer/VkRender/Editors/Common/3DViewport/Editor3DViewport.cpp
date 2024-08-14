//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"

namespace VkRender {

    void Editor3DViewport::onRender(CommandBuffer &drawCmdBuffers) {
        if (!m_activeScene)
            return;


        for (auto& pipeline : m_renderPipelines){
            pipeline.second->draw(drawCmdBuffers);
        }

    }

    void Editor3DViewport::onUpdate() {
        if (!m_activeScene)
            return;

        auto view = m_activeScene->getRegistry().view<VkRender::TransformComponent, OBJModelComponent>();
        for (auto entity : view) {
            auto& transform = view.get<VkRender::TransformComponent>(entity);

            if (!m_renderPipelines[entity]) { // Check if the pipeline already exists
                DefaultGraphicsPipelineComponent::RenderPassInfo renderPassInfo{};
                renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
                renderPassInfo.renderPass = m_renderPass->getRenderPass();

                auto pipeline = std::make_unique<DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);
                auto& model = view.get<VkRender::OBJModelComponent>(entity);
                pipeline->bind(model);
                m_renderPipelines[entity] = std::move(pipeline);
            }

            m_renderPipelines[entity]->updateTransform(transform);
        }

        m_activeScene->update(m_context->currentFrameIndex());

        for (auto entity : view) {
            auto& object = view.get<VkRender::TransformComponent>(entity);
            m_renderPipelines[entity]->updateTransform(object);
        }

        // active camera
        auto cameraView = m_activeScene->getRegistry().view<VkRender::CameraComponent>();
        Camera* camera;
        for (auto entity : cameraView) {
            auto& c = cameraView.get<VkRender::CameraComponent>(entity);
        }

        // Update all pipelines with the current view and frame index
        for (auto& pipeline : m_renderPipelines){
                pipeline.second->updateView(m_editorCamera);
                pipeline.second->update(m_context->currentFrameIndex());
            }
    }

    void Editor3DViewport::onEntityDestroyed(entt::entity entity) {
        m_renderPipelines.erase(entity);
    }

    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo) : Editor(createInfo) {

        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");

        // Grid and objects

    }

    void Editor3DViewport::onSceneLoad() {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline

        m_editorCamera = Camera(1280, 720);
        DefaultGraphicsPipelineComponent::RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        m_activeScene = m_context->activeScene();
        // Pass the destroy function to the scene
        m_activeScene->addDestroyFunction(this, [this](entt::entity entity) {

            onEntityDestroyed(entity);
        });

        auto view = m_activeScene->getRegistry().view<VkRender::OBJModelComponent>();
        // Iterate over the entities in the view
        for (entt::entity entity : view) {
            m_renderPipelines[entity] = std::make_unique<DefaultGraphicsPipelineComponent>(*m_context, renderPassInfo);
            auto& model = view.get<VkRender::OBJModelComponent>(entity);
            m_renderPipelines[entity]->bind(model);
        }
    }

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui().hovered && mouse.left){
            m_editorCamera.rotate(mouse.dx, mouse.dy);
            m_activeScene->onMouseEvent(mouse);
        }
    }

    void Editor3DViewport::onMouseScroll(float change) {

        /*
        auto view = m_registry.view<CameraComponent>();
        for (auto entity : view) {
            auto &cameraComponent = view.get<CameraComponent>(entity);
        }
        */
        if (ui().hovered)
            m_editorCamera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);

        m_activeScene->onMouseScroll(change);

    }
}
