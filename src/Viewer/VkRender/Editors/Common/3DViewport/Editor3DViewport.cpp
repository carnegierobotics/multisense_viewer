//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/RenderPipelines/UboGraphicsPipeline.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {

    void Editor3DViewport::onSceneLoad() {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_editorCamera = Camera(1280, 720);
        m_activeCamera = m_editorCamera;

        m_activeScene = m_context->activeScene();
        // Pass the destroy function to the scene
        m_activeScene->addDestroyFunction(this, [this](entt::entity entity) {
            onEntityDestroyed(entity);
        });


        generatePipelines();
        /*
        auto cameraModelView = m_activeScene->getRegistry().view<CameraModelComponent>();
        // Iterate over the entities in the view
        for (entt::entity entity: cameraModelView) {
            m_cameraRenderPipelines[entity] = std::make_unique<CameraModelGraphicsPipeline>(*m_context, renderPassInfo);
            auto &model = cameraModelView.get<CameraModelComponent>(entity);
            m_cameraRenderPipelines[entity]->bind(model);
        }
        */
    }


    void Editor3DViewport::generatePipelines() {
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        // Generate pipelines
        auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            if (!m_renderPipelines[entity]) { // Check if the pipeline already exists
                auto &model = view.get<MeshComponent>(entity);
                // Decide which pipeline to use
                if (model.usesUBOMesh()) {
                    m_renderPipelines[entity] = std::make_unique<UboGraphicsPipeline>(*m_context, renderPassInfo);
                } else {
                    m_renderPipelines[entity] = std::make_unique<DefaultGraphicsPipeline>(*m_context, renderPassInfo);
                }
                m_renderPipelines[entity]->bind(model);
            }
        }
    }

    void Editor3DViewport::onUpdate() {
        if (!m_activeScene)
            return;

        {
            auto view = m_activeScene->getRegistry().view<CameraComponent>(); // TODO make one specific component type for renderables in standard pipelines
                m_activeCamera = m_editorCamera;
                for (auto entity: view) {
                    if (m_createInfo.sharedUIContextData->setActiveCamera.contains(static_cast<uint32_t>(entity))){
                        // Assuming you have an entt::registry instance
                        if (!m_createInfo.sharedUIContextData->setActiveCamera[static_cast<uint32_t>(entity)] || !ui().setActiveCamera)
                            continue;

                        auto &registry = m_activeScene->getRegistry();
                        // Now you can check if the entity is valid and retrieve components
                        auto &cameraComponent = registry.get<CameraComponent>(entity);
                        // Set the active camera to the components transform
                        auto &transform = registry.get<TransformComponent>(entity);
                        cameraComponent().pose.pos = transform.getPosition();
                        cameraComponent().pose.q = transform.getQuaternion();
                        cameraComponent().updateViewMatrix();
                        //transform.getPosition() = cameraComponent().pose.pos;
                        //transform.getQuaternion() = cameraComponent().pose.q;
                        m_activeCamera = cameraComponent();
                    }
                }

        }

        generatePipelines();
        // Update model transforms:
        auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            if (m_renderPipelines.contains(entity)) {
                auto transform = view.get<TransformComponent>(entity);
                m_renderPipelines[entity]->updateTransform(transform);
                m_renderPipelines[entity]->updateView(m_activeCamera);
                m_renderPipelines[entity]->update(m_context->currentFrameIndex());
            }
        }
        /*
        auto view2 = m_activeScene->getRegistry().view<TransformComponent, CameraModelComponent>();
        for (auto entity: view2) {
            auto &transform = view2.get<TransformComponent>(entity);
            m_cameraRenderPipelines[entity]->updateTransform(transform);
        }
         */
        /*
        // Update all pipelines with the current view and frame index
        for (auto &pipeline: m_renderPipelines) {
            if (ui().setActiveCamera) {
                auto entity = m_activeScene->findEntityByName("DefaultCamera");
                auto& camera = entity.getComponent<CameraComponent>();
                pipeline.second->updateView(camera());
            } else
                pipeline.second->updateView(m_editorCamera);

            pipeline.second->update(m_context->currentFrameIndex());
        }
        // Update all pipelines with the current view and frame index
        for (auto &pipeline: m_cameraRenderPipelines) {
            if (ui().setActiveCamera) {
                auto entity = m_activeScene->findEntityByName("DefaultCamera");
                auto& camera = entity.getComponent<CameraComponent>();
                pipeline.second->updateView(camera());
            } else
                pipeline.second->updateView(m_editorCamera);
            pipeline.second->update(m_context->currentFrameIndex());
        }

         */
        m_activeScene->update(m_context->currentFrameIndex());
    }

    void Editor3DViewport::onRender(CommandBuffer &drawCmdBuffers) {
        if (!m_activeScene)
            return;

        for (auto &pipeline: m_renderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
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

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui().hovered && mouse.left) {
            m_editorCamera.rotate(mouse.dx, mouse.dy);
        }
        m_activeScene->onMouseEvent(mouse);
    }

    void Editor3DViewport::onMouseScroll(float change) {

        if (ui().hovered)
            m_editorCamera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);


        m_activeScene->onMouseScroll(change);

    }
}
