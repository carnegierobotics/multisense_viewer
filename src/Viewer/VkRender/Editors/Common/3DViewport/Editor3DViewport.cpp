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

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo) : Editor(createInfo) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        // Grid and objects
    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_editorCamera = Camera(m_createInfo.width, m_createInfo.height);
        ui().editorCamera = &m_editorCamera;
        m_activeCamera = m_editorCamera;
        m_activeScene = m_context->activeScene();
        // Pass the destroy function to the scene
        m_activeScene->addDestroyFunction(this, [this](entt::entity entity) {
            onEntityDestroyed(entity);
        });
        generatePipelines();
    }

    void Editor3DViewport::generatePipelines() {
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        // Generate pipelines
        auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            if (!m_renderPipelines.contains(entity)) { // Check if the pipeline already exists
                auto &model = view.get<MeshComponent>(entity);
                if (!model.hasMesh())
                    continue;
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
            auto view = m_activeScene->getRegistry().view<CameraComponent>();
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
                        cameraComponent().pose.q = glm::quat_cast(transform.getRotMat());
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
                if (!view.get<MeshComponent>(entity).hasMesh())
                    continue;
                auto transform = view.get<TransformComponent>(entity);
                m_renderPipelines[entity]->updateTransform(transform);
                m_renderPipelines[entity]->updateView(m_activeCamera);
                m_renderPipelines[entity]->update(m_context->currentFrameIndex());
            }
        }
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
