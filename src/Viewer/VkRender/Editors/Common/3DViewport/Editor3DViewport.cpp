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
        // Generate pipelines
        auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            auto &model = view.get<MeshComponent>(entity);
            const UUID &meshUUID = model.getUUID();
            if (!m_renderPipelines.contains(meshUUID)) {
                RenderPassInfo renderPassInfo{};
                renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
                renderPassInfo.renderPass = m_renderPass->getRenderPass();
                // Decide which pipeline to use
                if (model.usesUBOMesh()) {
                    m_renderPipelines[meshUUID] = std::make_unique<UboGraphicsPipeline>(*m_context, renderPassInfo);
                } else {
                    m_renderPipelines[meshUUID] = std::make_unique<DefaultGraphicsPipeline>(*m_context, renderPassInfo);
                }
                m_renderPipelines[meshUUID]->bind(model);
                cleanUpUnusedPipelines();
            }

            if (!m_depthOnlyRenderPipelines.contains(meshUUID)) {
                RenderPassInfo renderPassInfo{};
                renderPassInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT; // TODO we could infer this from render pass creation
                renderPassInfo.renderPass = m_depthRenderPass->getRenderPass();
                // Decide which pipeline to use
                if (model.usesUBOMesh()) {
                    m_depthOnlyRenderPipelines[meshUUID] = std::make_unique<UboGraphicsPipeline>(*m_context,
                                                                                                 renderPassInfo);
                } else {
                    m_depthOnlyRenderPipelines[meshUUID] = std::make_unique<DefaultGraphicsPipeline>(*m_context,
                                                                                                     renderPassInfo);
                }
                m_depthOnlyRenderPipelines[meshUUID]->bind(model);
                cleanUpUnusedPipelines();
            }
        }
    }

    void Editor3DViewport::cleanUpUnusedPipelines() {
        // Delete pipelines if the UUID no longer exists
        // CLEAN UP DEFAULT PIPELINES
        {
            // Step 1: Collect all active UUIDs
            std::unordered_set<UUID> activeUUIDs;
            auto view = m_activeScene->getRegistry().view<MeshComponent>();
            for (auto entity: view) {
                const UUID &meshUUID = m_activeScene->getRegistry().get<MeshComponent>(entity).getUUID();
                activeUUIDs.insert(meshUUID);
            }
            // Step 2: Iterate over existing pipelines and find unused UUIDs
            std::vector<UUID> unusedUUIDs;
            for (const auto &[uuid, pipeline]: m_renderPipelines) {
                if (activeUUIDs.find(uuid) == activeUUIDs.end()) {
                    // UUID not found in active UUIDs, mark it for deletion
                    unusedUUIDs.push_back(uuid);
                }
            }
            // Step 3: Remove the unused pipelines
            for (const UUID &uuid: unusedUUIDs) {
                m_renderPipelines.erase(uuid);
            }
        }
        // CLEAN UP DEPTH PIPELINES
        {  // Step 1: Collect all active UUIDs
            std::unordered_set<UUID> activeUUIDs;
            auto view = m_activeScene->getRegistry().view<MeshComponent>();
            for (auto entity: view) {
                const UUID &meshUUID = m_activeScene->getRegistry().get<MeshComponent>(entity).getUUID();
                activeUUIDs.insert(meshUUID);
            }
            // Step 2: Iterate over existing pipelines and find unused UUIDs
            std::vector<UUID> unusedUUIDs;
            for (const auto &[uuid, pipeline]: m_depthOnlyRenderPipelines) {
                if (activeUUIDs.find(uuid) == activeUUIDs.end()) {
                    // UUID not found in active UUIDs, mark it for deletion
                    unusedUUIDs.push_back(uuid);
                }
            }
            // Step 3: Remove the unused pipelines
            for (const UUID &uuid: unusedUUIDs) {
                m_depthOnlyRenderPipelines.erase(uuid);
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
                auto e = Entity(entity, m_activeScene.get());
                if (ui().setActiveCamera && e == m_createInfo.sharedUIContextData->m_selectedEntity) {
                    auto &registry = m_activeScene->getRegistry();
                    auto &cameraComponent = registry.get<CameraComponent>(entity);
                    auto &transform = registry.get<TransformComponent>(entity);
                    cameraComponent().pose.pos = transform.getPosition();
                    cameraComponent().pose.q = glm::quat_cast(transform.getRotMat());
                    cameraComponent().updateViewMatrix();
                    //transform.getPosition() = cameraComponent().pose.pos;
                    //transform.getQuaternion() = cameraComponent().pose.q;
                    m_activeCamera = cameraComponent();
                    m_ui.activeCamera = &m_activeCamera.get();
                }
            }
        }
        ui().saveRenderToFile = m_createInfo.sharedUIContextData->newFrame;
        if (m_createInfo.sharedUIContextData->m_selectedEntity){
            ui().renderToFileName = "scene_0000/viewport/" +  m_createInfo.sharedUIContextData->m_selectedEntity.getComponent<TagComponent>().Tag;
            ui().renderToFileName.replace_extension(".png");
        }
        generatePipelines();
        // Update model transforms:
        auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            auto &model = view.get<MeshComponent>(entity);
            const UUID &meshUUID = model.getUUID();
            if (m_renderPipelines.contains(meshUUID)) {
                auto transform = view.get<TransformComponent>(entity);
                m_renderPipelines[meshUUID]->updateTransform(transform);
                m_renderPipelines[meshUUID]->updateView(m_activeCamera);
                m_renderPipelines[meshUUID]->update(m_context->currentFrameIndex());
            }
            if (m_depthOnlyRenderPipelines.contains(meshUUID)) {
                Entity e(entity, m_activeScene.get());
                if (e.hasComponent<CameraComponent>())
                    continue;
                auto transform = view.get<TransformComponent>(entity);
                m_depthOnlyRenderPipelines[meshUUID]->updateTransform(transform);
                m_depthOnlyRenderPipelines[meshUUID]->updateView(m_activeCamera);
                m_depthOnlyRenderPipelines[meshUUID]->update(m_context->currentFrameIndex());
            }
        }
        m_activeScene->update(m_context->currentFrameIndex());
    }

    void Editor3DViewport::onRender(CommandBuffer &drawCmdBuffers) {
        for (auto &pipeline: m_renderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
        }
    }

    void Editor3DViewport::onRenderDepthOnly(CommandBuffer &drawCmdBuffers) {
        for (auto &pipeline: m_depthOnlyRenderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
        }
    }

    void Editor3DViewport::onEntityDestroyed(entt::entity entity) {
        Entity e(entity, m_activeScene.get());
        if (e.hasComponent<MeshComponent>()) {
            if (m_renderPipelines.contains(e.getComponent<MeshComponent>().getUUID()))
                m_renderPipelines.erase(e.getComponent<MeshComponent>().getUUID());
        }
    }

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui().hovered && mouse.left && !ui().resizeActive) {
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
