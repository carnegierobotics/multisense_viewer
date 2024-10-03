//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"

#include "Editor3DLayer.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/RenderPipelines/UboGraphicsPipeline.h"

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        // Grid and objects
        addUIData<Editor3DViewportUI>();
    }

    void Editor3DViewport::onEditorResize() {
        m_editorCamera.setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);
    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_editorCamera = Camera(m_createInfo.width, m_createInfo.height);
        m_activeCamera = m_editorCamera;
        m_activeScene = m_context->activeScene();
        // Pass the destroy function to the scene
        if (!m_activeScene)
            return;
        m_activeScene->addDestroyFunction(this, [this](entt::entity entity) {
            onEntityDestroyed(entity);
        });
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
                /*
                if (activeUUIDs.find(uuid) == activeUUIDs.end()) {
                    // UUID not found in active UUIDs, mark it for deletion
                    unusedUUIDs.push_back(uuid);
                }
                */
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
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        // Update model transforms:
        for (auto &pipeline: m_renderPipelines) {
            auto entity = Entity(static_cast<entt::entity>(pipeline.first), m_activeScene.get());
            auto &transform = entity.getComponent<TransformComponent>();
            pipeline.second->updateTransform(transform);
            pipeline.second->updateView(m_activeCamera);
            pipeline.second->update(m_context->currentFrameIndex());
        }
        m_activeScene->update(m_context->currentFrameIndex());
    }

    void Editor3DViewport::onComponentAdded(Entity entity, MeshComponent &meshComponent) {
        // add graphics pipeline for meshcomponent
        Log::Logger::getInstance()->info("Add meshcomponent for entity: {} in editor: {}",
                                         entity.getUUID().operator std::string(), getUUID().operator std::string());
        if (!m_renderPipelines.contains(entity)) {
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            // Decide which pipeline to use
            if (meshComponent.usesUBOMesh()) {
                m_renderPipelines[entity] = std::make_unique<UboGraphicsPipeline>(*m_context, renderPassInfo);
            } else {
                m_renderPipelines[entity] = std::make_unique<DefaultGraphicsPipeline>(*m_context, renderPassInfo);
            }
            m_renderPipelines[entity]->bind(meshComponent);
        }
    }

    void Editor3DViewport::onRender(CommandBuffer &drawCmdBuffers) {
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);

        for (auto &val: m_renderPipelines | std::views::values) {
            val->draw(drawCmdBuffers);
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
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera.rotate(mouse.dx, mouse.dy);
        }
        m_activeScene->onMouseEvent(mouse);
    }

    void Editor3DViewport::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        m_activeScene->onMouseScroll(change);
    }
};

