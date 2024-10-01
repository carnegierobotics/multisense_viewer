//
// Created by mgjer on 30/09/2024.
//


#include "Viewer/VkRender/Editors/Common/SceneRenderer/SceneRenderer.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/RenderPipelines/UboGraphicsPipeline.h"


namespace VkRender {
    SceneRenderer::SceneRenderer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
    }


    void SceneRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_activeScene = m_context->activeScene();
        // Pass the destroy function to the scene
        m_activeScene->addDestroyFunction(this, [this](entt::entity entity) {
            onEntityDestroyed(entity);
        });
        generatePipelines();
    }

    void SceneRenderer::generatePipelines() {
        // Generate pipelines
        auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>();
        // TODO make one specific component type for renderables in standard pipelines
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
                renderPassInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
                // TODO we could infer this from render pass creation
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

    void SceneRenderer::cleanUpUnusedPipelines() {
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

    void SceneRenderer::onUpdate() {
        if (!m_activeScene)
            return;
        m_renderToOffscreen = false;
        for (auto &[uuid, entity]: m_createInfo.sharedUIContextData->selectedEntityMap) {
            std::string entityTag = entity.getComponent<TagComponent>().Tag;
            if (uuid == getUUID()) {
                auto &cameraComponent = entity.getComponent<CameraComponent>();
                if (cameraComponent.renderFromViewpoint()) {
                    auto &transform = entity.getComponent<TransformComponent>();
                    cameraComponent.camera.setPerspective(
                        static_cast<float>(m_createInfo.width) / m_createInfo.height);
                    cameraComponent().pose.pos = transform.getPosition();
                    cameraComponent().pose.q = glm::quat_cast(transform.getRotMat());
                    cameraComponent().updateViewMatrix();
                    transform.getPosition() = cameraComponent().pose.pos;
                    transform.getQuaternion() = cameraComponent().pose.q;
                    m_activeCamera = std::make_shared<Camera>(cameraComponent.camera);
                    m_renderToOffscreen = true;
                }
            }
        }


        generatePipelines();
        // Update model transforms:
        if (m_activeCamera) {
            auto view = m_activeScene->getRegistry().view<TransformComponent, MeshComponent>();
            // TODO make one specific component type for renderables in standard pipelines
            for (auto entity: view) {
                auto &model = view.get<MeshComponent>(entity);
                const UUID &meshUUID = model.getUUID();
                if (m_renderPipelines.contains(meshUUID)) {
                    auto transform = view.get<TransformComponent>(entity);
                    m_renderPipelines[meshUUID]->updateTransform(transform);
                    m_renderPipelines[meshUUID]->updateView(*m_activeCamera);
                    m_renderPipelines[meshUUID]->update(m_context->currentFrameIndex());
                }
                if (m_depthOnlyRenderPipelines.contains(meshUUID)) {
                    Entity e(entity, m_activeScene.get());
                    if (e.hasComponent<CameraComponent>())
                        continue;
                    auto transform = view.get<TransformComponent>(entity);
                    m_depthOnlyRenderPipelines[meshUUID]->updateTransform(transform);
                    m_depthOnlyRenderPipelines[meshUUID]->updateView(*m_activeCamera);
                    m_depthOnlyRenderPipelines[meshUUID]->update(m_context->currentFrameIndex());
                }
            }
            m_activeScene->update(m_context->currentFrameIndex());
        }
    }

    void SceneRenderer::onRender(CommandBuffer &drawCmdBuffers) {
        if (m_activeCamera) {
            for (auto &val: m_renderPipelines | std::views::values) {
                val->draw(drawCmdBuffers);
            }
        }
    }

    void SceneRenderer::onRenderDepthOnly(CommandBuffer &drawCmdBuffers) {
        if (m_activeCamera) {
            for (auto &pipeline: m_depthOnlyRenderPipelines) {
                pipeline.second->draw(drawCmdBuffers);
            }
        }
    }

    void SceneRenderer::onEntityDestroyed(entt::entity entity) {
        Entity e(entity, m_activeScene.get());
        if (e.hasComponent<MeshComponent>()) {
            if (m_renderPipelines.contains(e.getComponent<MeshComponent>().getUUID()))
                m_renderPipelines.erase(e.getComponent<MeshComponent>().getUUID());
        }
    }
}
