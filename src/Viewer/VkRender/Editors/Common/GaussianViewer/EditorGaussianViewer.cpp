//
// Created by magnus on 8/15/24.
//

#include "EditorGaussianViewer.h"

#include "Viewer/VkRender/Components/GaussianModelComponent.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"

namespace VkRender {
    void VkRender::EditorGaussianViewer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();

        m_activeScene->addDestroyFunction(this, [this](entt::entity entity) {
            onEntityDestroyed(entity);
        });

        generatePipelines();
    }

    void EditorGaussianViewer::generatePipelines() {
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.swapchainImageCount = m_context->swapChainBuffers().size();
        // Generate pipelines
        auto view = m_activeScene->getRegistry().view<GaussianModelComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            if (!m_gaussianRenderPipelines[entity]) { // Check if the pipeline already exists
                auto &model = view.get<GaussianModelComponent>(entity);
                m_gaussianRenderPipelines[entity] = std::make_unique<GaussianModelGraphicsPipeline>(
                        m_context->vkDevice(), renderPassInfo, 1280, 720);
                m_gaussianRenderPipelines[entity]->bind(model);
                m_gaussianRenderPipelines[entity]->bind(model.getMeshComponent());
            }
        }
    }


    void EditorGaussianViewer::onEntityDestroyed(entt::entity entity) {
        m_gaussianRenderPipelines.erase(entity);
    }

    void VkRender::EditorGaussianViewer::onUpdate() {
        {
            auto view = m_activeScene->getRegistry().view<CameraComponent>();
            for (auto entity: view) {
                auto e = Entity(entity, m_activeScene.get());
                if (m_createInfo.sharedUIContextData->setActiveCamera.contains(e.getUUID())) {
                    // Assuming you have an entt::registry instance
                    auto &registry = m_activeScene->getRegistry();
                    auto &cameraComponent = registry.get<CameraComponent>(entity);
                    m_activeCamera = std::make_shared<Camera>(cameraComponent());
                }
            }
        }
        if (!m_activeCamera)
            return;

        generatePipelines();
        if (ui().render3DGSImage) {
            for (auto &pipeline: m_gaussianRenderPipelines) {
                pipeline.second->generateImage(*m_activeCamera, ui().render3dgsColor);
            }
        }
        auto view = m_activeScene->getRegistry().view<TransformComponent, GaussianModelComponent>(); // TODO make one specific component type for renderables in standard pipelines
        for (auto entity: view) {
            if (m_gaussianRenderPipelines.contains(entity)) {
                auto transform = view.get<TransformComponent>(entity);
                m_gaussianRenderPipelines[entity]->updateTransform(transform);
                m_gaussianRenderPipelines[entity]->updateView(*m_activeCamera);
                m_gaussianRenderPipelines[entity]->update(m_context->currentFrameIndex());
            }
        }
    }

    void VkRender::EditorGaussianViewer::onRender(CommandBuffer &drawCmdBuffers) {

        for (auto &pipeline: m_gaussianRenderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
        }

    }

    void VkRender::EditorGaussianViewer::onMouseMove(const VkRender::MouseButtons &mouse) {
        if (ui().hovered && mouse.left) {
            //m_activeCamera->rotate(mouse.dx, mouse.dy);
        }
        if (ui().hovered && mouse.right) {
            //m_activeCamera->translate(mouse.dx, mouse.dy);
        }
    }

    void VkRender::EditorGaussianViewer::onMouseScroll(float change) {
    }

    void EditorGaussianViewer::onKeyCallback(const Input &input) {
        m_activeCamera->keys.up = input.keys.up;
        m_activeCamera->keys.down = input.keys.down;
        m_activeCamera->keys.left = input.keys.left;
        m_activeCamera->keys.right = input.keys.right;

    }

}