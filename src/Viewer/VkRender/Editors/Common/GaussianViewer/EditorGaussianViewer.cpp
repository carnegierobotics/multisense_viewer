//
// Created by magnus on 8/15/24.
//

#include "EditorGaussianViewer.h"

#include "Viewer/VkRender/Components/GaussianModelComponent.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"

namespace VkRender {
    void VkRender::EditorGaussianViewer::onSceneLoad() {
        m_activeScene = m_context->activeScene();
        // We'll find the gaussian objects in the scene and render them

        auto cameraEntity = m_activeScene->findEntityByName("DefaultCamera");
        if (cameraEntity) {
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            auto &camera = cameraEntity.getComponent<CameraComponent>()();
            camera.pose.pos = transform.getPosition();
            camera.updateViewMatrix();
            m_activeCamera = &camera;
        }

        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        auto cameraModelView = m_activeScene->getRegistry().view<GaussianModelComponent, OBJModelComponent>();
        // Iterate over the entities in the view
        for (entt::entity entity: cameraModelView) {
            m_gaussianRenderPipelines[entity] = std::make_unique<GaussianModelGraphicsPipeline>(m_context->vkDevice());
            m_2DRenderPipeline[entity] = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);

            m_2DRenderPipeline[entity]->bind(cameraModelView.get<OBJModelComponent>(entity));

            auto &model = cameraModelView.get<GaussianModelComponent>(entity);
            m_gaussianRenderPipelines[entity]->bind(model, *m_activeCamera);


            m_2DRenderPipeline[entity]->setTexture(
                    &m_gaussianRenderPipelines[entity]->getTextureRenderTarget()->m_descriptor);
        }


    }

    void VkRender::EditorGaussianViewer::onUpdate() {

        auto cameraModelView = m_activeScene->getRegistry().view<GaussianModelComponent, OBJModelComponent>();
        for (auto entity: cameraModelView) {
            if (!m_gaussianRenderPipelines[entity]) { // Check if the pipeline already exists
                RenderPassInfo renderPassInfo{};
                renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
                renderPassInfo.renderPass = m_renderPass->getRenderPass();
                m_gaussianRenderPipelines[entity] = std::make_unique<GaussianModelGraphicsPipeline>(m_context->vkDevice());
                m_2DRenderPipeline[entity] = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);

                m_2DRenderPipeline[entity]->bind(cameraModelView.get<OBJModelComponent>(entity));

                auto &model = cameraModelView.get<GaussianModelComponent>(entity);
                m_gaussianRenderPipelines[entity]->bind(model, *m_activeCamera);
                m_2DRenderPipeline[entity]->setTexture(&m_gaussianRenderPipelines[entity]->getTextureRenderTarget()->m_descriptor);
            }
        }

        uint8_t *image = nullptr;
        uint32_t imageSize = 0;
        if (ui().render3DGSImage) {
            for (auto &pipeline: m_gaussianRenderPipelines) {
                pipeline.second->generateImage(*m_activeCamera);

            }
        }


        for (auto &pipeline: m_2DRenderPipeline) {

            pipeline.second->updateView(*m_activeCamera);
            pipeline.second->update(m_context->currentFrameIndex());
        }


        auto cameraEntity = m_activeScene->findEntityByName("DefaultCamera");
        if (cameraEntity) {
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setQuaternion(m_activeCamera->pose.q);
            transform.setPosition(m_activeCamera->pose.pos);
        }

        if (ui().hovered)
            m_activeCamera->update(m_context->deltaTime());

    }

    void VkRender::EditorGaussianViewer::onRender(CommandBuffer &drawCmdBuffers) {

        for (auto &pipeline: m_gaussianRenderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
        }

        for (auto &pipeline: m_2DRenderPipeline) {
            pipeline.second->draw(drawCmdBuffers);
        }
    }

    void VkRender::EditorGaussianViewer::onMouseMove(const VkRender::MouseButtons &mouse) {

        if (ui().hovered && mouse.left) {
            m_activeCamera->rotate(mouse.dx, mouse.dy);
        }

        if (ui().hovered && mouse.right){
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