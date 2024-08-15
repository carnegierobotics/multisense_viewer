//
// Created by magnus on 8/15/24.
//

#include "EditorGaussianViewer.h"

#include "Viewer/VkRender/Components/GaussianModelComponent.h"
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

        auto cameraModelView = m_activeScene->getRegistry().view<GaussianModelComponent>();
        // Iterate over the entities in the view
        for (entt::entity entity: cameraModelView) {
            m_gaussianRenderPipelines[entity] = std::make_unique<GaussianModelGraphicsPipeline>();
            auto &model = cameraModelView.get<GaussianModelComponent>(entity);
            m_gaussianRenderPipelines[entity]->bind(model, *m_activeCamera);
        }

    }

    void VkRender::EditorGaussianViewer::onUpdate() {
        if (ui().render3DGSImage){
            for (auto &pipeline: m_gaussianRenderPipelines) {
                pipeline.second->generateImage(*m_activeCamera);
            }
        }

    }

    void VkRender::EditorGaussianViewer::onRender(CommandBuffer &drawCmdBuffers) {

        for (auto &pipeline: m_gaussianRenderPipelines) {
            pipeline.second->draw(drawCmdBuffers);
        }
    }

    void VkRender::EditorGaussianViewer::onMouseMove(const VkRender::MouseButtons &mouse) {
    }

    void VkRender::EditorGaussianViewer::onMouseScroll(float change) {
    }

}