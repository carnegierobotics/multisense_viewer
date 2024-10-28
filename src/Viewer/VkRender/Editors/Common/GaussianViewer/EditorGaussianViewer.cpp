//
// Created by magnus on 8/15/24.
//

#include "Viewer/VkRender/Editors/Common/GaussianViewer/EditorGaussianViewer.h"

#include <Viewer/VkRender/Editors/Common/CommonEditorFunctions.h>

#include "Viewer/VkRender/Editors/Common/GaussianViewer/EditorGaussianViewerLayer.h"

#include "Viewer/Application/Application.h"

namespace VkRender {

    EditorGaussianViewer::EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid),
                                                                                          m_syclGaussianGfx(
                                                                                                  m_deviceSelector.getQueue()) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("EditorGaussianViewerLayer");

        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);

        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_UNORM, m_context);
        m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());

    }

    void VkRender::EditorGaussianViewer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);

        m_activeScene = m_context->activeScene();
        addUIData<EditorGaussianViewerUI>();
        m_syclGaussianGfx.setActiveCamera(m_editorCamera);

    }

    void EditorGaussianViewer::onEditorResize() {
        m_editorCamera->setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);

        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_UNORM, m_context);
        m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());
        m_syclGaussianGfx.setActiveCamera(m_editorCamera);

    }


    void VkRender::EditorGaussianViewer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_ui);
        if (imageUI->useImageFrom3DViewport) {

        }
        m_syclGaussianGfx.setActiveCamera(m_editorCamera);


        auto& e = m_context->getSelectedEntity();
        if (e && e.hasComponent<CameraComponent>()) {
            auto& camera = e.getComponent<CameraComponent>();
            if (camera.renderFromViewpoint() ) {
                // If the selected entity has a camera with renderFromViewpoint, use it
                camera.camera->setCameraResolution(m_createInfo.width ,m_createInfo.height);
                m_syclGaussianGfx.setActiveCamera(camera.camera);
                m_lastActiveCamera = &camera; // Update the last active camera
            }
        } else if (m_lastActiveCamera && m_lastActiveCamera->renderFromViewpoint()) {
            // Use the last active camera if it still has renderFromViewpoint enabled
            m_syclGaussianGfx.setActiveCamera(m_lastActiveCamera->camera);
        }


    }

    void VkRender::EditorGaussianViewer::onRender(CommandBuffer &commandBuffer) {
        auto scene = m_context->activeScene();

        auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_ui);

        bool updateRender = false;
        m_activeScene->getRegistry().view<GaussianComponent>().each([&](auto entity, GaussianComponent &gaussianComp) {
            auto e = Entity(entity, m_activeScene.get());
            if (e.getComponent<GaussianComponent>().addToRenderer)
                updateRender = true;
        });

        if (imageUI->render3dgsImage || updateRender)
            m_syclGaussianGfx.render(scene, m_colorTexture);

        m_renderPipelines->draw(commandBuffer);

    }

    void EditorGaussianViewer::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera->rotate(mouse.dx, mouse.dy);
        }
    }

    void EditorGaussianViewer::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera->setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }

    void EditorGaussianViewer::onKeyCallback(const Input &input) {

        m_editorCamera->keys.up = input.keys.up;
        m_editorCamera->keys.down = input.keys.down;
        m_editorCamera->keys.left = input.keys.left;
        m_editorCamera->keys.right = input.keys.right;

    }
}