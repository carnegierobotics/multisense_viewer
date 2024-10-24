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


        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_SRGB, m_context);
        m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());

        m_activeCamera = Camera(m_createInfo.width, m_createInfo.height);

    }

    void VkRender::EditorGaussianViewer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();
        addUIData<EditorGaussianViewerUI>();

    }

    void EditorGaussianViewer::onEditorResize() {

        m_activeCamera.setPerspective(static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height));
        m_activeCamera = Camera(m_createInfo.width, m_createInfo.height);
        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_B8G8R8A8_SRGB, m_context);
        m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());

    }


    void VkRender::EditorGaussianViewer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_ui);
        if (imageUI->useImageFrom3DViewport) {

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
            m_syclGaussianGfx.render(scene, m_colorTexture, m_activeCamera);

        m_renderPipelines->draw(commandBuffer);

    }

    void EditorGaussianViewer::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_activeCamera.rotate(mouse.dx, mouse.dy);
        }
    }

    void EditorGaussianViewer::onMouseScroll(float change) {
        if (ui()->hovered)
            m_activeCamera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }

    void EditorGaussianViewer::onKeyCallback(const Input &input) {

        m_activeCamera.keys.up = input.keys.up;
        m_activeCamera.keys.down = input.keys.down;
        m_activeCamera.keys.left = input.keys.left;
        m_activeCamera.keys.right = input.keys.right;

    }
}