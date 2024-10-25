//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"
#include "Editor3DLayer.h"

#include "Viewer/VkRender/Components/MaterialComponent.h"
#include "Viewer/VkRender/Editors/Common/CommonEditorFunctions.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        addUIData<Editor3DViewportUI>();

        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);

        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(uuid);

        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();

        m_renderPipelines = std::make_unique<GraphicsPipeline2D>(*m_context, renderPassInfo);
        m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());

    }

    void Editor3DViewport::onEditorResize() {
        m_editorCamera->setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);

        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(getUUID());
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        m_renderPipelines->setTexture(&m_colorTexture->getDescriptorInfo());
        m_sceneRenderer->setActiveCamera(m_editorCamera);

    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);
        m_activeScene = m_context->activeScene();
        m_sceneRenderer->setActiveCamera(m_editorCamera);

    }

    void Editor3DViewport::onUpdate() {
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        m_sceneRenderer->setActiveCamera(m_editorCamera);

        auto& e = m_context->getSelectedEntity();
        if (e && e.hasComponent<CameraComponent>()) {
            auto& camera = e.getComponent<CameraComponent>();
            if (camera.renderFromViewpoint()) {
                // If the selected entity has a camera with renderFromViewpoint, use it
                m_sceneRenderer->setActiveCamera(camera.camera);
                m_lastActiveCamera = &camera; // Update the last active camera
            }
        } else if (m_lastActiveCamera && m_lastActiveCamera->renderFromViewpoint()) {
            // Use the last active camera if it still has renderFromViewpoint enabled
            m_sceneRenderer->setActiveCamera(m_lastActiveCamera->camera);
        }


        m_sceneRenderer->update();
    }


    void Editor3DViewport::onRender(CommandBuffer &commandBuffer) {
        m_renderPipelines->draw(commandBuffer);
    }

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera->rotate(mouse.dx, mouse.dy);
        }
    }

    void Editor3DViewport::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera->setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }

    void Editor3DViewport::onKeyCallback(const Input &input) {

        if (input.lastKeyPress == GLFW_KEY_KP_0 && input.action == GLFW_PRESS) {
            auto &e = m_context->getSelectedEntity();
            if (e && e.hasComponent<CameraComponent>()) {
                auto cameraPtr = e.getComponent<CameraComponent>().camera;
                if (cameraPtr) {
                    m_sceneRenderer->setActiveCamera(cameraPtr);
                }
            }
        }

        if (input.lastKeyPress == GLFW_KEY_KP_1 && input.action == GLFW_PRESS) {
            m_sceneRenderer->setActiveCamera(m_editorCamera);

        }
    }
};
