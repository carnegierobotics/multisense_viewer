//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scenes/Renderer3D/DefaultScene.h"
#include "Viewer/VkRender/ImGui/Widgets.h"
#include "Viewer/VkRender/Components/CustomModels.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/VkRender/Components/MeshComponent.h"

void DefaultScene::setup() {

    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

    if (showDepthView) {
        auto quad = m_context->createEntity("depthImageView");
        auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "quad.obj",
                m_context->data().device);
        quad.addComponent<VkRender::ImageViewComponent>();
        auto &res = quad.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->data(),
                                                                                   "default2D.vert.spv",
                                                                                   "default2D.frag.spv");
        res.bind(modelComponent);
        res.setTexture(&m_context->data().depthRenderPass->depthImageInfo);
    }
}


void DefaultScene::update() {
    if (showDepthView) {
        auto depthImageView = m_context->findEntityByName("depthImageView");
        if (depthImageView) {
            auto &transform = depthImageView.getComponent<VkRender::TransformComponent>();
            float xOffsetPx = (m_context->data().width - 150.0) / m_context->data().width;
            float translationX = xOffsetPx * 2 - 1;
            float translationY = xOffsetPx * 2 - 1;
            transform.translation = glm::vec3(translationX, translationY, 0.0f);
            float scaleX = 300.0f / m_context->data().width;
            transform.scale =glm::vec3(scaleX, scaleX, 1.0f);
        }
    }

}


void DefaultScene::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
}

void DefaultScene::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    if (showDepthView) {
        // Re-create the depth image view if we resize the viewport
        auto oldEntity = m_context->findEntityByName("depthImageView");
        if (oldEntity) {
            oldEntity.getComponent<VkRender::DefaultGraphicsPipelineComponent2>().cleanUp(0, true);
            m_context->destroyEntity(m_context->findEntityByName("depthImageView"));
        }
        auto quad = m_context->createEntity("depthImageView");
        auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "quad.obj",
                m_context->data().device);
        auto &res = quad.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->data(),
                                                                                   "default2D.vert.spv",
                                                                                   "default2D.frag.spv");
        quad.addComponent<VkRender::ImageViewComponent>();
        res.bind(modelComponent);
        res.setTexture(&m_context->data().depthRenderPass->depthImageInfo);
    }
}
