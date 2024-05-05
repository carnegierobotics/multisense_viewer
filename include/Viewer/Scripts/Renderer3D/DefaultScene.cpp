//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/DefaultScene.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Renderer/Components/CustomModels.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"

void DefaultScene::setup() {

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Grid", &enable);

    auto grid = m_context->createEntity("3DViewerGrid");
    grid.addComponent<VkRender::CustomModelComponent>(&m_context->renderUtils);
    auto uuid = grid.getUUID();

    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void DefaultScene::update() {
    //Log::Logger::getInstance()->traceWithFrequency("Tag", 5000, "Update from {}", GetFactoryName());
    // Update UBO data
    auto e = m_context->findEntityByName("3DViewerGrid");
    if (!e)
        return;
    auto id = e.getUUID();
    auto& camera = m_context->getCamera();

    VkRender::UBOMatrix d;
    d.model = glm::mat4(1.0f);
    d.view = camera.matrices.view;
    d.projection = camera.matrices.perspective;

    auto &c = e.getComponent<VkRender::CustomModelComponent>();
    c.update(m_context->renderUtils.swapchainIndex, &d);

}

void DefaultScene::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {
    Log::Logger::getInstance()->traceWithFrequency("Tag_Draw", 60, "Drawing from {}", GetFactoryName());

}


void DefaultScene::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    auto& currentCamera = m_context->getCamera();

    // Loop through cameras and update the camera gizmo position
    for (auto entity: m_context->m_registry.view<VkRender::CameraGraphicsPipelineComponent>()) {
        /*
        auto &obj = m_context->m_registry.get<VkRender::CameraGraphicsPipelineComponent>(entity);
        auto &camComponent = m_context->m_registry.get<VkRender::CameraComponent>(entity);
        obj.mvp.projection = currentCamera.matrices.perspective;
        obj.mvp.view = currentCamera.matrices.view;

        glm::mat4 model = glm::translate(glm::mat4(1.0f), camComponent.camera.pose.pos);
        model = model * glm::mat4_cast(camComponent.camera.pose.q);
        model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
        obj.mvp.model = model;
        obj.mvp.camPos = camComponent.camera.pose.pos;
        */
    }
}