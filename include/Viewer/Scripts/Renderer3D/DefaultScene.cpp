//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/DefaultScene.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/ModelLoaders/CustomModels.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"

void DefaultScene::setup() {

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Grid", &enable);

    auto grid = m_context->createEntity("3DViewerGrid");
    grid.addComponent<VkRender::CameraComponent>();
    grid.addComponent<CustomModelComponent>(&m_context->renderUtils);
    auto uuid = grid.getUUID();

    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void DefaultScene::update() {
    Log::Logger::getInstance()->traceWithFrequency("Tag", 5000, "Update from {}", GetFactoryName());
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

    auto &c = e.getComponent<CustomModelComponent>();
    c.update(m_context->renderUtils.swapchainIndex, &d);

}

void DefaultScene::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {
    Log::Logger::getInstance()->traceWithFrequency("Tag_Draw", 60, "Drawing from {}", GetFactoryName());

}


void DefaultScene::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

}