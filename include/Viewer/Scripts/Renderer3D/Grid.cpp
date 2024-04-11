//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/Grid.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/ModelLoaders/CustomModels.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"

void Grid::setup() {

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Grid", &enable);

    auto grid = m_context->createEntity("3DViewerGrid");
    grid.addComponent<VkRender::CameraComponent>();
    grid.addComponent<CustomModelComponent>(&m_context->renderUtils);
    auto uuid = grid.getUUID();

    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void Grid::update() {
    Log::Logger::getInstance()->traceWithFrequency("Tag", 5000, "Update from {}", GetFactoryName());
    // Update UBO data
    auto e = m_context->findEntityByName("3DViewerGrid");
    auto id = e.getUUID();
    auto& c = e.getComponent<CustomModelComponent>();

    auto& camera = e.getComponent<VkRender::CameraComponent>();
    camera.camera.type = VkRender::Camera::arcball;
    camera.camera.setPerspective(60.0f, static_cast<float>(1280) / static_cast<float>(720), 0.01f, 100.0f);
    camera.camera.resetPosition();
    camera.camera.resetRotation();

    VkRender::UBOMatrix d;
    d.model = glm::mat4(1.0f);
    d.view = camera.camera.matrices.view;
    d.projection = camera.camera.matrices.perspective;

    c.update(m_context->renderUtils.swapchainIndex, &d);

}

void Grid::draw(CommandBuffer * commandBuffer, uint32_t i, bool b) {
    Log::Logger::getInstance()->traceWithFrequency("Tag_Draw", 60, "Drawing from {}", GetFactoryName());

}



void Grid::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

}