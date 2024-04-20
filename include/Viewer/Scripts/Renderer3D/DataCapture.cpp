//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/DataCapture.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Renderer/Components/CustomModels.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"

void DataCapture::setup() {

    auto entity = m_context->createEntity("3DGSModel");
    auto &component = entity.addComponent<VkRender::OBJModelComponent>(Utils::getModelsPath() / "ks21_pbr.gltf",
                                                                     m_context->renderUtils.device);

    auto uuid = entity.getUUID();
    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void DataCapture::update() {
    Log::Logger::getInstance()->traceWithFrequency("Tag", 5000, "Update from {}", GetFactoryName());


}

void DataCapture::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {
    Log::Logger::getInstance()->traceWithFrequency("Tag_Draw", 60, "Drawing from {}", GetFactoryName());

}


void DataCapture::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

}