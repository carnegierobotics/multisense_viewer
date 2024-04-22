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
#include "Viewer/Renderer/Components/DefaultGraphicsPipelineComponent.h"

void DataCapture::setup() {

    auto entity = m_context->createEntity("3DGSModel");
    auto &component = entity.addComponent<VkRender::OBJModelComponent>(Utils::getModelsPath() / "Test" / "viking_room.obj",
                                                                     m_context->renderUtils.device);



    auto& pbrComponent = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent>(&m_context->renderUtils, component);

    auto uuid = entity.getUUID();
    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void DataCapture::update() {
    auto gsMesh = m_context->findEntityByName("3DGSModel");
    if (gsMesh) {
        auto &camera = m_context->getCamera();

        auto &obj = gsMesh.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (size_t i = 0; i < m_context->renderUtils.UBCount; ++i) {

            obj.resources[i].uboMatrix.projection = camera.matrices.perspective;
            obj.resources[i].uboMatrix.view = camera.matrices.view;
            obj.resources[i].uboMatrix.model = glm::mat4(1.0f);

            obj.resources[i].uboMatrix.model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0)); // z-up rotation

        }
        obj.update();
    }

}


void DataCapture::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

}

void DataCapture::draw(CommandBuffer *cb, uint32_t i, bool b) {
}
