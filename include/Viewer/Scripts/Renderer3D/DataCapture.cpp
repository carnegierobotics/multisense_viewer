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

    auto entity = m_context->createEntity("viking_room");
    auto &component = entity.addComponent<VkRender::OBJModelComponent>(
            Utils::getModelsPath() / "obj" / "3dgs.obj",
            m_context->renderUtils.device);
    entity.addComponent<VkRender::DefaultGraphicsPipelineComponent>(&m_context->renderUtils, component, true);
    entity.addComponent<VkRender::SecondaryRenderPassComponent>();

    auto quad = m_context->createEntity("quad");
    auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(Utils::getModelsPath() / "obj" / "quad.obj",
                                                                          m_context->renderUtils.device);

    modelComponent.objTexture.m_Descriptor = m_context->renderUtils.secondaryRenderPasses->front().depthImageInfo;
    quad.addComponent<VkRender::DefaultGraphicsPipelineComponent>(&m_context->renderUtils, modelComponent, false,
                                                                  "default2D.vert.spv", "default2D.frag.spv");


    auto uuid = entity.getUUID();
    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void DataCapture::update() {
    auto &camera = m_context->getCamera();
    glm::mat4 invView = glm::inverse(camera.matrices.view);
    glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    auto cameraWorldPosition = glm::vec3(cameraPos4);

    auto gsMesh = m_context->findEntityByName("viking_room");
    if (gsMesh) {
        auto &obj = gsMesh.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (auto & i : obj.renderData) {

            i.uboMatrix.projection = camera.matrices.perspective;
            i.uboMatrix.view = camera.matrices.view;
            i.uboMatrix.model = glm::mat4(1.0f);
            i.uboMatrix.camPos = cameraWorldPosition;
        }
        obj.update();
    }
    auto quad = m_context->findEntityByName("quad");
    if (quad) {
        auto &obj = quad.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (size_t i = 0; i < obj.renderData.size(); ++i) {

            obj.renderData[i].uboMatrix.projection = camera.matrices.perspective;
            obj.renderData[i].uboMatrix.view = camera.matrices.view;
            obj.renderData[i].uboMatrix.camPos = cameraWorldPosition;

            auto model = glm::mat4(1.0f);

            float xOffsetPx = (m_context->renderData.width - 150.0) / m_context->renderData.width;

            float translationX = xOffsetPx * 2 - 1;
            float translationY = xOffsetPx * 2 - 1;

            // Apply translation after scaling
            model = glm::translate(model, glm::vec3(translationX, translationY, 0.0f));
            // Convert 300 pixels from the right edge into NDC
            float scaleX = 300.0f / m_context->renderData.width;

            model = glm::scale(model, glm::vec3(scaleX, scaleX, 1.0f)); // Uniform scaling in x and y
            //model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)); // Uniform scaling in x and y

            obj.renderData[i].uboMatrix.model = model;

        }
        obj.update();
    }

}


void DataCapture::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

}

void DataCapture::draw(CommandBuffer *cb, uint32_t i, bool b) {
}
