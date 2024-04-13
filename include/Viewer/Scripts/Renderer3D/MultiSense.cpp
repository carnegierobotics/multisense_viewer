//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/MultiSense.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/DefaultPBRGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/GLTFModelComponent.h"

void MultiSense::setup() {
    {
        skybox = m_context->createEntitySharedPtr("Skybox");
        skybox->addComponent<VkRender::CameraComponent>();
        auto &modelComponent = skybox->addComponent<VkRender::GLTFModelComponent>(
                Utils::getModelsPath() / "Box" / "Box.gltf", m_context->renderUtils.device);
        skybox->addComponent<RenderResource::SkyboxGraphicsPipelineComponent>(&m_context->renderUtils, modelComponent);
        //auto uuid = skybox->getUUID();
    }

    {
        auto ent = m_context->createEntity("Humvee");
        ent.addComponent<VkRender::CameraComponent>();
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(
                "/home/magnus/Downloads/glTF-Sample-Assets-main/Models/ABeautifulGame/glTF/ABeautifulGame.gltf",
                m_context->renderUtils.device);

        ent.addComponent<RenderResource::DefaultPBRGraphicsPipelineComponent>(&m_context->renderUtils, component,
                                                                              m_context->findEntityByName(
                                                                                      "Skybox").getComponent<RenderResource::SkyboxGraphicsPipelineComponent>());

    }

    {

        auto ent = m_context->createEntity("Coordinates");
        ent.addComponent<VkRender::CameraComponent>();
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(
                Utils::getModelsPath() / "coordinates.gltf", m_context->renderUtils.device);

        ent.addComponent<RenderResource::DefaultPBRGraphicsPipelineComponent>(&m_context->renderUtils, component,
                                                                              m_context->findEntityByName(
                                                                                      "Skybox").getComponent<RenderResource::SkyboxGraphicsPipelineComponent>());

    }

}


void MultiSense::update() {
    auto &camera = m_context->getCamera();
    glm::mat4 invView = glm::inverse(camera.matrices.view);
    glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    auto cameraWorldPosition = glm::vec3(cameraPos4);

    if (skybox && skybox->hasComponent<RenderResource::SkyboxGraphicsPipelineComponent>()) {
        auto &obj = skybox->getComponent<RenderResource::SkyboxGraphicsPipelineComponent>();
        // Skybox
        obj.uboMatrix.projection = camera.matrices.perspective;
        obj.uboMatrix.model = glm::mat4(glm::mat3(camera.matrices.view));
        obj.uboMatrix.model = glm::rotate(obj.uboMatrix.model, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0)); // z-up rotation
        obj.update();
    }


    auto model = m_context->findEntityByName("Humvee");
    if (model) {
        auto &obj = model.getComponent<RenderResource::DefaultPBRGraphicsPipelineComponent>();

        obj.uboMatrix.projection = camera.matrices.perspective;
        obj.uboMatrix.view = camera.matrices.view;
        obj.uboMatrix.model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0)); // z-up rotation
        //obj.uboMatrix.model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 0.0f)); // z-up rotation
        obj.uboMatrix.model = glm::scale(obj.uboMatrix.model, glm::vec3(0.5f, 0.5f, 0.5f)); // z-up rotation



        obj.uboMatrix.camPos = cameraWorldPosition;
        obj.shaderValuesParams.lightDir = glm::vec4(
                sin(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                sin(glm::radians(lightSource.rotation.y)),
                cos(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                0.0f);

        obj.update();
    }


    auto coordinates = m_context->findEntityByName("Coordinates");
    if (coordinates) {
        auto &obj = coordinates.getComponent<RenderResource::DefaultPBRGraphicsPipelineComponent>();

        obj.uboMatrix.projection = camera.matrices.perspective;
        obj.uboMatrix.view = camera.matrices.view;
        obj.uboMatrix.model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0)); // z-up rotation
        //obj.uboMatrix.model = glm::scale(obj.uboMatrix.model, glm::vec3(0.2f, 0.2f, 0.2f)); // z-up rotation

        obj.uboMatrix.camPos = cameraWorldPosition;

        obj.shaderValuesParams.lightDir = glm::vec4(
                sin(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                sin(glm::radians(lightSource.rotation.y)),
                cos(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                0.0f);

        obj.update();
    }

}

void MultiSense::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {


}
