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
#include "Viewer/Renderer/Components/SecondaryCameraComponent.h"

void MultiSense::setup() {
    {
        auto skybox = m_context->createEntity("Skybox");
        auto &modelComponent = skybox.addComponent<VkRender::GLTFModelComponent>(
                Utils::getModelsPath() / "Box" / "Box.gltf", m_context->renderUtils.device);
        skybox.addComponent<VkRender::SkyboxGraphicsPipelineComponent>(&m_context->renderUtils, modelComponent);
    }

    {
        auto ent = m_context->createEntity("KS21");
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(Utils::getModelsPath() / "ks21_pbr.gltf",
                                                                         m_context->renderUtils.device);

        auto &sky = m_context->findEntityByName(
                "Skybox").getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
        ent.addComponent<VkRender::DefaultPBRGraphicsPipelineComponent>(&m_context->renderUtils, component, sky);



    }

    {
        auto ent = m_context->createEntity("Coordinates");
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(Utils::getModelsPath() / "coordinates.gltf",
                                                                         m_context->renderUtils.device);
        ent.addComponent<VkRender::DepthRenderPassComponent>();

        auto &sky = m_context->findEntityByName(
                "Skybox").getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
        ent.addComponent<VkRender::DefaultPBRGraphicsPipelineComponent>(&m_context->renderUtils, component, sky);
    }
}


void MultiSense::update() {
    auto &camera = m_context->getCamera();
    glm::mat4 invView = glm::inverse(camera.matrices.view);
    glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    auto cameraWorldPosition = glm::vec3(cameraPos4);

    auto skybox = m_context->findEntityByName("Skybox");
    if (skybox) {
        auto &obj = skybox.getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
        // Skybox
        obj.uboMatrix.projection = camera.matrices.perspective;
        obj.uboMatrix.model = glm::mat4(glm::mat3(camera.matrices.view));
        obj.uboMatrix.model = glm::rotate(obj.uboMatrix.model, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0)); // z-up rotation

        obj.update();
    }


    auto model = m_context->findEntityByName("KS21");
    if (model) {
        auto &obj = model.getComponent<VkRender::DefaultPBRGraphicsPipelineComponent>();
        auto &transform = model.getComponent<VkRender::TransformComponent>();
        transform.setPosition( glm::vec3(0.25f, 0.0f, 0.25f));
        transform.setEuler(90.0f, 0.0f, 0.0f);

        for (size_t i = 0; i < m_context->renderUtils.swapchainImages; ++i) {
            //obj.resources[i].uboMatrix.model = glm::scale(obj.resources[i].uboMatrix.model, glm::vec3(0.5f, 0.5f, 0.5f));
            obj.resources[i].uboMatrix.camPos = cameraWorldPosition;
            obj.resources[i].shaderValuesParams.lightDir = glm::vec4(
                    sin(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                    sin(glm::radians(lightSource.rotation.y)),
                    cos(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                    0.0f);
        }
    }


    auto coordinates = m_context->findEntityByName("Coordinates");
    if (coordinates) {
        auto &obj = coordinates.getComponent<VkRender::DefaultPBRGraphicsPipelineComponent>();
        auto &transform = coordinates.getComponent<VkRender::TransformComponent>();
        transform.scale = glm::vec3(0.25f, 0.25f, 0.25f);
        transform.setEuler(90.0f, 0.0f, 0.0f);

        for (size_t i = 0; i < m_context->renderUtils.swapchainImages; ++i) {

            obj.resources[i].uboMatrix.camPos = cameraWorldPosition;
            obj.resources[i].shaderValuesParams.lightDir = glm::vec4(
                    sin(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                    sin(glm::radians(lightSource.rotation.y)),
                    cos(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                    0.0f);

            obj.resources[i].shaderValuesParams.gamma += 6.0f;
            obj.resources[i].shaderValuesParams.exposure += 12.0f;
        }
        obj.update(m_context->renderUtils.swapchainIndex);
    }
}