//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scenes/Default/Scripts/MultiSense.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/Components.h"


void MultiSense::setup() {
    Log::Logger::getInstance()->info("Setup from the MultiSense Script");

    /*
    {
        auto ent = m_context->createEntity("KS21");
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(Utils::getModelsPath() / "ks21_pbr.gltf",
                                                                         m_context->data().device);

        auto &sky = m_context->findEntityByName(
                "Skybox").getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
        ent.addComponent<VkRender::DefaultPBRGraphicsPipelineComponent>(&m_context->data(), component, sky);

    }


    {
        auto ent = m_context->createEntity("Coordinates");
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(Utils::getModelsPath() / "coordinates.gltf",
                                                                         m_context->data().device);
        ent.addComponent<VkRender::DepthRenderPassComponent>();

        auto &sky = m_context->findEntityByName(
                "Skybox").getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
        ent.addComponent<VkRender::DefaultPBRGraphicsPipelineComponent>(&m_context->data(), component, sky);
    }

     */
}


void MultiSense::update() {

    //auto &camera = m_context->getCamera();
    //glm::mat4 invView = glm::inverse(camera.matrices.view);
    //glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    //auto cameraWorldPosition = glm::vec3(cameraPos4);

    /*
    auto model = m_context->findEntityByName("KS21");
    if (model) {
        auto &obj = model.getComponent<VkRender::DefaultPBRGraphicsPipelineComponent>();
        auto &transform = model.getComponent<VkRender::TransformComponent>();
        transform.setPosition( glm::vec3(0.25f, 0.0f, 0.25f));
        transform.setEuler(90.0f, 0.0f, 0.0f);

        for (size_t i = 0; i < m_context->data().swapchainImages; ++i) {
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
        //transform.scale = glm::vec3(0.25f, 0.25f, 0.25f);
        transform.setEuler(90.0f, 0.0f, 0.0f);

        for (size_t i = 0; i < m_context->data().swapchainImages; ++i) {

            obj.resources[i].uboMatrix.camPos = cameraWorldPosition;
            obj.resources[i].shaderValuesParams.lightDir = glm::vec4(
                    sin(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                    sin(glm::radians(lightSource.rotation.y)),
                    cos(glm::radians(lightSource.rotation.x)) * cos(glm::radians(lightSource.rotation.y)),
                    0.0f);

            obj.resources[i].shaderValuesParams.gamma += 6.0f;
            obj.resources[i].shaderValuesParams.exposure += 12.0f;
        }
        obj.update(m_context->data().swapchainIndex);
    }
     */

}