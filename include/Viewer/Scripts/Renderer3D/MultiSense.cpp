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
        humvee = m_context->createEntitySharedPtr("Skybox");
        humvee->addComponent<VkRender::CameraComponent>();
        auto &modelComponent = humvee->addComponent<VkRender::GLTFModelComponent>(
                Utils::getModelsPath() / "Box" / "Box.gltf", m_context->renderUtils.device);
        humvee->addComponent<RenderResource::SkyboxGraphicsPipelineComponent>(&m_context->renderUtils, modelComponent);
        //auto uuid = humvee->getUUID();
    }

    {
        auto ent = m_context->createEntity("Humvee");
        ent.addComponent<VkRender::CameraComponent>();
        auto &component = ent.addComponent<VkRender::GLTFModelComponent>(
                Utils::getModelsPath() / "humvee.gltf", m_context->renderUtils.device);
//
        ent.addComponent<RenderResource::DefaultPBRGraphicsPipelineComponent>(&m_context->renderUtils, component);

    }
}


void MultiSense::update() {

    if (humvee && humvee->hasComponent<RenderResource::SkyboxGraphicsPipelineComponent>()){
        auto& obj = humvee->getComponent<RenderResource::SkyboxGraphicsPipelineComponent>();
        auto& camera = m_context->getCamera();
        // Skybox
        obj.uboMatrix.projection = camera.matrices.perspective;
        obj.uboMatrix.model = glm::mat4(glm::mat3(camera.matrices.view));

        obj.uboMatrix.view = camera.matrices.view;

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
