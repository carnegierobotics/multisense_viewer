//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/Example/Example3D.h"


void Example3D::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("spv/object.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("spv/object.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};

    KS21 = std::make_unique<GLTFModel::Model>(&renderUtils, renderUtils.device);
    KS21->loadFromFile(Utils::getAssetsPath().append("Models/ks21_pbr.gltf").string(), renderUtils.device,
                       renderUtils.device->m_TransferQueue, 1.0f);
    KS21->createRenderPipeline(renderUtils, shaders);


}


void Example3D::update() {
    auto &d = ubo[0].mvp;


    d->model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
    d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    d->model = glm::rotate(d->model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    d->model = glm::scale(d->model, glm::vec3(0.001f, 0.001f, 0.001f));

    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
    d->camPos = glm::vec3(
            static_cast<double>(-renderData.camera->m_Position.z) * sin(
                    static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
            static_cast<double>(-renderData.camera->m_Position.z) * sin(
                    static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
            static_cast<double>(renderData.camera->m_Position.z) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x)))
    );

    auto &d2 = ubo[0].fragShader;
    d2->lightDir = glm::vec4(
            static_cast<double>(sinf(glm::radians(lightSource.rotation.x))) * cos(
                    static_cast<double>(glm::radians(lightSource.rotation.y))),
            sin(static_cast<double>(glm::radians(lightSource.rotation.y))),
            cos(static_cast<double>(glm::radians(lightSource.rotation.x))) * cos(
                    static_cast<double>(glm::radians(lightSource.rotation.y))),
            0.0f);


    auto *ptr = reinterpret_cast<VkRender::FragShaderParams *>(sharedData->data);
    d2->gamma = ptr->gamma;
    d2->exposure = ptr->exposure;
    d2->scaleIBLAmbient = ptr->scaleIBLAmbient;
    d2->debugViewInputs = ptr->debugViewInputs;
    d2->prefilteredCubeMipLevels = renderUtils.skybox.prefilteredCubeMipLevels;

}

void Example3D::draw(CommandBuffer * commandBuffer, uint32_t i, bool b) {
    if (b)
        KS21->draw(commandBuffer, i);
}