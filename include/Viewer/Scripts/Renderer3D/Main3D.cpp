//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/Main3D.h"


void Main3D::setup() {
    std::string fileName;
    //m_Model.loadFromFile(Utils::getAssetsPath() + "Models/DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf", renderUtils.m_Device, renderUtils.m_Device->m_TransferQueue, 1.0f);


    // Shader loading
    VkPipelineShaderStageCreateInfo vs = loadShader("Scene/spv/helmet.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("Scene/spv/helmet.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};

    // Obligatory call to prepare render resources for glTFModel.
    //glTFModel::createRenderPipeline(renderUtils, shaders);
}


void Main3D::update() {

    // Update Uniform buffers for this draw call
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(4.0f, -5.0f, -1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
}