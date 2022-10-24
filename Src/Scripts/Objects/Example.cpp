

#include "Example.h"

void Example::setup() {
    printf("MyModelExample setup\n");

    std::string fileName;
    //m_Model.loadFromFile(Utils::getAssetsPath() + "Models/DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf", renderUtils.m_Device, renderUtils.m_Device->m_TransferQueue, 1.0f);


    // Shader loading
    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/helmet.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/helmet.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};

    // Obligatory call to prepare render resources for glTFModel.
    //glTFModel::createRenderPipeline(renderUtils, shaders);
}


void Example::update() {

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

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -2.0f, -3.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;
}