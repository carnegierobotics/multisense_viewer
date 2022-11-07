//
// Created by magnus on 11/4/22.
//

#include "Gizmos.h"

void Gizmos::setup() {
    m_Model = std::make_unique<glTFModel::Model>(renderUtils.device);
    m_Model->loadFromFile(Utils::getAssetsPath() + "Models/coordinates.gltf", renderUtils.device,
                          renderUtils.device->m_TransferQueue, 1.0f);


    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("myScene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("myScene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};


    // Obligatory call to prepare render resources for glTFModel.
    m_Model->createRenderPipeline(renderUtils, shaders);
}

void Gizmos::draw(VkCommandBuffer commandBuffer, uint32_t i, bool primaryDraw) {
    if (primaryDraw)
        m_Model->draw(commandBuffer, i);
}

void Gizmos::update() {
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(0.01f, 0.01f, 0.01f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;

}


void Gizmos::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;
    }
}