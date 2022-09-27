#include "MultiSenseCamera.h"

void MultiSenseCamera::setup() {
    std::string fileName;
    //loadFromFile(fileName);
    model.loadFromFile(Utils::getAssetsPath() + "Models/camera.gltf", renderUtils.device,
                       renderUtils.device->transferQueue, 1.0f);


    // Shader creation
    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/box.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/box.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    renderUtils.shaders = {{vs},
                           {fs}};

    // Obligatory call to prepare render resources for glTFModel.
    glTFModel::createRenderPipeline(renderUtils);
}

void MultiSenseCamera::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (previewTab == TAB_3D_POINT_CLOUD && b)
        glTFModel::draw(commandBuffer, i);
}

void MultiSenseCamera::update() {
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(0.001f, 0.001f, 0.001f));
    mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    auto *d = (VkRender::UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (VkRender::FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;

}


void MultiSenseCamera::onUIUpdate(MultiSense::GuiObjectHandles uiHandle) {

    for (const auto &d: *uiHandle.devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

        previewTab = d.selectedPreviewTab;

    }

}