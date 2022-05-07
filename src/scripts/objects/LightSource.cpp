#include <MultiSense/src/imgui/Layer.h>
#include "LightSource.h"

void LightSource::setup() {
    printf("MyModelExample setup\n");

    std::string fileName;
    //loadFromFile(fileName);
    model.loadFromFile(Utils::getAssetsPath() + "Models/Box/glTF-Embedded/Box.gltf", renderUtils.device,
                       renderUtils.device->transferQueue, 1.0f);


    // Shader creation
    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/box.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/box.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

    renderUtils.shaders = {{vs},
                           {fs}};

    // Obligatory call to prepare render resources for glTFModel.
    glTFModel::createRenderPipeline(renderUtils);
}

void LightSource::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    glTFModel::draw(commandBuffer, i);
}

void LightSource::update() {
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, -3.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(0.1f, 0.1f, 0.1f));

    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

}



void LightSource::onUIUpdate(GuiObjectHandles uiHandle) {

}