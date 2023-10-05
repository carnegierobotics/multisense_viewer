//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/Grid.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"
#include "Viewer/ImGui/ScriptUIAddons.h"

void Grid::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/grid.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/grid.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};
    model = std::make_unique<CustomModels>(&renderUtils);
    VkRender::ScriptUtils::ImageData imgData{};
    model->model->uploadMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
    model->createDescriptorSetLayout();
    model->createDescriptorPool();
    model->createDescriptorSets();
    model->createGraphicsPipeline(shaders);

    Widgets::make()->checkbox("Renderer3D", "Grid", &enable);


}


void Grid::update() {
    auto &d = bufferOneData;

    if (enable && !hide)
        drawMethod = CRL_SCRIPT_DRAW;
    else
        drawMethod = CRL_SCRIPT_DONT_DRAW;
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

}

void Grid::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b) {
        model->draw(commandBuffer, i);
    }
}



void Grid::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    if (uiHandle->devices.empty()){
        hide = false;
    } else
        hide = true;
}