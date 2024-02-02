//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/Grid.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"
#include "Viewer/ImGui/Widgets.h"

void Grid::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("spv/grid.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("spv/grid.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};

    resourceTracker.resize(renderUtils.UBCount);
    model = std::make_unique<CustomModels>(&renderUtils);
    VkRender::ScriptUtils::ImageData imgData{};
    model->model->uploadMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
    model->createDescriptorSetLayout();
    model->createDescriptorPool();
    model->createDescriptorSets();
    model->createGraphicsPipeline(shaders);

    for (size_t i = 0; i < renderUtils.UBCount; ++i) {
        resourceTracker[i].pipeline = model->pipelines[i];
        resourceTracker[i].pipelineLayout = model->pipelineLayouts[i];
        resourceTracker[i].descriptorSetLayout = model->descriptorSetLayouts[i];
        resourceTracker[i].descriptorPool = model->descriptorPools[i];
    }


    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Grid", &enable);
}


void Grid::update() {
    auto &d = bufferOneData;
    d->model = glm::mat4(1.0f);
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
}

void Grid::draw(CommandBuffer * commandBuffer, uint32_t i, bool b) {
    if (b && enable) {
        model->draw(commandBuffer, i);
    }
}



void Grid::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    if (uiHandle->devices.empty()){
        hide = false;
    } else
        hide = true;
}