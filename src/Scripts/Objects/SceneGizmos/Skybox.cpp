//
// Created by magnus on 2/25/23.
//

#include "Viewer/Scripts/Objects/SceneGizmos/Skybox.h"
#include "Viewer/ImGui/ScriptUIAddons.h"

void Skybox::setup() {
    // Create Skybox
    skybox = std::make_unique<GLTFModel::Model>(renderUtils.device);
    std::vector<VkPipelineShaderStageCreateInfo> envShaders = {{loadShader("Scene/spv/filtercube.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("Scene/spv/irradiancecube.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("Scene/spv/prefilterenvmap.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("Scene/spv/genbrdflut.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("Scene/spv/genbrdflut.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("Scene/spv/skybox.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("Scene/spv/skybox.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)}};



/*
    Widgets::make()->slider("Exposure", &exposure, 0.0f, 25.0f);
    Widgets::make()->slider("Gamma", &gamma, 0.0f, 8.0f);
    Widgets::make()->slider("IBL", &ibl, 0.0f, 8.0f);
    Widgets::make()->slider("debugView", &debugViewInputs, 0, 6);
    Widgets::make()->slider("Skybox - LOD", &lod, 0, 10.0f);
    */

    skyboxTextures.environmentMap.loadFromFile(Utils::getAssetsPath().append("Textures/Environments/skies.ktx2"), renderUtils.device);

    skybox->createSkybox(envShaders, renderUtils.uniformBuffers, renderUtils.renderPass, &skyboxTextures);
    sharedData->destination = "All";
}

void Skybox::draw(VkCommandBuffer commandBuffer, uint32_t i, bool primaryDraw) {
    if (selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD && primaryDraw)
        skybox->drawSkybox(commandBuffer, i);
}

void Skybox::update() {
    auto &d = bufferOneData;
    d->model = glm::mat4(glm::mat3(renderData.camera->matrices.view));
    d->model = glm::rotate(d->model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;


    auto &d2 = bufferTwoData;


    d2->exposure = exposure;
    d2->gamma = gamma;
    d2->debugViewInputs = 0.0f;
    d2->scaleIBLAmbient = ibl;
    d2->lod = lod;

    sharedData->put(d2.get(), sizeof(VkRender::FragShaderParams));

}


void Skybox::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    for (const auto &d: uiHandle->devices) {
        if (d.state != CRL_STATE_ACTIVE)
            continue;
        selectedPreviewTab = d.selectedPreviewTab;

    }
}