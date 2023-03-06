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



    Widgets::make()->slider("Exposure", &exposure, 0.0f, 25.0f);
    Widgets::make()->slider("Gamma", &gamma, 0.0f, 8.0f);
    Widgets::make()->slider("IBL", &ibl, 0.0f, 8.0f);
    Widgets::make()->slider("debugView", &debugViewInputs, 0, 6);
    Widgets::make()->slider("Skybox - LOD", &lod, 0, 10.0f);

    skyboxTextures.environmentMap.loadFromFile(Utils::getAssetsPath() + "Textures/Environments/papermill.ktx", renderUtils.device);

    skybox->createSkybox(envShaders, renderUtils.uniformBuffers, renderUtils.renderPass, &skyboxTextures);
    sharedData->destination = "All";
}

void Skybox::draw(VkCommandBuffer commandBuffer, uint32_t i, bool primaryDraw) {
    if (primaryDraw)
        skybox->drawSkybox(commandBuffer, i);
}

void Skybox::update() {
    auto &d = bufferOneData;
    d->model = glm::mat4(glm::mat3(renderData.camera->matrices.view));
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;


    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;


    d2->exposure = exposure;
    d2->gamma = gamma;
    d2->debugViewInputs = 0;
    d2->debugViewEquation = 0;
    d2->scaleIBLAmbient = ibl;
    d2->debugViewInputs = debugViewInputs;
    d2->debugViewEquation = lod;

    sharedData->put(d2.get(), sizeof(VkRender::FragShaderParams));

}


void Skybox::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != CRL_STATE_ACTIVE)
            continue;

    }
}