//
// Created by magnus on 2/25/23.
//

#include "Viewer/Scripts/MultiSenseRenderer/SceneGizmos/Skybox.h"
#include "Viewer/ImGui/Widgets.h"

void Skybox::setup() {
    // Create Skybox
    skybox = std::make_unique<GLTFModel::Model>(renderUtils.device);
    std::vector<VkPipelineShaderStageCreateInfo> envShaders = {{loadShader("spv/filtercube.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("spv/irradiancecube.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("spv/prefilterenvmap.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("spv/genbrdflut.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("spv/genbrdflut.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)},
                                                               {loadShader("spv/skybox.vert",
                                                                           VK_SHADER_STAGE_VERTEX_BIT)},
                                                               {loadShader("spv/skybox.frag",
                                                                           VK_SHADER_STAGE_FRAGMENT_BIT)}};



    /*
    Widgets::make()->slider("default", "Exposure", &exposure, 0.0f, 25.0f);
    Widgets::make()->slider("default", "Gamma", &gamma, 0.0f, 8.0f);
    Widgets::make()->slider("default", "IBL", &ibl, 0.0f, 8.0f);
    Widgets::make()->slider("default", "debugView", &debugViewInputs, 0, 6);
    Widgets::make()->slider("default", "Skybox - LOD", &lod, 0, 10.0f);
    */
    skyboxTextures.environmentMap = std::make_shared<TextureCubeMap>();
    skyboxTextures.irradianceCube = std::make_shared<TextureCubeMap>();
    skyboxTextures.prefilterEnv = std::make_shared<TextureCubeMap>();
    skyboxTextures.lutBrdf = std::make_shared<Texture2D>();

    skyboxTextures.environmentMap->loadFromFile(Utils::getAssetsPath().append("Textures/Environments/skies.ktx2"),
                                                renderUtils.device);

    skybox->createSkybox(envShaders, renderUtils.uniformBuffers, renderUtils.renderPass, &skyboxTextures,
                         renderUtils.msaaSamples);
    sharedData->destination = "All";
}

void Skybox::draw(CommandBuffer *commandBuffer, uint32_t i, bool primaryDraw) {
    if (selectedPreviewTab == VkRender::CRL_TAB_3D_POINT_CLOUD && primaryDraw)
        skybox->drawSkybox(commandBuffer, i);
}

void Skybox::update() {
    auto &d = ubo[0].mvp;
    // we can remove the translation section of transformation matrices by taking the upper-left 3x3 matrix of the
    // 4x4 matrix. We can achieve this by converting the view matrix to a 3x3 matrix (removing translation)
    // and converting it back to a 4x4 matrix:
    d->model = glm::mat4(glm::mat3(renderData.camera->matrices.view));
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;


    auto &d2 = ubo[0].fragShader;


    d2->exposure = exposure;
    d2->gamma = gamma;
    d2->debugViewInputs = 0.0f;
    d2->scaleIBLAmbient = ibl;
    d2->lod = lod;

    sharedData->put(d2.get(), sizeof(VkRender::FragShaderParams));

}


void Skybox::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    for (const auto &d: uiHandle->devices) {
        if (d.state != VkRender::CRL_STATE_ACTIVE)
            continue;
        selectedPreviewTab = d.selectedPreviewTab;

    }
}