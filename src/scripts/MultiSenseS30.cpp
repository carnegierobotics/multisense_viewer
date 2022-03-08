//
// Created by magnus on 3/1/22.
//

#include "MultiSenseS30.h"

void MultiSenseS30::setup() {
    /**
     * Create UI Elements
     */

    // UI cretion


    renderUtils.ui->createButton({"Connect Camera", 175.0f, 30.0f});


    /**
     * Create and load Mesh elements
     */
    model = new MeshModel::Model(1, renderUtils.device);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    camera = new CRLPhysicalCamera(CrlImage);

    CRLBaseCamera::ImageData *imgData = camera->getImageData();
    model->createMeshDeviceLocal((MeshModel::Model::Vertex *) imgData->quad.vertices, imgData->quad.vertexCount,
                                 imgData->quad.indices, imgData->quad.indexCount);

    //model->setVideoTexture( Utils::getTexturePath() + "Video/earth/ezgif-frame-001.jpg");

    model->loadTextures(); // TODO TEMP-remove it

    MeshModel::createRenderPipeline(renderUtils, shaders, model, type);
}


void MultiSenseS30::update() {
    //camera->update(renderData);

    int runTimeInMS = (int) (renderData.runTime * 1000);
    if ((runTimeInMS % 50) < 20) {

        model->setVideoTexture("Utils::getTexturePath() + file");
        count += 1;
        if (count >= 101)
            count = 1;

    }


    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(5.0f, -5.0f, -5.0f));

    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    mat.model = glm::scale(mat.model, glm::vec3(5.0f, 5.0f, 5.0f));


    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;


}

void MultiSenseS30::onUIUpdate(UISettings uiSettings) {

    if (uiSettings.buttons.empty())
        return;

    for (auto &button: uiSettings.buttons) {
        if (button.name == "Connect Camera")
            if (button.clicked) {
                printf("%s: Clicked\n", button.name.c_str());
                button.clicked = false;
            }
    }
}


void MultiSenseS30::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    MeshModel::draw(commandBuffer, i, model);

}
