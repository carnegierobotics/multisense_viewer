//
// Created by magnus on 3/1/22.
//

#include "MultiSenseS30.h"

void MultiSenseS30::setup() {
    model = new MeshModel::Model(1, renderUtils.device);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    camera = new CRLVirtualCamera();
    camera->initialize(CrlImage);
    CRLBaseCamera::ImageData *imgData = camera->getImageData();
    model->createMeshDeviceLocal((MeshModel::Model::Vertex *) imgData->quad.vertices, imgData->quad.vertexCount,
                                 imgData->quad.indices, imgData->quad.indexCount);

    model->setTexture(Utils::getTexturePath() + "moon.jpg", false);

    MeshModel::createRenderPipeline(renderUtils, shaders, model, type);
}

int count = 1;
void MultiSenseS30::update() {
    //camera->update(renderData);

    /*
    int runTimeInMS = renderData.runTime * 1000;
    if (runTimeInMS % 500  > 2){
        model->setTexture(Utils::getTexturePath() + "Video/earth/ezgif-frame-00" + std::to_string(count) + ".jpg",false);
        count += 1;
        if (count >= 10)
            count = 1;

    }
*/

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
}


void MultiSenseS30::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    MeshModel::draw(commandBuffer, i, model);

}
