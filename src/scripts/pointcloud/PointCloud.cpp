//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"


void PointCloud::setup() {

    model = new CRLCameraModels::Model(renderUtils.device, CrlPointCloud);
    model->draw = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");


}


void PointCloud::update() {

    if (camera == nullptr)
        return;
    camera->update();

    if (camera->stream == nullptr)
        return;


    if (camera->stream->source != crl::multisense::Source_Disparity_Left)
        return;

    if (model->draw == false || camera->modeChange) {
        auto imgConf = camera->getImageConfig();

        model->prepareTextureImage(imgConf.width(), imgConf.height(), CrlDisparityImage);
        camera->setup(imgConf.width(), imgConf.height());

        VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        renderUtils.shaders = shaders;

        CRLCameraModels::createRenderPipeline(renderUtils, renderUtils.shaders, model, type);

        auto *buf = (PointCloudParam *) bufferThreeData;
        buf->kInverse = camera->kInverseMatrix;
        buf->height = imgConf.height();
        buf->width = imgConf.width();

        model->draw = true;
    }

    if (camera->play && model->draw) {
        CRLBaseCamera::PointCloudData *meshData = camera->getStream();
        model->setGrayscaleTexture(&camera->getImage()[crl::multisense::Source_Disparity_Left]);
        model->createMesh((CRLCameraModels::Model::Vertex *) meshData->vertices, meshData->vertexCount);
    }
    // Transform pointcloud
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    //mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
}


void PointCloud::onUIUpdate(GuiObjectHandles uiHandle) {
    //camera = (CRLPhysicalCamera *) uiSettings->physicalCamera;

}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (this->model->draw)
        CRLCameraModels::draw(commandBuffer, i, this->model);
}