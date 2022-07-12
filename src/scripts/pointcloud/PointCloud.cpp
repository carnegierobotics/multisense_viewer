//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"


void PointCloud::setup() {

    model = new CRLCameraModels::Model(renderUtils.device, CrlPointCloud);
    model->draw = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");


}


void PointCloud::update(CameraConnection *conn) {
    CRLBaseInterface *camPtr = conn->camPtr;

    if (model->draw == false) {

        auto imgConf = camPtr->getCameraInfo().imgConf;
        camPtr->preparePointCloud(imgConf.width(), imgConf.height());

        model->prepareTextureImage(imgConf.width(), imgConf.height(), CrlPointCloud);

        VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        renderUtils.shaders = shaders;

        CRLCameraModels::createRenderPipeline(renderUtils, renderUtils.shaders, model, type);

        auto *buf = (PointCloudParam *) bufferThreeData;
        buf->kInverse = camPtr->kInverseMatrix;
        buf->height = imgConf.height();
        buf->width = imgConf.width();

        model->draw = true;

    }


    if (model->draw) {
        //CRLBaseCamera::PointCloudData *meshData = camera->getStream();
        crl::multisense::image::Header *disp;
        camPtr->getCameraStream("Disparity Left", &disp);
        model->setGrayscaleTexture(disp);

        const int vertexCount = 960 * 600;

        ArEngine::Vertex* meshData = new ArEngine::Vertex[vertexCount]; // Don't forget to delete [] a; when you're done!

        int v = 0;
        for (int i = 0; i < 960; ++i) {
            for (int j = 0; j < 600; ++j) {
                meshData[v].pos = glm::vec3((float) i / 100.0f, (float) j / 100.0f, 0.0f);
                meshData[v].uv0 = glm::vec2((float) i / 960.0f, (float) j / 600.0f);

                v++;
            }
        }


        model->createMesh((ArEngine::Vertex *) meshData, vertexCount);

        free(meshData);
    }

    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, -5.0f));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    //mat.model = glm::translate(mat.model, glm::vec3(2.8, 0.4, -5));
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


void PointCloud::onUIUpdate(GuiObjectHandles uiHandle) {
    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        if (dev.streams.find(PREVIEW_DISPARITY) == dev.streams.end())
            continue;

        playbackSate = dev.streams.find(PREVIEW_DISPARITY)->second.playbackStatus;

        if (dev.selectedPreviewTab != TAB_3D_POINTCLOUD)
            playbackSate = PREVIEW_NONE;
        else
            playbackSate = PREVIEW_PLAYING;
    }

}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != PREVIEW_NONE)
        CRLCameraModels::draw(commandBuffer, i, model);
}