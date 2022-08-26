//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"
#include <glm/gtx/string_cast.hpp>


void PointCloud::setup(Base::Render r) {

    model = new CRLCameraModels::Model(renderUtils.device, AR_POINT_CLOUD);
    model->draw = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");

    for (auto dev: *r.gui) {
        if (dev.streams.find(AR_PREVIEW_POINT_CLOUD) == dev.streams.end()  || dev.state != AR_STATE_ACTIVE )
            continue;

        auto opt = dev.streams.find(AR_PREVIEW_POINT_CLOUD)->second;
        r.crlCamera->get()->camPtr->start(opt.selectedStreamingMode, "Disparity Left");
    }


}


void PointCloud::update(CameraConnection *conn) {
    if (playbackSate != AR_PREVIEW_PLAYING && TAB_3D_POINT_CLOUD == selectedPreviewTab) return;
    CRLBaseInterface *camPtr = conn->camPtr;

    if (model->draw == false) {

        auto imgConf = camPtr->getCameraInfo().imgConf;
        camPtr->preparePointCloud(imgConf.width(), imgConf.height());

        model->prepareTextureImage(imgConf.width(), imgConf.height(), AR_POINT_CLOUD);

        VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

        auto *buf = (PointCloudParam *) bufferThreeData;
        buf->kInverse = camPtr->getCameraInfo().kInverseMatrix;
        buf->height = static_cast<float>(imgConf.height());
        buf->width = static_cast<float>(imgConf.width());
        std::cout << glm::to_string( buf->kInverse) << std::endl;
        model->draw = true;

    }


    if (model->draw) {
        //CRLBaseCamera::PointCloudData *meshData = camera->getStream();
        crl::multisense::image::Header disp;
        //camPtr->getCameraStream(nullptr);
        model->setGrayscaleTexture(&disp);

        const int vertexCount = 960 * 600;
        auto *meshData = new ArEngine::Vertex[vertexCount]; // Don't forget to delete [] a; when you're done!

        int v = 0;
        for (int i = 0; i < 960; ++i) {
            for (int j = 0; j < 600; ++j) {
                meshData[v].pos = glm::vec3((float) i / 100.0f, (float) j / 100.0f, 0.0f);
                meshData[v].uv0 = glm::vec2((float) 1 - ((float) i / 960.0f), (float) 1 - ((float) j / 600.0f));
                v++;
            }
        }

        model->createMesh((ArEngine::Vertex *) meshData, vertexCount);

        delete[] meshData;
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


void PointCloud::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        if (dev.streams.find(AR_PREVIEW_POINT_CLOUD) == dev.streams.end())
            continue;

        playbackSate = dev.streams.find(AR_PREVIEW_POINT_CLOUD)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;
    }

}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_3D_POINT_CLOUD)
        CRLCameraModels::draw(commandBuffer, i, model);
}