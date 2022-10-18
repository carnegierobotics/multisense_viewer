//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"
#include <glm/gtx/string_cast.hpp>


void PointCloud::setup() {
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    model->draw = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");
    model->modelType = AR_POINT_CLOUD;

}

void PointCloud::update() {

    if (renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf.width() != width) {
        model->draw = false;
        Log::Logger::getInstance()->info("Size mismatch with image size and imgconf in pointcloud");
        prepareTexture();
        return;
    }

    if (model->draw) {
        const auto& conf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
        auto tex = std::make_unique<VkRender::TextureData>(AR_POINT_CLOUD, conf.width(), conf.height());
        model->getTextureDataPointer(tex.get());
        if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Left", tex.get(), remoteHeadIndex)) {
            model->updateTexture(tex->m_Type);
        }

        auto depthTex = std::make_unique<VkRender::TextureData>(AR_DISPARITY_IMAGE, conf.width(), conf.height());
        model->getTextureDataPointer(depthTex.get());
        if (renderData.crlCamera->get()->getCameraStream("Disparity Left", depthTex.get(), remoteHeadIndex)) {
            model->updateTexture(depthTex->m_Type);
        }
    }
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 0.0f));

    // 24 degree rotation to compensate for MultiSense S27 24 degree camera slant.
    //mat.model = glm::rotate(mat.model, glm::radians(24.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    //mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    //mat.model = glm::translate(mat.model, glm::vec3(2.8, 0.4, -5));
    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;
}


void PointCloud::onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: *uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        selectedPreviewTab = dev.selectedPreviewTab;

        auto &preview = dev.win.at(AR_PREVIEW_POINT_CLOUD);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedMode;

        if ((currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            prepareTexture();
        }
    }
}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && selectedPreviewTab == TAB_3D_POINT_CLOUD)
        CRLCameraModels::draw(commandBuffer, i, model.get(), b);
}


void PointCloud::prepareTexture() {
    auto imgConf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
    width = imgConf.width();
    height = imgConf.height();
    std::vector<VkRender::Vertex> meshData{};
    meshData.resize(width * height);
    int v = 0;
    // first few rows and cols (20) are discarded in the shader anyway
    for (int i = 20; i < width - 20; ++i) {
        for (int j = 20; j < height - 20; ++j) {
            meshData[v].pos = glm::vec3((float) i, (float) j, 0.0f);
            meshData[v].uv0 = glm::vec2(1.0f - ((float) i / (float) width), 1.0f - ((float) j / (float) height));
            v++;
        }
    }
    model->createMeshDeviceLocal(meshData);

    renderData.crlCamera->get()->preparePointCloud(width, height);
    model->createEmtpyTexture(width, height, AR_POINT_CLOUD);
    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    CRLCameraModels::createRenderPipeline(shaders, model.get(), &renderUtils);
    auto *buf = (VkRender::PointCloudParam *) bufferThreeData.get();
    buf->kInverse = renderData.crlCamera->get()->getCameraInfo(0).kInverseMatrix;
    buf->height = static_cast<float>(height);
    buf->width = static_cast<float>(width);
    model->draw = true;
}