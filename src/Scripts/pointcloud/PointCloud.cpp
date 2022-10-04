//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"
#include <glm/gtx/string_cast.hpp>


void PointCloud::setup(Base::Render r) {
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    model->draw = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");
}

void PointCloud::update() {

    if (renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf.width() != width) {
        model->draw = false;
        return;
    }

    if (model->draw) {
        auto *tex = new VkRender::TextureData(AR_DISPARITY_IMAGE);
        if (renderData.crlCamera->get()->getCameraStream(src, tex, remoteHeadIndex)) {
            model->setTexture(tex);
            free(tex->data);
        }

        auto *tex2 = new VkRender::TextureData(AR_POINT_CLOUD);
        if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Left", tex2, remoteHeadIndex)) {
            model->setTexture(tex2);
            free(tex2->data);
        }
        delete tex;
        delete tex2;
    }
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 0.0f));

    // 24 degree rotation to compensate for MultiSense S27 24 degree camera slant.
    //mat.model = glm::rotate(mat.model, glm::radians(24.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    //mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    //mat.model = glm::translate(mat.model, glm::vec3(2.8, 0.4, -5));
    auto *d = (VkRender::UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (VkRender::FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;
}


void PointCloud::onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: *uiHandle->devices) {

        auto &preview = dev.win.at(AR_PREVIEW_POINT_CLOUD);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedMode;
        if (preview.selectedSource == "Source") {
            // dont draw or update
            model->draw = false;
        }


        if ((src != preview.selectedSource || currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            src = preview.selectedSource;
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            selectedPreviewTab = dev.selectedPreviewTab;
            textureType = AR_POINT_CLOUD;
            prepareTexture();
        }
    }
}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && selectedPreviewTab == TAB_3D_POINT_CLOUD)
        CRLCameraModels::draw(commandBuffer, i, model.get(), b);
}


void PointCloud::prepareTexture() {
    model->modelType = textureType;
    if (textureType == AR_CAMERA_IMAGE_NONE)
        return;

    auto imgConf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
    width = imgConf.width();
    height = imgConf.height();
    meshData = new VkRender::Vertex[width * height];
    int v = 0;
    // first few rows and cols (20) are discarded in the shader anyway
    for (int i = 20; i < width-20; ++i) {
        for (int j = 20; j < height - 20; ++j) {
            meshData[v].pos = glm::vec3((float) i, (float) j, 0.0f);
            meshData[v].uv0 = glm::vec2(1.0f- ((float) i / (float) width), 1.0f - ((float) j / (float) height));
            v++;
        }
    }
    const uint32_t vtxBufSize = width * height;
    model->createMeshDeviceLocal((VkRender::Vertex *) meshData, vtxBufSize, nullptr, 0);
    delete[] meshData;

    renderData.crlCamera->get()->preparePointCloud(width, height);
    model->createEmtpyTexture(width, height, textureType);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};

    CRLCameraModels::createRenderPipeline(shaders, model.get(), type, &renderUtils);

    auto *buf = (VkRender::PointCloudParam *) bufferThreeData;
    buf->kInverse = renderData.crlCamera->get()->getCameraInfo(0).kInverseMatrix;
    buf->height = static_cast<float>(height);
    buf->width = static_cast<float>(width);
    std::cout << glm::to_string(buf->kInverse) << std::endl;

    model->draw = true;
}