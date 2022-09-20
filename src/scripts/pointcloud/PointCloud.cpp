//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"
#include <glm/gtx/string_cast.hpp>


void PointCloud::setup(Base::Render r) {

    model = new CRLCameraModels::Model(renderUtils.device, AR_POINT_CLOUD, nullptr);
    model->draw = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");


    const int vertexCount = 960 * 600;
    meshData = new ArEngine::Vertex[vertexCount]; // Don't forget to delete [] when you're done!

    int v = 0;
    for (int i = 0; i < 960; ++i) {
        for (int j = 0; j < 600; ++j) {
            meshData[v].pos = glm::vec3((float) i, (float) j, 0.0f);
            meshData[v].uv0 = glm::vec2((float) 1 - ((float) i / 960.0f), (float) 1 - ((float) j / 600.0f));
            v++;
        }
    }

    model->createMesh((ArEngine::Vertex *) meshData, vertexCount);

}


void PointCloud::update() {
    if (playbackSate != AR_PREVIEW_PLAYING || TAB_3D_POINT_CLOUD != selectedPreviewTab)
        return;
    /*
    CRLBaseInterface *camPtr = conn->camPtr;

    if (camPtr->getCameraInfo().imgConf.width() != width){
        model->draw = false;
    }

    if (model->draw == false) {
        auto imgConf = camPtr->getCameraInfo().imgConf;
        width = imgConf.width();
        height = imgConf.height();

        camPtr->preparePointCloud(width, height);
        model->createEmtpyTexture(width, height, AR_POINT_CLOUD);

        VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

        auto *buf = (ArEngine::PointCloudParam *) bufferThreeData;
        buf->kInverse = camPtr->getCameraInfo().kInverseMatrix;
        buf->height = static_cast<float>(height);
        buf->width = static_cast<float>(width);
        std::cout << glm::to_string(buf->kInverse) << std::endl;
        model->draw = true;

    }


    if (model->draw) {
        auto *tex = new ArEngine::TextureData();
        if (camPtr->getCameraStream(src, tex)) {
            model->setGrayscaleTexture(tex, AR_GRAYSCALE_IMAGE);
            free(tex->data);
        }

        if (camPtr->getCameraStream("Luma Rectified Left", tex)) {
            model->setGrayscaleTexture(tex, AR_POINT_CLOUD);
            free(tex->data);
        }
        delete tex;

    }
     */

    ArEngine::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 0.0f));

    // 24 degree rotation to compensate for MultiSense S27 24 degree camera slant.
    mat.model = glm::rotate(mat.model, glm::radians(24.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    //mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    //mat.model = glm::translate(mat.model, glm::vec3(2.8, 0.4, -5));
    auto *d = (ArEngine::UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (ArEngine::FragShaderParams *) bufferTwoData;
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

        src = dev.streams.find(AR_PREVIEW_POINT_CLOUD)->second.selectedStreamingSource;

        playbackSate = dev.streams.find(AR_PREVIEW_POINT_CLOUD)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;
    }

}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_3D_POINT_CLOUD)
        CRLCameraModels::draw(commandBuffer, i, model, false);
}
