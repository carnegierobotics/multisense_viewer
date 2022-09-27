//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"
#include <glm/gtx/string_cast.hpp>


void PointCloud::setup(Base::Render r) {
    model = new CRLCameraModels::Model(&renderUtils);
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

    if (renderData.crlCamera->get()->getCameraInfo().imgConf.width() != width) {
        model->draw = false;
        return;
    }

    if (model->draw) {
        auto *tex = new ArEngine::TextureData(AR_DISPARITY_IMAGE);
        if (renderData.crlCamera->get()->getCameraStream(src, tex)) {
            model->setTexture(tex);
            free(tex->data);
        }

        auto *tex2 = new ArEngine::TextureData(AR_POINT_CLOUD);
        if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Left", tex2)) {
            model->setTexture(tex2);
            free(tex2->data);
        }
        delete tex;
        delete tex2;
    }
    ArEngine::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 0.0f));

    // 24 degree rotation to compensate for MultiSense S27 24 degree camera slant.
    //mat.model = glm::rotate(mat.model, glm::radians(24.0f), glm::vec3(1.0f, 0.0f, 0.0f));
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

        if (!dev.selectedSourceMap.contains(AR_PREVIEW_POINT_CLOUD))
            break;

        src = dev.selectedSourceMap.at(AR_PREVIEW_POINT_CLOUD);

        if ((dev.selectedMode != res ||
             dev.selectedPreviewTab != selectedPreviewTab))
        {
            textureType = AR_POINT_CLOUD;
            res = dev.selectedMode;
            selectedPreviewTab = dev.selectedPreviewTab;
            prepareTexture();
        }
    }
}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && selectedPreviewTab == TAB_3D_POINT_CLOUD)
        CRLCameraModels::draw(commandBuffer, i, model, b);
}


void PointCloud::prepareTexture() {
    model->modelType = textureType;
    auto imgConf = renderData.crlCamera->get()->getCameraInfo().imgConf;
    width = imgConf.width();
    height = imgConf.height();

    renderData.crlCamera->get()->preparePointCloud(width, height);
    model->createEmtpyTexture(width, height, textureType);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};

    CRLCameraModels::createRenderPipeline(shaders, model, type, &renderUtils);

    auto *buf = (ArEngine::PointCloudParam *) bufferThreeData;
    buf->kInverse = renderData.crlCamera->get()->getCameraInfo().kInverseMatrix;
    buf->height = static_cast<float>(height);
    buf->width = static_cast<float>(width);
    std::cout << glm::to_string(buf->kInverse) << std::endl;

    model->draw = true;
}