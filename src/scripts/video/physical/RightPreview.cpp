//
// Created by magnus on 7/8/22.
//

#include "RightPreview.h"
#include "GLFW/glfw3.h"

void RightPreview::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, AR_CAMERA_DATA_IMAGE);

    // Don't draw it before we create the texture in update()
    model->draw = false;
    for (auto dev: *r.gui) {
        if (dev.streams.find(AR_PREVIEW_RIGHT) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        auto opt = dev.streams.find(AR_PREVIEW_RIGHT)->second;
        r.crlCamera->get()->camPtr->start(opt.selectedStreamingMode, opt.selectedStreamingSource);
    }
}


void RightPreview::update(CameraConnection *conn) {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    auto *camera = conn->camPtr;
    assert(camera != nullptr);


    if (!model->draw && coordinateTransformed) {
        auto imgConf = camera->getCameraInfo().imgConf;


        std::string vertexShaderFileName;
        std::string fragmentShaderFileName;

        vertexShaderFileName = "myScene/spv/depth.vert";
        fragmentShaderFileName = "myScene/spv/depth.frag";

        model->prepareTextureImage(imgConf.width(), imgConf.height(), AR_GRAYSCALE_IMAGE);

        auto *imgData = new ImageData(posXMin, posXMax, posYMin, posYMax);

        //auto *imgData = new ImageData(((float) imgConf.width() / (float) imgConf.height()), 1);


        // Load shaders
        VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        // Create quad and store it locally on the GPU
        model->createMeshDeviceLocal((ArEngine::Vertex *) imgData->quad.vertices,
                                     imgData->quad.vertexCount, imgData->quad.indices, imgData->quad.indexCount);


        // Create graphics render pipeline
        CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

        model->draw = true;
    }

    if (model->draw) {
        auto* image = new crl::multisense::image::Header();
        camera->getCameraStream(src, image);
        model->setGrayscaleTexture(image);

        delete image;
    }


    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));

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


void RightPreview::onUIUpdate(AR::GuiObjectHandles uiHandle) {

    for (const auto &dev: *uiHandle.devices) {

        if (dev.streams.find(AR_PREVIEW_RIGHT) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        src = dev.streams.find(AR_PREVIEW_RIGHT)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_RIGHT)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;


    }

    if (playbackSate == AR_PREVIEW_PLAYING) {
        posY -= (float) uiHandle.mouseBtns.wheel * 0.1f * 0.557 * (720.0f / (float)renderData.height);
        // center of viewing area box.

        //posX =  2*;

        for (auto &dev: *uiHandle.devices) {
            if (prevOrder != dev.streams.find(AR_PREVIEW_RIGHT)->second.streamingOrder) {
                transformToUISpace(uiHandle, dev);

            }
            prevOrder = dev.streams.find(AR_PREVIEW_RIGHT)->second.streamingOrder;

            if (!model->draw && !coordinateTransformed) {
                renderData.crlCamera->get()->camPtr->start(src, AR_PREVIEW_RIGHT);

                transformToUISpace(uiHandle, dev);
                coordinateTransformed = true;

            }
        }
    }


    //printf("Pos %f, %f, %f\n", posX, posY, posZ);

}

void RightPreview::transformToUISpace(AR::GuiObjectHandles uiHandle, AR::Element dev) {
    posXMin = -1 + 2*((uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + 40.0f) / (float) renderData.width);
    posXMax = (uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + uiHandle.info->viewingAreaWidth - 80.0f) / (float) renderData.width;

    int order = dev.streams.find(AR_PREVIEW_RIGHT)->second.streamingOrder;
    float orderOffset =  uiHandle.info->viewAreaElementPositionsY[order];

    posYMin = -1.0f + 2*(orderOffset / (float) renderData.height);
    posYMax = -1.0f + 2*((uiHandle.info->viewAreaElementSizeY + (orderOffset)) / (float) renderData.height);                // left anchor

}


void RightPreview::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model);

}

