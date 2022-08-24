//
// Created by magnus on 6/27/22.
//


#include <execution>
#include "DecodeVideo.h"
#include "GLFW/glfw3.h"


void DecodeVideo::setup(Base::Render r) {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, AR_CAMERA_DATA_COLOR_IMAGE);
    // Don't draw it before we create the texture in update()
    model->draw = false;
    this->camHandle = r.crlCamera->get();

    start = std::chrono::steady_clock::now();

}


void DecodeVideo::update() {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(time - start);
    if ((time_span.count() < 0.015f)){
        return;
    }
    start = std::chrono::steady_clock::now();


    if (model->draw) {
        crl::multisense::image::Header stream;
        ArEngine::MP4Frame frame{};

        bool ret = camHandle->camPtr->getCameraStream(&frame);

        if (ret)
            model->setColorTexture(&frame);

        free(frame.plane0);
        free(frame.plane1);
        free(frame.plane2);
    }

    transformToUISpace();
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);


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

void DecodeVideo::prepareTextureAfterDecode() {
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;
    vertexShaderFileName = "myScene/spv/quad.vert";
    fragmentShaderFileName = "myScene/spv/quad.frag";

    auto inf = camHandle->camPtr->getCameraInfo();

    width = inf.imgConf.width();
    height = inf.imgConf.height();

    model->prepareTextureImage(width, height, AR_YUV_PLANAR_FRAME);
    auto *imgData = new ImageData(posXMin, posXMax, posYMin, posYMax);


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

}

void DecodeVideo::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        if (dev.streams.find(AR_PREVIEW_VIRTUAL) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        src = dev.streams.find(AR_PREVIEW_VIRTUAL)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_VIRTUAL)->second.playbackStatus;

    }

    if (playbackSate == AR_PREVIEW_PLAYING) {

        switch (uiHandle.keypress) {
            case GLFW_KEY_M:
                speed -= 0.01;
                break;
            case GLFW_KEY_N:
                speed += 0.01;
                break;
        }


        posY -= uiHandle.mouseBtns.wheel * 0.1f * (1/ speed);

        // center of viewing area box.

        //posX =  2*;

        for (auto &dev: *uiHandle.devices) {
            if (dev.cameraName == "Virtual Camera" && !model->draw) {

                camHandle->camPtr->start(src, " ");
                posXMin = -1 + 2*((uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + 40.0f) / (float) renderData.width);
                posXMax = (uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + uiHandle.info->viewingAreaWidth - 80.0f) / (float) renderData.width;

                posYMin = -1.0f + 2*(75.0f / (float) renderData.height);
                posYMax = -1.0f + 2*((75.0f + 300.0f) / (float) renderData.height);
                // left anchor
                prepareTextureAfterDecode();


            }
            model->draw = true;
        }
    } else if (playbackSate == AR_PREVIEW_STOPPED && model->draw == true) {
        model->draw = false;
        camHandle->camPtr->stop("");

    }

}


void DecodeVideo::transformToUISpace(){


    int width = 1280;
    int height = 720;

    //scaleHorizontal = renderData.width / width;
    //scaleVertical = renderData.height / height;

}


void DecodeVideo::draw(VkCommandBuffer commandBuffer, uint32_t i) {

    if (model->draw && playbackSate != AR_PREVIEW_NONE)
        CRLCameraModels::draw(commandBuffer, i, model);
}
