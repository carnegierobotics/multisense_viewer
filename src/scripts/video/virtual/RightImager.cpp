//
// Created by magnus on 8/24/22.
//


#include <execution>
#include "RightImager.h"
#include "GLFW/glfw3.h"


void RightImager::setup(Base::Render r) {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, AR_CAMERA_DATA_IMAGE);
    // Don't draw it before we create the texture in update()
    model->draw = false;
    this->camHandle = r.crlCamera->get();

    start = std::chrono::steady_clock::now();

}


void RightImager::update() {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(time - start);
    if ((time_span.count() < 0.015f)) {
        return;
    }
    start = std::chrono::steady_clock::now();


    if (model->draw) {
        crl::multisense::image::Header stream;
        ArEngine::MP4Frame frame{};

        // Todo On destructor of connected device in sidebar, camPtr is set to nullptr. This function update() should not get called then.
        bool ret = camHandle->camPtr->getCameraStream(&frame, AR_PREVIEW_VIRTUAL_RIGHT);

        if (ret)
            model->setColorTexture(&frame);

        free(frame.plane0);
        free(frame.plane1);
        free(frame.plane2);
    }

    ArEngine::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX * (1 /scaleX), centerY * (1 /scaleY), 0.0f));

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

void RightImager::prepareTextureAfterDecode() {
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;
    vertexShaderFileName = "myScene/spv/quad.vert";
    fragmentShaderFileName = "myScene/spv/quad.frag";

    auto inf = camHandle->camPtr->getCameraInfo();

    width = inf.imgConf.width();
    height = inf.imgConf.height();

    model->createEmtpyTexture(width, height, AR_YUV_PLANAR_FRAME);
    //auto *imgData = new ImageData(posXMin, posXMax, posYMin, posYMax);

    ImageData imgData;


    // Load shaders
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    // Create quad and store it locally on the GPU
    model->createMeshDeviceLocal((ArEngine::Vertex *) imgData.quad.vertices,
                                 imgData.quad.vertexCount, imgData.quad.indices, imgData.quad.indexCount);

    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

}


void RightImager::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        if (dev.streams.find(AR_PREVIEW_VIRTUAL_RIGHT) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        src = dev.streams.find(AR_PREVIEW_VIRTUAL_RIGHT)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_VIRTUAL_RIGHT)->second.playbackStatus;


    }

    if (playbackSate == AR_PREVIEW_PLAYING) {
        posY = uiHandle.mouseBtns->wheel * 0.05 * 0.1f * 0.557 * (720.0f / (float)renderData.height);

        for (auto &dev: *uiHandle.devices) {
            if (dev.state != AR_STATE_ACTIVE)
                continue;
            if (prevOrder != dev.streams.find(AR_PREVIEW_VIRTUAL_RIGHT)->second.streamingOrder) {
                transformToUISpace(uiHandle, dev);

                prepareTextureAfterDecode();
            }
            prevOrder = dev.streams.find(AR_PREVIEW_VIRTUAL_RIGHT)->second.streamingOrder;

            if (dev.cameraName == "Virtual Camera" && !model->draw) {
                camHandle->camPtr->start(src, AR_PREVIEW_VIRTUAL_RIGHT);

                transformToUISpace(uiHandle, dev);

                prepareTextureAfterDecode();


            }
            model->draw = true;
        }
    }
}


void RightImager::transformToUISpace(AR::GuiObjectHandles uiHandle, AR::Element dev) {
    posXMin = -1 + 2*((uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth) / (float) renderData.width);
    posXMax = (uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + uiHandle.info->viewingAreaWidth) / (float) renderData.width;

    // 1280x 720 is the original aspect

    float scaleUniform = ((float) renderData.width/ 1280.0f); // Scale by width of screen.
    centerX = (posXMax - posXMin) / 2 + posXMin; // center of the quad in the given view area.
    scaleX = (1280.0f / (float) renderData.width) * 0.25f * scaleUniform;
    scaleY = (720.0f / (float) renderData.height) * 0.25f * scaleUniform;

    int order = dev.streams.find(AR_PREVIEW_VIRTUAL_RIGHT)->second.streamingOrder;
    float orderOffset =  uiHandle.info->viewAreaElementPositionsY[order] - (uiHandle.mouseBtns->wheel );
    posYMin = -1.0f + 2*(orderOffset / (float) renderData.height);
    posYMax = -1.0f + 2*((uiHandle.info->viewAreaElementSizeY + (orderOffset)) / (float) renderData.height);                // left anchor
    centerY = (posYMax - posYMin) / 2 + posYMin;          // left anchor

}


void RightImager::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE)
        CRLCameraModels::draw(commandBuffer, i, model);
}


void RightImager::onWindowResize(AR::GuiObjectHandles uiHandle) {
    for (auto &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        transformToUISpace(uiHandle, dev);
        model->draw = false;
    }
}
