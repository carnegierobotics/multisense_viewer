//
// Created by magnus on 8/25/22.
//

#include "AuxImager.h"

#include <execution>
#include "GLFW/glfw3.h"


void AuxImager::setup(Base::Render r) {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(nullptr);
    // Don't draw it before we create the texture in update()
    model->draw = false;

    start = std::chrono::steady_clock::now();

}


void AuxImager::update() {
    if (playbackSate != AR_PREVIEW_PLAYING || camHandle->camPtr == nullptr)
        return;

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span = std::chrono::duration_cast<std::chrono::duration<float>>(time - start);
    if ((time_span.count() < 0.015f)){
        return;
    }
    start = std::chrono::steady_clock::now();


    if (model->draw) {
        crl::multisense::image::Header stream;
        VkRender::MP4Frame frame{};

        /*
        bool ret = camHandle->camPtr->getCameraStream(&frame, AR_PREVIEW_VIRTUAL_AUX);

        if (ret)
            model->setTexture(&frame);

         */
        free(frame.plane0);
        free(frame.plane1);
        free(frame.plane2);
    }

    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX * (1 /scaleX), centerY * (1 /scaleY), 0.0f));

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

void AuxImager::prepareTextureAfterDecode() {
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
    model->createMeshDeviceLocal((VkRender::Vertex *) imgData.quad.vertices,
                                 imgData.quad.vertexCount, imgData.quad.indices, imgData.quad.indexCount);

    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, model, type, nullptr);

}


void AuxImager::onUIUpdate(MultiSense::GuiObjectHandles uiHandle) {
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        /*
        //if (dev.streams.find(AR_PREVIEW_VIRTUAL_AUX) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
        //    continue;

        src = dev.streams.find(AR_PREVIEW_VIRTUAL_AUX)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_VIRTUAL_AUX)->second.playbackStatus;

    }

    if (playbackSate == AR_PREVIEW_PLAYING) {
        if (input->getButton(GLFW_KEY_LEFT_CONTROL)){
            std::cout << "Btn down: " << uiHandle.mouseBtns->wheel / 100.0f << std::endl;
            auto * d2 = (VkRender::ZoomParam *) bufferFourData;
            d2->zoom = uiHandle.mouseBtns->wheel / 1000.0f;

        } else if (!uiHandle.info->hoverState ) {
            posY = (float) uiHandle.accumulatedActiveScroll * 0.05f * 0.1f * 0.557f * (720.0f / (float) renderData.height);
        }
        // center of viewing area box.

        //posX =  2*;

        for (auto &dev: *uiHandle.devices) {
            if (dev.state != AR_STATE_ACTIVE)
                continue;
            if (prevOrder != dev.streams.find(AR_PREVIEW_VIRTUAL_AUX)->second.streamingOrder) {
                transformToUISpace(uiHandle, dev);
                prepareTextureAfterDecode();
            }
            prevOrder = dev.streams.find(AR_PREVIEW_VIRTUAL_AUX)->second.streamingOrder;

            if (dev.cameraName == "Virtual Camera" && !model->draw) {
                camHandle->camPtr->start(src, AR_PREVIEW_VIRTUAL_AUX);

                transformToUISpace(uiHandle, dev);

                prepareTextureAfterDecode();


            }
            model->draw = true;
        }*/
    }

}


void AuxImager::transformToUISpace(MultiSense::GuiObjectHandles uiHandle, MultiSense::Device dev) {
    posXMin = -1 + 2*((uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth) / (float) renderData.width);
    posXMax = (uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + uiHandle.info->viewingAreaWidth) / (float) renderData.width;

    // 1280x 720 is the original aspect

    float scaleUniform = ((float) renderData.width/ 1280.0f); // Scale by width of screen.
    centerX = (posXMax - posXMin) / 2 + posXMin; // center of the quad in the given view area.
    scaleX = (1280.0f / (float) renderData.width) * 0.25f * scaleUniform;
    scaleY = (720.0f / (float) renderData.height) * 0.25f * scaleUniform;

    /*
    int order = dev.streams.find(AR_PREVIEW_VIRTUAL_AUX)->second.streamingOrder;
    float orderOffset = uiHandle.info->viewAreaElementPositionsY[order] - uiHandle.accumulatedActiveScroll;
    posYMin = -1.0f + 2*(orderOffset / (float) renderData.height);
    //posYMax = -1.0f + 2*((uiHandle.info->viewAreaElementSizeY + (orderOffset)) / (float) renderData.height);                // left anchor
    centerY = (posYMax - posYMin) / 2 + posYMin;
    */
     }


void AuxImager::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {

    if (model->draw && playbackSate != AR_PREVIEW_NONE)
        CRLCameraModels::draw(commandBuffer, i, model, false);
}


void AuxImager::onWindowResize(MultiSense::GuiObjectHandles uiHandle) {
    for (auto &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        transformToUISpace(uiHandle, dev);
        model->draw = false;
    }
}