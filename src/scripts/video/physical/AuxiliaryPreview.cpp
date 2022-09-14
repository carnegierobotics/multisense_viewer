//
// Created by magnus on 7/11/22.
//

#include "AuxiliaryPreview.h"
#include "GLFW/glfw3.h"


void AuxiliaryPreview::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, AR_CAMERA_DATA_IMAGE);

    // Don't draw it before we create the texture in update()
    model->draw = false;
    for (auto dev: *r.gui) {
        if (dev.streams.find(AR_PREVIEW_AUXILIARY) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        auto opt = dev.streams.find(AR_PREVIEW_AUXILIARY)->second;
        r.crlCamera->get()->camPtr->start(opt.selectedStreamingMode, opt.selectedStreamingSource);
        startedSources.push_back(opt.selectedStreamingSource);

    }

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
}


void AuxiliaryPreview::update(CameraConnection *conn) {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    auto *camera = conn->camPtr;
    assert(camera != nullptr);

    if (camera->getCameraInfo().imgConf.width() != width){
        model->draw = false;
    }

    if (!model->draw && coordinateTransformed) {
        auto imgConf = camera->getCameraInfo().imgConf;


        std::string vertexShaderFileName;
        std::string fragmentShaderFileName;

        vertexShaderFileName = "myScene/spv/quad.vert";
        fragmentShaderFileName = "myScene/spv/quad.frag";

        width = imgConf.width();
        height = imgConf.height();

        model->createEmtpyTexture(width, height, AR_COLOR_IMAGE_YUV420);

        //ImageData imgData(((float) imgConf.width() / (float) imgConf.height()), 1);
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

        model->draw = true;
    }

    if (model->draw) {

        auto *tex = new ArEngine::YUVTexture();
        if (camera->getCameraStream(tex)) {
            model->setColorTexture(tex);
            free(tex->data[0]);
            free(tex->data[1]);

        }
        delete tex;
    }


    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX * (1 /scaleX), centerY * (1 /scaleY), 0.0f));

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


void AuxiliaryPreview::onUIUpdate(AR::GuiObjectHandles uiHandle) {

    for (const auto &dev: *uiHandle.devices) {

        if (dev.streams.find(AR_PREVIEW_AUXILIARY) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        src = dev.streams.find(AR_PREVIEW_AUXILIARY)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_AUXILIARY)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;


    }

    if (playbackSate == AR_PREVIEW_PLAYING) {


        posY = uiHandle.mouseBtns->wheel * 0.05 * 0.1f * 0.557 * (720.0f / (float)renderData.height);
        // center of viewing area box.

        //posX =  2*;

        for (auto &dev: *uiHandle.devices) {
            if (dev.state != AR_STATE_ACTIVE)
                continue;
            if (prevOrder != dev.streams.find(AR_PREVIEW_AUXILIARY)->second.streamingOrder) {
                transformToUISpace(uiHandle, dev);
                model->draw = false;
                coordinateTransformed = true;

            }
            prevOrder = dev.streams.find(AR_PREVIEW_AUXILIARY)->second.streamingOrder;

            if (!model->draw && !coordinateTransformed) {
                renderData.crlCamera->get()->camPtr->start(src, AR_PREVIEW_AUXILIARY);

                transformToUISpace(uiHandle, dev);
                coordinateTransformed = true;

            }
        }
    }


    //printf("Pos %f, %f, %f\n", posX, posY, posZ);

}

void AuxiliaryPreview::transformToUISpace(AR::GuiObjectHandles uiHandle, AR::Element dev) {
    posXMin = -1 + 2*((uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth) / (float) renderData.width);
    posXMax = (uiHandle.info->sidebarWidth + uiHandle.info->controlAreaWidth + uiHandle.info->viewingAreaWidth) / (float) renderData.width;

    // 1280x 720 is the original aspect

    float scaleUniform = ((float) renderData.width/ 1280.0f); // Scale by width of screen.
    centerX = (posXMax - posXMin) / 2 + posXMin; // center of the quad in the given view area.
    scaleX = (1280.0f / (float) renderData.width) * 0.25f * scaleUniform;
    scaleY = (720.0f / (float) renderData.height) * 0.25f * scaleUniform * 1.11f;

    int order = dev.streams.find(AR_PREVIEW_AUXILIARY)->second.streamingOrder;
    float orderOffset =  uiHandle.info->viewAreaElementPositionsY[order] - (uiHandle.mouseBtns->wheel );
    posYMin = -1.0f + 2*(orderOffset / (float) renderData.height);
    posYMax = -1.0f + 2*((uiHandle.info->viewAreaElementSizeY + (orderOffset)) / (float) renderData.height);                // left anchor
    centerY = (posYMax - posYMin) / 2 + posYMin;

}


void AuxiliaryPreview::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model);

}

void AuxiliaryPreview::onWindowResize(AR::GuiObjectHandles uiHandle) {
    for (auto &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        transformToUISpace(uiHandle, dev);
        model->draw = false;
        coordinateTransformed = true;
    }
}
