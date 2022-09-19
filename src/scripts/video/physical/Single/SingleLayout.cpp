//
// Created by magnus on 5/8/22.
//

#include "SingleLayout.h"
#include "GLFW/glfw3.h"


void SingleLayout::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
}

void SingleLayout::update(CameraConnection *conn) {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    auto *camera = conn->camPtr;

    if (camera->getCameraInfo().imgConf.width() != width) {
        model->draw = false;
    }

    if (model->draw) {
        auto *tex = new ArEngine::TextureData();
        if (camera->getCameraStream(src, tex)) {
            model->setGrayscaleTexture(tex, src == "Disparity Left" ? AR_DISPARITY_IMAGE : AR_GRAYSCALE_IMAGE);
            model->setZoom();
            free(tex->data);
        }
        delete tex;
    }

    ArEngine::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX * (1 / scaleX), centerY * (1 / scaleY), 0.0f));

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


void SingleLayout::prepareTexture() {

    model = new CRLCameraModels::Model(renderUtils.device,
                                       src == "Disparity Left" ? AR_DISPARITY_IMAGE : AR_GRAYSCALE_IMAGE);
    model->draw = false;


    auto imgConf = renderData.crlCamera->get()->camPtr->getCameraInfo().imgConf;
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;

    if (src == "Disparity Left") {
        vertexShaderFileName = "myScene/spv/depth.vert";
        fragmentShaderFileName = "myScene/spv/depth.frag";
    } else {
        vertexShaderFileName = "myScene/spv/preview.vert";
        fragmentShaderFileName = "myScene/spv/preview.frag";
    }


    width = imgConf.width();
    height = imgConf.height();

    model->createEmtpyTexture(width, height, src == "Disparity Left" ? AR_DISPARITY_IMAGE : AR_GRAYSCALE_IMAGE);
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
    model->draw = true;
}

void SingleLayout::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    for (const AR::Element &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;


        if (src != dev.selectedSource || res != dev.selectedMode) {
            src = dev.selectedSource;
            selectedPreviewTab = dev.selectedPreviewTab;
            playbackSate = dev.playbackStatus;
            res = dev.selectedMode;
            prepareTexture();
        }



        transformToUISpace(uiHandle, dev);

    }


    /*
    for (const auto &dev: *uiHandle.devices) {


        src = dev.streams.find(AR_PREVIEW_DISPARITY)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_DISPARITY)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;


    }

    if (playbackSate == AR_PREVIEW_PLAYING) {

        if (input->getButton(GLFW_KEY_LEFT_CONTROL)){
            std::cout << "Btn down: " << uiHandle.mouseBtns->wheel / 100.0f << std::endl;
            auto * d2 = (ArEngine::ZoomParam *) bufferFourData;
            d2->zoom = uiHandle.mouseBtns->wheel / 1000.0f;

        } else if (!uiHandle.info->hoverState ) {
            posY = uiHandle.accumulatedActiveScroll * 0.05f * 0.1f * 0.557 * (720.0f / (float) renderData.height);
        }

        // center of viewing area box.

        //posX =  2*;

        for (auto &dev: *uiHandle.devices) {
            if (dev.state != AR_STATE_ACTIVE)
                continue;
            if (prevOrder != dev.streams.find(AR_PREVIEW_DISPARITY)->second.streamingOrder) {
                transformToUISpace(uiHandle, dev);
                model->draw = false;
                coordinateTransformed = true;
            }
            prevOrder = dev.streams.find(AR_PREVIEW_DISPARITY)->second.streamingOrder;

            if (!model->draw && !coordinateTransformed) {
                renderData.crlCamera->get()->camPtr->start(src, AR_PREVIEW_DISPARITY);

                transformToUISpace(uiHandle, dev);
                coordinateTransformed = true;

            }
        }
    }


    //printf("Pos %f, %f, %f\n", posX, posY, posZ);
*/
}

void SingleLayout::transformToUISpace(AR::GuiObjectHandles uiHandle, AR::Element dev) {
    auto *info = uiHandle.info;
    centerX = 2 * ((info->width - (info->viewingAreaWidth / 2)) / info->width) - 1; // map between -1 to 1q
    centerY = 2 * (info->tabAreaHeight +
                   ((info->viewAreaElementSizeY / 2) + ((dev.row[0]) * info->viewAreaElementSizeY) +
                    ((dev.row[0]) * 10.0f))) / info->height- 1; // map between -1 to 1

    scaleX = (info->viewAreaElementSizeX / 1280.0f) * (1280.0f / info->width);
    scaleY = (info->viewAreaElementSizeY / 720.0f) * (720 / info->height);
}


void SingleLayout::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model);

}

void SingleLayout::onWindowResize(AR::GuiObjectHandles uiHandle) {
    for (auto &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        //transformToUISpace(uiHandle, dev);
    }
}
