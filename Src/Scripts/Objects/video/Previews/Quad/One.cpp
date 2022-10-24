//
// Created by magnus on 5/8/22.
//

#include "One.h"
#include "GLFW/glfw3.h"


void One::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    model->draw = false;

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
}

void One::update() {
    if (playbackSate != AR_PREVIEW_PLAYING || selectedPreviewTab != TAB_2D_PREVIEW)
        return;

    // There might be some delay for when the camera actually sets the resolution therefore add this check so we dont render to a texture that does not match the actual camere frame size
    if (model->draw) {
        if (renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf.width() != width) {
            model->draw = false;
            prepareTexture();
            return;
        }
        const auto& conf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
        auto tex = std::make_unique<VkRender::TextureData>(textureType, conf.width(), conf.height());
        model->getTextureDataPointer(tex.get());
        if (renderData.crlCamera->get()->getCameraStream(src, tex.get(), remoteHeadIndex)) {
            model->updateTexture(textureType);
        }
    }

    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX * (1 / scaleX), centerY * (1 / scaleY), 0.0f));

    auto& d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto& d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;

}


void One::prepareTexture() {
    model->modelType = textureType;
    auto imgConf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;

    switch (textureType) {
        case AR_GRAYSCALE_IMAGE:
            vertexShaderFileName = "myScene/spv/preview.vert";
            fragmentShaderFileName = "myScene/spv/preview.frag";
            break;
        case AR_COLOR_IMAGE_YUV420:
        case AR_YUV_PLANAR_FRAME:
            vertexShaderFileName = "myScene/spv/quad.vert";
            fragmentShaderFileName = "myScene/spv/quad.frag";

            break;
        case AR_DISPARITY_IMAGE:
            vertexShaderFileName = "myScene/spv/depth.vert";
            fragmentShaderFileName = "myScene/spv/depth.frag";
            break;
        default:
            std::cerr << "Invalid Texture type" << std::endl;
            return;
    }

    width = imgConf.width();
    height = imgConf.height();

    model->createEmtpyTexture(width, height, textureType);
    //auto *imgData = new ImageData(posXMin, posXMax, posYMin, posYMax);


    // Load shaders
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    // Create quad and store it locally on the GPU
    ImageData imgData;
    model->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);


    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, model.get(), &renderUtils);
    model->draw = true;
}

void One::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;
        playbackSate = dev.playbackStatus;

        auto &preview = dev.win.at(AR_PREVIEW_ONE);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedMode;
        if (preview.selectedSource == "Source") {
            // dont draw or update
            model->draw = false;
        }

        if ((src != preview.selectedSource || currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            src = preview.selectedSource;
            textureType = Utils::CRLSourceToTextureType(src);
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            prepareTexture();
        }
        transformToUISpace(uiHandle, dev);
    }
}

void One::transformToUISpace(const VkRender::GuiObjectHandles * uiHandle, VkRender::Device dev) {
    float row = dev.row[0];
    float col = dev.col[0];
    scaleX = ((uiHandle->info->viewAreaElementSizeX - uiHandle->info->previewBorderPadding)/ 1280.0f) * (1280.0f / uiHandle->info->width);
    scaleY = ((uiHandle->info->viewAreaElementSizeY  - uiHandle->info->previewBorderPadding )/ 720.0f) * (720 / uiHandle->info->height);
    float offsetX = (uiHandle->info->controlAreaWidth + uiHandle->info->sidebarWidth + 5.0f);
    float viewAreaElementPosX = offsetX + (uiHandle->info->viewAreaElementSizeX/2) + (col * uiHandle->info->viewAreaElementSizeX) + (col * 10.0f);
    centerX = 2 * (viewAreaElementPosX) / uiHandle->info->width - 1; // map between -1 to 1q
    centerY = 2 * (uiHandle->info->tabAreaHeight + (uiHandle->info->viewAreaElementSizeY/2.0f)  + ((row) * uiHandle->info->viewAreaElementSizeY) + ((row) * 10.0f)) / uiHandle->info->height - 1; // map between -1 to 1
}



void One::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model.get(), b);

}

void One::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    for (auto &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        //transformToUISpace(uiHandle, dev);
    }
}
