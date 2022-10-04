//
// Created by magnus on 9/19/22.
//

#include "Three.h"
#include "GLFW/glfw3.h"


void Three::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    model->draw = false;

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
}

void Three::update(){
    if (model->draw) {
        if (renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf.width() != width) {
            model->draw = false;
            return;
        }

        auto* tex = new VkRender::TextureData(textureType);
        if (renderData.crlCamera->get()->getCameraStream(src, tex, remoteHeadIndex)) {
            model->setTexture(tex);
            model->setZoom();
            if (tex->type == AR_DISPARITY_IMAGE || tex->type == AR_GRAYSCALE_IMAGE)
                free(tex->data);
            else {
                free(tex->planar.data[0]);
                free(tex->planar.data[1]);
            }
        }
        delete tex;
    }

    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX * (1 / scaleX), centerY * (1 / scaleY), 0.0f));

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


void Three::prepareTexture() {
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
    model->createMeshDeviceLocal((VkRender::Vertex *) imgData.quad.vertices,
                                 imgData.quad.vertexCount, imgData.quad.indices, imgData.quad.indexCount);

    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, model.get(), type, &renderUtils);
    model->draw = true;
}

void Three::onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
    for (const MultiSense::Device &dev: *uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;
        playbackSate = dev.playbackStatus;

        auto &preview = dev.win.at(AR_PREVIEW_THREE);
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

void Three::transformToUISpace(const MultiSense::GuiObjectHandles * uiHandle, MultiSense::Device dev) {
    float row = dev.row[2];
    float col = dev.col[0];
    scaleX = (uiHandle->info->viewAreaElementSizeX / 1280.0f) * (1280.0f / uiHandle->info->width);
    scaleY = (uiHandle->info->viewAreaElementSizeY / 720.0f) * (720 / uiHandle->info->height);
    float offsetX = (uiHandle->info->controlAreaWidth + uiHandle->info->sidebarWidth + 5.0f);
    float viewAreaElementPosX = offsetX + (uiHandle->info->viewAreaElementSizeX/2) + (col * uiHandle->info->viewAreaElementSizeX) + (col * 10.0f);
    centerX = 2 * (viewAreaElementPosX) / uiHandle->info->width - 1; // map between -1 to 1q
    centerY = 2 * (uiHandle->info->tabAreaHeight + (uiHandle->info->viewAreaElementSizeY/2.0f)  + ((row) * uiHandle->info->viewAreaElementSizeY) + ((row) * 10.0f)) / uiHandle->info->height - 1; // map between -1 to 1
}


void Three::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model.get(), b);

}

void Three::onWindowResize(const MultiSense::GuiObjectHandles *uiHandle) {
    for (auto &dev: *uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        //transformToUISpace(uiHandle, dev);
    }
}
