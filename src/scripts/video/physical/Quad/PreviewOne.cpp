//
// Created by magnus on 5/8/22.
//

#include "PreviewOne.h"
#include "GLFW/glfw3.h"


void PreviewOne::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
}

void PreviewOne::update(){
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    if (model->draw) {
        if (renderData.crlCamera->getCameraInfo().imgConf.width() != width) {
            model->draw = false;
        }

        auto *tex = new ArEngine::TextureData();
        if (renderData.crlCamera->getCameraStream(src, tex)) {
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


void PreviewOne::prepareTexture() {
    model = new CRLCameraModels::Model(renderUtils.device,
                                       src == "Disparity Left" ? AR_DISPARITY_IMAGE : AR_GRAYSCALE_IMAGE);
    model->draw = false;

    auto imgConf = renderData.crlCamera->getCameraInfo().imgConf;
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

void PreviewOne::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    for (const AR::Element &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        if (!dev.selectedSourceMap.contains(AR_PREVIEW_ONE))
            break;

        if (dev.selectedSourceMap.at(AR_PREVIEW_ONE) == "None"){
            // dont draw or update
            if (model)
                model->draw = false;
        }

        if ((src != dev.selectedSourceMap.at(AR_PREVIEW_ONE) || dev.selectedMode != res)) {
            src = dev.selectedSourceMap.at(AR_PREVIEW_ONE);
            selectedPreviewTab = dev.selectedPreviewTab;
            playbackSate = dev.playbackStatus;
            res = dev.selectedMode;
            prepareTexture();
        }

        transformToUISpace(uiHandle, dev);
    }
}

void PreviewOne::transformToUISpace(AR::GuiObjectHandles uiHandle, AR::Element dev) {
    auto *info = uiHandle.info;
    scaleX = (info->viewAreaElementSizeX / 1280.0f) * (1280.0f / info->width);
    scaleY = (info->viewAreaElementSizeY / 720.0f) * (720 / info->height);

    float offsetX = (info->controlAreaWidth + info->sidebarWidth + 5.0f);

    float viewAreaElementPosX = offsetX + (info->viewAreaElementSizeX/2) + (dev.col[0] * info->viewAreaElementSizeX) + (dev.col[0] * 10.0f);

    centerX = 2 * (viewAreaElementPosX) / info->width - 1; // map between -1 to 1q
    centerY = 2 * (info->tabAreaHeight + (info->viewAreaElementSizeY/2.0f)  + ((dev.row[0]) * info->viewAreaElementSizeY) + ((dev.row[0]) * 10.0f)) / info->height - 1; // map between -1 to 1
}


void PreviewOne::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (!model)
        return;

    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model);

}

void PreviewOne::onWindowResize(AR::GuiObjectHandles uiHandle) {
    for (auto &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        //transformToUISpace(uiHandle, dev);
    }
}
