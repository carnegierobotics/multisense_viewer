//
// Created by magnus on 5/8/22.
//

#include "SingleLayout.h"
#include "GLFW/glfw3.h"


void SingleLayout::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = new CRLCameraModels::Model(&renderUtils);
    model->draw = false;

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
}

void SingleLayout::update() {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    if (model->draw) {
        if (renderData.crlCamera->get()->getCameraInfo().imgConf.width() != width) {
            model->draw = false;
            return;
        }
        auto* tex = new VkRender::TextureData(textureType);
        if (renderData.crlCamera->get()->getCameraStream(src, tex)) {
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


void SingleLayout::prepareTexture() {
    model->modelType = textureType;
    auto imgConf = renderData.crlCamera->get()->getCameraInfo().imgConf;
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
    CRLCameraModels::createRenderPipeline(shaders, model, type, &renderUtils);
    model->draw = true;
}

void SingleLayout::onUIUpdate(MultiSense::GuiObjectHandles uiHandle) {
    for (const MultiSense::Device &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;
        playbackSate = dev.playbackStatus;
        if (!dev.selectedSourceMap.contains(AR_PREVIEW_ONE))
            break;

        if (dev.selectedSourceMap.at(AR_PREVIEW_ONE) == "None") {
            // dont draw or update
            model->draw = false;
        }

        if ((src != dev.selectedSourceMap.at(AR_PREVIEW_ONE) || dev.selectedMode != res)) {
            src = dev.selectedSourceMap.at(AR_PREVIEW_ONE);
            textureType =  Utils::CRLSourceToTextureType(src);
            res = dev.selectedMode;
            prepareTexture();
        }

        transformToUISpace(uiHandle, dev);
    }
}

void SingleLayout::transformToUISpace(MultiSense::GuiObjectHandles uiHandle, MultiSense::Device dev) {
    auto *info = uiHandle.info;
    centerX = 2 * ((info->width - (info->viewingAreaWidth / 2)) / info->width) - 1; // map between -1 to 1q
    centerY = 2 * (info->tabAreaHeight +
                   ((info->viewAreaElementSizeY / 2) + ((dev.row[0]) * info->viewAreaElementSizeY) +
                    ((dev.row[0]) * 10.0f))) / info->height - 1; // map between -1 to 1

    scaleX = (info->viewAreaElementSizeX / 1280.0f) * (1280.0f / info->width);
    scaleY = (info->viewAreaElementSizeY / 720.0f) * (720 / info->height);
}


void SingleLayout::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model, b);
} 

void SingleLayout::onWindowResize(MultiSense::GuiObjectHandles uiHandle) {
    for (auto &dev: *uiHandle.devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
    }
}
