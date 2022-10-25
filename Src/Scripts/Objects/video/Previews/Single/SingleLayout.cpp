//
// Created by magnus on 5/8/22.
//

#include "SingleLayout.h"
#include "GLFW/glfw3.h"


void SingleLayout::setup() {
    // Prepare a m_Model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);

    // Create quad and store it locally on the GPU
    ImageData imgData{};
    model->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
    // Create texture m_Image if not created
    pixels = stbi_load((Utils::getTexturePath() + "no_image_tex.png").c_str(), &texWidth, &texHeight, &texChannels,
                       STBI_rgb_alpha);
    if (!pixels) {
        Log::Logger::getInstance()->info("Failed to log texture image {}",
                                         (Utils::getTexturePath() + "no_image_tex.png"));
    }
    prepareDefaultTexture();
    updateLog();
}

void SingleLayout::update() {
    if (selectedPreviewTab != TAB_2D_PREVIEW)
        return;

    auto tex = VkRender::TextureData(textureType, width, height);
    model->getTextureDataPointer(&tex);
    // If we get an image successfully render it, otherwise revert to default image
    if (renderData.crlCamera->get()->getCameraStream(src, &tex, remoteHeadIndex)) {
        model->updateTexture(textureType);
    } else {
        auto defTex = VkRender::TextureData(AR_COLOR_IMAGE, texWidth, texHeight);
        prepareDefaultTexture();
        model->getTextureDataPointer(&defTex);
        std::memcpy(defTex.data, pixels, texWidth * texHeight * texChannels);
        model->updateTexture(defTex.m_Type);
    }

    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, posY, 0.0f));
    mat.model = glm::translate(mat.model, glm::vec3(centerX, centerY, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(scaleX, scaleY, 0.25f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;
    updateLog();
}

void SingleLayout::prepareDefaultTexture() {

    if (!usingDefaultTexture) {
        model->modelType = AR_COLOR_IMAGE;
        textureType = AR_COLOR_IMAGE;
        model->createEmtpyTexture(texWidth, texHeight, AR_COLOR_IMAGE);
        std::string vertexShaderFileName = "myScene/spv/quad.vert";
        std::string fragmentShaderFileName = "myScene/spv/quad.frag";
        VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        // Create graphics render pipeline
        CRLCameraModels::createRenderPipeline(shaders, model.get(), &renderUtils);
        usingDefaultTexture = true;
    }


}

void SingleLayout::prepareMultiSenseTexture() {
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


    auto imgConf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
    width = imgConf.width();
    height = imgConf.height();
    model->modelType = textureType;
    model->createEmtpyTexture(width, height, textureType);
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, model.get(), &renderUtils);

    usingDefaultTexture = false;
}

void SingleLayout::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;
        playbackSate = dev.playbackStatus;
        auto &preview = dev.win.at(AR_PREVIEW_ONE);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedMode;
        textureType = Utils::CRLSourceToTextureType(src);

        if ((src != preview.selectedSource || currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            src = preview.selectedSource;
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            prepareMultiSenseTexture();
        }

        transformToUISpace(uiHandle, dev);
    }
}

void SingleLayout::transformToUISpace(const VkRender::GuiObjectHandles *uiHandle, const VkRender::Device &dev) {
    centerX = 2 * ((uiHandle->info->width - (uiHandle->info->viewingAreaWidth / 2)) / uiHandle->info->width) -
              1; // map between -1 to 1q
    centerY = 2 * (uiHandle->info->tabAreaHeight +
                   ((uiHandle->info->viewAreaElementSizeY / 2) + ((dev.row[0]) * uiHandle->info->viewAreaElementSizeY) +
                    ((dev.row[0]) * 10.0f))) / uiHandle->info->height - 1; // map between -1 to 1

    scaleX = ((uiHandle->info->viewAreaElementSizeX - uiHandle->info->previewBorderPadding) / 1280.0f) *
             (1280.0f / uiHandle->info->width);
    scaleY = ((uiHandle->info->viewAreaElementSizeY - uiHandle->info->previewBorderPadding) / 720.0f) *
             (720 / uiHandle->info->height);
}


void SingleLayout::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model.get(), b);
}

void SingleLayout::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    for (auto &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
    }
}

void SingleLayout::updateLog(){
    auto* met = Log::Logger::getLogMetrics();
    met->preview.height = height;
    met->preview.width = width;
    met->preview.texHeight = texHeight;
    met->preview.texWidth = texWidth;
    met->preview.src = src;
    met->preview.textureType = textureType;
    met->preview.usingDefaultTexture = usingDefaultTexture;
    met->preview.res = res;
}
