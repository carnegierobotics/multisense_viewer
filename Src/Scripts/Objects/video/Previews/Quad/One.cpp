//
// Created by magnus on 5/8/22.
//

#include "One.h"
#include "GLFW/glfw3.h"


void One::setup() {
    // Prepare a m_Model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    noImageModel = std::make_unique<CRLCameraModels::Model>(&renderUtils);

    // Create quad and store it locally on the GPU
    ImageData imgData{};
    model->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
    noImageModel->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);

    // Create texture m_Image if not created
    pixels = stbi_load((Utils::getTexturePath() + "no_image_tex.png").c_str(), &texWidth, &texHeight, &texChannels,
                       STBI_rgb_alpha);
    if (!pixels) {
        Log::Logger::getInstance()->info("Failed to load texture image {}",
                                         (Utils::getTexturePath() + "no_image_tex.png"));
    }

    lastPresentTime = std::chrono::steady_clock::now();
    prepareDefaultTexture();
}

void One::update() {
    if (selectedPreviewTab != TAB_2D_PREVIEW)
        return;

    auto tex = VkRender::TextureData(textureType, res);

    model->getTextureDataPointers(&tex);
    // If we get an image attempt to update the GPU buffer
    if (renderData.crlCamera->get()->getCameraStream(src, &tex, remoteHeadIndex)) {
        // If we have already presented this frame id and
        std::chrono::duration<float> time_span =
                std::chrono::duration_cast<std::chrono::duration<float>>(
                        std::chrono::steady_clock::now() - lastPresentTime);
        float frameTime = 1.0f / renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf.fps();
        if (time_span.count() > (frameTime * TOLERATE_FRAME_NUM_SKIP) &&
            lastPresentedFrameID == tex.m_Id){
            drawDefaultTexture = true;
            return;
        }

        // update timer
        if (lastPresentedFrameID != tex.m_Id){
            lastPresentTime = std::chrono::steady_clock::now();
        }
        // If we get MultiSense images then
        // Update the texture or update the GPU Texture
        if (model->updateTexture(textureType)) {
            drawDefaultTexture = false;
            lastPresentedFrameID = tex.m_Id;
        } else {
            prepareMultiSenseTexture();
            return;
        }
        // If we didn't receive a valid MultiSense image then draw default texture
    } else {
        drawDefaultTexture = true;
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
}


void One::prepareDefaultTexture() {
    noImageModel->modelType = AR_COLOR_IMAGE;
    noImageModel->createEmptyTexture(texWidth, texHeight, AR_COLOR_IMAGE);
    std::string vertexShaderFileName = "Scene/spv/quad.vert";
    std::string fragmentShaderFileName = "Scene/spv/quad_sampler.frag";
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},{fs}};
    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, noImageModel.get(), &renderUtils);
    auto defTex = std::make_unique<VkRender::TextureData>(AR_COLOR_IMAGE, texWidth, texHeight);
    if (noImageModel->getTextureDataPointers(defTex.get())) {
        std::memcpy(defTex->data, pixels, texWidth * texHeight * texChannels);
        noImageModel->updateTexture(defTex->m_Type);
    }
}

void One::prepareMultiSenseTexture() {
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;
    switch (textureType) {
        case AR_GRAYSCALE_IMAGE:
            vertexShaderFileName = "Scene/spv/preview.vert";
            fragmentShaderFileName = "Scene/spv/preview.frag";
            break;
        case AR_COLOR_IMAGE_YUV420:
        case AR_YUV_PLANAR_FRAME:
            vertexShaderFileName = "Scene/spv/quad.vert";
            fragmentShaderFileName = vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME) ?
                                     "Scene/spv/quad_sampler.frag" :  "Scene/spv/quad.frag";

            break;
        case AR_DISPARITY_IMAGE:
            vertexShaderFileName = "Scene/spv/depth.vert";
            fragmentShaderFileName = "Scene/spv/depth.frag";
            break;
        default:
            return;
    }
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};

    uint32_t width = 0, height = 0, depth = 0;
    Utils::cameraResolutionToValue(res, &width, &height, &depth);
    if (width == 0 || height == 0){
        Log::Logger::getInstance()->error("Attempted to create texture with dimmensions {}x{}", width, height);
        return;
    }
    model->modelType = textureType;
    model->createEmptyTexture(width, height, textureType);
    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, model.get(), &renderUtils);
}

void One::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;
        playbackSate = dev.playbackStatus;

        auto &preview = dev.win.at(AR_PREVIEW_ONE);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedMode;

        if ((src != preview.selectedSource || currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            src = preview.selectedSource;
            textureType = Utils::CRLSourceToTextureType(src);
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            prepareMultiSenseTexture();
        }
        transformToUISpace(uiHandle, dev);
    }
}

void One::transformToUISpace(const VkRender::GuiObjectHandles * uiHandle, const VkRender::Device& dev) {
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
    if (selectedPreviewTab == TAB_2D_PREVIEW) {
        CRLCameraModels::draw(commandBuffer, i, drawDefaultTexture ? noImageModel.get() : model.get(), b);
    }
}

void One::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    for (auto &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        //transformToUISpace(uiHandle, dev);
    }
}
