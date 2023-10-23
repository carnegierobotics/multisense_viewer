/**
 * @file: MultiSense-Viewer/src/Scripts/Objects/Video/Previews/Quad/Three.cpp
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/

#include "Viewer/Scripts/Objects/Video/Previews/Quad/Three.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"

void Three::setup() {
    // Prepare a m_Model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    m_Model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    m_NoDataModel = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    m_NoSourceModel = std::make_unique<CRLCameraModels::Model>(&renderUtils);

    // Create quad and store it locally on the GPU
    VkRender::ScriptUtils::ImageData imgData{};
    m_Model->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
    m_NoDataModel->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);
    m_NoSourceModel->createMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);

    // Create texture m_Image if not created
    m_NoDataTex = stbi_load((Utils::getTexturePath().append("no_image_tex.png")).string().c_str(), &texWidth, &texHeight, &texChannels,
                            STBI_rgb_alpha);
    if (!m_NoDataTex) {
        Log::Logger::getInstance()->info("Failed to load texture image {}",
                                         (Utils::getTexturePath().append("no_image_tex.png")).string());
    }
    // Create texture m_Image if not created
    m_NoSourceTex = stbi_load((Utils::getTexturePath().append("no_source_selected.png")).string().c_str(), &texWidth, &texHeight,
                              &texChannels,
                              STBI_rgb_alpha);
    if (!m_NoSourceTex) {
        Log::Logger::getInstance()->info("Failed to load texture image {}",
                                         (Utils::getTexturePath().append("no_source_selected.png")).string());
    }

    lastPresentTime = std::chrono::steady_clock::now();
    prepareDefaultTexture();
}

void Three::update() {
    if (selectedPreviewTab != CRL_TAB_2D_PREVIEW)
        return;

    auto tex = VkRender::TextureData(textureType, res);

    m_Model->getTextureDataPointers(&tex);
    // If we get an image attempt to update the GPU buffer
    if (renderData.crlCamera->getCameraStream(src, &tex, remoteHeadIndex)) {
        // If we have already presented this frame id and
        std::chrono::duration<float> time_span =
                std::chrono::duration_cast<std::chrono::duration<float>>(
                        std::chrono::steady_clock::now() - lastPresentTime);
        float frameTime = 1.0f / renderData.crlCamera->getCameraInfo(remoteHeadIndex).imgConf.fps();
        if (time_span.count() > (frameTime * TOLERATE_FRAME_NUM_SKIP) &&
            lastPresentedFrameID == tex.m_Id) {
            state = DRAW_NO_DATA;
            return;
        }
        if (lastPresentedFrameID != tex.m_Id){
            lastPresentTime = std::chrono::steady_clock::now();
        }

        // If we get MultiSense images then
        // Update the texture or update the GPU Texture
        if (m_Model->updateTexture(textureType)) {
            state = DRAW_MULTISENSE;
            lastPresentedFrameID = tex.m_Id;
        } else {
            prepareMultiSenseTexture();
            return;
        }
        // If we didn't receive a valid MultiSense image then draw default texture
    } else {
        // if a valid source was selected but not drawn
        if (Utils::stringToDataSource(src)) {
            state = DRAW_NO_DATA;
        }
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
    VkRender::ScriptUtils::handleZoom(&zoom);
    d2->zoomCenter = glm::vec4(0.0f, zoom.offsetY, zoom.zoomValue, zoom.offsetX);
    d2->zoomTranslate = glm::vec4(zoom.translateX, zoom.translateY, 0.0f, 0.0f);
    d2->disparityNormalizer = glm::vec4(options->normalize, options->data.minDisparityValue,
                                        options->data.maxDisparityValue, options->interpolation);
    d2->kernelFilters = glm::vec4(options->edgeDetection, options->blur, options->emboss, options->sharpening);    d2->pad.x = options->depthColorMap;
}


void Three::prepareDefaultTexture() {
    m_NoDataModel->cameraDataType = CRL_COLOR_IMAGE_RGBA;
    m_NoDataModel->createEmptyTexture(texWidth, texHeight, CRL_COLOR_IMAGE_RGBA, false, 0);
    std::string vertexShaderFileName = "Scene/spv/color.vert";
    std::string fragmentShaderFileName = "Scene/spv/color_default_sampler.frag";
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, m_NoDataModel.get(), &renderUtils);
    auto defTex = std::make_unique<VkRender::TextureData>(CRL_COLOR_IMAGE_RGBA, texWidth, texHeight);
    if (m_NoDataModel->getTextureDataPointers(defTex.get())) {
        std::memcpy(defTex->data, m_NoDataTex, texWidth * texHeight * texChannels);
        m_NoDataModel->updateTexture(defTex->m_Type);
    }

    m_NoSourceModel->cameraDataType = CRL_COLOR_IMAGE_RGBA;
    m_NoSourceModel->createEmptyTexture(texWidth, texHeight, CRL_COLOR_IMAGE_RGBA, false, 0);
    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, m_NoSourceModel.get(), &renderUtils);
    auto tex = std::make_unique<VkRender::TextureData>(CRL_COLOR_IMAGE_RGBA, texWidth, texHeight);
    if (m_NoSourceModel->getTextureDataPointers(tex.get())) {
        std::memcpy(tex->data, m_NoSourceTex, texWidth * texHeight * texChannels);
        m_NoSourceModel->updateTexture(tex->m_Type);
    }
}

void Three::prepareMultiSenseTexture() {
    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;
    switch (textureType) {
        case CRL_GRAYSCALE_IMAGE:
            vertexShaderFileName = "Scene/spv/grayscale.vert";
            fragmentShaderFileName = "Scene/spv/grayscale.frag";
            break;
        case CRL_COLOR_IMAGE_YUV420:
            vertexShaderFileName = "Scene/spv/color.vert";
            fragmentShaderFileName = vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME) ?
                                     "Scene/spv/color_default_sampler.frag" :  "Scene/spv/color_ycbcr_sampler.frag";
            break;
        case CRL_DISPARITY_IMAGE:
            vertexShaderFileName = "Scene/spv/disparity.vert";
            fragmentShaderFileName = "Scene/spv/disparity.frag";
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
    m_Model->cameraDataType = textureType;
    m_Model->createEmptyTexture(width, height, textureType, false, 0);
    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(shaders, m_Model.get(), &renderUtils);
}

void Three::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    for (VkRender::Device &dev: uiHandle->devices) {


        if (dev.state != CRL_STATE_ACTIVE)
            continue;
        selectedPreviewTab = dev.selectedPreviewTab;

        auto &preview = dev.win.at(CRL_PREVIEW_THREE);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedResolutionMode;

        if (src == "Idle") {
            state = DRAW_NO_SOURCE;
        } else {
            state = DRAW_NO_DATA;
        }
        zoom.resChanged = currentRes != res;
        uint32_t width = 0, height = 0, depth = 0;
        Utils::cameraResolutionToValue(currentRes, &width, &height, &depth);
        if ((src != preview.selectedSource || currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex)) {
            src = preview.selectedSource;
            textureType = Utils::CRLSourceToTextureType(src);
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            zoom.resolutionUpdated(width, height);

            prepareMultiSenseTexture();
        }
        transformToUISpace(uiHandle, dev);
        options = &preview.effects;
        zoomEnabled = preview.enableZoom;
        VkRender::ScriptUtils::setZoomValue(zoom, &uiHandle->previewZoom, CRL_PREVIEW_THREE);

        glm::vec2 deltaMouse(uiHandle->mouse->dx, uiHandle->mouse->dy);
        VkRender::ScriptUtils::handleZoomUiLoop(&zoom, dev, CRL_PREVIEW_THREE, deltaMouse,
                                                (uiHandle->mouse->left && preview.isHovered), options->magnifyZoomMode,
                                                preview.enableZoom);

    }
}

void Three::transformToUISpace(const VkRender::GuiObjectHandles * uiHandle, const VkRender::Device& dev) {
    float row = dev.win.at(CRL_PREVIEW_THREE).row;
    float col = dev.win.at(CRL_PREVIEW_THREE).col;
    scaleX = ((uiHandle->info->viewAreaElementSizeX - uiHandle->info->previewBorderPadding)/ 1280.0f) * (1280.0f / uiHandle->info->width);
    scaleY = ((uiHandle->info->viewAreaElementSizeY  - uiHandle->info->previewBorderPadding )/ 720.0f) * (720 / uiHandle->info->height);
    float offsetX = (uiHandle->info->controlAreaWidth + uiHandle->info->sidebarWidth + 5.0f);
    float viewAreaElementPosX = offsetX + (uiHandle->info->viewAreaElementSizeX/2) + (col * uiHandle->info->viewAreaElementSizeX) + (col * 10.0f);
    centerX = 2 * (viewAreaElementPosX) / uiHandle->info->width - 1; // map between -1 to 1q
    centerY = 2 * (uiHandle->info->tabAreaHeight + (uiHandle->info->viewAreaElementSizeY/2.0f)  + ((row) * uiHandle->info->viewAreaElementSizeY) + ((row) * 10.0f)) / uiHandle->info->height - 1; // map between -1 to 1
}


void Three::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (selectedPreviewTab == CRL_TAB_2D_PREVIEW) {
        switch (state) {
            case DRAW_NO_SOURCE:
                CRLCameraModels::draw(commandBuffer, i, m_NoSourceModel.get(), b);
                break;
            case DRAW_NO_DATA:
                CRLCameraModels::draw(commandBuffer, i, m_NoDataModel.get(), b);
                break;
            case DRAW_MULTISENSE:
                CRLCameraModels::draw(commandBuffer, i, m_Model.get(), b);
                break;
        }
    }
}

void Three::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    for (auto &dev: uiHandle->devices) {
        if (dev.state != CRL_STATE_ACTIVE)
            continue;
    }
}
