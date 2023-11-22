/**
 * @file: MultiSense-Viewer/src/Scripts/Objects/Pointcloud/PointCloud.cpp
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
#include <glm/gtx/string_cast.hpp>

#include "Viewer/Scripts/MultiSenseRenderer/Pointcloud/PointCloud.h"
#include "Viewer/ImGui/Widgets.h"

void PointCloud::setup() {
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils, renderUtils.UBCount);
    model->m_Draw = false;
    model->m_CameraDataType = CRL_POINT_CLOUD;
    for(uint32_t i = 0; i < renderUtils.UBCount; ++i){
        renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &model->m_ColorPointCloudBuffer[i], sizeof(VkRender::ColorPointCloudParams));
        model->m_ColorPointCloudBuffer[i].map();
    }


    Widgets::make()->text(WIDGET_PLACEMENT_POINTCLOUD, "Set Point Size:");
    Widgets::make()->slider(WIDGET_PLACEMENT_POINTCLOUD, "##Set Point size", &pointSize, 0, 10);
    // Widgets::make()->text("Flip pointcloud: ");
    //Widgets::make()->checkbox("##Flip pc", &flipPointCloud);


    pc = std::make_unique<PointCloudLoader>(&renderUtils);
    VkPipelineShaderStageCreateInfo vs = loadShader("Scene/spv/pc3D.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("Scene/spv/pc3D.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    pc->createGraphicsPipeline(shaders);

}

void PointCloud::update() {


    if (prepareTexFuture.valid() && prepareTexFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        if (prepareTexFuture.get())
            model->m_Draw = true;
    }

    if (model->m_Draw && selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD) {
        const auto &conf = renderData.crlCamera->getCameraInfo(remoteHeadIndex).imgConf;
        auto tex = VkRender::TextureData(lumaOrColor ? CRL_COLOR_IMAGE_YUV420 : CRL_GRAYSCALE_IMAGE, conf.width(),
                                         conf.height(), true);
        model->getTextureDataPointers(&tex, renderData.index);
        if (renderData.crlCamera->getCameraStream(lumaOrColor ? "Color Rectified Aux" : "Luma Rectified Left", &tex,
                                                  remoteHeadIndex)) {
            model->updateTexture(&tex, renderData.index);
        }

        auto depthTex = VkRender::TextureData(CRL_DISPARITY_IMAGE, conf.width(), conf.height());
        model->getTextureDataPointers(&depthTex, renderData.index);
        if (renderData.crlCamera->getCameraStream("Disparity Left", &depthTex, remoteHeadIndex)) {
            model->updateTexture(depthTex.m_Type, renderData.index);
        }

        VkRender::UBOMatrix mat{};
        mat.model = glm::mat4(1.0f);
        mat.model = glm::translate(mat.model, glm::vec3(0.1f, 0.0f, 0.0f));
        mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        // 24 degree m_Rotation to compensate for VkRender S27 24 degree camera slant.
        //mat.m_Model = glm::rotate(mat.m_Model, glm::radians(24.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        //mat.m_Model = glm::translate(mat.m_Model, glm::vec3(2.8, 0.4, -5));
        auto &d = bufferOneData;
        d->model = mat.model;
        d->projection = renderData.camera->matrices.perspective;
        d->view = renderData.camera->matrices.view;

        VkRender::ColorPointCloudParams data{};
        data.instrinsics = renderData.crlCamera->getCameraInfo(0).KColorMat;
        data.extrinsics = renderData.crlCamera->getCameraInfo(0).KColorMatExtrinsic;
        data.useColor = lumaOrColor == 1;
        data.hasSampler = renderUtils.device->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
        memcpy(model->m_ColorPointCloudBuffer[renderData.index].mapped, &data, sizeof(VkRender::ColorPointCloudParams));

        auto *buf = bufferThreeData.get();
        buf->pointSize = pointSize;


    }
}


void PointCloud::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: uiHandle->devices) {
        if (dev.state != CRL_STATE_ACTIVE)
            continue;

        selectedPreviewTab = dev.selectedPreviewTab;

        auto &preview = dev.win.at(CRL_PREVIEW_POINT_CLOUD);
        auto currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedResolutionMode;

        if (dev.simulatedDevice) {
            currentRes = CRL_RESOLUTION_1920_1200_128;
            model->m_Draw = false;
        }

        if ((currentRes != res ||
             remoteHeadIndex != preview.selectedRemoteHeadIndex || lumaOrColor != dev.useAuxForPointCloudColor)) {
            res = currentRes;
            remoteHeadIndex = preview.selectedRemoteHeadIndex;
            lumaOrColor = dev.useAuxForPointCloudColor;
            model->m_Draw = false;
            auto& device = const_cast<VkRender::Device &>(dev);
            prepareTexFuture = std::async(std::launch::async, &PointCloud::prepareTexture, this, std::ref(device));
        }
    }
}


void PointCloud::draw(CommandBuffer * commandBuffer, uint32_t i, bool b) {
    if (selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD && b){
        //pc->draw(commandBuffer, i);

        if (model->m_Draw ) {
            CRLCameraModels::draw(commandBuffer, i, model.get(), b);
        }
    }
}


bool PointCloud::prepareTexture(VkRender::Device &dev) {
    uint32_t width = 0, height = 0, depth = 0;
    Utils::cameraResolutionToValue(res, &width, &height, &depth);

    std::vector<VkRender::Vertex> meshData{};
    meshData.resize(width * height);
    int v = 0;
    // first few rows and cols (20) are discarded in the shader anyway
    for (uint32_t i = 20; i < width - 20; ++i) {
        for (uint32_t j = 20; j < height - 20; ++j) {
            meshData[v].pos = glm::vec3(static_cast<float>(i), static_cast<float>(j), 0.0f);
            meshData[v].uv0 = glm::vec2(1.0f - (static_cast<float>(i) / static_cast<float>(width)),
                                        1.0f - (static_cast<float>(j) / static_cast<float>(height)));
            v++;
        }
    }
    model->createMeshDeviceLocal(meshData);
    model->createEmptyTexture(width, height, CRL_DISPARITY_IMAGE, true, lumaOrColor);
    VkPipelineShaderStageCreateInfo vs = loadShader("Scene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("Scene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    CRLCameraModels::createRenderPipeline(shaders, model.get(), &renderUtils);


    if (renderData.crlCamera->updateCameraInfo(&dev, 0)){

        auto *buf = bufferThreeData.get();
        buf->Q = renderData.crlCamera->getCameraInfo(0).QMat;
        buf->height = static_cast<float>(height);
        buf->width = static_cast<float>(width);
        buf->disparity = static_cast<float>(depth);
        buf->focalLength = renderData.crlCamera->getCameraInfo(0).focalLength;
        buf->scale = renderData.crlCamera->getCameraInfo(0).pointCloudScale;
        buf->pointSize = pointSize;
        Log::Logger::getInstance()->info(
                "Transforming depth image to point cloud with focal length {} and point cloud scale {}. Image size is ({}, {})",
                renderData.crlCamera->getCameraInfo(0).focalLength, renderData.crlCamera->getCameraInfo(0).pointCloudScale, renderData.crlCamera->getCameraInfo(0).imgConf.width(), renderData.crlCamera->getCameraInfo(0).imgConf.height());
        return true;
    } else {
        Log::Logger::getInstance()->error("Failed to update camera info for pointcloud!");
    }
    return false;
}