/**
 * @file: MultiSense-Viewer/src/Renderer/Renderer.cpp
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
 *   2022-4-9, mgjerde@carnegierobotics.com, Created file.
 **/


#ifdef WIN32
//#include <WinRegEditor.h>
#else

#include <sys/socket.h>
#include <netinet/in.h>
#include <sstream>

#endif

#include <array>

#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"

void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::arcball;
    camera.setPerspective(60.0f, (float) m_Width / (float) m_Height, 0.01f, 1024.0f);
    camera.setPosition(defaultCameraPosition);
    camera.setRotation(yaw, pitch);
    createSelectionImages();
    createSelectionFramebuffer();
    createSelectionBuffer();
    cameraConnection = std::make_unique<VkRender::MultiSense::CameraConnection>();

    // Load Object Scripts from file
    buildScripts();

}

void Renderer::addDeviceFeatures() {
    if (deviceFeatures.fillModeNonSolid) {
        enabledFeatures.fillModeNonSolid = VK_TRUE;
        // Wide lines must be present for line m_Width > 1.0f
        if (deviceFeatures.wideLines) {
            enabledFeatures.wideLines = VK_TRUE;
        }
        if (deviceFeatures.samplerAnisotropy) {
            enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
    }
}

void Renderer::buildCommandBuffers() {
    VkCommandBufferBeginInfo cmdBufInfo = Populate::commandBufferBeginInfo();
    cmdBufInfo.flags = 0;
    cmdBufInfo.pInheritanceInfo = nullptr;

    std::vector<VkClearValue> clearValues{};
    clearValues.resize(2);

    clearValues[0] = {{guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                       guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]}};
    clearValues[1] = {{1.0f, 0.0f}};

    VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = m_Width;
    renderPassBeginInfo.renderArea.extent.height = m_Height;
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    const VkViewport viewport = Populate::viewport((float) m_Width, (float) m_Height, 0.0f, 1.0f);
    const VkRect2D scissor = Populate::rect2D((int32_t) m_Width, (int32_t) m_Height, 0, 0);

    for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i) {
        renderPassBeginInfo.framebuffer = frameBuffers[i];
        vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo);
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        // Draw scripts that must be drawn first
        for (auto &script: scripts) {
            if (script.second->getType() == CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE)
                script.second->drawScript(drawCmdBuffers[i], i, true);
        }

        /** Generate Script draw commands **/
        for (auto &script: scripts) {
            if (script.second->getType() != CRL_SCRIPT_TYPE_DISABLED &&
                script.second->getType() != CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE) {
                script.second->drawScript(drawCmdBuffers[i], i, true);
            }
        }
        /** Generate UI draw commands **/
        guiManager->drawFrame(drawCmdBuffers[i]);
        vkCmdEndRenderPass(drawCmdBuffers[i]);
        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}


void Renderer::buildScripts() {
    std::ifstream infile(Utils::getAssetsPath().append("Generated/Scripts.txt").string());
    std::string scriptName;
    while (std::getline(infile, scriptName)) {
        // Skip comment # line
        if (scriptName.find('#') != std::string::npos)
            continue;
        // Do not recreate script if already created
        auto it = std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName);
        if (it != builtScriptNames.end())
            return;
        builtScriptNames.emplace_back(scriptName);
        scripts[scriptName] = VkRender::ComponentMethodFactory::Create(scriptName);

        if (scripts[scriptName].get() == nullptr) {
            pLogger->error("Failed to register script {}.", scriptName);
            builtScriptNames.erase(std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName));
            return;
        }
        pLogger->info("Registered script: {} in factory", scriptName.c_str());
    }

    // Run Once
    VkRender::RenderUtils vars{};
    vars.device = vulkanDevice.get();
    vars.renderPass = &renderPass;
    vars.UBCount = swapchain->imageCount;
    vars.picking = &selection;
    vars.queueSubmitMutex = &queueSubmitMutex;
    // create first set of scripts for TOP OF PIPE
    for (auto &script: scripts) {
        if (script.second->getType() == CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE) {
            script.second->createUniformBuffers(vars, renderData);
        }
    }
    // Copy data generated from TOP OF PIPE scripts
    vars.skybox.irradianceCube = &scripts["Skybox"]->skyboxTextures.irradianceCube;
    vars.skybox.lutBrdf = &scripts["Skybox"]->skyboxTextures.lutBrdf;
    vars.skybox.prefilterEnv = &scripts["Skybox"]->skyboxTextures.prefilterEnv;
    vars.skybox.prefilteredCubeMipLevels = scripts["Skybox"]->skyboxTextures.prefilteredCubeMipLevels;

    // Run script setup function
    for (auto &script: scripts) {
        if (script.second->getType() != CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE) {
            script.second->createUniformBuffers(vars, renderData);
        }
    }
}

void Renderer::deleteScript(const std::string &scriptName) {
    if (builtScriptNames.empty())
        return;
    auto it = std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName);
    if (it != builtScriptNames.end())
        builtScriptNames.erase(it);
    else
        return;
    pLogger->info("Deleting Script: {}", scriptName.c_str());
    scripts[scriptName].get()->onDestroyScript();
    scripts[scriptName].reset();
    scripts.erase(scriptName);
}


void Renderer::render() {
    pLogger->frameNumber = frameID;
    if (keyPress == GLFW_KEY_SPACE) {
        camera.setPosition(defaultCameraPosition);
        camera.setRotation(yaw, pitch);
    }

    if (guiManager->handles.showDebugWindow) {
        auto &cam = Log::Logger::getLogMetrics()->camera;
        cam.pitch = camera.yAngle;
        cam.pos = camera.m_Position;
        cam.rot = camera.m_Rotation;
        cam.cameraFront = camera.cameraFront;
    }

    camera.viewportHeight = static_cast<float>(m_Height);
    camera.viewportWidth = static_cast<float>(m_Width);

    // RenderData
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentBuffer;
    renderData.height = m_Height;
    renderData.width = m_Width;
    renderData.input = &input;
    renderData.crlCamera = &cameraConnection->camPtr;
    // Update GUI
    guiManager->handles.info->frameID = frameID;
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
    // Update Camera connection based on Actions from GUI
    cameraConnection->onUIUpdate(guiManager->handles.devices, guiManager->handles.configureNetwork);

    // Enable scripts depending on gui layout chosen
    bool noActivePreview = true;
    for (auto &dev: guiManager->handles.devices) {
        if (dev.state == CRL_STATE_ACTIVE) {
            noActivePreview = false;
            switch (dev.layout) {
                case CRL_PREVIEW_LAYOUT_SINGLE:
                    scripts.at("SingleLayout")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    scripts.at("DoubleTop")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleBot")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("One")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Two")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Three")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Four")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    break;
                case CRL_PREVIEW_LAYOUT_DOUBLE:
                    scripts.at("DoubleTop")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    scripts.at("DoubleBot")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);

                    scripts.at("SingleLayout")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("One")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Two")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Three")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Four")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    break;
                case CRL_PREVIEW_LAYOUT_QUAD:
                    scripts.at("One")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Two")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Three")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Four")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);

                    scripts.at("SingleLayout")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleTop")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleBot")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    break;
                default:
                    scripts.at("SingleLayout")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleTop")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleBot")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("One")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Two")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Three")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Four")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    break;
            }

            switch (dev.selectedPreviewTab) {
                case CRL_TAB_3D_POINT_CLOUD:
                    scripts.at("PointCloud")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    //scripts.at("Gizmos")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Skybox")->setDrawMethod(CRL_SCRIPT_TYPE_RENDER_TOP_OF_PIPE);
                    scripts.at("MultiSenseCamera")->setDrawMethod(CRL_SCRIPT_TYPE_DEFAULT);
                    break;
                default:
                    scripts.at("PointCloud")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Gizmos")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("Skybox")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    scripts.at("MultiSenseCamera")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
                    break;
            }

            if (dev.cameraType == 0)
                camera.type = Camera::arcball;
            if (dev.cameraType == 1)
                camera.type = Camera::flycam;
            if (dev.resetCamera) {
                camera.setPosition(defaultCameraPosition);
                camera.setRotation(yaw, pitch);
            }
        }
    }
    if (noActivePreview) {
        scripts.at("SingleLayout")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("DoubleTop")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("DoubleBot")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("One")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("Two")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("Three")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("Four")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("PointCloud")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("Skybox")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("Gizmos")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);
        scripts.at("MultiSenseCamera")->setDrawMethod(CRL_SCRIPT_TYPE_DISABLED);

    }
    // Run update function on active camera Scripts and build them if not built
    for (size_t i = 0; i < guiManager->handles.devices.size(); ++i) {
        if (guiManager->handles.devices.at(i).state == CRL_STATE_REMOVE_FROM_LIST)
            guiManager->handles.devices.erase(guiManager->handles.devices.begin() + i);
    }

    // Scripts that share dataa
    for (auto &script: scripts) {
        if (script.second->getType() != CRL_SCRIPT_TYPE_DISABLED) {
            if (!script.second->sharedData->destination.empty()) {
                // Send to destination script
                if (script.second->sharedData->destination == "All") {
                    // Copy shared data to all
                    auto &shared = script.second->sharedData;

                    for (auto &s: scripts) {
                        if (s == script)
                            continue;
                        memcpy(s.second->sharedData->data, shared->data, SHARED_MEMORY_SIZE_1MB);

                    }
                }
            }
        }
    }

    // UIUpdateFunction on Scripts with const handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != CRL_SCRIPT_TYPE_DISABLED)
            script.second->uiUpdate(&guiManager->handles);
    }
    // run update function for camera connection
    for (auto &dev: guiManager->handles.devices) {
        if (dev.state != CRL_STATE_ACTIVE)
            continue;
        cameraConnection->update(dev);

    }

    // Run update function on Scripts
    for (auto &script: scripts) {
        if (script.second->getType() != CRL_SCRIPT_TYPE_DISABLED) {
            script.second->updateUniformBufferData(&renderData);
        }
    }

    /** Generate Draw Commands **/
    guiManager->updateBuffers();
    buildCommandBuffers();
    /** IF WE SHOULD RENDER SECOND IMAGE FOR MOUSE PICKING EVENTS (Reason: let user see PerPixelInformation) **/
    if (renderSelectionPass) {
        VkCommandBuffer renderCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0] = {{guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                           guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]}};
        clearValues[1].depthStencil = {1.0f, 0};
        const VkViewport viewport = Populate::viewport((float) m_Width, (float) m_Height, 0.0f, 1.0f);
        const VkRect2D scissor = Populate::rect2D((int32_t) m_Width, (int32_t) m_Height, 0, 0);

        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = m_Width;
        renderPassBeginInfo.renderArea.extent.height = m_Height;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.renderPass = selection.renderPass;
        renderPassBeginInfo.framebuffer = selection.frameBuffer;
        vkCmdBeginRenderPass(renderCmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(renderCmd, 0, 1, &viewport);
        vkCmdSetScissor(renderCmd, 0, 1, &scissor);
        for (auto &script: scripts) {
            if (script.second->getType() != CRL_SCRIPT_TYPE_DISABLED) {
                script.second->drawScript(renderCmd, 0, false);
            }
        }
        vkCmdEndRenderPass(renderCmd);
        vulkanDevice->flushCommandBuffer(renderCmd, queue);
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkMemoryBarrier memoryBarrier = {
                VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT,   // srcAccessMask
                VK_ACCESS_HOST_READ_BIT};     // dstAccessMask
        vkCmdPipelineBarrier(
                copyCmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,          // srcStageMask
                VK_PIPELINE_STAGE_HOST_BIT,                     // dstStageMask
                VK_DEPENDENCY_BY_REGION_BIT,
                1,                                    // memoryBarrierCount
                &memoryBarrier,
                0, nullptr,
                0, nullptr);                     // pMemoryBarriers);

        // Copy mip levels from staging buffer
        vkCmdCopyImageToBuffer(
                copyCmd,
                selection.colorImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                selectionBuffer,
                1,
                &bufferCopyRegion
        );
        vulkanDevice->flushCommandBuffer(copyCmd, queue);
        // Copy texture data into staging buffer
        uint8_t *data = nullptr;
        CHECK_RESULT(
                vkMapMemory(vulkanDevice->m_LogicalDevice, selectionMemory, 0, m_MemReqs.size, 0, (void **) &data));
        vkUnmapMemory(vulkanDevice->m_LogicalDevice, selectionMemory);
        for (auto &dev: guiManager->handles.devices) {
            if (dev.state != CRL_STATE_ACTIVE || dev.notRealDevice)
                continue;

            uint32_t idx = uint32_t((mousePos.x + (m_Width * mousePos.y)) * 4);
            if (idx > m_Width * m_Height * 4)
                continue;

            if (dev.state == CRL_STATE_ACTIVE) {
                for (auto &win: dev.win) {
                    // Skip second render pass if we dont have a source selected or if the source is point cloud related
                    if (win.second.selectedSource == "Idle" || win.first == CRL_PREVIEW_POINT_CLOUD)
                        continue;

                    auto windowIndex = win.first;
                    auto tex = VkRender::TextureData(Utils::CRLSourceToTextureType(win.second.selectedSource),
                                                     dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedResolutionMode,
                                                     true);

                    if (renderData.crlCamera->getCameraStream(win.second.selectedSource, &tex,
                                                              win.second.selectedRemoteHeadIndex)) {
                        uint32_t width = 0, height = 0, depth = 0;
                        Utils::cameraResolutionToValue(
                                dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedResolutionMode,
                                &width, &height, &depth);

                        float viewAreaElementPosX = win.second.xPixelStartPos;
                        float viewAreaElementPosY = win.second.yPixelStartPos;
                        float imGuiPosX = (float) mousePos.x - viewAreaElementPosX -
                                          (guiManager->handles.info->previewBorderPadding / 2.0f);
                        float imGuiPosY = (float) mousePos.y - viewAreaElementPosY -
                                          (guiManager->handles.info->previewBorderPadding / 2.0f);
                        float maxInRangeX = guiManager->handles.info->viewAreaElementSizeX -
                                            guiManager->handles.info->previewBorderPadding;
                        float maxInRangeY = guiManager->handles.info->viewAreaElementSizeY -
                                            guiManager->handles.info->previewBorderPadding;
                        if (imGuiPosX > 0 && imGuiPosX < maxInRangeX
                            && imGuiPosY > 0 && imGuiPosY < maxInRangeY) {
                            uint32_t w = 0, h = 0, d = 0;
                            Utils::cameraResolutionToValue(
                                    dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedResolutionMode, &w, &h,
                                    &d);

                            auto x = (uint32_t) ((float) w * (imGuiPosX) / maxInRangeX);
                            auto y = (uint32_t) ((float) h * (imGuiPosY) / maxInRangeY);
                            // Add one since we are not counting from zero anymore :)
                            dev.pixelInfo[windowIndex].x = x + 1;
                            dev.pixelInfo[windowIndex].y = y + 1;

                            // Check that we are within bounds
                            if (dev.pixelInfoZoomed[windowIndex].y > h)
                                dev.pixelInfoZoomed[windowIndex].y = 0;
                            if (dev.pixelInfoZoomed[windowIndex].x > h)
                                dev.pixelInfoZoomed[windowIndex].x = 0;

                            switch (Utils::CRLSourceToTextureType(win.second.selectedSource)) {
                                case CRL_POINT_CLOUD:
                                    break;
                                case CRL_GRAYSCALE_IMAGE: {
                                    uint8_t intensity = tex.data[(w * y) + x];
                                    dev.pixelInfo[windowIndex].intensity = intensity;

                                    intensity = tex.data[(w * dev.pixelInfoZoomed[windowIndex].y) +
                                                         dev.pixelInfoZoomed[windowIndex].x];
                                    dev.pixelInfoZoomed[windowIndex].intensity = intensity;
                                }
                                    break;
                                case CRL_COLOR_IMAGE_RGBA:
                                    break;
                                case CRL_COLOR_IMAGE_YUV420:
                                    break;
                                case CRL_CAMERA_IMAGE_NONE:
                                    break;
                                case CRL_DISPARITY_IMAGE: {
                                    float disparity = 0;
                                    auto *p = (uint16_t *) tex.data;
                                    disparity = (float) p[(w * y) + x] / 16.0f;

                                    // get focal length
                                    float fx = cameraConnection->camPtr.getCameraInfo(
                                            win.second.selectedRemoteHeadIndex).calibration.left.P[0][0];
                                    float tx = cameraConnection->camPtr.getCameraInfo(
                                            win.second.selectedRemoteHeadIndex).calibration.right.P[0][3] /
                                               (fx * (1920.0f / (float) w));
                                    if (disparity > 0) {
                                        float dist = (fx * abs(tx)) / disparity;
                                        dev.pixelInfo[windowIndex].depth = dist;
                                    } else {
                                        dev.pixelInfo[windowIndex].depth = 0;
                                    }
                                    float disparityDisplayed = (float) p[(w * dev.pixelInfoZoomed[windowIndex].y) +
                                                                         dev.pixelInfoZoomed[windowIndex].x] / 16.0f;
                                    if (disparityDisplayed > 0) {
                                        float dist = (fx * abs(tx)) / disparityDisplayed;
                                        dev.pixelInfoZoomed[windowIndex].depth = dist;
                                    } else {
                                        dev.pixelInfoZoomed[windowIndex].depth = 0;
                                    }
                                }
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void Renderer::windowResized() {

    // Clear script and scriptnames before rebuilding
    for (const auto &scriptName: builtScriptNames) {
        pLogger->info("Deleting Script: {}", scriptName.c_str());
        scripts[scriptName].get()->onDestroyScript();
        scripts[scriptName].reset();
        scripts.erase(scriptName);
    }
    builtScriptNames.clear();

    // Recreate to fit new dimensions
    vkDestroyFramebuffer(device, selection.frameBuffer, nullptr);
    vkDestroyImage(device, selection.colorImage, nullptr);
    vkDestroyImage(device, selection.depthImage, nullptr);
    vkDestroyImageView(device, selection.colorView, nullptr);
    vkDestroyImageView(device, selection.depthView, nullptr);
    vkFreeMemory(device, selection.colorMem, nullptr);
    vkFreeMemory(device, selection.depthMem, nullptr);
    createSelectionImages();
    createSelectionFramebuffer();
    destroySelectionBuffer();
    createSelectionBuffer();

    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentBuffer;
    renderData.height = m_Height;
    renderData.width = m_Width;
    renderData.crlCamera = &cameraConnection->camPtr;

    Widgets::clear();
    // Update gui with new res
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
    // Update general Scripts with handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != CRL_SCRIPT_TYPE_DISABLED)
            script.second->windowResize(&renderData, &guiManager->handles);
    }


    buildScripts();
}


void Renderer::cleanUp() {
    usageMonitor->sendUsageLog();

    for (auto &dev: guiManager->handles.devices) {
        dev.interruptConnection = true; // Disable all current connections if user wants to exit early
        cameraConnection->saveProfileAndDisconnect(&dev);
    }

/** REVERT NETWORK SETTINGS **/
#ifdef WIN32
    // Reset Windows registry from backup file
    for (const auto &dev: guiManager->handles.devices) {
        //WinRegEditor regEditor(dev.interfaceName, dev.interfaceDescription, dev.interfaceIndex);
        //regEditor.resetJumbo();
        //regEditor.restartNetAdapters(); // Make changes into effect
    }

#endif
    // Shutdown GUI manually since it contains thread. Not strictly necessary but nice to have
    guiManager.reset();

    // Clear script and scriptnames
    for (const auto &scriptName: builtScriptNames) {
        pLogger->info("Deleting Script: {}", scriptName.c_str());
        scripts[scriptName].get()->onDestroyScript();
        scripts[scriptName].reset();
        scripts.erase(scriptName);
    }
    builtScriptNames.clear();
    destroySelectionBuffer();

    Log::LOG_ALWAYS("<=============================== END OF PROGRAM ===========================>");
}


void Renderer::createSelectionFramebuffer() {
    // Depth/Stencil attachment is the same for all frame buffers
    std::array<VkImageView, 2> attachments{};
    attachments[0] = selection.colorView;
    attachments[1] = selection.depthView;
    VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(m_Width, m_Height,
                                                                                    attachments.data(),
                                                                                    attachments.size(),
                                                                                    selection.renderPass);
    VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &selection.frameBuffer);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
}

void Renderer::createSelectionImages() {
    // Create picking images
    {
        // Create optimal tiled target m_Image
        VkImageCreateInfo colorImage = Populate::imageCreateInfo();
        colorImage.imageType = VK_IMAGE_TYPE_2D;
        colorImage.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorImage.mipLevels = 1;
        colorImage.arrayLayers = 1;
        colorImage.samples = VK_SAMPLE_COUNT_1_BIT;
        colorImage.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorImage.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        colorImage.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorImage.extent = {m_Width, m_Height, 1};
        colorImage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        CHECK_RESULT(vkCreateImage(device, &colorImage, nullptr, &selection.colorImage));
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, selection.colorImage, &memReqs);
        VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &selection.colorMem));
        CHECK_RESULT(vkBindImageMemory(device, selection.colorImage, selection.colorMem, 0));

        VkImageViewCreateInfo colorAttachmentView = {};
        colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorAttachmentView.pNext = nullptr;
        colorAttachmentView.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorAttachmentView.components = {
                VK_COMPONENT_SWIZZLE_R,
                VK_COMPONENT_SWIZZLE_G,
                VK_COMPONENT_SWIZZLE_B,
                VK_COMPONENT_SWIZZLE_A
        };
        colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorAttachmentView.subresourceRange.baseMipLevel = 0;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.baseArrayLayer = 0;
        colorAttachmentView.subresourceRange.layerCount = 1;
        colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorAttachmentView.flags = 0;
        colorAttachmentView.image = selection.colorImage;

        VkResult result = vkCreateImageView(device, &colorAttachmentView, nullptr, &selection.colorView);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create swapchain m_Image views");

        /**DEPTH IMAGE*/
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = depthFormat;
        imageCI.extent = {m_Width, m_Height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        result = vkCreateImage(device, &imageCI, nullptr, &selection.depthImage);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image");

        vkGetImageMemoryRequirements(device, selection.depthImage, &memReqs);

        VkMemoryAllocateInfo memAllloc{};
        memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(device, &memAllloc, nullptr, &selection.depthMem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
        result = vkBindImageMemory(device, selection.depthImage, selection.depthMem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        VkImageViewCreateInfo imageViewCI{};
        imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = selection.depthImage;
        imageViewCI.format = depthFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(device, &imageViewCI, nullptr, &selection.depthView);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image m_View");
    }
}

void Renderer::createSelectionBuffer() {
    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_Width * m_Height * 4,
            &selectionBuffer,
            &selectionMemory));

    // Create the memory backing up the buffer handle
    vkGetBufferMemoryRequirements(vulkanDevice->m_LogicalDevice, selectionBuffer, &m_MemReqs);
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = m_Width;
    bufferCopyRegion.imageExtent.height = m_Height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;
}

void Renderer::destroySelectionBuffer() {
    // Clean up staging resources
    vkFreeMemory(vulkanDevice->m_LogicalDevice, selectionMemory, nullptr);
    vkDestroyBuffer(vulkanDevice->m_LogicalDevice, selectionBuffer, nullptr);
}

void Renderer::mouseMoved(float x, float y, bool &handled) {
    float dx = mousePos.x - (float) x;
    float dy = mousePos.y - (float) y;

    mouseButtons.dx = dx;
    mouseButtons.dy = dy;

    if (mouseButtons.left && guiManager->handles.info->isViewingAreaHovered) { // && !mouseButtons.middle) {
        camera.rotate(dx, dy);
    }
    if (mouseButtons.right) {
    }
    if (mouseButtons.middle && camera.type == Camera::flycam) {
        camera.translate(glm::vec3((float) -dx * 0.01f, (float) -dy * 0.01f, 0.0f));
    } else if (mouseButtons.middle && camera.type == Camera::arcball) {
        //camera.orbitPan((float) -dx * 0.01f, (float) -dy * 0.01f);
    }
    mousePos = glm::vec2((float) x, (float) y);

    handled = true;
}

void Renderer::mouseScroll(float change) {
    for (const auto &item: guiManager->handles.devices) {
        if (item.state == CRL_STATE_ACTIVE && item.selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD &&
            guiManager->handles.info->isViewingAreaHovered) {
            camera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        }
    }

}
