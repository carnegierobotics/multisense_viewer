// Created by magnus on 9/4/21.
//
//


#include "Renderer.h"

#ifdef WIN32
#include <WinRegEditor.h>
#else

#include <sys/socket.h>
#include <netinet/in.h>
#include <sstream>
#include <fstream>

#endif

#include <array>


void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) m_Width / (float) m_Height, 0.001f, 1024.0f);
    camera.m_RotationSpeed = 0.2f;
    camera.m_MovementSpeed = 1.0f;
    camera.setPosition(defaultCameraPosition);
    camera.setRotation(defaultCameraRotation);
    createSelectionImages();
    createSelectionFramebuffer();
    createSelectionBuffer();
    cameraConnection = std::make_unique<VkRender::MultiSense::CameraConnection>();

    // Prefer to load the m_Model only once, so load it in first setup
    // Load Object Scripts from file
    std::ifstream infile(Utils::getAssetsPath() + "Generated/Scripts.txt");
    std::string line;
    while (std::getline(infile, line)) {
        // Skip comment # line
        if (line.find('#') != std::string::npos)
            continue;

        buildScript(line);
    }
}


void Renderer::addDeviceFeatures() {
    if (deviceFeatures.fillModeNonSolid) {
        enabledFeatures.fillModeNonSolid = VK_TRUE;
        // Wide lines must be present for line m_Width > 1.0f
        if (deviceFeatures.wideLines) {
            enabledFeatures.wideLines = VK_TRUE;
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

        /** Generate Script draw commands **/
        for (auto &script: scripts) {
            if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED) {
                script.second->drawScript(drawCmdBuffers[i], i, true);
            }
        }
        /** Generate UI draw commands **/
        guiManager->drawFrame(drawCmdBuffers[i]);
        vkCmdEndRenderPass(drawCmdBuffers[i]);
        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}


void Renderer::buildScript(const std::string &scriptName) {
    // Do not recreate script if already created
    auto it = std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName);
    if (it != builtScriptNames.end())
        return;
    builtScriptNames.emplace_back(scriptName);
    scripts[scriptName] = VkRender::ComponentMethodFactory::Create(scriptName);

    if (scripts[scriptName].get() == nullptr) {
        pLogger->error("Failed to register script {}. Did you remember to include it in renderer.h?", scriptName);
        builtScriptNames.erase(std::find(builtScriptNames.begin(), builtScriptNames.end(), scriptName));
        return;
    }
    pLogger->info("Registered script: {} in factory", scriptName.c_str());
    // Run Once
    VkRender::RenderUtils vars{};
    vars.device = vulkanDevice.get();
    vars.renderPass = &renderPass;
    vars.UBCount = swapchain->imageCount;
    vars.picking = &selection;
    renderData.scriptName = scriptName;
    // Run script setup function
    scripts[scriptName]->createUniformBuffers(vars, renderData);
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
        camera.setRotation(defaultCameraRotation);
    }

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
    bool previewActive = false;
    for (auto &dev: guiManager->handles.devices) {
        if (dev.state == AR_STATE_ACTIVE) {
            renderSelectionPass = dev.pixelInfoEnable;
            previewActive = true;
            switch (dev.layout) {
                case PREVIEW_LAYOUT_SINGLE:
                    scripts.at("SingleLayout")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);
                    scripts.at("DoubleTop")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleBot")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("One")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Two")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Three")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Four")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    break;
                case PREVIEW_LAYOUT_DOUBLE:
                    scripts.at("DoubleTop")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);
                    scripts.at("DoubleBot")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);

                    scripts.at("SingleLayout")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("One")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Two")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Three")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Four")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    break;
                case PREVIEW_LAYOUT_QUAD:
                    scripts.at("One")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Two")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Three")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);
                    scripts.at("Four")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);

                    scripts.at("SingleLayout")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleTop")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleBot")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    break;
                default:
                    scripts.at("SingleLayout")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleTop")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("DoubleBot")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("One")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Two")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Three")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    scripts.at("Four")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    break;
            }

            switch (dev.selectedPreviewTab) {
                case TAB_3D_POINT_CLOUD:
                    scripts.at("PointCloud")->setDrawMethod(AR_SCRIPT_TYPE_DEFAULT);
                    break;
                default:
                    scripts.at("PointCloud")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
                    break;
            }
        }
    }
    if (!previewActive) {
        scripts.at("SingleLayout")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("DoubleTop")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("DoubleBot")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("One")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("Two")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("Three")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("Four")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
        scripts.at("PointCloud")->setDrawMethod(AR_SCRIPT_TYPE_DISABLED);
    }
    // Run update function on active camera Scripts and build them if not built
    for (size_t i = 0; i < guiManager->handles.devices.size(); ++i) {
        if (guiManager->handles.devices.at(i).state == AR_STATE_REMOVE_FROM_LIST)
            guiManager->handles.devices.erase(guiManager->handles.devices.begin() + i);
    }


    // UiUpdate on Scripts with const handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED)
            script.second->uiUpdate(&guiManager->handles);
    }

    // Run update function on Scripts
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED) {
            script.second->updateUniformBufferData(&renderData);
        }
    }

    /** Generate Draw Commands **/
    guiManager->updateBuffers();
    buildCommandBuffers();

    /** IF WE SHOULD RENDER SECOND IMAGE FOR MOUSE PICKING EVENTS (Reason: PerPixelInformation) **/
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
            if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED) {
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
            if (dev.state != AR_STATE_ACTIVE)
                continue;
            auto idx = uint32_t((mousePos.x + (m_Width * mousePos.y)) * 4);
            if (idx > m_Width * m_Height * 4)
                continue;

            // Required
            // - What resolution is currently set
            // - Handle to each preview window
            // - What type is currently set on each window

            // If we get an image attempt to update the GPU buffer
            if (dev.state == AR_STATE_ACTIVE) {
                for (auto &win: dev.win) {
                    if (win.second.selectedSource == "Source")
                        continue;



                    auto tex = VkRender::TextureData(Utils::CRLSourceToTextureType(win.second.selectedSource),
                                                     dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedMode,
                                                     true);

                    if (renderData.crlCamera->get()->getCameraStream(win.second.selectedSource, &tex,
                                                                     win.second.selectedRemoteHeadIndex)) {
                        uint32_t width = 0, height = 0, depth = 0;
                        Utils::cameraResolutionToValue(dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedMode,
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
                                    dev.channelInfo[win.second.selectedRemoteHeadIndex].selectedMode, &w, &h,
                                    &d);

                            auto x = (uint32_t) ((float) w * (imGuiPosX) / maxInRangeX);
                            auto y = (uint32_t) ((float) h * (imGuiPosY) / maxInRangeY);
                            // Add one since we are not counting from zero anymore :)
                            dev.pixelInfo.x = x + 1;
                            dev.pixelInfo.y = y + 1;

                            switch (Utils::CRLSourceToTextureType(win.second.selectedSource)) {
                                case AR_POINT_CLOUD:
                                    break;
                                case AR_GRAYSCALE_IMAGE:
                                {
                                    uint8_t intensity = tex.data[(w * y) + x];
                                    dev.pixelInfo.intensity = intensity;
                                }
                                    break;
                                case AR_COLOR_IMAGE:
                                    break;
                                case AR_COLOR_IMAGE_YUV420:
                                    break;
                                case AR_YUV_PLANAR_FRAME:
                                    break;
                                case AR_CAMERA_IMAGE_NONE:
                                    break;
                                case AR_DISPARITY_IMAGE:
                                {
                                    float disparity = 0;
                                    auto *p = (uint16_t *) tex.data;
                                    disparity = (float) p[(w * y) + x] / 16.0f;
                                    // get focal length
                                    float fx = cameraConnection->camPtr->getCameraInfo(
                                            win.second.selectedRemoteHeadIndex).calibration.left.P[0][0];
                                    float tx = cameraConnection->camPtr->getCameraInfo(
                                            win.second.selectedRemoteHeadIndex).calibration.right.P[0][3] / fx;
                                    if (disparity > 0) {
                                        float dist = (fx * abs(tx)) / disparity;
                                        dev.pixelInfo.depth = dist;
                                    } else {
                                        dev.pixelInfo.depth = 0;
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

    // Update gui with new res
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
    // Update general Scripts with handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED)
            script.second->windowResize(&renderData, &guiManager->handles);
    }
}


void Renderer::cleanUp() {
    for (auto &dev: guiManager->handles.devices)
        cameraConnection->saveProfileAndDisconnect(&dev);
/** REVERT NETWORK SETTINGS **/
#ifdef WIN32
    // Reset Windows registry from backup file
    for (const auto& dev : guiManager->handles.devices) {
        WinRegEditor regEditor(dev.interfaceName, dev.interfaceDescription, dev.interfaceIndex);
        regEditor.resetJumbo();
        regEditor.restartNetAdapters(); // Make changes into effect
    }
#endif
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

    if (mouseButtons.left && !guiManager->handles.disableCameraRotationFromGUI) {
        glm::vec3 rot(dy * camera.m_RotationSpeed, -dx * camera.m_RotationSpeed, 0.0f);
        camera.rotate(rot);
    }
    if (mouseButtons.right) {
    }
    if (mouseButtons.middle) {
        camera.translate(glm::vec3((float) -dx * 0.01f, (float) -dy * 0.01f, 0.0f));
    }
    mousePos = glm::vec2((float) x, (float) y);

    handled = true;
}
