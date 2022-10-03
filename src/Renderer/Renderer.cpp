// Created by magnus on 9/4/21.
//
//

#include "Renderer.h"
#include "MultiSense/external/simpleini/SimpleIni.h"

void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) width / (float) height, 0.001f, 1024.0f);
    camera.rotationSpeed = 0.2f;
    camera.movementSpeed = 1.0f;
    camera.setPosition(defaultCameraPosition);
    camera.setRotation(defaultCameraRotation);
    createSelectionImages();
    createSelectionFramebuffer();
    createSelectionBuffer();
    cameraConnection = std::make_unique<CameraConnection>();

    /** LOAD PREVIOUS CONNECTION PROFILE IF THEY EXIST*/
    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile("crl.ini");
    if (rc < 0) { /* handle error */ }
    else {
        // A serial number is the section identifier of a profile of the ini file
        CSimpleIniA::TNamesDepend sections;
        ini.GetAllSections(sections);
        for (auto& section : sections) {
            std::string profileName = ini.GetValue(section.pItem, "ProfileName");
            std::string IP = ini.GetValue(section.pItem, "IP");
            std::string cameraName = ini.GetValue(section.pItem, "CameraName");
            int interfaceIndex = std::stoi(ini.GetValue(section.pItem, "AdapterIndex"));
            std::string adapterName = ini.GetValue(section.pItem, "AdapterName");
            MultiSense::Device el;
            el.name = profileName;
            el.IP = IP;
            el.state = AR_STATE_JUST_ADDED;
            el.cameraName = cameraName;
            el.interfaceName = adapterName;
            el.clicked = true;
            el.interfaceIndex = interfaceIndex;
            el.serialName = section.pItem;
            guiManager->handles.devices->emplace_back(el);
        }
    }

    // Prefer to load the model only once, so load it in first setup
    buildScript("MultiSenseCamera");
}


void Renderer::addDeviceFeatures() {
    if (deviceFeatures.fillModeNonSolid) {
        enabledFeatures.fillModeNonSolid = VK_TRUE;
        // Wide lines must be present for line width > 1.0f
        if (deviceFeatures.wideLines) {
            enabledFeatures.wideLines = VK_TRUE;
        }
    }

}

void Renderer::buildCommandBuffers() {
    VkCommandBufferBeginInfo cmdBufInfo = Populate::commandBufferBeginInfo();

    std::array< VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.870f, 0.878f, 0.862f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    const VkViewport viewport = Populate::viewport((float) width, (float) height, 0.0f, 1.0f);
    const VkRect2D scissor = Populate::rect2D((int32_t) width, (int32_t) height, 0, 0);

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
    auto it = std::find(scriptNames.begin(), scriptNames.end(), scriptName);
    if (it != scriptNames.end())
        return;
    scriptNames.emplace_back(scriptName);
    scripts[scriptName] = ComponentMethodFactory::Create(scriptName);

    if (scripts[scriptName].get() == nullptr) {
        pLogger->error("Failed to register script {}. Did you remember to include it in renderer.h?", scriptName);
        scriptNames.erase(std::find(scriptNames.begin(), scriptNames.end(), scriptName));
        return;
    }
    pLogger->info("Registered script: {} in factory", scriptName.c_str());
    // Run Once
    Base::RenderUtils vars{};
    vars.device = vulkanDevice;
    vars.renderPass = &renderPass;
    vars.UBCount = swapchain.imageCount;
    vars.picking = &selection;
    renderData.scriptName = scriptName;
    // Run script setup function
    scripts[scriptName]->createUniformBuffers(vars, renderData);
}

void Renderer::deleteScript(const std::string &scriptName) {
    if (scriptNames.empty())
        return;
    auto it = std::find(scriptNames.begin(), scriptNames.end(), scriptName);
    if (it != scriptNames.end())
        scriptNames.erase(it);
    else
        return;
    pLogger->info("Deleting Script: {}", scriptName.c_str());
    scripts[scriptName].get()->onDestroyScript();
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
    renderData.pLogger = pLogger;
    renderData.height = height;
    renderData.width = width;
    renderData.input = &input;
    renderData.crlCamera = &cameraConnection->camPtr;
    guiManager->handles.mouseBtns = &mouseButtons;
    // Update GUI
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
    // Update Camera connection based on Actions from GUI
    cameraConnection->onUIUpdate(guiManager->handles.devices, guiManager->handles.configureNetworkForNextConnection);
    // Run update function on active camera Scripts and build them if not built
    for (int i = 0; i < guiManager->handles.devices->size(); ++i){
        if (guiManager->handles.devices->at(i).state == AR_STATE_REMOVE_FROM_LIST)
            guiManager->handles.devices->erase(guiManager->handles.devices->begin() + i);
    }
    // TODO Rework conditions to when Scripts are built to more human readable code
    for (auto &dev: *guiManager->handles.devices) {
        if (dev.state == AR_STATE_ACTIVE) {
            renderSelectionPass = dev.pixelInfoEnable;
            if (dev.selectedPreviewTab == TAB_3D_POINT_CLOUD) {

                std::string scriptName = "PointCloud";
                buildScript(scriptName);
                if (!Utils::isInVector(dev.attachedScripts, scriptName))
                    dev.attachedScripts.emplace_back(scriptName);


            } else {
                deleteScript("PointCloud");

                if (dev.layout == PREVIEW_LAYOUT_SINGLE) {
                    std::string scriptName = "SingleLayout";
                    buildScript(scriptName);
                    if (!Utils::isInVector(dev.attachedScripts, scriptName))
                        dev.attachedScripts.emplace_back(scriptName);
                } else {
                    deleteScript("SingleLayout");
                }

                if (dev.layout == PREVIEW_LAYOUT_DOUBLE) {
                    std::string scriptName = "DoubleLayout";
                    buildScript(scriptName);

                    if (!Utils::isInVector(dev.attachedScripts, "DoubleLayout"))
                        dev.attachedScripts.emplace_back("DoubleLayout");
                    scriptName = "DoubleLayoutBot";
                    buildScript(scriptName);

                    if (!Utils::isInVector(dev.attachedScripts, "DoubleLayoutBot"))
                        dev.attachedScripts.emplace_back("DoubleLayoutBot");
                } else {
                    deleteScript("DoubleLayout");
                    deleteScript("DoubleLayoutBot");

                }

                if (dev.layout == PREVIEW_LAYOUT_QUAD) {
                    buildScript("PreviewOne");
                    buildScript("PreviewTwo");
                    buildScript("Three");
                    buildScript("Four");
                    if (!Utils::isInVector(dev.attachedScripts, "PreviewOne"))
                        dev.attachedScripts.emplace_back("PreviewOne");
                    if (!Utils::isInVector(dev.attachedScripts, "PreviewTwo"))
                        dev.attachedScripts.emplace_back("PreviewTwo");
                    if (!Utils::isInVector(dev.attachedScripts, "Three"))
                        dev.attachedScripts.emplace_back("Three");
                    if (!Utils::isInVector(dev.attachedScripts, "Four"))
                        dev.attachedScripts.emplace_back("Four");
                } else {
                    deleteScript("PreviewOne");
                    deleteScript("PreviewTwo");
                    deleteScript("Three");
                    deleteScript("Four");
                }
            }
        }
        // Check if camera connection was MultiSense RESET and clean up all Scripts attached to that camera connection
        if (dev.state == AR_STATE_RESET) {
            // delete all Scripts attached to device
            for (const std::string &script: dev.attachedScripts)
                deleteScript(script);
        }
    }
    // UIupdaet on Scripts with const handle to GUI
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
        std::array< VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.870f, 0.878f, 0.862f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };
        const VkViewport viewport = Populate::viewport((float) width, (float) height, 0.0f, 1.0f);
        const VkRect2D scissor = Populate::rect2D((int32_t) width, (int32_t) height, 0, 0);

        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
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
        CHECK_RESULT(vkMapMemory(vulkanDevice->logicalDevice, selectionMemory, 0, memReqs.size, 0, (void **) &data));
        vkUnmapMemory(vulkanDevice->logicalDevice, selectionMemory);
        for (auto &dev: *guiManager->handles.devices) {
            if (dev.state != AR_STATE_ACTIVE)
                continue;
            uint32_t idx = uint32_t((mousePos.x + (width * mousePos.y)) * 4);
            if (idx > width * height * 4)
                continue;

            uint32_t val = data[idx];
            if (dev.state == AR_STATE_ACTIVE) {
                dev.pixelInfo.x = static_cast<uint32_t>(mousePos.x);
                dev.pixelInfo.y = static_cast<uint32_t>(mousePos.y);
                dev.pixelInfo.intensity = val;
            }
        }


    }
}

void Renderer::windowResized() {
    // Recreate to fit new dimensions
    createSelectionImages();
    createSelectionFramebuffer();

    destroySelectionBuffer();
    createSelectionBuffer();

    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentBuffer;
    renderData.pLogger = pLogger;
    renderData.height = height;
    renderData.width = width;
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
    for (auto& dev : *guiManager->handles.devices)
        CameraConnection::disconnectCRLCameraTask(cameraConnection.get(), &dev); // TODO Note: potentially unsafe usage. Casting smart pointer cameraConnection to void* then back to CameraConnection * with uses context in static function


    destroySelectionBuffer();

     Log::LOG_ALWAYS("<=============================== END OF PROGRAM ===========================>");

    }


void Renderer::createSelectionFramebuffer() {
    // Depth/Stencil attachment is the same for all frame buffers
    std::array<VkImageView, 2> attachments{};
    attachments[0] = selection.colorView;
    attachments[1] = selection.depthView;
    VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(width, height, attachments.data(),
                                                                                    attachments.size(),
                                                                                    selection.renderPass);
    VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &selection.frameBuffer);
    if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
}

void Renderer::createSelectionImages() {
    // Create picking images
    {
        // Create optimal tiled target image
        VkImageCreateInfo colorImage = Populate::imageCreateInfo();
        colorImage.imageType = VK_IMAGE_TYPE_2D;
        colorImage.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorImage.mipLevels = 1;
        colorImage.arrayLayers = 1;
        colorImage.samples = VK_SAMPLE_COUNT_1_BIT;
        colorImage.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorImage.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        colorImage.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorImage.extent = {width, height, 1};
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
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create swapchain image views");

        /**DEPTH IMAGE*/
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = depthFormat;
        imageCI.extent = {width, height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        result = vkCreateImage(device, &imageCI, nullptr, &selection.depthImage);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image");

        vkGetImageMemoryRequirements(device, selection.depthImage, &memReqs);

        VkMemoryAllocateInfo memAllloc{};
        memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(device, &memAllloc, nullptr, &selection.depthMem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth image memory");
        result = vkBindImageMemory(device, selection.depthImage, selection.depthMem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth image memory");

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
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image view");
    }
}

void Renderer::createSelectionBuffer() {
    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            width * height * 4,
            &selectionBuffer,
            &selectionMemory));


    // Create the memory backing up the buffer handle
    vkGetBufferMemoryRequirements(vulkanDevice->logicalDevice, selectionBuffer, &memReqs);
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    memAlloc.allocationSize = memReqs.size;

    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;
}

void Renderer::destroySelectionBuffer() {
    // Clean up staging resources
    vkFreeMemory(vulkanDevice->logicalDevice, selectionMemory, nullptr);
    vkDestroyBuffer(vulkanDevice->logicalDevice, selectionBuffer, nullptr);
}