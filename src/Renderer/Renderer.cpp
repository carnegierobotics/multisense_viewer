// Created by magnus on 9/4/21.
//
//

#include <armadillo>
#include "Renderer.h"
#include "MultiSense/src/imgui/InteractionMenu.h"

void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) width / (float) height, 0.001f, 1024.0f);
    camera.rotationSpeed = 0.2f;
    camera.movementSpeed = 10.0f;
    camera.setPosition(defaultCameraPosition);
    camera.setRotation(defaultCameraRotation);

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

    {
        // Setup picking render pass
        std::array<VkAttachmentDescription, 2> attachments = {};
        // Color attachment
        attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        // Depth attachment
        attachments[1].format = depthFormat;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkAttachmentReference colorReference = {};
        colorReference.attachment = 0;
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthReference = {};
        depthReference.attachment = 1;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;
        subpassDescription.pDepthStencilAttachment = &depthReference;
        subpassDescription.inputAttachmentCount = 0;
        subpassDescription.pInputAttachments = nullptr;
        subpassDescription.preserveAttachmentCount = 0;
        subpassDescription.pPreserveAttachments = nullptr;
        subpassDescription.pResolveAttachments = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies{};

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

        VkResult result = (vkCreateRenderPass(device, &renderPassInfo, nullptr, &selection.renderPass));
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create render pass");
    }

    // Frame buffers
    {
        VkImageView attachments[2];
        // Depth/Stencil attachment is the same for all frame buffers'
        attachments[0] = selection.colorView;
        attachments[1] = selection.depthView;
        VkFramebufferCreateInfo frameBufferCreateInfo = {};
        frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        frameBufferCreateInfo.pNext = NULL;
        frameBufferCreateInfo.renderPass = selection.renderPass;
        frameBufferCreateInfo.attachmentCount = 2;
        frameBufferCreateInfo.pAttachments = attachments;
        frameBufferCreateInfo.width = width;
        frameBufferCreateInfo.height = height;
        frameBufferCreateInfo.layers = 1;


        VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &selection.frameBuffer);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");

    }
    //generateScriptClasses();

    cameraConnection = std::make_unique<CameraConnection>();
}


void Renderer::UIUpdate(AR::GuiObjectHandles *uiSettings) {
    //printf("Index: %d, name: %s\n", uiSettings.getSelectedItem(), uiSettings.listBoxNames[uiSettings.getSelectedItem()].c_str());



    camera.setMovementSpeed(200.0f);

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

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.870f, 0.878, 0.862, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    const VkViewport viewport = Populate::viewport((float) width, (float) height, 0.0f, 1.0f);
    const VkRect2D scissor = Populate::rect2D((int32_t) width, (int32_t) height, 0, 0);

    for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i) {
        renderPassBeginInfo.framebuffer = frameBuffers[i];
        vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo);
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);


        for (auto &script: scripts) {
            if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED) {
                script.second->drawScript(drawCmdBuffers[i], i, true);
            }
        }
        guiManager->drawFrame(drawCmdBuffers[i]);


        vkCmdEndRenderPass(drawCmdBuffers[i]);


        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }

    static bool activated = false;
    if (!activated)
        activated = input.getButtonUp(GLFW_KEY_SPACE);
    if (activated) {
        VkCommandBuffer renderCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
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

        // Create a host-visible staging buffer that contains the raw image data
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;

        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                1280 * 720 * 4,
                &stagingBuffer,
                &stagingMemory));


        // Create the memory backing up the buffer handle
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(vulkanDevice->logicalDevice, stagingBuffer, &memReqs);
        VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
        memAlloc.allocationSize = memReqs.size;


        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bufferCopyRegion.imageSubresource.mipLevel = 0;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.width = width;
        bufferCopyRegion.imageExtent.height = height;
        bufferCopyRegion.imageExtent.depth = 1;
        bufferCopyRegion.bufferOffset = 0;


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
                stagingBuffer,
                1,
                &bufferCopyRegion
        );
        vulkanDevice->flushCommandBuffer(copyCmd, queue);

        // Copy texture data into staging buffer
        uint8_t *data;
        CHECK_RESULT(vkMapMemory(vulkanDevice->logicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));

        vkUnmapMemory(vulkanDevice->logicalDevice, stagingMemory);
        uint32_t val = data[int((mousePos.x + (1280 * mousePos.y)) * 4)];
        printf("Vulkan: XY: %f, %f, value: %d, depth: %f\n", mousePos.x, mousePos.y, val,
               (((600.0f / 960.0f) * 300.0f) / (float) val));



        // Clean up staging resources
        vkFreeMemory(vulkanDevice->logicalDevice, stagingMemory, nullptr);
        vkDestroyBuffer(vulkanDevice->logicalDevice, stagingBuffer, nullptr);

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
    VulkanRenderer::prepareFrame();

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
    cameraConnection->onUIUpdate(guiManager->handles.devices);
    // Create/delete scripts after use
    buildScript("LightSource");
    // Run update function on active camera scripts and build them if not built
    for (auto &dev: *guiManager->handles.devices) {
        if (dev.state == AR_STATE_ACTIVE) {
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
        // Check if camera connection was AR RESET and disable all scripts
        if (dev.state == AR_STATE_RESET) {
            // delete all scripts attached to device
            for (const std::string &script: dev.attachedScripts)
                deleteScript(script);
        }
    }
    // Update general scripts with handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED)
            script.second->uiUpdate(guiManager->handles);
    }

    // Run update function on scripts
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED) {
            script.second->updateUniformBufferData(&renderData);
        }
    }
    // Generate draw commands
    guiManager->updateBuffers();
    buildCommandBuffers();
    VulkanRenderer::submitFrame();

}

void Renderer::windowResized() {
    Base::Render renderData{};
    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentBuffer;
    renderData.pLogger = pLogger;
    renderData.height = height;
    renderData.width = width;
    renderData.crlCamera = &cameraConnection->camPtr;

    // Update gui with new res
    guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
    // Update general scripts with handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED)
            script.second->windowResize(&renderData, guiManager->handles);
    }
}


void Renderer::cleanUp() {
    Log::LOG_ALWAYS("<=============================== END OF PROGRAM ===========================>");
    cameraConnection.reset();
}



