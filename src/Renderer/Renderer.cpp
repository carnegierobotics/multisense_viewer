// Created by magnus on 9/4/21.
//
//

#include "Renderer.h"


void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) width / (float) height, 0.001f, 1024.0f);
    camera.rotationSpeed = 0.2f;
    camera.movementSpeed = 10.0f;
    camera.setPosition(defaultCameraPosition);
    camera.setRotation(defaultCameraRotation);


    //generateScriptClasses();

    // Generate UI from Layers
    guiManager->pushLayer<SideBar>();
    guiManager->pushLayer<InteractionMenu>();

    cameraConnection = std::make_unique<CameraConnection>();
}


void Renderer::viewChanged() {
    updateUniformBuffers();
}


void Renderer::UIUpdate(AR::GuiObjectHandles *uiSettings) {
    //printf("Index: %d, name: %s\n", uiSettings.getSelectedItem(), uiSettings.listBoxNames[uiSettings.getSelectedItem()].c_str());



    camera.setMovementSpeed(20.0f);

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
    clearValues[0].color = {{0.054, 0.137, 0.231, 1.0f}};
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
                script.second->drawScript(drawCmdBuffers[i], i);
            }
        }
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
        pLogger->error("Failed to register script. Did you remember to include it in renderer.h?");
        scriptNames.erase(std::find(scriptNames.begin(), scriptNames.end(), scriptName));
        return;
    }

    pLogger->info("Registered script: {} in factory", scriptName.c_str());

    // Run Once
    Base::RenderUtils vars{};
    vars.device = vulkanDevice;
    vars.renderPass = &renderPass;
    vars.UBCount = swapchain.imageCount;


    Base::Render renderData{};
    renderData.crlCamera = &cameraConnection;
    renderData.gui = guiManager->handles.devices;
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

    pLogger->info("RENDERER::deleted Script: {}", scriptName.c_str());

    scripts.erase(scriptName);

}


void Renderer::render() {
    VulkanRenderer::prepareFrame();

    if (keypress == GLFW_KEY_SPACE) {
        camera.setPosition(defaultCameraPosition);
        camera.setRotation(defaultCameraRotation);
    }

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    Base::Render renderData{};
    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentBuffer;
    renderData.pLogger = pLogger;
    renderData.height = height;
    renderData.width = width;

    guiManager->handles.keypress = keypress;
    guiManager->handles.mouseBtns.wheel = mouseButtons.wheel;
    guiManager->handles.mouseBtns.left = mouseButtons.left;
    guiManager->handles.mouseBtns.right = mouseButtons.right;


    drawDataExt.texInfo = &guiManager->handles.info->imageElements;

    // Update GUI
    guiManager->update((frameCounter == 0), frameTimer, width, height);

    // Update Camera connection based on Actions from GUI
    cameraConnection->onUIUpdate(guiManager->handles.devices);
    renderData.crlCamera = &cameraConnection;

    // Create/delete scripts after use
    buildScript("LightSource");

    // Run update function on active camera scripts and build them if not built
    for (auto &dev: *guiManager->handles.devices) {
            for (const auto &i: dev.streams) {
                if (i.second.playbackStatus == AR_PREVIEW_PLAYING) {
                    switch (i.second.streamIndex) {
                        case AR_PREVIEW_LEFT:
                            buildScript("DefaultPreview");
                            break;
                        case AR_PREVIEW_RIGHT:
                            buildScript("RightPreview");
                            break;
                        case AR_PREVIEW_DISPARITY:
                            buildScript("DisparityPreview");
                            break;
                        case AR_PREVIEW_AUXILIARY:
                            buildScript("AuxiliaryPreview");
                            break;
                        case AR_PREVIEW_VIRTUAL_LEFT:
                            buildScript("LeftImager");
                            break;
                        case AR_PREVIEW_POINT_CLOUD:
                            buildScript("PointCloud");
                            break;
                        case AR_PREVIEW_VIRTUAL_POINT_CLOUD:
                            buildScript("VirtualPointCloud");
                            break;
                        case AR_PREVIEW_VIRTUAL_RIGHT:
                            buildScript("RightImager");
                            break;
                        case AR_PREVIEW_VIRTUAL_AUX:
                            buildScript("AuxImager");
                            break;
                    }
                }


            if (i.second.playbackStatus == AR_PREVIEW_NONE) {
                switch (i.second.streamIndex) {
                    case AR_PREVIEW_LEFT:
                        deleteScript("DefaultPreview");
                        break;
                    case AR_PREVIEW_RIGHT:
                        deleteScript("RightPreview");
                        break;
                    case AR_PREVIEW_DISPARITY:
                        deleteScript("DisparityPreview");
                        break;
                    case AR_PREVIEW_AUXILIARY:
                        deleteScript("AuxiliaryPreview");
                        break;
                    case AR_PREVIEW_VIRTUAL_LEFT:
                        deleteScript("LeftImager");
                        break;
                    case AR_PREVIEW_POINT_CLOUD:
                        deleteScript("PointCloud");
                        break;
                    case AR_PREVIEW_VIRTUAL_POINT_CLOUD:
                        deleteScript("VirtualPointCloud");
                        break;
                    case AR_PREVIEW_VIRTUAL_RIGHT:
                        deleteScript("RightImager");
                        break;
                    case AR_PREVIEW_VIRTUAL_AUX:
                        deleteScript("AuxImager");
                        break;
                }
            }
        }
    }


    // Run update function on scripts
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED) {
            script.second->updateUniformBufferData(&renderData);
        }
    }

    // Update general scripts with handle to GUI
    for (auto &script: scripts) {
        if (script.second->getType() != AR_SCRIPT_TYPE_DISABLED)
            script.second->uiUpdate(guiManager->handles);
    }

    // Generate draw commands
    guiManager->updateBuffers();
    buildCommandBuffers();

    vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]);
    VulkanRenderer::submitFrame();

}

void Renderer::draw() {


}


void Renderer::updateUniformBuffers() {


}

void Renderer::cleanUp() {
    Log::LOG_ALWAYS("<=============================== END OF PROGRAM ===========================>");
    cameraConnection.reset();

}

