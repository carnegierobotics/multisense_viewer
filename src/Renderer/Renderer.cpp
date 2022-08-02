// Created by magnus on 9/4/21.
//
//

#include "Renderer.h"


void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) width / (float) height, 0.001f, 1024.0f);
    camera.rotationSpeed = 0.2f;
    camera.movementSpeed = 10.0f;
    camera.setPosition({0.0f, 0.0f, 0.0f});
    camera.setRotation({0.0f, 0.0f, 0.0f});


    //generateScriptClasses();

    // Generate UI from Layers
    guiManager->pushLayer<SideBar>();
    guiManager->pushLayer<InteractionMenu>();

    cameraConnection = std::make_unique<CameraConnection>();
}


void Renderer::viewChanged() {
    updateUniformBuffers();
}


void Renderer::UIUpdate(GuiObjectHandles *uiSettings) {
    //printf("Index: %d, name: %s\n", uiSettings.getSelectedItem(), uiSettings.listBoxNames[uiSettings.getSelectedItem()].c_str());



    camera.setMovementSpeed(20.0f);

}

void Renderer::addDeviceFeatures() {
    printf("Overriden function\n");
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
    const VkRect2D scissor = Populate::rect2D(width, height, 0, 0);

    for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i) {
        renderPassBeginInfo.framebuffer = frameBuffers[i];
        vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo);
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);


        for (auto &script: scripts) {
            if (script.second->getType() != ArDisabled) {
                script.second->draw(drawCmdBuffers[i], i);
            }
        }
        guiManager->drawFrame(drawCmdBuffers[i]);


        vkCmdEndRenderPass(drawCmdBuffers[i]);
        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}


void Renderer::buildScript(const std::string& scriptName){

    // Do not recreate script if already created
    auto end = std::remove(scriptNames.begin(), scriptNames.end(), scriptName);
    if (end != scriptNames.end()){
        return;
    }

    scriptNames.emplace_back(scriptName);
    scripts[scriptName] = ComponentMethodFactory::Create(scriptName);


    // Run Once
    Base::RenderUtils vars{};
    vars.device = vulkanDevice;
    vars.renderPass = &renderPass;
    vars.UBCount = swapchain.imageCount;

    Base::Render renderData{};
    renderData.crlCamera = &cameraConnection;


    // Run script setup function
    for (auto &script: scripts) {
        assert(script.second);
        script.second->createUniformBuffers(vars, renderData, script.second->getType());
    }
    printf("Setup finished\n");
}

void Renderer::deleteScript(const std::string& scriptName){
    auto end = std::remove(scriptNames.begin(), scriptNames.end(), scriptName);
    if (end == scriptNames.end()){
        return;
    }

    scripts.erase(scriptName);

}


void Renderer::generateScriptClasses() {
    std::cout << "Generate script classes" << std::endl;
    std::vector<std::string> classNames;

    /*
    std::string path = Utils::getScriptsPath();


    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::string file = entry.path().generic_string();

        // Delete path from filename
        auto n = file.find(path);
        if (n != std::string::npos)
            file.erase(n, path.length());

        // Ensure we have the header file by looking for .h extension
        std::string extension = file.substr(file.find('.') + 1, file.length());
        if (extension == "h") {
            std::string className = file.substr(0, file.find('.'));
            classNames.emplace_back(className);
        }
    }
    */
    //classNames.emplace_back("Example");
    classNames.emplace_back("LightSource");
    //classNames.emplace_back("VirtualPointCloud");

    // TODO INSERT AND RUN SETUP AS THE ITEMS ARE PLACED INTO THE SCENE
    classNames.emplace_back("DisparityPreview");
    //classNames.emplace_back("DefaultPreview");
    classNames.emplace_back("RightPreview");
    classNames.emplace_back("AuxiliaryPreview");
    classNames.emplace_back("PointCloud");
    //classNames.emplace_back("DecodeVideo");

    /*
    // Also add class names to listbox
    //UIOverlay->uiSettings->listBoxNames = classNames;
    scripts.reserve(classNames.size());
    // Create class instances of scripts
    for (auto &className: classNames) {
        scripts.push_back(ComponentMethodFactory::Create(className));
    }

    // Run Once
    Base::RenderUtils vars{};
    vars.device = vulkanDevice;
    vars.renderPass = &renderPass;
    vars.UBCount = swapchain.imageCount;

    Base::Render renderData{};
    renderData.crlCamera = &cameraConnection;


    // Run script setup function
    for (auto &script: scripts) {
        assert(script);
        script->createUniformBuffers(vars, renderData, script->getType());
    }
    printf("Setup finished\n");
*/
     }

void Renderer::render() {
    VulkanRenderer::prepareFrame();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    Base::Render renderData{};
    renderData.camera = &camera;
    renderData.deltaT = frameTimer;
    renderData.index = currentBuffer;
    renderData.runTime = runTime;

    // Update GUI
    guiManager->update((frameCounter == 0), frameTimer, width, height);

    // Update Camera connection based on Actions from GUI
    cameraConnection->onUIUpdate(guiManager->handles.devices);
    renderData.crlCamera = &cameraConnection;

    // Create/delete scripts after use
    // Run update function on scripts
    for (auto &dev: *guiManager->handles.devices) {

        for (const auto& i : dev.streams){
            if(i.second.playbackStatus == AR_PREVIEW_PLAYING){
                switch (i.second.streamIndex) {
                    case AR_PREVIEW_LEFT:
                        buildScript("DefaultPreview");
                        break;
                    case AR_PREVIEW_RIGHT:
                        break;
                    case AR_PREVIEW_DISPARITY:
                        break;
                    case AR_PREVIEW_AUXILIARY:
                        break;
                    case AR_PREVIEW_VIRTUAL:
                        buildScript("DecodeVideo");
                        break;
                    case AR_PREVIEW_PLAYING:
                        break;
                    case AR_PREVIEW_PAUSED:
                        break;
                    case AR_PREVIEW_STOPPED:
                        break;
                    case AR_PREVIEW_NONE:
                        break;
                    case AR_PREVIEW_RESET:
                        break;
                }
            }

            if (i.second.playbackStatus == AR_PREVIEW_NONE){
                switch (i.second.streamIndex) {
                    case AR_PREVIEW_LEFT:
                        deleteScript("DefaultPreview");
                        break;
                    case AR_PREVIEW_RIGHT:
                        break;
                    case AR_PREVIEW_DISPARITY:
                        break;
                    case AR_PREVIEW_AUXILIARY:
                        break;
                    case AR_PREVIEW_VIRTUAL:
                        deleteScript("DecodeVideo");
                        break;
                    case AR_PREVIEW_PLAYING:
                        break;
                    case AR_PREVIEW_PAUSED:
                        break;
                    case AR_PREVIEW_STOPPED:
                        break;
                    case AR_PREVIEW_NONE:
                        break;
                    case AR_PREVIEW_RESET:
                        break;
                }
            }
        }

    }


    // Run update function on scripts
    for (auto &script: scripts) {
        if (script.second->getType() != ArDisabled) {
            renderData.type = script.second->getType();
            script.second->updateUniformBufferData(renderData);
        }
    }

    // Update general scripts with handle to GUI
    for (auto &script: scripts) {
        script.second->onUIUpdate(guiManager->handles);
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

