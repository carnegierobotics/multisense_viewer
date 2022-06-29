// Created by magnus on 9/4/21.
//
//

#include <MultiSense/src/imgui/SideBar.h>
#include <MultiSense/src/imgui/InteractionMenu.h>

#include "Renderer.h"


void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) width / (float) height, 0.001f, 1024.0f);
    camera.rotationSpeed = 0.0f;
    camera.movementSpeed = 0.0f;
    camera.setPosition({0.0f, 0.0f, 0.0f});
    camera.setRotation({0.0f, 0.0f, 0.0f});


    generateScriptClasses();

    // Generate UI from Layers
    guiManager->pushLayer<SideBar>();
    guiManager->pushLayer<InteractionMenu>();

    cameraConnection = new CameraConnection();

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
    clearValues[0].color = {{0.06f, 0.05f, 0.05f, 1.0f}};
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
        (vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));
        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);


        for (auto &script: scripts) {
            if (script->getType() != ArDisabled) {
                script->draw(drawCmdBuffers[i], i);
            }
        }

        guiManager->drawFrame(drawCmdBuffers[i]);

        vkCmdEndRenderPass(drawCmdBuffers[i]);
        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
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
    // TODO: Create a list of renderable classnames
    //classNames.emplace_back("Example");
    //classNames.emplace_back("LightSource");
    //classNames.emplace_back("VirtualPointCloud");
    classNames.emplace_back("PreviewBar");
    classNames.emplace_back("DefaultPreview");
    classNames.emplace_back("PointCloud");
    classNames.emplace_back("DecodeVideo");

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

    // Run script setup function
    for (auto &script: scripts) {
        assert(script);
        script->createUniformBuffers(vars, script->getType());
    }
    printf("Setup finished\n");
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
    renderData.crlCamera = cameraConnection;

    // Run update function on scripts
    for (auto &script: scripts) {
        if (script->getType() != ArDisabled) {
            renderData.type = script->getType();
            script->updateUniformBufferData(renderData);
        }
    }

    // Update general scripts with handle to GUI
    for (auto &script: scripts) {
        script->onUIUpdate(guiManager->handles);
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

