//
// Created by magnus on 9/4/21.
//

#include <MultiSense/src/tools/Macros.h>
#include "Renderer.h"


void Renderer::prepareRenderer() {
    camera.type = Camera::CameraType::firstperson;
    camera.setPerspective(60.0f, (float) width / (float) height, 0.001f, 1024.0f);
    camera.rotationSpeed = 0.25f;
    camera.movementSpeed = 0.1f;
    camera.setPosition({-6.0f, 7.0f, -5.5f});
    camera.setRotation({-35.0f, -45.0f, 0.0f});

    prepareUniformBuffers();
}


void Renderer::viewChanged() {
    updateUniformBuffers();
}


void Renderer::UIUpdate(UISettings uiSettings) {
    //printf("Index: %d, name: %s\n", uiSettings.getSelectedItem(), uiSettings.listBoxNames[uiSettings.getSelectedItem()].c_str());

    camera.setMovementSpeed(uiSettings.movementSpeed);

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
    clearValues[0].color = {{0.25f, 0.25f, 0.25f, 1.0f}};;
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

        UIOverlay->drawFrame(drawCmdBuffers[i]);

        vkCmdEndRenderPass(drawCmdBuffers[i]);
        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}


void Renderer::render() {
    draw();

}

void Renderer::draw() {
    VulkanRenderer::prepareFrame();
    buildCommandBuffers();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];


    vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]);
    VulkanRenderer::submitFrame();


}


void Renderer::updateUniformBuffers() {
    // Scene
    UBOFrag->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    UBOFrag->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    UBOFrag->lightPos = glm::vec4(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
    UBOFrag->viewPos = camera.viewPos;

    UBOVert->projection = camera.matrices.perspective;
    UBOVert->view = camera.matrices.view;
    UBOVert->model = glm::mat4(1.0f);

}

void Renderer::prepareUniformBuffers() {
    // TODO REMEMBER TO CLEANUP
    UBOVert = new UBOMatrix();
    UBOFrag = new FragShaderParams();

    updateUniformBuffers();
}

