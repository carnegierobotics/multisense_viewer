//
// Created by magnus on 11/7/23.
//

#include "Viewer/Scripts/MultiSenseRenderer/StereoSim.h"

void StereoSim::setup() {

    Log::Logger::getInstance()->info("Setup called from script: {}", GetFactoryName());
    // Load Shaders

    // Load L/R image
    lastPresentTime = std::chrono::steady_clock::now();

    computeShader.m_VulkanDevice = renderUtils.device;
    computeShader.createBuffers(renderUtils.UBCount);
    computeShader.createTextureTarget(960, 600, renderUtils.UBCount);
    computeShader.createDescriptorSetLayout();
    computeShader.createDescriptorSetPool(renderUtils.UBCount);
    computeShader.createDescriptorSets(renderUtils.UBCount, renderUtils.uniformBuffers);

    computeShader.createComputePipeline(loadShader("Scene/spv/stereo_sim.comp.spv",
                                                   VK_SHADER_STAGE_COMPUTE_BIT));
    // Execute compute pipeline
    topLevelData->compute.computeBuffer = &computeShader.m_Buffer;
    topLevelData->compute.textureComputeTarget = &computeShader.m_TextureComputeTargets;
    topLevelData->compute.valid = true;
    // Read Result
}


void StereoSim::update() {

    if (topLevelData->compute.valid) {
        auto lTex = VkRender::TextureData(CRL_GRAYSCALE_IMAGE, CRL_RESOLUTION_960_600_256);
        auto rTex = VkRender::TextureData(CRL_GRAYSCALE_IMAGE, CRL_RESOLUTION_960_600_256);
        lTex.data = computeShader.m_TextureComputeLeftInput[renderData.index]->m_DataPtr;
        rTex.data = computeShader.m_TextureComputeRightInput[renderData.index]->m_DataPtr;
        bool gotLeft = renderData.crlCamera->getCameraStream("Luma Rectified Left", &lTex, 0);
        bool gotRight = renderData.crlCamera->getCameraStream("Luma Rectified Right", &rTex, 0);
        if (gotLeft && gotRight) {
            computeShader.m_TextureComputeLeftInput[renderData.index]->updateTextureFromBuffer(VK_IMAGE_LAYOUT_GENERAL,
                                                                                               VK_IMAGE_LAYOUT_GENERAL);
            computeShader.m_TextureComputeRightInput[renderData.index]->updateTextureFromBuffer(VK_IMAGE_LAYOUT_GENERAL,
                                                                                                VK_IMAGE_LAYOUT_GENERAL);
        }
    }
    // Update Uniform buffers for this draw call
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(4.0f, -5.0f, -1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->dt = renderData.deltaT;
}

void StereoSim::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &dev: uiHandle->devices) {
        if (dev.state != CRL_STATE_ACTIVE)
            continue;

        auto &preview = dev.win.at(CRL_PREVIEW_ONE);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedResolutionMode;
        // Setup compute pipeline
        if (!topLevelData->compute.valid && currentRes != CRL_RESOLUTION_NONE) {
            uint32_t width, height, depth;
            Utils::cameraResolutionToValue(currentRes, &width, &height, &depth);

        }
    }
}


void StereoSim::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b && topLevelData->compute.valid) {
        computeShader.recordDrawCommands(commandBuffer, i);
    }


}