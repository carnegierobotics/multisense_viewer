//
// Created by magnus on 11/7/23.
//

#include "Viewer/Scripts/MultiSenseRenderer/StereoSim.h"
#include "Viewer/ImGui/Widgets.h"


void StereoSim::setup() {

    Log::Logger::getInstance()->info("Setup called from script: {}", GetFactoryName());
    // Load Shaders

    // Load L/R image
    lastPresentTime = std::chrono::steady_clock::now();
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("spv/stereo_sim_pass_1.comp.spv",
                                                                        VK_SHADER_STAGE_COMPUTE_BIT)}};

    computeShader.m_VulkanDevice = renderUtils.device;
    computeShader.createBuffers(renderUtils.UBCount);
    computeShader.createTextureTarget(960, 600, 256, renderUtils.UBCount);
    computeShader.prepareDescriptors(renderUtils.UBCount, renderUtils.uniformBuffers, shaders);

    // Execute compute pipeline
    topLevelData->compute.computeBuffer = &computeShader.m_Buffer;
    topLevelData->compute.textureComputeTarget = &computeShader.m_TextureDisparityTarget;
    topLevelData->compute.textureComputeTarget3D = &computeShader.m_TextureComputeTargets3D;

    topLevelData->compute.valid = true;


    Widgets::make()->checkbox(WIDGET_PLACEMENT_MULTISENSE_RENDERER, "Enable compute", &enable);

    // Read Result
}


void StereoSim::update() {

    if (topLevelData->compute.valid) {
        auto lTex = VkRender::TextureData(VkRender::CRL_GRAYSCALE_IMAGE,VkRender::CRL_RESOLUTION_960_600_256);
        auto rTex = VkRender::TextureData(VkRender::CRL_GRAYSCALE_IMAGE,VkRender::CRL_RESOLUTION_960_600_256);
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
        if (dev.state !=VkRender::CRL_STATE_ACTIVE)
            continue;

        auto &preview = dev.win.at(VkRender::CRL_PREVIEW_ONE);
        auto &currentRes = dev.channelInfo[preview.selectedRemoteHeadIndex].selectedResolutionMode;
        // Setup compute pipeline
        if (!topLevelData->compute.valid && currentRes !=VkRender::CRL_RESOLUTION_NONE) {
            uint32_t width, height, depth;
            Utils::cameraResolutionToValue(currentRes, &width, &height, &depth);

        }
    }
}


void StereoSim::draw(CommandBuffer * commandBuffer, uint32_t i, bool b) {
    if (b && topLevelData->compute.valid && enable) {
        computeShader.recordDrawCommands(commandBuffer, i);
        commandBuffer->hasWork[i] = true;

    }


}