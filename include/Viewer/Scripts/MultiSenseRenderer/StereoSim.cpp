//
// Created by magnus on 11/7/23.
//

#include "Viewer/Scripts/MultiSenseRenderer/StereoSim.h"

void StereoSim::setup() {

    Log::Logger::getInstance()->info("Setup called from script: {}", GetFactoryName());
    // Load Shaders

    auto computeShaderModule = loadShader("Scene/spv/particle.comp.spv",
                                    VK_SHADER_STAGE_COMPUTE_BIT);
    // Load L/R image

    // Setup compute pipeline
    computeShader.vulkanDevice = renderUtils.device;
    computeShader.createBuffers(renderUtils.UBCount);
    computeShader.createDescriptorSetLayout();
    computeShader.createDescriptorSetPool(renderUtils.UBCount);
    computeShader.createDescriptorSets(renderUtils.UBCount, renderUtils.uniformBuffers);
    topLevelData->computeBuffer = &computeShader.buffer;

    computeShader.createComputePipeline(computeShaderModule);
    // Execute compute pipeline

    // Read Result
}


void StereoSim::update() {

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


void StereoSim::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b) {
        computeShader.recordDrawCommands(commandBuffer, i);
    }


}