//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/GaussianSplatScript.h"


void GaussianSplatScript::setup() {
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("spv/default.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("spv/default.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};
    uniformBuffers.resize(renderUtils.UBCount);
    for (size_t i = 0; i < renderUtils.UBCount; ++i) {
        renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &uniformBuffers[i], sizeof(VkRender::UBOMatrix));
        uniformBuffers[i].map();
    }

    // setup descriptors and such
    RenderResource::Mesh::Config meshConf;
    meshConf.device = renderUtils.device;
    mesh = RenderResource::Mesh::createMesh(meshConf);
    RenderResource::Pipeline::Config pConf;
    pConf.device = renderUtils.device;
    pConf.shaders = shaders;
    pConf.UboCount = renderUtils.UBCount;
    pConf.msaaSamples = renderUtils.msaaSamples;
    pConf.renderPass = renderUtils.renderPass;
    pConf.ubo = uniformBuffers.data();
    pipeline = RenderResource::Pipeline::createRenderPipeline(pConf);

}


void GaussianSplatScript::update() {
    auto &d = bufferOneData;

    float time = static_cast<float>(renderData.scriptRuntime / 1e9);
    mvpMat.model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f + glm::sin(time * 0.7f), 0.0f, 0.0f));
    mvpMat.projection = renderData.camera->matrices.perspective;
    mvpMat.view = renderData.camera->matrices.view;

    memcpy(uniformBuffers[renderData.index].mapped, &mvpMat, sizeof(VkRender::UBOMatrix));


}

void GaussianSplatScript::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {
    if (b) {

        vkCmdBindDescriptorSets(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipelineLayout, 0,
                                1,
                                &pipeline.descriptors[i], 0, nullptr);
        vkCmdBindPipeline(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
        const VkDeviceSize offsets[1] = {0};

        vkCmdBindVertexBuffers(commandBuffer->buffers[i], 0, 1, &mesh.vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer->buffers[i], mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer->buffers[i], mesh.indexCount, 1, mesh.firstIndex, 0, 0);

    }
}

void GaussianSplatScript::onDestroy() {
    Base::onDestroy();
}
