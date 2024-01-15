//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/GaussianSplatScript.h"

#include "Viewer/ModelLoaders/GaussianSplat.h"

void GaussianSplatScript::setup() {
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("spv/default.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("spv/default.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};
    uniformBuffers.resize(renderUtils.UBCount);
    textures.resize(renderUtils.UBCount);
    // Create texture m_Image if not created
    int texWidth = 0, texHeight = 0, texChannels = 0;

    auto pixels = stbi_load((Utils::getTexturePath().append("rover.png")).string().c_str(), &texWidth, &texHeight, &texChannels,
                            STBI_rgb_alpha);
    if (!pixels) {
        Log::Logger::getInstance()->error("Failed to load texture image {}",
                                         (Utils::getTexturePath().append("rover.png")).string());
    }

    for (size_t i = 0; i < renderUtils.UBCount; ++i) {
        renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &uniformBuffers[i], sizeof(VkRender::UBOMatrix));
        uniformBuffers[i].map();

        textures[i].fromBuffer(pixels, texWidth * texHeight * 4, VK_FORMAT_R8G8B8A8_UNORM, texWidth, texHeight, renderUtils.device,
                               renderUtils.device->m_TransferQueue);
    }
    stbi_image_free(pixels);
    // setup descriptors and such
    RenderResource::MeshConfig meshConf;
    meshConf.device = renderUtils.device;
    mesh = std::make_unique<RenderResource::Mesh>(meshConf);

    RenderResource::PipelineConfig pConf;
    pConf.device = renderUtils.device;
    pConf.shaders = &shaders;
    pConf.textures = &textures;
    pConf.UboCount = renderUtils.UBCount;
    pConf.msaaSamples = renderUtils.msaaSamples;
    pConf.renderPass = renderUtils.renderPass;
    pConf.ubo = uniformBuffers.data();
    pipeline = std::make_unique<RenderResource::Pipeline>(pConf);

    cudaTexture = std::make_unique<Texture2D>(renderUtils.device);

    splat = std::make_unique<GaussianSplat>(renderUtils.device);
    int device = splat->setCudaVkDevice(renderUtils.vkDeviceUUID);
    cudaStream_t streamToRun;
    checkCudaErrors(cudaStreamCreate(&streamToRun));


}


void GaussianSplatScript::update() {
    mvpMat.model = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, 0.0f));
    mvpMat.projection = renderData.camera->matrices.perspective;
    mvpMat.view = renderData.camera->matrices.view;
    memcpy(uniformBuffers[renderData.index].mapped, &mvpMat, sizeof(VkRender::UBOMatrix));
}

void GaussianSplatScript::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {
    if (b) {
        vkCmdBindDescriptorSets(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->data.pipelineLayout, 0,
                                1,
                                &pipeline->data.descriptors[i], 0, nullptr);
        vkCmdBindPipeline(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->data.pipeline);
        const VkDeviceSize offsets[1] = {0};

        vkCmdBindVertexBuffers(commandBuffer->buffers[i], 0, 1, &mesh->model.vertices.buffer, offsets);
        if (mesh->model.indexCount){
            vkCmdBindIndexBuffer(commandBuffer->buffers[i], mesh->model.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer->buffers[i], mesh->model.indexCount, 1, mesh->model.firstIndex, 0, 0);
        } else {
            vkCmdDraw(commandBuffer->buffers[i], mesh->model.vertexCount, 1, 0, 0);
        }
    }
}

void GaussianSplatScript::onDestroy() {
    Base::onDestroy();
}


