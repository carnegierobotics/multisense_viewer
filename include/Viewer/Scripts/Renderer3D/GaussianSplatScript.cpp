//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/GaussianSplatScript.h"

#include <Viewer/ImGui/Widgets.h>

#ifdef WIN32
#endif
void GaussianSplatScript::setup() {
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {
        {
            loadShader("spv/default.vert",
                       VK_SHADER_STAGE_VERTEX_BIT)
        },
        {
            loadShader("spv/default.frag",
                       VK_SHADER_STAGE_FRAGMENT_BIT)
        }
    };
    uniformBuffers.resize(renderUtils.UBCount);
    textures.resize(renderUtils.UBCount);
    // Create texture m_Image if not created
    int texWidth = 1280, texHeight = 720, texChannels = 4;
    auto* pixels = malloc(texWidth * texHeight * 16);
    memset(pixels, 0xFF, texWidth * texHeight * 16);

    handles.resize(textures.size());
    uint32_t bytesPerChannel = sizeof(uint8_t);
    uint32_t bufferSize = texWidth * texHeight * texChannels * bytesPerChannel;
    for (size_t i = 0; i < renderUtils.UBCount; ++i) {
        renderUtils.device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         &uniformBuffers[i], sizeof(VkRender::UBOMatrix));
        uniformBuffers[i].map();

        textures[i].fromBuffer(pixels, bufferSize, VK_FORMAT_R8G8B8A8_UNORM, texWidth, texHeight,
                               renderUtils.device,
                               renderUtils.device->m_TransferQueue, &cudaRequestedMemorySize);
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

    auto camParams = renderData.camera->getFocalParams(texWidth, texHeight);

    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set scale modifier");
    Widgets::make()->slider(WIDGET_PLACEMENT_RENDERER3D, "##scale modifier", &scaleModifier, 0.1f, 5.0f);

    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set scale modifier");
    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Render Gaussians", &renderGaussians);
    /*
    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set camera pos");
    Widgets::make()->vec3(WIDGET_PLACEMENT_RENDERER3D, "##camera pos", &cameraPos);

    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set camera target");
    Widgets::make()->vec3(WIDGET_PLACEMENT_RENDERER3D, "##camera target", &target);

    Widgets::make()->text(WIDGET_PLACEMENT_RENDERER3D, "Set camera up");
    Widgets::make()->vec3(WIDGET_PLACEMENT_RENDERER3D, "##camera up", &up);
    */

    Widgets::make()->fileDialog(WIDGET_PLACEMENT_RENDERER3D, "load model", &plyFileFolder);

}


void GaussianSplatScript::update() {

    mvpMat.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    mvpMat.projection = renderData.camera->matrices.perspective;
    mvpMat.view = renderData.camera->matrices.view;
    memcpy(uniformBuffers[renderData.index].mapped, &mvpMat, sizeof(VkRender::UBOMatrix));

}

void GaussianSplatScript::draw(CommandBuffer* commandBuffer, uint32_t i, bool b) {
    if (b && renderGaussians) {
        vkCmdBindDescriptorSets(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline->data.pipelineLayout, 0,
                                1,
                                &pipeline->data.descriptors[i], 0, nullptr);
        vkCmdBindPipeline(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->data.pipeline);
        const VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(commandBuffer->buffers[i], 0, 1, &mesh->model.vertices.buffer, offsets);
        //vkCmdDraw(commandBuffer->buffers[i], 6, 1, 0, 0);
        if (mesh->model.indexCount) {
            vkCmdBindIndexBuffer(commandBuffer->buffers[i], mesh->model.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer->buffers[i], mesh->model.indexCount, 1, mesh->model.firstIndex, 0, 0);
        }
        else {
        }
    }
}

void GaussianSplatScript::onDestroy() {
    Base::onDestroy();
}
