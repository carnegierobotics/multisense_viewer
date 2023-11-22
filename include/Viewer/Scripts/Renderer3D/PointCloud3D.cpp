//
// Created by magnus on 10/15/23.
//


#include "Viewer/Scripts/Renderer3D/PointCloud3D.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"

void PointCloud3D::setup() {

/*
    uint32_t width = 960, height = 600, depth = 255;

    pc = std::make_unique<PointCloudLoader>(&renderUtils);
    VkPipelineShaderStageCreateInfo vs = loadShader("Scene/spv/pc3D.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("Scene/spv/pc3D.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
               {fs}};


    std::vector<VkRender::Vertex> meshData{};
    meshData.resize(width * height);
    int v = 0;
    // first few rows and cols (20) are discarded in the shader anyway
    for (uint32_t i = 20; i < width - 20; ++i) {
        for (uint32_t j = 20; j < height - 20; ++j) {
            meshData[v].pos = glm::vec3(static_cast<float>(i), static_cast<float>(j), 0.0f);
            meshData[v].uv0 = glm::vec2(1.0f - (static_cast<float>(i) / static_cast<float>(width)),
                                        1.0f - (static_cast<float>(j) / static_cast<float>(height)));
            v++;
        }
    }
    pc->model->createMeshDeviceLocal(meshData);
    pc->model->createTexture(width, height);

    pc->createDescriptorSetLayout();
    pc->createDescriptorPool();
    pc->createDescriptorSets();
    pc->createGraphicsPipeline(shaders);


    Widgets::make()->inputText("Renderer3D", "Draw pc", buf);
*/
}

void PointCloud3D::update() {
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 5.0f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    VkRender::ColorPointCloudParams data{};
    data.instrinsics = renderData.crlCamera->getCameraInfo(0).KColorMat;
    data.extrinsics = renderData.crlCamera->getCameraInfo(0).KColorMatExtrinsic;
    data.hasSampler = renderUtils.device->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
    //memcpy(model->colorPointCloudBuffer.mapped, &data, sizeof(VkRender::ColorPointCloudParams));

    auto *buf = bufferThreeData.get();
    buf->pointSize = 1.8f;


}

void PointCloud3D::draw(CommandBuffer * commandBuffer, uint32_t i, bool b) {
    if (b) {
        //pc->draw(commandBuffer, i);
    }
}
