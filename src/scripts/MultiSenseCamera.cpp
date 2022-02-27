#include "MultiSenseCamera.h"

void MultiSenseCamera::setup() {

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    virtualCamera = new CRLVirtualCamera();
    virtualCamera->initialize();

    MeshModel::createRenderPipeline(renderUtils, shaders);
}


void MultiSenseCamera::update() {
    virtualCamera->update(renderData);
    CRLBaseCamera::MeshData *meshData = virtualCamera->getStream();
    transferData((MeshModel::Model::Vertex *)meshData->vertices, meshData->vertexCount);

    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (FragShaderParams *) bufferTwoData;
    d2->objectColor =  glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;


}

void MultiSenseCamera::onUIUpdate(UISettings uiSettings) {
}


void MultiSenseCamera::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    MeshModel::draw(commandBuffer, i);

}
