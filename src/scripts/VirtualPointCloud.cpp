#include "VirtualPointCloud.h"

void VirtualPointCloud::setup() {
    model = new MeshModel::Model(1, renderUtils.device);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    virtualCamera = new CRLVirtualCamera(CrlPointCloud);
    virtualCamera->connect(CrlPointCloud);

    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");

    MeshModel::createRenderPipeline(renderUtils, shaders, model, type);
}


void VirtualPointCloud::update() {
    virtualCamera->update(renderData);
    CRLBaseCamera::PointCloudData *meshData = virtualCamera->getStream();
    model->createMesh((MeshModel::Model::Vertex *)meshData->vertices, meshData->vertexCount);

    // Transform pointcloud
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

}

void VirtualPointCloud::onUIUpdate(UISettings *uiSettings) {
}


void VirtualPointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i) {

    MeshModel::draw(commandBuffer, i, this->model);
}
