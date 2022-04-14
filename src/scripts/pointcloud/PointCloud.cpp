//
// Created by magnus on 4/11/22.
//

#include "PointCloud.h"


void PointCloud::setup() {

    model = new CRLCameraModels::Model(renderUtils.device);
    this->drawModel = false;
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");


    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);
}


void PointCloud::update() {

    if (camera == nullptr)
        return;
    camera->update();

    if (camera->stream == nullptr)
        return;

    if (camera->stream->source != crl::multisense::Source_Disparity_Left)
        return;

    auto *imageP = camera->stream;
    cv::Vec3f CloudPt;
    CRLBaseCamera::PointCloudData *meshData = camera->getStream();
    auto *p = (CRLCameraModels::Model::Vertex *) meshData->vertices;
    float min = 1000.0F;
    float max = 0.0F;
    glm::vec4 coords(0.0F, 0.0F, 0.0F, 1.0F);
    auto *disparity = (uint16_t *) imageP->imageDataP;
    int col = 1;
    int row = 1;

    for (int i = 0; i < (imageP->imageLength / 2); ++i) {
        point = (960 * (row - 1)) + (col - 1);
        p[point].pos = glm::vec3(0.0f);

        auto val = (float) disparity[i] / 16.0F;
        if (val != 0) {
            glm::vec4 imgCoords(col, row, 1.0F, 1.0F / val);
            coords = val * glm::transpose(camera->kInverseMatrix) * imgCoords;

            float x = coords.x;
            float y = coords.y;
            float z = coords.z;
            p[point].pos = glm::vec3(x, y, z);

        }
        col++;
        if (col == 961) {
            col = 1;
            row++;
        }
    }

    printf("Min/Max %f %f\n", min, max);



    model->createMesh((CRLCameraModels::Model::Vertex *) meshData->vertices, meshData->vertexCount);
    // Transform pointcloud
    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    //mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
    drawModel = true;
}


void PointCloud::onUIUpdate(UISettings *uiSettings) {
    camera = (CRLPhysicalCamera *) uiSettings->physicalCamera;

    if (camera != nullptr)
        camera->setup();
}


void PointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (this->drawModel)
        CRLCameraModels::draw(commandBuffer, i, this->model);
}