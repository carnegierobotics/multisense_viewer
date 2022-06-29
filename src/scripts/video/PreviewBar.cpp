//
// Created by magnus on 5/8/22.
//

#include "PreviewBar.h"


void PreviewBar::setup() {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    left = new CRLCameraModels::Model(renderUtils.device, CrlImage);
    right = new CRLCameraModels::Model(renderUtils.device, CrlImage);
    // Don't draw it before we create the texture in update()
    left->draw = false;
    right->draw = false;

}


void PreviewBar::update(CameraConnection *conn) {
    auto *camera = conn->camPtr;
    assert(camera != nullptr);

    /*
    if (camera->modeChange){
        createRenderResource(camera);
    }

    if (!conn->camPreviewBar.active)
        return;

    if (camera->play){
        left->setGrayscaleTexture(nullptr);
    }
    */

    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.5f, 1.0f, -2.0f));
    mat.model = glm::rotate(mat.model,glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));


    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;
}


void PreviewBar::onUIUpdate(GuiObjectHandles uiHandle) {
    //camera = (CRLPhysicalCamera *) uiSettings->physicalCamera;

}


void PreviewBar::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (left->draw)
        CRLCameraModels::draw(commandBuffer, i, left);
    if (right->draw)
        CRLCameraModels::draw(commandBuffer, i, right);
}

void PreviewBar::createRenderResource(CRLPhysicalCamera *pCamera) {

    std::string vertexShaderFileName;
    std::string fragmentShaderFileName;


    vertexShaderFileName = "myScene/spv/depth.vert";
    fragmentShaderFileName = "myScene/spv/depth.frag";

    left->prepareTextureImage(1920, 1080, CrlDisparityImage);
    auto *imgData = new ImageData(((float) 1920 / (float) 1080), 1);


    // Load shaders
    VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    // Create quad and store it locally on the GPU
    left->createMeshDeviceLocal((ArEngine::Vertex *) imgData->quad.vertices,
                                imgData->quad.vertexCount, imgData->quad.indices, imgData->quad.indexCount);

    // Create graphics render pipeline
    CRLCameraModels::createRenderPipeline(renderUtils, shaders, left, type);

    left->draw = true;

}
