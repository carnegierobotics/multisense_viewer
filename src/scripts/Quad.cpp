//
// Created by magnus on 3/10/22.
//

#include "Quad.h"

void Quad::setup() {
    /**
     * Create and load Mesh elements
     */
    model = new CRLCameraModels::Model(1, renderUtils.device);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/quad.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/quad.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    uint32_t width = 1920;
    uint32_t height = 1200;
    model->prepareTextureImage(width, height, width * height, CrlGrayscaleImage); //
    auto *imgData = new ImageData((1920.0f / 1200.0f), 1);
    model->createMeshDeviceLocal((CRLCameraModels::Model::Vertex *) imgData->quad.vertices, imgData->quad.vertexCount,
                                 imgData->quad.indices, imgData->quad.indexCount);


    CRLCameraModels::createRenderPipeline(renderUtils,  renderUtils.shaders, model, type);
}


void Quad::update() {
    if (camera == nullptr)
        return;
    // If mode changes

    if (camera->modeChange) {
        auto imgConf = camera->getImageConfig();
        model->prepareTextureImage(imgConf.width(), imgConf.height(), imgConf.width() * imgConf.height(), CrlColorImage);
        auto *imgData = new ImageData(((float) imgConf.width() / (float) imgConf.height()), 1);
        model->createMeshDeviceLocal((CRLCameraModels::Model::Vertex *) imgData->quad.vertices, imgData->quad.vertexCount,imgData->quad.indices, imgData->quad.indexCount);
        VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/quad.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/quad.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};

        CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

        camera->modeChange = false;
    }



    int runTimeInMS = (int) (renderData.runTime * 1000);
    if ((runTimeInMS % 50) < 20 && camera->play) {

        model->setVideoTexture(camera->getImage()[crl::multisense::Source_Chroma_Rectified_Aux],camera->getImage()[crl::multisense::Source_Luma_Rectified_Aux] );
        count += 1;
        if (count >= 100)
            count = 1;

    }


    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(5.0f, -5.0f, -5.0f));

    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    mat.model = glm::scale(mat.model, glm::vec3(5.0f, 5.0f, 5.0f));


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


void Quad::onUIUpdate(UISettings *uiSettings) {
    camera = (CRLPhysicalCamera *) uiSettings->sharedData;
}


void Quad::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    CRLCameraModels::draw(commandBuffer, i, model);
}
